"""
Stage 1: LoRA SFT training.

Loss = CE + supcon_weight * SupCon + triplet_weight * Triplet
  - SupCon (optional): contrastive loss on EOS embeddings, comparing the current
    micro-batch embedding against a memory bank of recent micro-batches. Works
    even with batch_size=1 because the bank supplies the second class.
  - Triplet (optional): hallucination-grounding triplet loss on a secondary
    DataLoader of (anchor_image, positive_attr_text, negative_attr_text). One
    triplet sample is consumed per optimizer step (i.e. every grad_accum
    micro-batches) to keep wall-clock overhead low.

Both auxiliary losses default to 0.05 / 0.03 weights (matching docs); pass
`--supcon_weight 0` / `--triplet_weight 0` to disable.

Resume:
  --resume_lora_from <ckpt_dir>   Load LoRA adapter weights instead of injecting
                                  fresh adapters; useful for picking up a run
                                  whose optimizer state was not persisted.
  --start_epoch N                 Offset the epoch counter in logs/checkpoint
                                  names so the resumed run lines up with the
                                  original run's history.

Usage:
    python -m src.stage1_sft.train \
        --model_path models/pretrained/Qwen3-VL-8B-Instruct \
        --train_parquet data/sft/train.parquet \
        --triplet_parquet data/sft/triplets.parquet \
        --out_dir models/sft_ckpt \
        --epochs 3 --batch_size 1 --grad_accum 16
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.stage1_sft.dataset import SFTDataset, sft_collate_fn
from src.stage1_sft.losses import supcon_loss
from src.stage1_sft.triplet_dataset import TripletDataset, triplet_collate_fn
from src.utils.model_loader import load_model_and_processor, freeze_vision_encoder
from src.utils.tracking import finish_run, init_swanlab, log_metrics

DEFAULT_SUPCON_WEIGHT = 0.05
DEFAULT_TRIPLET_WEIGHT = 0.03


def _eos_embedding(input_ids: torch.Tensor, hidden: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Return the last-EOS hidden state for each sample in the batch.

    Falls back to the last non-padding position when EOS is absent (e.g. when
    the response was truncated).
    """
    is_eos = (input_ids == eos_id)
    has_eos = is_eos.any(dim=1)
    flipped = is_eos.flip(dims=[1])
    eos_pos = input_ids.shape[1] - 1 - flipped.long().argmax(dim=1)
    # When no EOS in row, fall back to last token position.
    fallback = torch.full_like(eos_pos, input_ids.shape[1] - 1)
    eos_pos = torch.where(has_eos, eos_pos, fallback)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), eos_pos]


def _encode_for_text_embed(processor, image, text: str, device: torch.device) -> dict:
    """Tokenise `<image><text>` for triplet anchor/pos/neg embedding."""
    from src.schema import SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        },
    ]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = processor(
        text=[rendered],
        images=[image],
        return_tensors="pt",
        padding=False,
    )
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in enc.items()}


def _embed_image_text(model, processor, image, text: str, device: torch.device) -> torch.Tensor:
    """Forward `(image, text)` through the model and return the last-token hidden state."""
    enc = _encode_for_text_embed(processor, image, text, device)
    fwd_kwargs = dict(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        pixel_values=enc.get("pixel_values"),
        image_grid_thw=enc.get("image_grid_thw"),
        output_hidden_states=True,
    )
    if "mm_token_type_ids" in enc:
        fwd_kwargs["mm_token_type_ids"] = enc["mm_token_type_ids"]
    out = model(**fwd_kwargs)
    last = out.hidden_states[-1]
    seq_len = enc["attention_mask"].sum(dim=1) - 1
    pooled = last[torch.arange(last.shape[0], device=last.device), seq_len]
    return pooled.float().squeeze(0)


def _maybe_load_lora(model, resume_dir: str | None) -> None:
    """Overwrite the freshly-injected LoRA weights with weights from `resume_dir`."""
    if not resume_dir:
        return
    from peft import set_peft_model_state_dict
    print(f"Resuming LoRA adapter weights from {resume_dir}")
    safetensors_path = Path(resume_dir) / "adapter_model.safetensors"
    bin_path = Path(resume_dir) / "adapter_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        state = load_file(str(safetensors_path))
    elif bin_path.exists():
        state = torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No adapter weights found in {resume_dir} (expected adapter_model.safetensors or .bin)"
        )
    res = set_peft_model_state_dict(model, state)
    missing = getattr(res, "missing_keys", None) or getattr(res, "unexpected_keys", None) or []
    if missing:
        print(f"  set_peft_model_state_dict diagnostics: {res}")


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    model, processor = load_model_and_processor(
        args.model_path,
        use_flash_attn=args.flash_attn,
        apply_lora=True,
        device_map="auto" if not args.no_device_map else None,
    )
    freeze_vision_encoder(model)
    _maybe_load_lora(model, args.resume_lora_from)

    if args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        gc_kwargs = {"use_reentrant": False}
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
        print("Enabled gradient checkpointing (use_reentrant=False)")

    dataset = SFTDataset(
        parquet_path=args.train_parquet,
        processor=processor,
        max_prompt_len=args.max_prompt_len,
        max_response_len=args.max_response_len,
    )
    pad_id = getattr(processor, "pad_token_id", None)
    if pad_id is None and hasattr(processor, "tokenizer"):
        pad_id = processor.tokenizer.pad_token_id or 0
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: sft_collate_fn(b, pad_token_id=pad_id or 0),
        num_workers=2,
        pin_memory=True,
    )

    triplet_iter = None
    if args.triplet_parquet and args.triplet_weight > 0 and Path(args.triplet_parquet).exists():
        tri_ds = TripletDataset(args.triplet_parquet, image_dir=args.image_dir)
        tri_loader = DataLoader(
            tri_ds,
            batch_size=1,
            shuffle=True,
            collate_fn=triplet_collate_fn,
            num_workers=1,
            pin_memory=False,
        )
        triplet_iter = _cycle(tri_loader)
        print(f"Triplet loader ready ({len(tri_ds)} triplets), 1 sample per optimizer step")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps = max(len(loader) * args.epochs // args.grad_accum, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    tracker = init_swanlab(
        stage="stage1-sft",
        config={
            **vars(args),
            "supcon_weight": args.supcon_weight,
            "triplet_weight": args.triplet_weight,
            "triplet_enabled": triplet_iter is not None,
        },
        project=args.project_name,
        experiment_name=args.experiment_name,
        tags=["stage1", "sft"],
        description="LoRA SFT training",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eos_id = None
    if hasattr(processor, "tokenizer"):
        eos_id = processor.tokenizer.eos_token_id
    eos_id = eos_id if eos_id is not None else getattr(processor, "eos_token_id", None)

    # Memory bank for SupCon: stores recent (embedding, label) pairs *detached*
    # so we can pull positives/negatives across micro-batches without retaining
    # graph state. Keeps max_bank_size newest entries.
    bank_embeds: list[torch.Tensor] = []
    bank_labels: list[torch.Tensor] = []
    bank_max = max(args.supcon_bank_size, 0)

    global_step = 0
    best_loss = float("inf")
    optimizer.zero_grad()
    should_stop = False

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_sc = 0.0
        epoch_tri = 0.0
        n_batches = 0

        for step, batch in enumerate(loader):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            violation_labels = batch.pop("violation_labels")

            model_kwargs = dict(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                labels=batch["labels"],
                output_hidden_states=True,
            )
            if "mm_token_type_ids" in batch:
                model_kwargs["mm_token_type_ids"] = batch["mm_token_type_ids"]
            outputs = model(**model_kwargs)

            ce_loss = outputs.loss

            # --- SupCon (memory-bank variant; tolerates batch_size=1) ---
            sc_loss = torch.tensor(0.0, device=device)
            if args.supcon_weight > 0 and eos_id is not None and bank_max > 0:
                last_hidden = outputs.hidden_states[-1]
                cur_embed = _eos_embedding(batch["input_ids"], last_hidden, eos_id).float().to(device)
                cur_labels = violation_labels.to(device)
                if bank_embeds:
                    bank_e = torch.stack(bank_embeds, dim=0).to(device)
                    bank_l = torch.stack(bank_labels, dim=0).to(device)
                    pool_e = torch.cat([cur_embed, bank_e], dim=0)
                    pool_l = torch.cat([cur_labels, bank_l], dim=0)
                    if (pool_l == 0).any() and (pool_l == 1).any():
                        sc_loss = supcon_loss(pool_e, pool_l)
                # Refresh the bank with detached embeddings.
                for i in range(cur_embed.shape[0]):
                    bank_embeds.append(cur_embed[i].detach().to(device))
                    bank_labels.append(cur_labels[i].detach().cpu())
                while len(bank_embeds) > bank_max:
                    bank_embeds.pop(0)
                    bank_labels.pop(0)

            total_loss = ce_loss + args.supcon_weight * sc_loss

            (total_loss / args.grad_accum).backward()
            epoch_loss += total_loss.item()
            epoch_ce += ce_loss.item()
            epoch_sc += sc_loss.item() if torch.is_tensor(sc_loss) else float(sc_loss)
            n_batches += 1

            if (step + 1) % args.grad_accum == 0:
                # --- Triplet (one sample per optimizer step) ---
                tri_val = 0.0
                if triplet_iter is not None:
                    try:
                        tri = next(triplet_iter)
                        anchor_emb = _embed_image_text(
                            model, processor, tri["image"], args.triplet_anchor_text, device
                        )
                        pos_emb = _embed_image_text(
                            model, processor, tri["image"], tri["positive_attr"], device
                        )
                        neg_emb = _embed_image_text(
                            model, processor, tri["image"], tri["negative_attr"], device
                        )
                        tri_loss = F.triplet_margin_with_distance_loss(
                            anchor_emb.unsqueeze(0),
                            pos_emb.unsqueeze(0),
                            neg_emb.unsqueeze(0),
                            distance_function=lambda a, b: 1 - F.cosine_similarity(a, b),
                            margin=args.triplet_margin,
                        )
                        (args.triplet_weight * tri_loss).backward()
                        tri_val = tri_loss.item()
                        epoch_tri += tri_val
                    except StopIteration:
                        triplet_iter = None
                    except Exception as exc:
                        print(f"[WARN] triplet step skipped: {exc}")

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    log = {
                        "epoch": epoch,
                        "step": global_step,
                        "ce_loss": ce_loss.item(),
                        "supcon_loss": float(sc_loss.item()) if torch.is_tensor(sc_loss) else float(sc_loss),
                        "triplet_loss": tri_val,
                        "total_loss": total_loss.item(),
                        "lr": scheduler.get_last_lr()[0],
                    }
                    print(log)
                    log_metrics(tracker, log)

                if args.max_steps and global_step >= args.max_steps:
                    should_stop = True
                    break

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_ce = epoch_ce / max(n_batches, 1)
        avg_sc = epoch_sc / max(n_batches, 1)
        avg_tri = epoch_tri / max(global_step, 1)
        print(
            f"Epoch {epoch+1} — avg_total: {avg_loss:.4f} "
            f"(ce={avg_ce:.4f}, supcon={avg_sc:.4f}, triplet={avg_tri:.4f})"
        )
        log_metrics(
            tracker,
            {
                "epoch": epoch,
                "epoch_avg_loss": avg_loss,
                "epoch_avg_ce": avg_ce,
                "epoch_avg_supcon": avg_sc,
                "epoch_avg_triplet": avg_tri,
                "global_step": global_step,
            },
        )

        ckpt_dir = out_dir / f"epoch-{epoch+1}"
        model.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)
        print(f"Saved checkpoint to {ckpt_dir}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_dir = out_dir / "best"
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"New best model (loss={best_loss:.4f}) saved to {best_dir}")

        if should_stop:
            print(f"Reached max_steps={args.max_steps}, stopping early.")
            break

    print("SFT training complete.")
    finish_run(tracker)


def _cycle(loader):
    """Endless iterator over a DataLoader."""
    while True:
        for batch in loader:
            yield batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_parquet", required=True)
    parser.add_argument("--triplet_parquet", default=None,
                        help="Path to triplet parquet for hallucination loss")
    parser.add_argument("--image_dir", default="data/raw/images",
                        help="Image root used to resolve triplet image_path entries")
    parser.add_argument("--out_dir", default="models/sft_ckpt")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs to train *this* invocation (combine with --start_epoch when resuming)")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Epoch index to start counting from (used when resuming a previous run)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_prompt_len", type=int, default=1024)
    parser.add_argument("--max_response_len", type=int, default=512)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--no_device_map", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable activation checkpointing to cut memory ~3-5x at ~20%% speed cost")
    parser.add_argument("--project_name", default="vlm-posttraining")
    parser.add_argument("--experiment_name", default="stage1-sft")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume_lora_from", default=None,
                        help="Directory containing previously-saved LoRA adapter (epoch checkpoint) to resume from")
    parser.add_argument("--supcon_weight", type=float, default=DEFAULT_SUPCON_WEIGHT,
                        help="Weight for SupCon auxiliary loss (0 disables)")
    parser.add_argument("--supcon_bank_size", type=int, default=16,
                        help="Number of past EOS embeddings kept as a memory bank for SupCon")
    parser.add_argument("--triplet_weight", type=float, default=DEFAULT_TRIPLET_WEIGHT,
                        help="Weight for triplet auxiliary loss (0 disables)")
    parser.add_argument("--triplet_margin", type=float, default=0.3)
    parser.add_argument("--triplet_anchor_text", default="请描述商品的视觉属性，包括颜色、材质和款式。",
                        help="Neutral prompt used to obtain the anchor embedding for triplet loss")
    train(parser.parse_args())
