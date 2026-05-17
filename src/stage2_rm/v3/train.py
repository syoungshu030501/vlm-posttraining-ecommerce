"""
Stage 2 v3 training: field-wise PRM with 5 heads + joint loss.

Backbone is frozen (sft_aux_merged). Only the trunk + 5 heads are trained.

Compared to v2 (single scalar Bradley-Terry):
    - input is pointwise (single response with field labels), NOT preference pairs
    - 5 independent supervision signals → no longer needs SupCon/Triplet
    - holdout still supports pairwise eval for sanity comparison vs v2-aux

Usage:
    python -m src.stage2_rm.v3.train \\
        --model_path models/sft_aux_merged \\
        --train_parquet data/preference_v3/preference_v3_train.parquet \\
        --holdout_parquet data/preference_v3/preference_v3_holdout.parquet \\
        --out_dir models/rm_v3_aux_ckpt \\
        --epochs 3 --batch_size 1 --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.stage2_rm.v3.dataset import (
    FieldWisePreferenceDataset,
    field_wise_collate,
    field_wise_pair_collate,
)
from src.stage2_rm.v3.model import (
    FieldWiseLossWeights,
    FieldWisePRM,
    field_wise_loss,
    field_wise_pair_metrics,
)
from src.utils.model_loader import load_model_and_processor

try:
    from src.utils.tracking import finish_run, init_swanlab, log_metrics
except ImportError:
    init_swanlab = None
    log_metrics = None
    finish_run = None


def _to_device(t, device):
    if isinstance(t, torch.Tensor):
        return t.to(device)
    if isinstance(t, dict):
        return {k: _to_device(v, device) for k, v in t.items()}
    return t


def evaluate_pairwise(rm, loader, device) -> dict:
    """Sanity eval against v2-aux RM: per-head pair-acc / margin on chosen vs rejected."""
    rm.eval()
    agg: dict[str, list[float]] = {}
    n = 0
    with torch.no_grad():
        for batch in loader:
            chosen_out = rm(
                input_ids=_to_device(batch["chosen_input_ids"], device),
                attention_mask=_to_device(batch["chosen_attention_mask"], device),
                pixel_values=_to_device(batch.get("chosen_pixel_values"), device),
                image_grid_thw=_to_device(batch.get("chosen_image_grid_thw"), device),
                mm_token_type_ids=_to_device(batch.get("chosen_mm_token_type_ids"), device),
            )
            rejected_out = rm(
                input_ids=_to_device(batch["rejected_input_ids"], device),
                attention_mask=_to_device(batch["rejected_attention_mask"], device),
                pixel_values=_to_device(batch.get("rejected_pixel_values"), device),
                image_grid_thw=_to_device(batch.get("rejected_image_grid_thw"), device),
                mm_token_type_ids=_to_device(batch.get("rejected_mm_token_type_ids"), device),
            )
            m = field_wise_pair_metrics(chosen_out, rejected_out)
            for k, v in m.items():
                agg.setdefault(k, []).append(v)
            n += chosen_out.violation_prob_logit.shape[0]
    return {k: float(sum(v) / max(len(v), 1)) for k, v in agg.items()} | {"holdout_n": n}


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Backbone — frozen, no LoRA
    model, processor = load_model_and_processor(
        args.model_path,
        apply_lora=False,
        use_flash_attn=args.flash_attn,
        device_map=None,
    )

    rm = FieldWisePRM(model, head_dropout=args.head_dropout).to(device)
    print(f"FieldWisePRM init: hidden={rm.hidden_size}, trunk={rm.trunk_dim}")
    n_trainable = sum(p.numel() for p in rm.parameters() if p.requires_grad)
    print(f"Trainable params (heads + trunk): {n_trainable / 1e6:.2f}M")

    # Datasets
    train_ds = FieldWisePreferenceDataset(
        args.train_parquet, processor, mode="pointwise",
    )
    pad_id = getattr(processor, "pad_token_id", None)
    if pad_id is None and hasattr(processor, "tokenizer"):
        pad_id = processor.tokenizer.pad_token_id or 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda b: field_wise_collate(b, pad_token_id=pad_id or 0),
    )

    holdout_loader = None
    if args.holdout_parquet and Path(args.holdout_parquet).exists():
        holdout_ds = FieldWisePreferenceDataset(
            args.holdout_parquet, processor, mode="pairwise",
        )
        holdout_loader = DataLoader(
            holdout_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=lambda b: field_wise_pair_collate(b, pad_token_id=pad_id or 0),
        )
        print(f"Holdout pairs: {len(holdout_ds)}")

    weights = FieldWiseLossWeights(
        category=args.w_category,
        attributes=args.w_attributes,
        violation_prob=args.w_violation_prob,
        violation_type=args.w_violation_type,
        reason_align=args.w_reason_align,
    )

    optimizer = torch.optim.AdamW(
        [p for p in rm.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    tracker = None
    if init_swanlab is not None and args.use_swanlab:
        tracker = init_swanlab(
            stage="stage2-prm-v3",
            config=vars(args),
            project=args.project_name,
            experiment_name=args.experiment_name or "stage2-prm-v3",
            tags=["stage2", "prm", "v3", "field-wise"],
            description="Field-wise PRM with 5 heads",
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    best_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        rm.train()
        running: dict[str, float] = {}
        n_steps = 0

        for step, batch in enumerate(train_loader):
            inputs = dict(
                input_ids=_to_device(batch["input_ids"], device),
                attention_mask=_to_device(batch["attention_mask"], device),
                pixel_values=_to_device(batch.get("pixel_values"), device),
                image_grid_thw=_to_device(batch.get("image_grid_thw"), device),
                mm_token_type_ids=_to_device(batch.get("mm_token_type_ids"), device),
            )
            out = rm(**inputs)

            labels = _to_device(batch["labels"], device)
            loss, metrics = field_wise_loss(out, labels, weights=weights)

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in rm.parameters() if p.requires_grad],
                    args.grad_clip,
                )
            optimizer.step()
            optimizer.zero_grad()

            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v
            n_steps += 1
            global_step += 1

            if log_metrics is not None and tracker is not None:
                log_metrics(tracker, {"step": global_step, "epoch": epoch, **metrics})

            if (step + 1) % args.log_every == 0:
                msg = f"E{epoch} S{step+1}: " + " ".join(
                    f"{k.split('/')[-1]}={metrics[k]:.4f}"
                    for k in ("loss/total", "loss/category", "loss/violation_prob",
                              "loss/violation_type", "loss/reason_align", "loss/attributes")
                    if k in metrics
                )
                print(msg)

            if args.max_steps and global_step >= args.max_steps:
                break

        # epoch summary
        avg = {k: v / max(n_steps, 1) for k, v in running.items() if k.startswith("loss/")}
        print(f"\nEpoch {epoch} avg: " + " ".join(
            f"{k.split('/')[-1]}={v:.4f}" for k, v in avg.items()
        ))

        # save head + trunk only
        ckpt = {
            "trunk": rm.trunk.state_dict(),
            "category_head": rm.category_head.state_dict(),
            "violation_prob_head": rm.violation_prob_head.state_dict(),
            "violation_type_head": rm.violation_type_head.state_dict(),
            "reason_align_head": rm.reason_align_head.state_dict(),
            "attributes_token_head": rm.attributes_token_head.state_dict(),
        }
        torch.save(ckpt, out_dir / f"prm_v3_epoch{epoch+1}.pt")
        avg_total = avg.get("loss/total", float("inf"))
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(ckpt, out_dir / "prm_v3_best.pt")
            print(f"  new best PRM v3 (loss/total={best_loss:.4f})")

        if holdout_loader is not None:
            ho = evaluate_pairwise(rm, holdout_loader, device)
            print("  holdout pair metrics:")
            for k, v in ho.items():
                print(f"    {k}: {v}")
            if log_metrics is not None and tracker is not None:
                log_metrics(tracker, {"epoch": epoch, **{f"holdout/{k}": v for k, v in ho.items()}})
            rm.train()

        if args.max_steps and global_step >= args.max_steps:
            print(f"reached max_steps={args.max_steps}, stopping")
            break

    print("\nField-wise PRM training complete.")
    if finish_run is not None and tracker is not None:
        finish_run(tracker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="HF model path (frozen backbone, e.g. models/sft_aux_merged)")
    parser.add_argument("--train_parquet", required=True)
    parser.add_argument("--holdout_parquet", default=None)
    parser.add_argument("--out_dir", default="models/rm_v3_aux_ckpt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=20)
    # loss weights
    parser.add_argument("--w_category", type=float, default=1.0)
    parser.add_argument("--w_attributes", type=float, default=0.5)
    parser.add_argument("--w_violation_prob", type=float, default=1.0)
    parser.add_argument("--w_violation_type", type=float, default=1.0)
    parser.add_argument("--w_reason_align", type=float, default=0.3)
    # tracking
    parser.add_argument("--use_swanlab", action="store_true")
    parser.add_argument("--project_name", default="vlm-posttraining")
    parser.add_argument("--experiment_name", default="stage2-prm-v3")
    train(parser.parse_args())
