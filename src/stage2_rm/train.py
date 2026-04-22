"""
Stage 2: Reward model training with Bradley-Terry loss.

Backbone: frozen (auto-detected Qwen VL family). Only the scalar head is trained.

Usage:
    python -m src.stage2_rm.train \
        --model_path models/pretrained/Qwen2.5-VL-7B-Instruct \
        --train_parquet data/preference/preference.parquet \
        --out_dir models/rm_ckpt \
        --epochs 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.stage2_rm.dataset import PreferenceDataset, preference_collate_fn
from src.stage2_rm.evaluate import evaluate_holdout
from src.stage2_rm.model import RewardModel, bradley_terry_loss
from src.utils.model_loader import load_model_and_processor
from src.utils.tracking import finish_run, init_swanlab, log_metrics


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, processor = load_model_and_processor(
        args.model_path,
        apply_lora=False,
        use_flash_attn=args.flash_attn,
        device_map=None,
    )
    rm = RewardModel(
        model,
        head_bias=args.head_bias,
        head_layernorm=args.head_layernorm,
        head_mlp=args.head_mlp,
        head_mlp_hidden=args.head_mlp_hidden,
        head_dropout=args.head_dropout,
    ).to(device)

    dataset = PreferenceDataset(args.train_parquet, processor)
    pad_id = getattr(processor, "pad_token_id", None)
    if pad_id is None and hasattr(processor, "tokenizer"):
        pad_id = processor.tokenizer.pad_token_id or 0
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda b: preference_collate_fn(b, pad_token_id=pad_id or 0),
    )

    holdout_loader = None
    holdout_strategies: list[str] = []
    if args.holdout_parquet and Path(args.holdout_parquet).exists():
        import pandas as pd
        holdout_df = pd.read_parquet(args.holdout_parquet)
        holdout_strategies = (
            holdout_df["pair_strategy"].astype(str).tolist()
            if "pair_strategy" in holdout_df.columns
            else ["unknown"] * len(holdout_df)
        )
        holdout_ds = PreferenceDataset(args.holdout_parquet, processor)
        holdout_loader = DataLoader(
            holdout_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=lambda b: preference_collate_fn(b, pad_token_id=pad_id or 0),
        )
        print(f"Loaded holdout: {len(holdout_ds)} pairs from {args.holdout_parquet}")

    optimizer = torch.optim.AdamW(rm.reward_head.parameters(), lr=args.lr)
    tracker = init_swanlab(
        stage="stage2-rm",
        config=vars(args),
        project=args.project_name,
        experiment_name=args.experiment_name,
        tags=["stage2", "rm"],
        description="Reward model training",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    global_step = 0
    should_stop = False

    for epoch in range(args.epochs):
        rm.train()
        total_loss = 0.0
        n_correct = 0
        n_total = 0
        n_steps = 0

        for step, batch in enumerate(loader):
            def to_device(t):
                return t.to(device) if isinstance(t, torch.Tensor) else t

            chosen_r = rm(
                input_ids=to_device(batch["chosen_input_ids"]),
                attention_mask=to_device(batch["chosen_attention_mask"]),
                pixel_values=to_device(batch.get("chosen_pixel_values")),
                image_grid_thw=to_device(batch.get("chosen_image_grid_thw")),
                mm_token_type_ids=to_device(batch.get("chosen_mm_token_type_ids")),
            )
            rejected_r = rm(
                input_ids=to_device(batch["rejected_input_ids"]),
                attention_mask=to_device(batch["rejected_attention_mask"]),
                pixel_values=to_device(batch.get("rejected_pixel_values")),
                image_grid_thw=to_device(batch.get("rejected_image_grid_thw")),
                mm_token_type_ids=to_device(batch.get("rejected_mm_token_type_ids")),
            )

            loss = bradley_terry_loss(chosen_r, rejected_r)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            global_step += 1
            n_steps += 1

            # Accuracy: chosen should score higher than rejected
            n_correct += (chosen_r > rejected_r).sum().item()
            n_total += chosen_r.shape[0]

            log_metrics(
                tracker,
                {
                    "epoch": epoch,
                    "step": global_step,
                    "loss": loss.item(),
                    "running_accuracy": n_correct / max(n_total, 1),
                },
            )

            if (step + 1) % 20 == 0:
                acc = n_correct / max(n_total, 1)
                print(f"Epoch {epoch} step {step+1}: loss={loss.item():.4f}, acc={acc:.3f}")

            if args.max_steps and global_step >= args.max_steps:
                should_stop = True
                break

        avg_loss = total_loss / max(n_steps, 1)
        acc = n_correct / max(n_total, 1)
        print(f"Epoch {epoch} — avg_loss: {avg_loss:.4f}, accuracy: {acc:.3f}")
        log_metrics(
            tracker,
            {
                "epoch": epoch,
                "epoch_avg_loss": avg_loss,
                "epoch_accuracy": acc,
            },
        )

        torch.save(rm.reward_head.state_dict(), out_dir / f"reward_head_epoch{epoch+1}.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(rm.reward_head.state_dict(), out_dir / "reward_head_best.pt")
            print(f"  New best RM (loss={best_loss:.4f})")

        if holdout_loader is not None:
            ho_metrics = evaluate_holdout(rm, holdout_loader, holdout_strategies, device)
            print(
                f"  holdout @ epoch {epoch}: acc={ho_metrics['pair_accuracy']:.4f}  "
                f"margin={ho_metrics['mean_margin']:.4f}  len_diff={ho_metrics['len_shortcut']:.2f}"
            )
            ho_log = {
                "epoch": epoch,
                "holdout_pair_accuracy": ho_metrics["pair_accuracy"],
                "holdout_mean_margin": ho_metrics["mean_margin"],
                "holdout_len_shortcut": ho_metrics["len_shortcut"],
            }
            for strat, vals in ho_metrics["by_strategy"].items():
                ho_log[f"holdout_acc/{strat}"] = vals["accuracy"]
                ho_log[f"holdout_n/{strat}"] = vals["n"]
            log_metrics(tracker, ho_log)
            rm.train()

        if should_stop:
            print(f"Reached max_steps={args.max_steps}, stopping early.")
            break

    print("RM training complete.")
    finish_run(tracker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_parquet", required=True)
    parser.add_argument("--out_dir", default="models/rm_ckpt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--project_name", default="vlm-posttraining")
    parser.add_argument("--experiment_name", default="stage2-rm")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument(
        "--head_bias",
        action="store_true",
        help="Add bias term to the scalar reward head (default off, preserves v0 behaviour).",
    )
    parser.add_argument(
        "--head_layernorm",
        action="store_true",
        help="Insert LayerNorm before the scalar reward head (default off).",
    )
    parser.add_argument(
        "--head_mlp",
        action="store_true",
        help="Use 2-layer MLP head (Linear→GELU→[Dropout]→Linear) instead of single Linear.",
    )
    parser.add_argument(
        "--head_mlp_hidden",
        type=int,
        default=None,
        help="Hidden dim for MLP head (default: hidden_size//2 ≈ 1024).",
    )
    parser.add_argument(
        "--head_dropout",
        type=float,
        default=0.0,
        help="Dropout inside MLP head (default 0.0; only active when --head_mlp).",
    )
    parser.add_argument(
        "--holdout_parquet",
        default=None,
        help="Optional held-out preference parquet; when given, run pair-accuracy eval per epoch.",
    )
    train(parser.parse_args())
