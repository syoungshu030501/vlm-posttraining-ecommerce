"""
Held-out RM evaluation.

Re-uses the training data path / model-loading code so a saved
`reward_head_*.pt` checkpoint can be scored against the holdout split with
per-`pair_strategy` accuracy. Writes a JSON report so it can be diffed across
ablations and tracked from `results/`.

Usage:
    python -m src.stage2_rm.evaluate \
        --model_path /path/to/backbone \
        --reward_head models/rm_ckpt/reward_head_best.pt \
        --holdout_parquet data/preference/preference_holdout.parquet \
        --out_json results/stage2_rm_holdout.json \
        [--head_bias --head_layernorm]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.stage2_rm.dataset import PreferenceDataset, preference_collate_fn
from src.stage2_rm.model import RewardModel
from src.utils.model_loader import load_model_and_processor


@torch.no_grad()
def evaluate_holdout(
    rm: RewardModel,
    loader: DataLoader,
    strategies: list[str],
    device: torch.device,
) -> Dict[str, float | dict]:
    rm.eval()
    n_correct = 0
    n_total = 0
    margin_sum = 0.0
    chosen_len_sum = 0
    rejected_len_sum = 0
    per_strat = defaultdict(lambda: {"correct": 0, "total": 0, "margin_sum": 0.0})

    idx_iter = iter(strategies)
    for batch in loader:
        def to_dev(t):
            return t.to(device) if isinstance(t, torch.Tensor) else t

        chosen_r = rm(
            input_ids=to_dev(batch["chosen_input_ids"]),
            attention_mask=to_dev(batch["chosen_attention_mask"]),
            pixel_values=to_dev(batch.get("chosen_pixel_values")),
            image_grid_thw=to_dev(batch.get("chosen_image_grid_thw")),
            mm_token_type_ids=to_dev(batch.get("chosen_mm_token_type_ids")),
        )
        rejected_r = rm(
            input_ids=to_dev(batch["rejected_input_ids"]),
            attention_mask=to_dev(batch["rejected_attention_mask"]),
            pixel_values=to_dev(batch.get("rejected_pixel_values")),
            image_grid_thw=to_dev(batch.get("rejected_image_grid_thw")),
            mm_token_type_ids=to_dev(batch.get("rejected_mm_token_type_ids")),
        )
        diff = (chosen_r - rejected_r).detach().cpu()
        correct = (diff > 0).float()
        n_correct += int(correct.sum().item())
        n_total += diff.numel()
        margin_sum += float(diff.sum().item())

        chosen_len_sum += int(batch["chosen_attention_mask"].sum().item())
        rejected_len_sum += int(batch["rejected_attention_mask"].sum().item())

        for j in range(diff.numel()):
            try:
                strat = next(idx_iter)
            except StopIteration:
                strat = "unknown"
            bucket = per_strat[strat]
            bucket["total"] += 1
            bucket["correct"] += int(correct[j].item())
            bucket["margin_sum"] += float(diff[j].item())

    overall = {
        "n_pairs": n_total,
        "pair_accuracy": n_correct / max(n_total, 1),
        "mean_margin": margin_sum / max(n_total, 1),
        "len_shortcut": (chosen_len_sum - rejected_len_sum) / max(n_total, 1),
    }
    by_strategy = {
        k: {
            "n": v["total"],
            "accuracy": v["correct"] / max(v["total"], 1),
            "mean_margin": v["margin_sum"] / max(v["total"], 1),
        }
        for k, v in per_strat.items()
    }
    return {**overall, "by_strategy": by_strategy}


def run(args: argparse.Namespace) -> Dict[str, float | dict]:
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
    state = torch.load(args.reward_head, map_location=device)
    rm.reward_head.load_state_dict(state)

    dataset = PreferenceDataset(args.holdout_parquet, processor)

    import pandas as pd
    df = pd.read_parquet(args.holdout_parquet)
    strategies = df["pair_strategy"].astype(str).tolist() if "pair_strategy" in df.columns else ["unknown"] * len(df)

    pad_id = getattr(processor, "pad_token_id", None) or getattr(processor.tokenizer, "pad_token_id", 0) or 0
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda b: preference_collate_fn(b, pad_token_id=pad_id),
    )

    metrics = evaluate_holdout(rm, loader, strategies, device)
    metrics["meta"] = {
        "model_path": args.model_path,
        "reward_head": args.reward_head,
        "holdout_parquet": args.holdout_parquet,
        "head_bias": args.head_bias,
        "head_layernorm": args.head_layernorm,
        "head_mlp": args.head_mlp,
        "head_mlp_hidden": args.head_mlp_hidden,
        "head_dropout": args.head_dropout,
        "strategy_distribution": dict(Counter(strategies)),
    }
    print(json.dumps({k: v for k, v in metrics.items() if k != "meta"}, indent=2, ensure_ascii=False))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.out_json}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--reward_head", required=True)
    parser.add_argument("--holdout_parquet", default="data/preference/preference_holdout.parquet")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--head_bias", action="store_true")
    parser.add_argument("--head_layernorm", action="store_true")
    parser.add_argument("--head_mlp", action="store_true")
    parser.add_argument("--head_mlp_hidden", type=int, default=None)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--out_json", default=None)
    run(parser.parse_args())
