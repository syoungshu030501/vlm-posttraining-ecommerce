"""
Stage 3: Build a *reward-aware* RL training set for FIPO v2.

Reads per-row offline scores produced by `mine_hard_samples.py`, classifies
each sample by a difficulty rule, and writes a new parquet that mixes hard
samples (where SFT-aux struggles → group variance > 0 in GRPO) with a small
fraction of easy samples (anchor distribution / catastrophic-forgetting guard).

Difficulty rules (any one triggers `is_hard=True`):
    * label_wrong         : pred_violation != gt_violation
    * lexicon_contradict  : breakdown.lexicon < 0
    * align_low           : breakdown.semantic_align_sim < align_thresh (0.5)
    * length_bad          : breakdown.reason_length < 0
    * total_low           : reward < total_thresh (default 4.5)
    * parse_failed        : parse_failure or missing_fields hit
The union of these rules forms the hard pool.

Usage:
    python -m src.stage3_fipo.build_rl_train \
        --scores_jsonl data/fipo/sft_aux_train_scores.jsonl \
        --train_parquet data/fipo/train.parquet \
        --out_parquet data/fipo/rl_train.parquet \
        --hard_frac 0.7 --total_size 1500
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _classify(rec: Dict[str, Any], align_thresh: float, total_thresh: float) -> Dict[str, bool]:
    bd = rec.get("breakdown", {}) or {}
    flags: Dict[str, bool] = {
        "parse_failed": ("parse_failure" in bd) or ("missing_fields" in bd),
        "label_wrong": False,
        "lexicon_contradict": (bd.get("lexicon", 0.0) or 0.0) < 0,
        "align_low": (bd.get("semantic_align_sim", 1.0) or 1.0) < align_thresh,
        "length_bad": (bd.get("reason_length", 0.0) or 0.0) < 0,
        "total_low": (rec.get("reward", 5.0) or 5.0) < total_thresh,
    }
    # label_wrong inferred from violation_match contribution sign
    vm = bd.get("violation_match", None)
    if vm is not None:
        flags["label_wrong"] = float(vm) < 0
    flags["is_hard"] = any(v for k, v in flags.items() if k != "label_wrong" or vm is not None)
    return flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_jsonl", default="data/fipo/sft_aux_train_scores.jsonl")
    parser.add_argument("--train_parquet", default="data/fipo/train.parquet")
    parser.add_argument("--out_parquet", default="data/fipo/rl_train.parquet")
    parser.add_argument("--hard_frac", type=float, default=0.7,
                        help="Fraction of the final mix that comes from the hard pool.")
    parser.add_argument("--total_size", type=int, default=1500,
                        help="Target number of rows in the output parquet.")
    parser.add_argument("--align_thresh", type=float, default=0.5)
    parser.add_argument("--total_thresh", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_only", action="store_true",
                        help="Print difficulty breakdown without writing parquet.")
    args = parser.parse_args()

    random.seed(args.seed)

    # ------------------------------------------------------------------
    # Load scores + classify
    # ------------------------------------------------------------------
    scores_path = Path(args.scores_jsonl)
    if not scores_path.exists():
        raise FileNotFoundError(
            f"{scores_path} not found. Run mine_hard_samples.py first."
        )
    records: List[Dict[str, Any]] = []
    with open(scores_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[build] loaded {len(records)} scored records")

    classified = []
    counter = Counter()
    for rec in records:
        flags = _classify(rec, args.align_thresh, args.total_thresh)
        classified.append((rec["index"], flags, rec["reward"]))
        for k, v in flags.items():
            if v:
                counter[k] += 1

    print("[build] difficulty stats (rows triggering each rule):")
    for k, v in counter.most_common():
        print(f"  {k:<20s} {v:5d}  ({100*v/max(len(records),1):.1f}%)")

    if args.report_only:
        return

    hard_idx = [idx for idx, fl, _ in classified if fl["is_hard"]]
    easy_idx = [idx for idx, fl, _ in classified if not fl["is_hard"]]
    print(f"[build] hard pool size: {len(hard_idx)}, easy pool size: {len(easy_idx)}")

    # ------------------------------------------------------------------
    # Mix hard + easy
    # ------------------------------------------------------------------
    target_hard = min(int(args.total_size * args.hard_frac), len(hard_idx))
    target_easy = min(args.total_size - target_hard, len(easy_idx))

    if target_hard < int(args.total_size * args.hard_frac):
        print(
            f"[build] WARN: hard pool ({len(hard_idx)}) smaller than requested "
            f"({int(args.total_size * args.hard_frac)}); upsampling with replacement"
        )
        sampled_hard = random.choices(hard_idx, k=int(args.total_size * args.hard_frac))
    else:
        sampled_hard = random.sample(hard_idx, target_hard)

    sampled_easy = random.sample(easy_idx, target_easy)
    final_idx = sampled_hard + sampled_easy
    random.shuffle(final_idx)
    print(f"[build] final mix: {len(sampled_hard)} hard + {len(sampled_easy)} easy = {len(final_idx)}")

    # ------------------------------------------------------------------
    # Materialize parquet
    # ------------------------------------------------------------------
    src = pd.read_parquet(args.train_parquet)
    src["__idx__"] = src["extra_info"].apply(lambda x: int(x.get("index", -1)))
    by_idx = src.set_index("__idx__")

    missing = [i for i in final_idx if i not in by_idx.index]
    if missing:
        print(f"[build] WARN: {len(missing)} indices not found in train.parquet (skipping)")
        final_idx = [i for i in final_idx if i in by_idx.index]

    out_df = by_idx.loc[final_idx].reset_index(drop=True)
    out_df = out_df[[c for c in out_df.columns if c != "__idx__"]]

    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"[build] wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
