"""
Build a held-out preference set for RM evaluation.

The default `data/preference/preference.parquet` is split into:
  - data/preference/preference_train.parquet
  - data/preference/preference_holdout.parquet

Splitting is grouped by `image_file` and seeded so the result is reproducible
and stays disjoint with future re-runs. `pair_strategy` is recovered from the
source jsonl (parquet currently lacks that column) so we can stratify the
holdout report.

Usage:
    python -m src.stage2_rm.holdout_split \
        --src_parquet data/preference/preference.parquet \
        --src_jsonl   data/preference/preference.jsonl \
        --out_dir     data/preference \
        --holdout_size 200 --seed 42
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def _load_strategy_map(jsonl_path: str) -> dict[str, str]:
    """Read pair_strategy by image_file from the source jsonl."""
    mapping: dict[str, str] = {}
    if not jsonl_path or not Path(jsonl_path).exists():
        return mapping
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            img = row.get("image_file") or row.get("image")
            strat = row.get("pair_strategy", "unknown")
            if img is not None:
                mapping[str(img)] = str(strat)
    return mapping


def split(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.src_parquet)
    print(f"Loaded {len(df)} pairs ({df['image_file'].nunique()} distinct images)")

    strat_map = _load_strategy_map(args.src_jsonl)
    if strat_map:
        df = df.assign(pair_strategy=df["image_file"].map(strat_map).fillna("unknown"))
        print(f"Strategy distribution (full): {dict(Counter(df['pair_strategy']))}")
    else:
        df = df.assign(pair_strategy="unknown")
        print("[WARN] pair_strategy not recovered (jsonl missing); holdout will be flat")

    # Group-by-image split — preference.parquet today is image-unique but we
    # do this anyway in case future re-runs introduce duplicates per image.
    images = df["image_file"].drop_duplicates().tolist()
    rng = np.random.RandomState(args.seed)
    rng.shuffle(images)

    holdout_target = min(args.holdout_size, len(images))
    holdout_images = set(images[:holdout_target])
    train_images = set(images[holdout_target:])

    train_df = df[df["image_file"].isin(train_images)].reset_index(drop=True)
    holdout_df = df[df["image_file"].isin(holdout_images)].reset_index(drop=True)
    assert len(set(train_df["image_file"]) & set(holdout_df["image_file"])) == 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "preference_train.parquet"
    holdout_path = out_dir / "preference_holdout.parquet"
    train_df.to_parquet(train_path, index=False)
    holdout_df.to_parquet(holdout_path, index=False)

    print(f"  train   → {train_path}  ({len(train_df)} pairs)")
    print(f"  holdout → {holdout_path}  ({len(holdout_df)} pairs)")
    print(f"  holdout strategy mix: {dict(Counter(holdout_df['pair_strategy']))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_parquet", default="data/preference/preference.parquet")
    parser.add_argument("--src_jsonl", default="data/preference/preference.jsonl")
    parser.add_argument("--out_dir", default="data/preference")
    parser.add_argument("--holdout_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    split(parser.parse_args())
