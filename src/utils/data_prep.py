"""
Data preparation utilities:
  - JSONL → Parquet conversion (with image embedding)
  - Train/val/test splitting
  - Data validation

Usage:
    python -m src.utils.data_prep \
        --annotation_file data/raw/annotations.jsonl \
        --image_dir data/raw/images \
        --out_dir data/sft \
        --split_ratio 0.8 0.1 0.1
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def strip_markdown_fences(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def normalise_json_text(text: str) -> Optional[str]:
    """Return compact JSON text after stripping markdown fences."""
    cleaned = strip_markdown_fences(text)
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return json.dumps(parsed, ensure_ascii=False)


def load_annotations(path: str) -> List[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def image_to_bytes(image_path: str) -> Optional[bytes]:
    """Read image file and return JPEG bytes for parquet storage."""
    try:
        img = Image.open(image_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    except Exception as e:
        print(f"[WARN] Failed to load {image_path}: {e}")
        return None


def build_sft_parquet(
    annotations: List[dict],
    image_dir: str,
    out_path: str,
    embed_images: bool = True,
) -> pd.DataFrame:
    """
    Convert JSONL annotations to parquet with columns:
      image (bytes or path), prompt (str), response (str), violation (bool)
    """
    rows = []
    for ann in tqdm(annotations, desc="Building parquet"):
        img_file = ann.get("image_file", ann.get("image", ""))
        img_path = os.path.join(image_dir, img_file)
        response = normalise_json_text(ann.get("response", ann.get("label_json", "")))

        if not os.path.exists(img_path):
            print(f"[WARN] Image not found: {img_path}, skipping")
            continue
        if response is None:
            print(f"[WARN] Invalid response JSON for {img_file}, skipping")
            continue

        if embed_images:
            img_data = image_to_bytes(img_path)
            if img_data is None:
                continue
        else:
            img_data = img_path

        row = {
            "image": img_data,
            "prompt": ann.get("description", ann.get("prompt", "")),
            "response": response,
            "violation": bool(
                ann.get("violation", json.loads(response).get("violation", False))
            ),
            "image_file": img_file,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # Drop helper column before saving the full parquet; splitter re-reads df in memory.
    df.drop(columns=["image_file"]).to_parquet(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    return df


def build_preference_parquet(
    annotations: List[dict],
    image_dir: str,
    out_path: str,
    embed_images: bool = True,
) -> pd.DataFrame:
    """
    Convert preference pair annotations to parquet with columns:
      image, prompt, chosen, rejected
    """
    rows = []
    for ann in tqdm(annotations, desc="Building preference parquet"):
        img_file = ann.get("image_file", ann.get("image", ""))
        img_path = os.path.join(image_dir, img_file)
        chosen = normalise_json_text(ann.get("chosen", ""))
        rejected = normalise_json_text(ann.get("rejected", ""))

        if not os.path.exists(img_path):
            continue
        if chosen is None or rejected is None:
            print(f"[WARN] Invalid preference JSON for {img_file}, skipping")
            continue

        if embed_images:
            img_data = image_to_bytes(img_path)
            if img_data is None:
                continue
        else:
            img_data = img_path

        row = {
            "image": img_data,
            "prompt": ann.get("description", ann.get("prompt", "")),
            "chosen": chosen,
            "rejected": rejected,
        }
        if row["chosen"] and row["rejected"]:
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} preference pairs to {out_path}")
    return df


def train_val_test_split(
    df: pd.DataFrame,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    out_dir: str = "data/sft",
    group_col: Optional[str] = "image_file",
) -> None:
    """Split a dataframe and save train/val/test parquet files.

    If `group_col` is present in df, split is done on unique group keys (e.g.
    image filenames) to prevent the same source appearing in multiple splits.
    Falls back to row-level random split when the column is absent.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6

    if group_col and group_col in df.columns:
        groups = df[group_col].drop_duplicates().tolist()
        idx = np.random.RandomState(seed).permutation(len(groups))
        n_train = int(len(groups) * ratios[0])
        n_val = int(len(groups) * ratios[1])
        train_g = set(groups[i] for i in idx[:n_train])
        val_g = set(groups[i] for i in idx[n_train : n_train + n_val])
        test_g = set(groups[i] for i in idx[n_train + n_val :])
        splits = {
            "train": df[df[group_col].isin(train_g)],
            "val": df[df[group_col].isin(val_g)],
            "test": df[df[group_col].isin(test_g)],
        }
    else:
        n = len(df)
        idx = np.random.RandomState(seed).permutation(n)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        splits = {
            "train": df.iloc[idx[:n_train]],
            "val": df.iloc[idx[n_train : n_train + n_val]],
            "test": df.iloc[idx[n_train + n_val :]],
        }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for name, split_df in splits.items():
        path = out_path / f"{name}.parquet"
        # Drop helper column before saving
        save_df = split_df.drop(columns=[group_col]) if group_col in split_df.columns else split_df
        save_df.to_parquet(path, index=False)
        print(f"  {name}: {len(split_df)} samples → {path}")


def validate_parquet(path: str) -> None:
    """Quick sanity check on a parquet file."""
    df = pd.read_parquet(path)
    print(f"\n--- Validation: {path} ---")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    for col in df.columns:
        n_null = df[col].isna().sum()
        if n_null > 0:
            print(f"  [WARN] {col}: {n_null} null values")
    if "violation" in df.columns:
        vc = df["violation"].value_counts()
        print(f"  violation distribution: {dict(vc)}")
    print("  OK" if len(df) > 0 else "  [ERROR] Empty dataframe!")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_annotations(args.annotation_file)
    print(f"Loaded {len(annotations)} annotations")

    if args.mode == "sft":
        parquet_path = str(out_dir / "all.parquet")
        df = build_sft_parquet(
            annotations, args.image_dir, parquet_path,
            embed_images=not args.no_embed,
        )
    elif args.mode == "preference":
        parquet_path = str(out_dir / "preference.parquet")
        df = build_preference_parquet(
            annotations, args.image_dir, parquet_path,
            embed_images=not args.no_embed,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    if args.split and args.mode == "sft":
        ratios = tuple(float(x) for x in args.split_ratio)
        train_val_test_split(df, ratios, out_dir=str(out_dir))

    validate_parquet(parquet_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_file", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--out_dir", default="data/sft")
    parser.add_argument("--mode", choices=["sft", "preference"], default="sft")
    parser.add_argument("--split", action="store_true", help="Split into train/val/test")
    parser.add_argument("--split_ratio", nargs=3, default=["0.8", "0.1", "0.1"])
    parser.add_argument("--no_embed", action="store_true", help="Store image paths instead of bytes")
    run(parser.parse_args())
