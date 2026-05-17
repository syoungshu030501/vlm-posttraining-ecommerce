"""
Convert SFT train/val parquet -> verl-format FIPO RL parquet.

verl multimodal RL schema (matches examples/data_preprocess/geo3k.py):
    data_source:    str
    prompt:         list[{"role": "user"|"system", "content": str | list}]
    images:         list[PIL.Image | bytes]
    reward_model:   {"style": "rule", "ground_truth": str}
    extra_info:     {"split": str, "index": int, ...}

We feed:
- system: SYSTEM_PROMPT (same as SFT, ensures consistency)
- user content: <image> + product title (Qwen3-VL multimodal format)
- ground_truth: chosen JSON string (used by our reward_manager)

Usage:
    cd /home/young/VLM-posttraining
    /home/young/miniconda3/envs/<fipo_env>/bin/python -m src.stage3_fipo.prepare_fipo_data \
        --in_train data/sft/train.parquet \
        --in_val   data/sft/val.parquet \
        --out_dir  data/fipo \
        --max_train 2000  # optional cap, FIPO needs fewer prompts than SFT
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from src.schema import SYSTEM_PROMPT


def _ensure_image_object(blob):
    """verl/Qwen processor accepts both PIL.Image and bytes; we standardise to PIL bytes (smaller)."""
    if isinstance(blob, (bytes, bytearray)):
        return {"bytes": bytes(blob)}
    if isinstance(blob, Image.Image):
        buf = io.BytesIO()
        blob.save(buf, format="JPEG", quality=92)
        return {"bytes": buf.getvalue()}
    raise TypeError(f"Unsupported image type: {type(blob)}")


def _row_to_verl(row, split: str, idx: int) -> dict:
    title = str(row["prompt"]).strip() or "无商品描述"
    chosen_json = str(row["response"]).strip()
    image_blob = _ensure_image_object(row["image"])

    # verl/RLDataset._build_messages expects content as a *string* with
    # `<image>` / `<video>` placeholders (it re.splits on them). If we pass
    # list-of-dicts (OpenAI/Qwen style), verl skips the message entirely and
    # asserts image_offset==len(images), failing as 0 != 1.
    user_content = f"<image>商品描述：{title}"
    return {
        "data_source": "vlm_audit",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "images": [image_blob],
        "ability": "compliance_audit",
        "reward_model": {"style": "rule", "ground_truth": chosen_json},
        "extra_info": {
            "split": split,
            "index": idx,
            "image_file": str(row.get("image_file", "")),
            "violation": bool(row.get("violation", False)),
        },
    }


def convert(in_parquet: Path, out_parquet: Path, split: str, max_rows: int | None) -> int:
    df = pd.read_parquet(in_parquet)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    rows = [_row_to_verl(row, split, i) for i, (_, row) in enumerate(df.iterrows())]
    out_df = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_parquet, index=False)
    print(f"[prepare_fipo_data] {in_parquet} -> {out_parquet}  rows={len(out_df)}")
    return len(out_df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_train", type=Path, default=Path("data/sft/train.parquet"))
    ap.add_argument("--in_val", type=Path, default=Path("data/sft/val.parquet"))
    ap.add_argument("--out_dir", type=Path, default=Path("data/fipo"))
    ap.add_argument("--max_train", type=int, default=2000,
                    help="cap train rows (FIPO needs fewer prompts than SFT). 0=no cap")
    ap.add_argument("--max_val", type=int, default=200)
    args = ap.parse_args()

    cap_t = args.max_train if args.max_train > 0 else None
    cap_v = args.max_val if args.max_val > 0 else None

    convert(args.in_train, args.out_dir / "train.parquet", "train", cap_t)
    convert(args.in_val, args.out_dir / "val.parquet", "val", cap_v)
    print("done.")


if __name__ == "__main__":
    main()
