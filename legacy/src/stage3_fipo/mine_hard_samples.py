"""
Stage 3: Mine hard samples for FIPO RL training.

Motivation
----------
The first FIPO run hit *reward saturation*: SFT-aux already scores 4.71/5 on the
train distribution, so GRPO group variance ≈ 0 and 8/9 steps had `actor/loss=0,
grad_norm=0`. The model literally has no informative samples to learn from on
the current train split.

This script runs the SFT-aux model over `data/fipo/train.parquet` (greedy
decode), scores each rollout offline with `reward_fn v2` (semantic alignment +
violation match + lexicon + length + format), and writes a per-row JSONL with
the full reward breakdown. Hard samples can then be filtered downstream by
`build_rl_train.py`.

Usage
-----
    python -m src.stage3_fipo.mine_hard_samples \
        --model_path models/sft_aux_merged \
        --train_parquet data/fipo/train.parquet \
        --out_jsonl data/fipo/sft_aux_train_scores.jsonl \
        --batch_size 4 --max_new_tokens 512 \
        [--limit N]   # for quick smoke test
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from PIL import Image

# allow `python -m src.stage3_fipo.mine_hard_samples` from project root
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.stage3_fipo.reward_fn import compute_reward, make_encoder  # noqa: E402
from src.utils.model_loader import load_model_and_processor  # noqa: E402


SYSTEM_PROMPT = (
    "你是一位专业的电商商品合规审核员。分析给定的商品图片和描述，"
    "输出一个JSON对象，包含以下字段：\n"
    '- "category" (str): 商品品类\n'
    '- "attributes" (dict): 从图片中提取的关键视觉属性（如颜色、材质、款式等）\n'
    '- "violation" (bool): 该商品是否违反平台规则\n'
    '- "reason" (str): 简明审核理由，必须引用具体的视觉属性作为证据\n'
    "只输出合法JSON，不要用markdown代码块或其他格式包裹。"
)


def _load_image(images_field: Any) -> Image.Image:
    """Decode the parquet `images` column into a PIL.Image (RGB)."""
    rec = images_field[0]
    if isinstance(rec, dict) and "bytes" in rec:
        img = Image.open(io.BytesIO(rec["bytes"]))
    elif isinstance(rec, bytes):
        img = Image.open(io.BytesIO(rec))
    else:
        raise TypeError(f"Unsupported image record: {type(rec)}")
    return img.convert("RGB")


def _row_to_messages(image: Image.Image, user_text: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def _user_text_from_row(prompt_field: Any) -> str:
    """Reconstruct user description from the parquet `prompt` column.

    `prompt` is a list of dicts with role/content entries. The user content was
    stored as a single string with `<image>` placeholder; we strip the
    placeholder so the original textual description survives.
    """
    user_msg = next((m for m in prompt_field if m.get("role") == "user"), None)
    if user_msg is None:
        return ""
    text = user_msg.get("content", "")
    if isinstance(text, list):
        text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
    return str(text).replace("<image>", "").strip()


@torch.inference_mode()
def _generate_batch(
    model,
    processor,
    images: List[Image.Image],
    user_texts: List[str],
    *,
    max_new_tokens: int,
) -> List[str]:
    texts = [
        processor.apply_chat_template(
            _row_to_messages(img, ut),
            tokenize=False,
            add_generation_prompt=True,
        )
        for img, ut in zip(images, user_texts)
    ]
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    )
    # strip the prompt prefix per-sample (left padding makes input lengths uniform)
    prompt_len = inputs["input_ids"].shape[1]
    gens = out[:, prompt_len:]
    return processor.tokenizer.batch_decode(gens, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/sft_aux_merged")
    parser.add_argument("--train_parquet", default="data/fipo/train.parquet")
    parser.add_argument(
        "--out_jsonl", default="data/fipo/sft_aux_train_scores.jsonl"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--limit", type=int, default=0,
        help="If >0, only score the first N rows (smoke test).",
    )
    parser.add_argument(
        "--encoder_model", default="BAAI/bge-small-zh-v1.5",
        help="Sentence encoder for semantic alignment (cached locally).",
    )
    parser.add_argument(
        "--encoder_device", default="cpu",
        help="cpu is usually fine; the policy model owns the GPU.",
    )
    parser.add_argument("--flash_attn", action="store_true", default=True)
    parser.add_argument("--no_flash_attn", dest="flash_attn", action="store_false")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip rows whose `index` already appears in the output JSONL.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = pd.read_parquet(args.train_parquet)
    if args.limit > 0:
        df = df.head(args.limit)
    print(f"[mine] loaded {len(df)} rows from {args.train_parquet}")

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[int] = set()
    if args.resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    seen.add(int(json.loads(line)["index"]))
                except Exception:
                    pass
        print(f"[mine] resume: {len(seen)} rows already scored, skipping")

    # ------------------------------------------------------------------
    # Load policy + encoder
    # ------------------------------------------------------------------
    print(f"[mine] loading policy from {args.model_path}")
    model, processor = load_model_and_processor(
        args.model_path,
        apply_lora=False,
        use_flash_attn=args.flash_attn,
    )
    model.eval()
    # Left padding is required for batched generate when prompts have different
    # lengths; otherwise positional ids get clobbered.
    processor.tokenizer.padding_side = "left"

    print(f"[mine] loading sentence encoder ({args.encoder_model})")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    encoder = make_encoder(args.encoder_model, device=args.encoder_device)

    # ------------------------------------------------------------------
    # Inference + scoring loop
    # ------------------------------------------------------------------
    fp = open(out_path, "a", buffering=1)
    n_done = len(seen)
    t0 = time.time()

    pending_rows: List[pd.Series] = []
    pending_imgs: List[Image.Image] = []
    pending_texts: List[str] = []

    def flush() -> None:
        nonlocal n_done, pending_rows, pending_imgs, pending_texts
        if not pending_rows:
            return
        gens = _generate_batch(
            model, processor, pending_imgs, pending_texts,
            max_new_tokens=args.max_new_tokens,
        )
        for row, gen in zip(pending_rows, gens):
            gt_str = row["reward_model"]["ground_truth"]
            try:
                gt = json.loads(gt_str)
            except Exception:
                gt = {"violation": bool(row["extra_info"].get("violation", False))}
            reward, breakdown = compute_reward(
                gen,
                gt_annotation=gt,
                rm_score=None,
                encoder=encoder,
                return_breakdown=True,
            )
            rec = {
                "index": int(row["extra_info"]["index"]),
                "image_file": row["extra_info"].get("image_file"),
                "gt_violation": bool(gt.get("violation", False)),
                "generation": gen,
                "reward": float(reward),
                "breakdown": {k: float(v) for k, v in breakdown.items()},
            }
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += 1
        elapsed = time.time() - t0
        print(
            f"[mine] {n_done}/{len(df)}  "
            f"elapsed={elapsed:.0f}s  "
            f"rate={n_done / max(elapsed, 1e-6):.2f} samp/s"
        )
        pending_rows, pending_imgs, pending_texts = [], [], []

    for _, row in df.iterrows():
        idx = int(row["extra_info"]["index"])
        if idx in seen:
            continue
        try:
            img = _load_image(row["images"])
        except Exception as exc:
            print(f"[mine] skip row {idx}: failed to load image ({exc})")
            continue
        pending_rows.append(row)
        pending_imgs.append(img)
        pending_texts.append(_user_text_from_row(row["prompt"]))
        if len(pending_rows) >= args.batch_size:
            flush()
    flush()
    fp.close()
    print(f"[mine] done. wrote {n_done} records to {out_path}")


if __name__ == "__main__":
    main()
