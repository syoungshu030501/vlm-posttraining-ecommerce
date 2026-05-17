"""
Calibrate the Stage 4 RAG confidence threshold against the held-out val set.

For each row in ``data/fipo/val.parquet``:
  1. greedy-decode with `sft_aux_merged` (no RAG)
  2. record the 4 confidence signals (`mean_max`, `min_max`, `mean_entropy`,
     `field_min`) emitted by `AuditPipeline._compute_confidence`
  3. derive `is_correct = (pred_violation == gt_violation)`

Then for each signal:
  * sweep the threshold; report (recall_of_errors, precision, fraction_triggered)
  * recommend a threshold at the elbow of the precision-recall curve (where
    triggering RAG yields the best ROI on inference compute)

The output JSON is small enough to commit and easy to diff across model
versions / SFT seeds. No RAG retrieval is run here — this only needs the
policy model + tokenizer; ~10 minutes on a single 48GB card.

Usage::

    python -m scripts.calibrate_confidence \
        --model_path models/sft_aux_merged \
        --val_parquet data/fipo/val.parquet \
        --out_json results/stage4_confidence_calibration.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.stage4_rag.inference import AuditPipeline  # noqa: E402  (uses static method)
from src.utils.json_utils import safe_loads as try_parse_json  # noqa: E402


# Reuse the prompt the rest of the pipeline uses (mining/training/inference)
SYSTEM_PROMPT = (
    "你是一位专业的电商商品合规审核员。分析给定的商品图片和描述，"
    "输出一个JSON对象，包含以下字段：\n"
    '- "category" (str): 商品品类\n'
    '- "attributes" (dict): 从图片中提取的关键视觉属性（如颜色、材质、款式等）\n'
    '- "violation" (bool): 该商品是否违反平台规则\n'
    '- "reason" (str): 简明审核理由，必须引用具体的视觉属性作为证据\n'
    "只输出合法JSON，不要用markdown代码块或其他格式包裹。"
)


def _load_image(images_field) -> Image.Image:
    rec = images_field[0]
    if isinstance(rec, dict) and "bytes" in rec:
        img = Image.open(io.BytesIO(rec["bytes"]))
    elif isinstance(rec, bytes):
        img = Image.open(io.BytesIO(rec))
    else:
        raise TypeError(type(rec))
    return img.convert("RGB")


def _user_text(prompt_field) -> str:
    user_msg = next((m for m in prompt_field if m.get("role") == "user"), None)
    if user_msg is None:
        return ""
    text = user_msg.get("content", "")
    if isinstance(text, list):
        text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
    return str(text).replace("<image>", "").strip()


@torch.inference_mode()
def _generate_one(model, processor, image, user_text, max_new_tokens):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    )
    gen_ids = out.sequences[0, inputs["input_ids"].shape[1] :]
    response = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
    conf = AuditPipeline._compute_confidence(out.scores, gen_ids)
    return response, conf


def _sweep(rows: List[dict], signal: str, lower_is_uncertain: bool) -> List[dict]:
    """Return list of (threshold, recall_errors, precision, frac_triggered)."""
    values = sorted({r["conf"][signal] for r in rows})
    n_total = len(rows)
    n_errors = sum(1 for r in rows if not r["is_correct"])
    out: List[dict] = []
    for v in values:
        if lower_is_uncertain:
            triggered = [r for r in rows if r["conf"][signal] < v]
        else:
            triggered = [r for r in rows if r["conf"][signal] > v]
        n_trig = len(triggered)
        n_caught = sum(1 for r in triggered if not r["is_correct"])
        out.append({
            "threshold": v,
            "recall_errors": n_caught / max(n_errors, 1),
            "precision": n_caught / max(n_trig, 1),
            "frac_triggered": n_trig / max(n_total, 1),
            "n_triggered": n_trig,
            "n_errors_caught": n_caught,
        })
    return out


def _recommend(curve: List[dict], target_recall: float = 0.8) -> dict:
    """Pick the highest-precision threshold that still hits target recall."""
    eligible = [c for c in curve if c["recall_errors"] >= target_recall]
    if not eligible:
        return curve[-1] if curve else {}
    eligible.sort(key=lambda c: (-c["precision"], -c["threshold"]))
    return eligible[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/sft_aux_merged")
    parser.add_argument("--val_parquet", default="data/fipo/val.parquet")
    parser.add_argument("--out_json", default="results/stage4_confidence_calibration.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--target_recall", type=float, default=0.8)
    parser.add_argument("--flash_attn", action="store_true", default=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.val_parquet)
    if args.limit > 0:
        df = df.head(args.limit)
    print(f"[calib] {len(df)} val rows from {args.val_parquet}")

    from src.utils.model_loader import load_model_and_processor
    model, processor = load_model_and_processor(
        args.model_path, apply_lora=False, use_flash_attn=args.flash_attn,
    )
    model.eval()

    rows: List[dict] = []
    t0 = time.time()
    for i, row in df.iterrows():
        img = _load_image(row["images"])
        ut = _user_text(row["prompt"])
        try:
            resp, conf = _generate_one(model, processor, img, ut, args.max_new_tokens)
        except Exception as exc:
            print(f"[calib] skip row {i}: {exc}")
            continue

        gt = json.loads(row["reward_model"]["ground_truth"])
        gt_v = bool(gt.get("violation", False))
        parsed = try_parse_json(resp)
        if parsed and "violation" in parsed:
            pred_v = bool(parsed["violation"])
            parse_ok = True
        else:
            pred_v = False
            parse_ok = False
        rows.append({
            "index": int(row["extra_info"].get("index", i)),
            "is_correct": pred_v == gt_v,
            "parse_ok": parse_ok,
            "conf": conf,
            "gt_violation": gt_v,
            "pred_violation": pred_v,
        })
        if (len(rows)) % 20 == 0:
            elapsed = time.time() - t0
            acc = sum(1 for r in rows if r["is_correct"]) / len(rows)
            print(f"[calib] {len(rows)}/{len(df)}  elapsed={elapsed:.0f}s  running_acc={acc:.3f}")

    overall_acc = sum(1 for r in rows if r["is_correct"]) / max(len(rows), 1)
    overall_parse = sum(1 for r in rows if r["parse_ok"]) / max(len(rows), 1)
    print(f"[calib] done. overall_acc={overall_acc:.4f}  parse_ok={overall_parse:.4f}")

    # mean_entropy: HIGHER = more uncertain (so flip sweep direction)
    signals = [
        ("mean_max", True),
        ("min_max", True),
        ("field_min", True),
        ("mean_entropy", False),
    ]
    sweeps = {}
    recommendations = {}
    for sig, lower_uncertain in signals:
        curve = _sweep(rows, sig, lower_uncertain)
        sweeps[sig] = curve
        recommendations[sig] = _recommend(curve, args.target_recall)

    out = {
        "n_samples": len(rows),
        "overall_accuracy": overall_acc,
        "overall_parse_rate": overall_parse,
        "target_recall": args.target_recall,
        "recommendations": recommendations,
        "sweeps": {k: v[::max(1, len(v) // 50)] for k, v in sweeps.items()},  # downsample for readability
        "samples": rows,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[calib] wrote {out_path}")

    print("\n=== Recommended thresholds (target recall {:.0%}) ===".format(args.target_recall))
    for sig in recommendations:
        rec = recommendations[sig]
        print(
            f"  {sig:<14s} threshold={rec.get('threshold', float('nan')):.4f}  "
            f"precision={rec.get('precision', 0):.3f}  "
            f"recall_errors={rec.get('recall_errors', 0):.3f}  "
            f"frac_triggered={rec.get('frac_triggered', 0):.3f}"
        )


if __name__ == "__main__":
    main()
