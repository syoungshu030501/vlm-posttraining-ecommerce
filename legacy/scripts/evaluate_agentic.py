"""Evaluate the Agentic RAG pipeline (Plan-Then-Retrieve + Verify-then-Rewrite).

This script is a thin wrapper around AgenticAuditPipeline that:
  * Reuses scripts/evaluate.py's metric definitions for apples-to-apples comparison.
  * Adds Agentic-specific telemetry: long-tail trigger rate, fallback rate,
    rewrite rate, verify_score distribution, per-category hallucination break-down.

Usage (small sample for quick interview demo):
    python scripts/evaluate_agentic.py \\
        --model_path models/fipo_v2_step160_merged \\
        --test_parquet data/sft/test.parquet \\
        --rag_index_dir data/rag_index \\
        --max_samples 200 --sample_seed 42 \\
        --rag_threshold 0.40 \\
        --vt_low 0.5 --vt_high 0.7 \\
        --out results/eval_fipo_step160_agentic.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schema import coarse_category
from src.stage4_rag.inference import AgenticAuditPipeline


def compute_metrics(
    predictions: List[Dict[str, Any] | None],
    ground_truths: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Identical to scripts/evaluate.py for direct comparability."""
    n = len(predictions)
    format_ok = sum(1 for p in predictions if p is not None) / max(n, 1)

    tp = fp = fn = 0
    halluc_count = 0
    for pred, gt in zip(predictions, ground_truths):
        gt_viol = bool(gt.get("violation", False))
        if pred is None:
            if gt_viol:
                fn += 1
            halluc_count += 1
            continue
        pred_viol = bool(pred.get("violation", False))
        if pred_viol and gt_viol:
            tp += 1
        elif pred_viol and not gt_viol:
            fp += 1
        elif not pred_viol and gt_viol:
            fn += 1
        attrs = set(pred.get("attributes", {}).keys())
        reason = pred.get("reason", "")
        if attrs and not any(k in reason for k in attrs):
            halluc_count += 1

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "n_samples": n,
        "json_format_accuracy": round(format_ok, 4),
        "violation_precision": round(precision, 4),
        "violation_recall": round(recall, 4),
        "violation_f1": round(f1, 4),
        "hallucination_rate": round(halluc_count / max(n, 1), 4),
    }


def stratified_metrics_by_coarse(
    predictions: List[Dict[str, Any] | None],
    ground_truths: List[Dict[str, Any]],
    debug_info: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Per-coarse-category breakdown of {hallucination, n}.

    `coarse` is taken from the *prediction* (what the model says it is) when
    available, falling back to GT-side prompt parsing. This matches how RAG
    routes retrieval at inference time.
    """
    bucket_pred: Dict[str, List] = defaultdict(list)
    bucket_gt: Dict[str, List] = defaultdict(list)
    for pred, gt, dbg in zip(predictions, ground_truths, debug_info):
        coarse = dbg.get("coarse_category") or coarse_category(
            pred.get("category") if pred else ""
        )
        bucket_pred[coarse].append(pred)
        bucket_gt[coarse].append(gt)

    out = {}
    for k in sorted(bucket_pred.keys()):
        if not bucket_pred[k]:
            continue
        m = compute_metrics(bucket_pred[k], bucket_gt[k])
        out[k] = {
            "n": m["n_samples"],
            "f1": m["violation_f1"],
            "halluc": m["hallucination_rate"],
        }
    return out


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = AgenticAuditPipeline(
        model_path=args.model_path,
        index_dir=args.rag_index_dir,
        confidence_threshold=args.rag_threshold,
        confidence_method=args.rag_signal,
        top_k_visual=args.rag_top_k_visual,
        top_k_text=args.rag_top_k_text,
        clip_model=args.rag_clip_model,
        device=device,
        long_tail_categories=args.long_tail.split(",") if args.long_tail else None,
        verify_threshold_low=args.vt_low,
        verify_threshold_high=args.vt_high,
        enable_verify=not args.disable_verify,
        enable_rewrite=not args.disable_rewrite,
        enable_routing=not args.disable_routing,
        enable_long_tail_trigger=not args.disable_long_tail,
    )

    print(
        "[agentic] config:"
        f" rag_threshold={args.rag_threshold} signal={args.rag_signal}"
        f" vt_low={args.vt_low} vt_high={args.vt_high}"
        f" routing={not args.disable_routing} long_tail={not args.disable_long_tail}"
        f" verify={not args.disable_verify} rewrite={not args.disable_rewrite}"
    )

    df = pd.read_parquet(args.test_parquet)
    if args.max_samples > 0:
        if args.sample_seed is not None and args.sample_seed >= 0:
            df = df.sample(
                n=min(args.max_samples, len(df)),
                random_state=args.sample_seed,
            ).reset_index(drop=True)
        else:
            df = df.head(args.max_samples).reset_index(drop=True)

    predictions: List[Dict[str, Any] | None] = []
    ground_truths: List[Dict[str, Any]] = []
    debug_records: List[Dict[str, Any]] = []
    t_start = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="agentic-eval"):
        img_val = row["image"]
        if isinstance(img_val, bytes):
            image = Image.open(io.BytesIO(img_val)).convert("RGB")
        else:
            image = Image.open(str(img_val)).convert("RGB")

        try:
            result, debug = pipeline.predict(image, str(row["prompt"]), return_debug=True)
        except Exception as e:  # pragma: no cover — keep eval going
            print(f"[warn] sample failed: {e}", file=sys.stderr)
            predictions.append(None)
            ground_truths.append({"violation": bool(row.get("violation", False))})
            debug_records.append({"error": repr(e)})
            continue

        predictions.append(result.__dict__ if result else None)
        ground_truths.append({"violation": bool(row.get("violation", False))})
        # Strip heavy fields from debug to keep memory in check
        slim = {
            k: v
            for k, v in debug.items()
            if k
            in (
                "coarse_category",
                "triggered_by_conf",
                "triggered_by_long_tail",
                "rag_triggered",
                "verify_score",
                "fallback_used",
                "rewrite_used",
                "gating_score",
                "retrieved_text_categories",
            )
        }
        debug_records.append(slim)

    elapsed = time.time() - t_start
    metrics = compute_metrics(predictions, ground_truths)

    n = len(predictions)
    n_triggered = sum(1 for d in debug_records if d.get("rag_triggered"))
    n_long_tail = sum(1 for d in debug_records if d.get("triggered_by_long_tail"))
    n_conf = sum(1 for d in debug_records if d.get("triggered_by_conf"))
    n_fallback = sum(1 for d in debug_records if d.get("fallback_used"))
    n_rewrite = sum(1 for d in debug_records if d.get("rewrite_used"))
    verify_scores = [
        d["verify_score"] for d in debug_records if d.get("verify_score") is not None
    ]

    metrics.update(
        {
            "rag_triggered_rate": round(n_triggered / max(n, 1), 4),
            "trigger_by_confidence_rate": round(n_conf / max(n, 1), 4),
            "trigger_by_long_tail_rate": round(n_long_tail / max(n, 1), 4),
            "fallback_to_v1_rate": round(n_fallback / max(n_triggered, 1), 4),
            "rewrite_rate": round(n_rewrite / max(n_triggered, 1), 4),
            "verify_score_mean": round(mean(verify_scores), 4) if verify_scores else None,
            "verify_score_median": round(median(verify_scores), 4) if verify_scores else None,
            "rag_signal": args.rag_signal,
            "rag_threshold": args.rag_threshold,
            "vt_low": args.vt_low,
            "vt_high": args.vt_high,
            "elapsed_sec": round(elapsed, 1),
            "sec_per_sample": round(elapsed / max(n, 1), 2),
        }
    )

    metrics["per_coarse_category"] = stratified_metrics_by_coarse(
        predictions, ground_truths, debug_records
    )

    # Trigger-source breakdown (which categories actually fire long-tail)
    long_tail_hits = Counter(
        d.get("coarse_category", "?")
        for d in debug_records
        if d.get("triggered_by_long_tail")
    )
    metrics["long_tail_trigger_by_category"] = dict(long_tail_hits)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"[agentic] results saved to {args.out}")

    if args.dump_debug:
        Path(args.dump_debug).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dump_debug).write_text(
            json.dumps(debug_records, indent=2, ensure_ascii=False)
        )
        print(f"[agentic] per-sample debug saved to {args.dump_debug}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_parquet", required=True)
    parser.add_argument("--out", default="results/eval_agentic.json")
    parser.add_argument("--dump_debug", default="")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--sample_seed", type=int, default=42,
                        help="If >=0, sample N rows with this seed; else take head(N).")
    # RAG knobs
    parser.add_argument("--rag_index_dir", default="data/rag_index")
    parser.add_argument("--rag_signal", default="field_min",
                        choices=["mean_max", "min_max", "field_min", "mean_entropy"])
    parser.add_argument("--rag_threshold", type=float, default=0.40)
    parser.add_argument("--rag_top_k_visual", type=int, default=3)
    parser.add_argument("--rag_top_k_text", type=int, default=3)
    parser.add_argument("--rag_clip_model", default="models/pretrained/clip-vit-base-patch32")
    # Agentic knobs
    parser.add_argument("--long_tail", default="医药,电子产品,食品,其他",
                        help="Comma-separated coarse categories that force-trigger RAG.")
    parser.add_argument("--vt_low", type=float, default=0.5,
                        help="verify_score < vt_low → fallback to v1")
    parser.add_argument("--vt_high", type=float, default=0.7,
                        help="vt_low ≤ verify_score < vt_high → rewrite once")
    parser.add_argument("--disable_verify", action="store_true",
                        help="Ablation: skip the verify step (treat v2 as final).")
    parser.add_argument("--disable_rewrite", action="store_true",
                        help="Ablation: low verify always falls back to v1, never rewrite.")
    parser.add_argument("--disable_routing", action="store_true",
                        help="Ablation: BM25 over all docs (no category route).")
    parser.add_argument("--disable_long_tail", action="store_true",
                        help="Ablation: only confidence-trigger, no long-tail force-trigger.")
    main(parser.parse_args())
