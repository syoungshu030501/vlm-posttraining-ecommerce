"""
Evaluation script: compute JSON format accuracy, violation F1, and hallucination rate.

Two evaluation modes:
  * baseline  (default): plain greedy decode with the policy model
  * rag       (--use_rag): Stage-4 AuditPipeline with confidence-gated retrieval

Usage:
    # baseline
    python scripts/evaluate.py \
        --model_path models/sft_aux_merged \
        --test_parquet data/sft/test.parquet \
        --out results/eval_sft_aux.json

    # with RAG (confidence-gated retrieval over FAISS+BM25)
    python scripts/evaluate.py \
        --model_path models/sft_aux_merged \
        --test_parquet data/sft/test.parquet \
        --use_rag --rag_index_dir data/rag_index \
        --rag_signal field_min --rag_threshold 0.85 \
        --out results/eval_sft_aux_rag.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schema import SYSTEM_PROMPT, try_parse
from src.utils.model_loader import load_model_and_processor
from src.utils.tracking import finish_run, init_swanlab, log_metrics


def run_inference(
    model,
    processor,
    image: Image.Image,
    description: str,
    device: str,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": description},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated = out[0, inputs["input_ids"].shape[1] :]
    return processor.decode(generated, skip_special_tokens=True)


def compute_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
) -> Dict:
    n = len(predictions)
    format_ok = sum(1 for p in predictions if p is not None) / n

    # Violation F1
    tp = fp = fn = 0
    halluc_count = 0
    for pred, gt in zip(predictions, ground_truths):
        gt_viol = bool(gt.get("violation", False))
        if pred is None:
            if gt_viol:
                fn += 1
            halluc_count += 1  # Parse failure counts as hallucination
            continue
        pred_viol = bool(pred.get("violation", False))
        if pred_viol and gt_viol:
            tp += 1
        elif pred_viol and not gt_viol:
            fp += 1
        elif not pred_viol and gt_viol:
            fn += 1

        # Hallucination: reason doesn't reference any attribute
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
        "hallucination_rate": round(halluc_count / n, 4),
    }


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tracker = init_swanlab(
        stage="evaluation",
        config=vars(args),
        project=args.project_name,
        experiment_name=args.experiment_name,
        tags=["evaluation", "rag" if args.use_rag else "baseline"],
        description=f"Model evaluation (use_rag={args.use_rag})",
    )

    pipeline = None
    model = processor = None
    if args.use_rag:
        from src.stage4_rag.inference import AuditPipeline

        pipeline = AuditPipeline(
            model_path=args.model_path,
            index_dir=args.rag_index_dir,
            confidence_threshold=args.rag_threshold,
            confidence_method=args.rag_signal,
            top_k_visual=args.rag_top_k_visual,
            top_k_text=args.rag_top_k_text,
            clip_model=args.rag_clip_model,
            device=device,
        )
        print(
            f"[eval] RAG enabled: signal={args.rag_signal} threshold={args.rag_threshold} "
            f"top_k_visual={args.rag_top_k_visual} top_k_text={args.rag_top_k_text}"
        )
    else:
        model, processor = load_model_and_processor(
            args.model_path,
            apply_lora=False,
            use_flash_attn=True,
        )
        model.eval()

    df = pd.read_parquet(args.test_parquet)
    if args.max_samples > 0:
        df = df.head(args.max_samples)

    predictions = []
    ground_truths = []
    rag_triggered_count = 0
    confidence_signals: List[Dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        import io
        img_val = row["image"]
        if isinstance(img_val, bytes):
            image = Image.open(io.BytesIO(img_val)).convert("RGB")
        else:
            image = Image.open(str(img_val)).convert("RGB")

        if pipeline is not None:
            result, debug = pipeline.predict(image, str(row["prompt"]), return_debug=True)
            parsed = result
            if debug.get("rag_triggered"):
                rag_triggered_count += 1
            confidence_signals.append(debug.get("confidence", {}))
            predictions.append(parsed.__dict__ if parsed else None)
        else:
            response = run_inference(model, processor, image, str(row["prompt"]), device)
            parsed = try_parse(response)
            predictions.append(parsed.__dict__ if parsed else None)
        ground_truths.append({"violation": row.get("violation", False)})

    metrics = compute_metrics(predictions, ground_truths)
    if pipeline is not None:
        n = len(predictions)
        metrics["rag_triggered_rate"] = round(rag_triggered_count / max(n, 1), 4)
        metrics["rag_signal"] = args.rag_signal
        metrics["rag_threshold"] = args.rag_threshold
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    log_metrics(tracker, metrics)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"Results saved to {args.out}")
    finish_run(tracker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_parquet", required=True)
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--project_name", default="vlm-posttraining")
    parser.add_argument("--experiment_name", default="evaluation")
    # RAG knobs
    parser.add_argument("--use_rag", action="store_true",
                        help="Run AuditPipeline with confidence-gated RAG retrieval")
    parser.add_argument("--rag_index_dir", default="data/rag_index")
    parser.add_argument("--rag_signal", default="field_min",
                        choices=["mean_max", "min_max", "field_min", "mean_entropy"])
    parser.add_argument("--rag_threshold", type=float, default=0.85,
                        help="Trigger RAG when the chosen signal crosses this threshold "
                             "(< for max-prob signals, > for entropy)")
    parser.add_argument("--rag_top_k_visual", type=int, default=3)
    parser.add_argument("--rag_top_k_text", type=int, default=3)
    parser.add_argument("--rag_clip_model", default="models/pretrained/clip-vit-base-patch32")
    main(parser.parse_args())
