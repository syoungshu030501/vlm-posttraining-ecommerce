"""
Stage 2 v3 PRM offline evaluation.

Two evaluation modes:
    1. pointwise: per-head accuracy / MSE on a holdout with field labels
       - category accuracy (top-1)
       - violation_prob: BCE + ECE (calibration)
       - violation_type: macro-F1 across 11 classes (subset support reported)
       - reason_align: MSE
       - attributes: AUROC (per-token grounded prediction)

    2. pairwise: chosen vs rejected per-head margin / pair-acc
       - mirrors v2 RM holdout evaluation for direct comparison

Usage:
    python -m src.stage2_rm.v3.evaluate \\
        --model_path models/sft_aux_merged \\
        --ckpt models/rm_v3_aux_ckpt/prm_v3_best.pt \\
        --eval_parquet data/preference_v3/preference_v3_holdout.parquet \\
        --mode pointwise

    python -m src.stage2_rm.v3.evaluate ... --mode pairwise
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.schema import VIOLATION_TYPES, COARSE_CATEGORIES
from src.stage2_rm.v3.dataset import (
    FieldWisePreferenceDataset,
    field_wise_collate,
    field_wise_pair_collate,
)
from src.stage2_rm.v3.model import (
    FieldWisePRM,
    field_wise_pair_metrics,
    NUM_CATEGORIES,
    NUM_VIOLATION_TYPES,
)
from src.utils.model_loader import load_model_and_processor


def _to_device(t, device):
    if isinstance(t, torch.Tensor):
        return t.to(device)
    if isinstance(t, dict):
        return {k: _to_device(v, device) for k, v in t.items()}
    return t


def _load_ckpt(rm: FieldWisePRM, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    rm.trunk.load_state_dict(ckpt["trunk"])
    rm.category_head.load_state_dict(ckpt["category_head"])
    rm.violation_prob_head.load_state_dict(ckpt["violation_prob_head"])
    rm.violation_type_head.load_state_dict(ckpt["violation_type_head"])
    rm.reason_align_head.load_state_dict(ckpt["reason_align_head"])
    rm.attributes_token_head.load_state_dict(ckpt["attributes_token_head"])
    print(f"loaded heads + trunk from {ckpt_path}")


# ---------------------------------------------------------------------------
# Macro-F1
# ---------------------------------------------------------------------------

def macro_f1(preds: list[int], labels: list[int], num_classes: int) -> dict:
    """Per-class precision/recall/F1, plus macro F1.

    Classes with 0 support have F1=NaN excluded from macro avg.
    """
    per: dict[int, dict] = {}
    for c in range(num_classes):
        tp = sum(1 for p, l in zip(preds, labels) if p == c and l == c)
        fp = sum(1 for p, l in zip(preds, labels) if p == c and l != c)
        fn = sum(1 for p, l in zip(preds, labels) if p != c and l == c)
        sup = sum(1 for l in labels if l == c)
        precision = tp / max(tp + fp, 1) if (tp + fp) else 0.0
        recall = tp / max(tp + fn, 1) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per[c] = {"precision": precision, "recall": recall, "f1": f1, "support": sup}
    valid = [v["f1"] for v in per.values() if v["support"] > 0]
    macro = sum(valid) / max(len(valid), 1)
    return {"macro_f1": macro, "per_class": per}


# ---------------------------------------------------------------------------
# ECE (Expected Calibration Error)
# ---------------------------------------------------------------------------

def expected_calibration_error(probs: list[float], labels: list[float], n_bins: int = 10) -> float:
    """Standard ECE for binary calibration. labels ∈ [0,1] (treated as soft positive rate)."""
    if not probs:
        return 0.0
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    total = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bucket = [(p, l) for p, l in zip(probs, labels) if lo <= p < hi or (i == n_bins - 1 and p == hi)]
        if not bucket:
            continue
        avg_p = sum(p for p, _ in bucket) / len(bucket)
        avg_l = sum(l for _, l in bucket) / len(bucket)
        total += (len(bucket) / n) * abs(avg_p - avg_l)
    return total


# ---------------------------------------------------------------------------
# Pointwise eval
# ---------------------------------------------------------------------------

def evaluate_pointwise(rm, loader, device) -> dict:
    rm.eval()
    cat_p, cat_l, cat_mask = [], [], []
    vp_p, vp_l, vp_mask = [], [], []
    vt_p, vt_l, vt_mask = [], [], []
    ra_p, ra_l, ra_mask = [], [], []
    attr_pred_all, attr_label_all, attr_mask_all = [], [], []

    with torch.no_grad():
        for batch in loader:
            inputs = dict(
                input_ids=_to_device(batch["input_ids"], device),
                attention_mask=_to_device(batch["attention_mask"], device),
                pixel_values=_to_device(batch.get("pixel_values"), device),
                image_grid_thw=_to_device(batch.get("image_grid_thw"), device),
                mm_token_type_ids=_to_device(batch.get("mm_token_type_ids"), device),
            )
            out = rm(**inputs)
            labels = batch["labels"]

            # category
            cat_pred = out.category_logits.argmax(-1).cpu().tolist()
            cat_p += cat_pred
            cat_l += labels["category_id"].cpu().tolist()
            cat_mask += labels["category_mask"].cpu().tolist()

            # violation_prob
            vp_p += out.violation_prob().cpu().tolist()
            vp_l += labels["violation_prob"].cpu().tolist()
            vp_mask += labels["violation_prob_mask"].cpu().tolist()

            # violation_type
            vt_pred = out.violation_type_logits.argmax(-1).cpu().tolist()
            vt_p += vt_pred
            vt_l += labels["violation_type_id"].cpu().tolist()
            vt_mask += labels["violation_type_mask"].cpu().tolist()

            # reason_align
            ra_p += out.reason_align_pred.cpu().tolist()
            ra_l += labels["reason_align"].cpu().tolist()
            ra_mask += labels["reason_align_mask"].cpu().tolist()

            # attributes (token-wise)
            attr_pred = torch.sigmoid(out.attributes_logits).cpu().tolist()
            attr_label = labels["attributes_logit_target"].cpu().tolist()
            attr_msk = labels["attributes_token_mask"].cpu().tolist()
            attr_pred_all += attr_pred
            attr_label_all += attr_label
            attr_mask_all += attr_msk

    metrics: dict = {}

    # 1. category accuracy (only on masked-in samples)
    masked = [(p, l) for p, l, m in zip(cat_p, cat_l, cat_mask) if m > 0.5]
    if masked:
        acc = sum(1 for p, l in masked if p == l) / len(masked)
        metrics["category_accuracy"] = acc
        metrics["category_support"] = len(masked)

    # 2. violation_prob: BCE + ECE
    masked_vp = [(p, l) for p, l, m in zip(vp_p, vp_l, vp_mask) if m > 0.5]
    if masked_vp:
        # BCE (using float labels in [0,1])
        import math
        bce = sum(
            -(l * math.log(p + 1e-8) + (1 - l) * math.log(1 - p + 1e-8))
            for p, l in masked_vp
        ) / len(masked_vp)
        metrics["violation_prob_bce"] = bce
        metrics["violation_prob_ece"] = expected_calibration_error(
            [p for p, _ in masked_vp], [l for _, l in masked_vp],
        )
        # binary acc treating l > 0.5 as positive
        pred_bin = [1 if p > 0.5 else 0 for p, _ in masked_vp]
        label_bin = [1 if l > 0.5 else 0 for _, l in masked_vp]
        metrics["violation_prob_binary_acc"] = sum(
            1 for p, l in zip(pred_bin, label_bin) if p == l
        ) / len(masked_vp)
        metrics["violation_prob_support"] = len(masked_vp)

    # 3. violation_type macro-F1
    masked_vt = [(p, l) for p, l, m in zip(vt_p, vt_l, vt_mask) if m > 0.5]
    if masked_vt:
        f1 = macro_f1(
            [p for p, _ in masked_vt],
            [l for _, l in masked_vt],
            NUM_VIOLATION_TYPES,
        )
        metrics["violation_type_macro_f1"] = f1["macro_f1"]
        metrics["violation_type_support"] = len(masked_vt)
        metrics["violation_type_per_class"] = {
            VIOLATION_TYPES[c]: f1["per_class"][c] for c in range(NUM_VIOLATION_TYPES)
        }

    # 4. reason_align MSE
    masked_ra = [(p, l) for p, l, m in zip(ra_p, ra_l, ra_mask) if m > 0.5]
    if masked_ra:
        mse = sum((p - l) ** 2 for p, l in masked_ra) / len(masked_ra)
        metrics["reason_align_mse"] = mse
        metrics["reason_align_support"] = len(masked_ra)

    # 5. attributes AUROC (rough; pure-Python median split)
    flat_pred, flat_label = [], []
    for p_seq, l_seq, m_seq in zip(attr_pred_all, attr_label_all, attr_mask_all):
        for p, l, m in zip(p_seq, l_seq, m_seq):
            if m > 0.5:
                flat_pred.append(p)
                flat_label.append(l)
    if flat_pred:
        # bucket-based AUROC approximation
        thresholds = sorted(set(flat_pred))[::max(1, len(set(flat_pred)) // 50)]
        if len(thresholds) < 2:
            thresholds = [0.3, 0.5, 0.7]
        best_acc = 0.0
        for thr in thresholds:
            pred_bin = [1 if p > thr else 0 for p in flat_pred]
            lbl_bin = [1 if l > 0.5 else 0 for l in flat_label]
            tp = sum(1 for p, l in zip(pred_bin, lbl_bin) if p == 1 and l == 1)
            tn = sum(1 for p, l in zip(pred_bin, lbl_bin) if p == 0 and l == 0)
            acc = (tp + tn) / max(len(pred_bin), 1)
            best_acc = max(best_acc, acc)
        metrics["attributes_token_best_acc"] = best_acc
        metrics["attributes_token_support"] = len(flat_pred)

    return metrics


# ---------------------------------------------------------------------------
# Pairwise eval
# ---------------------------------------------------------------------------

def evaluate_pairwise_full(rm, loader, device) -> dict:
    rm.eval()
    aggs: dict[str, list[float]] = defaultdict(list)
    n = 0
    with torch.no_grad():
        for batch in loader:
            chosen_out = rm(
                input_ids=_to_device(batch["chosen_input_ids"], device),
                attention_mask=_to_device(batch["chosen_attention_mask"], device),
                pixel_values=_to_device(batch.get("chosen_pixel_values"), device),
                image_grid_thw=_to_device(batch.get("chosen_image_grid_thw"), device),
                mm_token_type_ids=_to_device(batch.get("chosen_mm_token_type_ids"), device),
            )
            rejected_out = rm(
                input_ids=_to_device(batch["rejected_input_ids"], device),
                attention_mask=_to_device(batch["rejected_attention_mask"], device),
                pixel_values=_to_device(batch.get("rejected_pixel_values"), device),
                image_grid_thw=_to_device(batch.get("rejected_image_grid_thw"), device),
                mm_token_type_ids=_to_device(batch.get("rejected_mm_token_type_ids"), device),
            )
            m = field_wise_pair_metrics(chosen_out, rejected_out)
            for k, v in m.items():
                aggs[k].append(v)
            n += chosen_out.violation_prob_logit.shape[0]
    avg = {k: sum(v) / max(len(v), 1) for k, v in aggs.items()}
    avg["holdout_n"] = n
    return avg


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, processor = load_model_and_processor(
        args.model_path, apply_lora=False, use_flash_attn=args.flash_attn,
        device_map=None,
    )
    rm = FieldWisePRM(model, head_dropout=0.0).to(device)
    if args.ckpt and Path(args.ckpt).exists():
        _load_ckpt(rm, args.ckpt)
    else:
        print(f"[WARN] no ckpt at {args.ckpt}, evaluating randomly-initialized heads")

    pad_id = getattr(processor, "pad_token_id", None)
    if pad_id is None and hasattr(processor, "tokenizer"):
        pad_id = processor.tokenizer.pad_token_id or 0

    if args.mode == "pointwise":
        ds = FieldWisePreferenceDataset(args.eval_parquet, processor, mode="pointwise")
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, num_workers=2,
            collate_fn=lambda b: field_wise_collate(b, pad_token_id=pad_id or 0),
        )
        metrics = evaluate_pointwise(rm, loader, device)
    else:
        ds = FieldWisePreferenceDataset(args.eval_parquet, processor, mode="pairwise")
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, num_workers=2,
            collate_fn=lambda b: field_wise_pair_collate(b, pad_token_id=pad_id or 0),
        )
        metrics = evaluate_pairwise_full(rm, loader, device)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\nsaved → {args.out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--ckpt", required=False, default=None)
    parser.add_argument("--eval_parquet", required=True)
    parser.add_argument("--mode", choices=["pointwise", "pairwise"], default="pointwise")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--out_json", default=None)
    main(parser.parse_args())
