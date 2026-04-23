"""
Stage 3: Multi-dimensional rule-based reward function (v2).

Plugs into veRL's custom reward_manager via callable interface.

v2 upgrades over v1
-------------------
- (NEW) Reason ↔ Attributes **semantic alignment** via local sentence encoder
        (replaces v1's brittle keyword-substring hallucination proxy)
- (NEW) Reason ↔ Violation **lexicon consistency** check
- (NEW) Reason length sanity bounds
- (KEPT) JSON structure / violation label / RM score
- (CHANGED) RM is now passed as precomputed `rm_score` (decoupled from this fn)

All weights are configurable via DEFAULT_WEIGHTS or `weights=` arg.

Reward composition (typical range [-3, 5]):
    JSON parseable + complete  : +1.0
    violation label match      : +2.0  /  mismatch -1.0
    reason length OK           :  0     (penalties below)
    reason too short  (<8)     : -1.0
    reason too long   (>250)   : -0.5
    lexicon  matches violation : +0.5  /  contradicts -0.3
    semantic alignment (sim>=0.6) : +1.5
                       (sim<=0.2) : -0.5
                       (linear in between)
    RM score (passed in)       : +rm_score

Hard early-exits:
    JSON parse failure     : -3.0
    missing required fields: -2.0
"""
from __future__ import annotations

import json
from typing import Any, Iterable, List, Optional

import numpy as np

REQUIRED_KEYS = frozenset({"category", "attributes", "violation", "reason"})

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------
VIOLATION_KEYWORDS: frozenset[str] = frozenset({
    # explicit violation
    "违规", "不合规", "违反", "禁止", "未标注", "缺少", "未注明",
    # exaggeration / absolute claims (中文电商高频违规词)
    "夸大", "虚假", "误导", "诱导",
    "极限", "最", "首选", "唯一", "保证",
    "100%", "国家级", "顶级", "世界级", "最佳",
    "绝对", "完全", "彻底", "永久", "根治", "速效", "特效",
})

NON_VIOLATION_KEYWORDS: frozenset[str] = frozenset({
    "合规", "正常", "符合", "规范", "标注完整",
    "无违规", "未发现", "正确", "合理", "如实",
})

# ---------------------------------------------------------------------------
# Default weights (override via `weights=` arg)
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "format_base": 1.0,
    "violation_match_correct": 2.0,
    "violation_match_wrong": -1.0,
    "reason_too_short": -1.0,
    "reason_too_long": -0.5,
    "lexicon_match": 0.5,
    "lexicon_contradict": -0.3,
    "semantic_align_high": 1.5,    # sim >= align_high_thresh
    "semantic_align_low": -0.5,    # sim <= align_low_thresh
    "rm_weight": 1.0,
    # hard penalties (early exit)
    "parse_failure": -3.0,
    "missing_fields": -2.0,
}

REASON_MIN_LEN = 8
REASON_MAX_LEN = 250
ALIGN_HIGH_THRESH = 0.60
ALIGN_LOW_THRESH = 0.20

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return text


def _attributes_to_text(attributes: dict) -> str:
    """Flatten an attributes dict to a single string for sentence encoding."""
    if not isinstance(attributes, dict) or not attributes:
        return ""
    parts: List[str] = []
    for k, v in attributes.items():
        if isinstance(v, (list, tuple)):
            v = "、".join(str(x) for x in v)
        parts.append(f"{k}: {v}")
    return "；".join(parts)


def _lexicon_score(reason: str, pred_violation: bool, w: dict) -> float:
    has_v = any(kw in reason for kw in VIOLATION_KEYWORDS)
    has_nv = any(kw in reason for kw in NON_VIOLATION_KEYWORDS)
    if pred_violation:
        if has_v and not has_nv:
            return w["lexicon_match"]
        if has_nv and not has_v:
            return w["lexicon_contradict"]
    else:
        if has_nv and not has_v:
            return w["lexicon_match"]
        if has_v and not has_nv:
            return w["lexicon_contradict"]
    return 0.0


def _length_score(reason: str, w: dict) -> float:
    L = len(reason.strip())
    if L < REASON_MIN_LEN:
        return w["reason_too_short"]
    if L > REASON_MAX_LEN:
        return w["reason_too_long"]
    return 0.0


def _semantic_alignment(reason: str, attr_text: str, encoder) -> float:
    """Cosine similarity in [-1, 1] between reason and attribute text."""
    if not reason or not attr_text:
        return 0.0
    embs = encoder.encode([reason, attr_text], normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))


def _alignment_score(sim: float, w: dict) -> float:
    """Map cosine similarity to a reward contribution (linear interp between thresholds)."""
    if sim >= ALIGN_HIGH_THRESH:
        return w["semantic_align_high"]
    if sim <= ALIGN_LOW_THRESH:
        return w["semantic_align_low"]
    # linear interp between low and high thresholds
    t = (sim - ALIGN_LOW_THRESH) / (ALIGN_HIGH_THRESH - ALIGN_LOW_THRESH)
    return w["semantic_align_low"] + t * (w["semantic_align_high"] - w["semantic_align_low"])


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------
def compute_reward(
    output_text: str,
    *,
    gt_annotation: Optional[dict] = None,
    rm_score: Optional[float] = None,
    encoder=None,
    weights: Optional[dict] = None,
    return_breakdown: bool = False,
) -> float | tuple[float, dict]:
    """
    Compute scalar reward for a single rollout response.

    Args:
        output_text:    Raw text generated by the policy model.
        gt_annotation:  Ground-truth dict with at least {"violation": bool}.
        rm_score:       Precomputed RM score (scalar). VeRL reward manager
                        runs RM in batch and passes the scalar in.
        encoder:        sentence-transformers encoder (e.g. BAAI/bge-small-zh-v1.5).
                        If None, semantic alignment component is skipped.
        weights:        Override DEFAULT_WEIGHTS partially.
        return_breakdown: if True, return (reward, breakdown_dict) for debugging.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    bd: dict = {}

    text = _strip_code_fences(output_text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        bd["parse_failure"] = w["parse_failure"]
        return (w["parse_failure"], bd) if return_breakdown else w["parse_failure"]

    if not REQUIRED_KEYS.issubset(parsed.keys()):
        bd["missing_fields"] = w["missing_fields"]
        return (w["missing_fields"], bd) if return_breakdown else w["missing_fields"]

    reward = w["format_base"]
    bd["format_base"] = w["format_base"]

    pred_violation = bool(parsed.get("violation", False))
    reason = str(parsed.get("reason", "") or "")
    attributes = parsed.get("attributes", {}) or {}

    # 1. Violation label correctness
    if gt_annotation is not None and "violation" in gt_annotation:
        gt_violation = bool(gt_annotation["violation"])
        delta = w["violation_match_correct"] if pred_violation == gt_violation else w["violation_match_wrong"]
        reward += delta
        bd["violation_match"] = delta

    # 2. Reason length sanity
    delta = _length_score(reason, w)
    if delta != 0.0:
        reward += delta
        bd["reason_length"] = delta

    # 3. Lexicon consistency: reason words ↔ violation label
    if reason:
        delta = _lexicon_score(reason, pred_violation, w)
        if delta != 0.0:
            reward += delta
            bd["lexicon"] = delta

    # 4. Semantic alignment: reason embedding ↔ attributes embedding
    #    KEY upgrade: replaces v1's substring hallucination proxy
    if encoder is not None and reason and attributes:
        attr_text = _attributes_to_text(attributes)
        sim = _semantic_alignment(reason, attr_text, encoder)
        delta = _alignment_score(sim, w)
        reward += delta
        bd["semantic_align_sim"] = round(sim, 4)
        bd["semantic_align"] = round(delta, 4)

    # 5. Learned RM score
    if rm_score is not None:
        delta = w["rm_weight"] * float(rm_score)
        reward += delta
        bd["rm"] = round(delta, 4)

    bd["total"] = round(reward, 4)
    return (reward, bd) if return_breakdown else reward


# ---------------------------------------------------------------------------
# Encoder factory
# ---------------------------------------------------------------------------
def make_encoder(model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu"):
    """Lazy-load a sentence encoder. Cached per process."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=device)


# ---------------------------------------------------------------------------
# Batch interface (veRL custom reward manager)
# ---------------------------------------------------------------------------
def batch_compute_reward(
    output_texts: List[str],
    gt_annotations: Optional[List[Optional[dict]]] = None,
    rm_scores: Optional[List[Optional[float]]] = None,
    encoder=None,
    weights: Optional[dict] = None,
) -> List[float]:
    n = len(output_texts)
    if gt_annotations is None:
        gt_annotations = [None] * n
    if rm_scores is None:
        rm_scores = [None] * n
    return [
        compute_reward(text, gt_annotation=gt, rm_score=rm, encoder=encoder, weights=weights)
        for text, gt, rm in zip(output_texts, gt_annotations, rm_scores)
    ]
