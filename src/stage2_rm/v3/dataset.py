"""
Stage 2 v3 dataset: field-wise PRM training data.

Parquet schema (forwards-compatible with v1/v2 preference parquet):

    # ---- 通用列（v1 已有）-----------------------------------
    image                 bytes | str path
    prompt                str
    response              str           # 单条 response（field-wise PRM 单样本训练）

    # ---- v1/v2 兼容列（pair 模式时填）-----------------------
    chosen                str           # 可选 — 仅 pair-eval 用
    rejected              str           # 可选

    # ---- v3 field-wise label 列（per-response）-------------
    label_category_coarse           str | None         # ∈ COARSE_CATEGORIES
    label_attributes_json           str | None         # JSON-list of {key, val, grounded, mc}
    label_violation_prob            float32 | None     # ∈ [0, 1]
    label_violation_type            str | None         # ∈ VIOLATION_TYPES
    label_reason_align              float32 | None     # ∈ [0, 1]

    # ---- 可选 split / meta ---------------------------------
    pair_strategy                   str (legacy)
    image_file                      str
    policy_ckpt                     str

支持两种训练模式：
    mode='pointwise'  : __getitem__ 返回单条 (response, labels) — field-wise PRM 主训练
    mode='pairwise'   : 返回 (chosen, rejected, ...) — 跟 v2 holdout 兼容做 sanity 评测

设计取舍：
    - attributes_logit_target / attributes_token_mask 在 dataset 层即生成；
      做法：从 label_attributes_json 解出 [(key, val, grounded), ...]，
      然后定位每个 (key val) 在 response 中的 token span，把 grounded 标到所有 span 的
      token 位置上。其余位置 token_mask=0（CE 不算这条 token）。
    - 这是经典的 span-tagging 思路（IO BIO 之外的简化版）。允许 fuzzy 匹配的版本以后再做。
"""
from __future__ import annotations

import io
import json
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.schema import (
    COARSE_CATEGORIES,
    VIOLATION_TYPES,
    coarse_category,
    normalize_violation_type,
    SYSTEM_PROMPT,
)

CATEGORY_TO_ID = {c: i for i, c in enumerate(COARSE_CATEGORIES)}
VIOLATION_TYPE_TO_ID = {t: i for i, t in enumerate(VIOLATION_TYPES)}


def _load_image(val) -> Image.Image:
    if isinstance(val, bytes):
        return Image.open(io.BytesIO(val)).convert("RGB")
    return Image.open(str(val)).convert("RGB")


def _safe_get(row, name, default=None):
    """pandas Series .get with NaN → default."""
    if name not in row.index:
        return default
    v = row[name]
    if v is None:
        return default
    if isinstance(v, float) and v != v:  # NaN
        return default
    return v


def _find_token_spans(
    response_token_ids: List[int],
    needle_text: str,
    tokenizer,
) -> List[tuple[int, int]]:
    """Find token-id contiguous spans in response_token_ids that decode to text
    containing `needle_text`. Returns list of (start, end_exclusive).

    Greedy first-match per overall response; we don't search all variants
    (cost not worth it for v3 v0).
    """
    if not needle_text:
        return []
    full = tokenizer.decode(response_token_ids, skip_special_tokens=False)
    if needle_text not in full:
        return []

    # Char positions of all occurrences
    char_positions = []
    start = 0
    while True:
        i = full.find(needle_text, start)
        if i < 0:
            break
        char_positions.append((i, i + len(needle_text)))
        start = i + max(1, len(needle_text))

    if not char_positions:
        return []

    # Decode each token to find char offsets
    char_to_tok: list[int] = []
    cursor = 0
    for tok_idx, tok_id in enumerate(response_token_ids):
        piece = tokenizer.decode([tok_id], skip_special_tokens=False)
        # piece may map to >1 chars; assign all those chars to this tok_idx
        for _ in range(len(piece)):
            char_to_tok.append(tok_idx)
        cursor += len(piece)

    spans: list[tuple[int, int]] = []
    for cs, ce in char_positions:
        if cs >= len(char_to_tok):
            continue
        ce = min(ce, len(char_to_tok))
        if ce <= cs:
            continue
        ts = char_to_tok[cs]
        te = char_to_tok[ce - 1] + 1
        spans.append((ts, te))
    return spans


class FieldWisePreferenceDataset(Dataset):
    """Pointwise training dataset for FieldWisePRM."""

    def __init__(
        self,
        parquet_path: str,
        processor,
        max_len: int = 1536,
        mode: str = "pointwise",   # 'pointwise' | 'pairwise'
    ):
        import pandas as pd
        self.df = pd.read_parquet(parquet_path)
        self.processor = processor
        self.max_len = max_len
        self.mode = mode
        if mode not in ("pointwise", "pairwise"):
            raise ValueError(f"unknown mode: {mode}")

    def __len__(self) -> int:
        return len(self.df)

    # -----------------------------------------------------------
    # encoding
    # -----------------------------------------------------------
    def _encode_response(self, image: Image.Image, prompt: str, response: str) -> Dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": response},
        ]
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        return self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
            padding=False,
        )

    # -----------------------------------------------------------
    # label parsing
    # -----------------------------------------------------------
    def _parse_labels(self, row, encoded: Dict, response_text: str) -> Dict[str, torch.Tensor]:
        """Parse v3 field labels into model-compatible tensors."""
        T = encoded["input_ids"].shape[1]
        labels: dict = {}

        # 1. category
        cat = _safe_get(row, "label_category_coarse")
        if cat is None:
            # 兜底：从原 category 字段映射
            cat_raw = _safe_get(row, "category")
            cat = coarse_category(cat_raw) if cat_raw else None
        if cat is not None and cat in CATEGORY_TO_ID:
            labels["category_id"] = torch.tensor(CATEGORY_TO_ID[cat], dtype=torch.long)
            labels["category_mask"] = torch.tensor(1.0, dtype=torch.float)
        else:
            labels["category_id"] = torch.tensor(0, dtype=torch.long)
            labels["category_mask"] = torch.tensor(0.0, dtype=torch.float)

        # 2. violation_prob
        vp = _safe_get(row, "label_violation_prob")
        if vp is not None:
            labels["violation_prob"] = torch.tensor(float(vp), dtype=torch.float).clamp(0, 1)
            labels["violation_prob_mask"] = torch.tensor(1.0, dtype=torch.float)
        else:
            # Legacy fallback：v1/v2 的 chosen 视 prob=0.1，rejected 视 prob=0.9
            labels["violation_prob"] = torch.tensor(0.5, dtype=torch.float)
            labels["violation_prob_mask"] = torch.tensor(0.0, dtype=torch.float)

        # 3. violation_type
        vt = _safe_get(row, "label_violation_type")
        if vt is not None:
            t_norm = normalize_violation_type(vt)
            labels["violation_type_id"] = torch.tensor(VIOLATION_TYPE_TO_ID[t_norm], dtype=torch.long)
            labels["violation_type_mask"] = torch.tensor(1.0, dtype=torch.float)
        else:
            labels["violation_type_id"] = torch.tensor(0, dtype=torch.long)
            labels["violation_type_mask"] = torch.tensor(0.0, dtype=torch.float)

        # 4. reason_align
        ra = _safe_get(row, "label_reason_align")
        if ra is not None:
            labels["reason_align"] = torch.tensor(float(ra), dtype=torch.float).clamp(0, 1)
            labels["reason_align_mask"] = torch.tensor(1.0, dtype=torch.float)
        else:
            labels["reason_align"] = torch.tensor(0.5, dtype=torch.float)
            labels["reason_align_mask"] = torch.tensor(0.0, dtype=torch.float)

        # 5. attributes (token-wise)
        attrs_target = torch.zeros(T, dtype=torch.float)
        attrs_mask = torch.zeros(T, dtype=torch.float)
        attrs_json = _safe_get(row, "label_attributes_json")
        if attrs_json:
            try:
                attrs = json.loads(attrs_json) if isinstance(attrs_json, str) else attrs_json
            except (json.JSONDecodeError, TypeError):
                attrs = None
            if attrs:
                # tokenize response only (find spans in the response portion of input_ids)
                tokenizer = getattr(self.processor, "tokenizer", self.processor)
                ids = encoded["input_ids"][0].tolist()
                # response tokens = anything after the last <|im_start|>assistant prefix
                im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
                last_start = max((i for i, t in enumerate(ids) if t == im_start), default=-1)
                if last_start >= 0:
                    response_ids = ids[last_start:]
                    offset = last_start
                else:
                    response_ids = ids
                    offset = 0
                for a in attrs:
                    needle = f"{a.get('val', '')}"
                    grounded = float(a.get("grounded", a.get("mc", 0.0)))
                    if not needle:
                        continue
                    spans = _find_token_spans(response_ids, str(needle), tokenizer)
                    for ts, te in spans:
                        attrs_target[offset + ts : offset + te] = grounded
                        attrs_mask[offset + ts : offset + te] = 1.0
        labels["attributes_logit_target"] = attrs_target
        labels["attributes_token_mask"] = attrs_mask

        return labels

    # -----------------------------------------------------------
    # __getitem__
    # -----------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        image = _load_image(row["image"])
        prompt = str(row["prompt"])

        if self.mode == "pointwise":
            response = str(_safe_get(row, "response") or _safe_get(row, "chosen") or "")
            enc = self._encode_response(image, prompt, response)
            labels = self._parse_labels(row, enc, response)
            return {
                "input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "pixel_values": enc.get("pixel_values"),
                "image_grid_thw": enc.get("image_grid_thw"),
                "mm_token_type_ids": (
                    enc["mm_token_type_ids"][0] if "mm_token_type_ids" in enc else None
                ),
                "labels": labels,
            }
        else:
            # pairwise: encode chosen + rejected separately, no per-token attr labels for rejected
            chosen = str(_safe_get(row, "chosen") or "")
            rejected = str(_safe_get(row, "rejected") or "")
            chosen_enc = self._encode_response(image, prompt, chosen)
            rejected_enc = self._encode_response(image, prompt, rejected)
            chosen_labels = self._parse_labels(row, chosen_enc, chosen)
            return {
                "chosen_input_ids": chosen_enc["input_ids"][0],
                "chosen_attention_mask": chosen_enc["attention_mask"][0],
                "chosen_pixel_values": chosen_enc.get("pixel_values"),
                "chosen_image_grid_thw": chosen_enc.get("image_grid_thw"),
                "chosen_mm_token_type_ids": (
                    chosen_enc["mm_token_type_ids"][0] if "mm_token_type_ids" in chosen_enc else None
                ),
                "rejected_input_ids": rejected_enc["input_ids"][0],
                "rejected_attention_mask": rejected_enc["attention_mask"][0],
                "rejected_pixel_values": rejected_enc.get("pixel_values"),
                "rejected_image_grid_thw": rejected_enc.get("image_grid_thw"),
                "rejected_mm_token_type_ids": (
                    rejected_enc["mm_token_type_ids"][0] if "mm_token_type_ids" in rejected_enc else None
                ),
                "labels": chosen_labels,  # 仅 chosen 有 field labels；pair-eval 不需要 rejected labels
            }


# ---------------------------------------------------------------------------
# collate
# ---------------------------------------------------------------------------

def _pad_1d(t: torch.Tensor, target_len: int, pad_val: int) -> torch.Tensor:
    pad = target_len - t.shape[0]
    if pad <= 0:
        return t[:target_len]
    return torch.cat([torch.full((pad,), pad_val, dtype=t.dtype), t])


def field_wise_collate(batch: List[Dict], pad_token_id: int = 0) -> Dict:
    """Pointwise collate. Stacks labels into batched tensors with masks."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    out: dict = {}

    out["input_ids"] = torch.stack(
        [_pad_1d(b["input_ids"], max_len, pad_token_id) for b in batch]
    )
    out["attention_mask"] = torch.stack(
        [_pad_1d(b["attention_mask"], max_len, 0) for b in batch]
    )

    pix = [b["pixel_values"] for b in batch if b["pixel_values"] is not None]
    if pix:
        out["pixel_values"] = torch.cat(pix, dim=0)
    grid = [b["image_grid_thw"] for b in batch if b["image_grid_thw"] is not None]
    if grid:
        out["image_grid_thw"] = torch.cat(grid, dim=0)
    if all(b["mm_token_type_ids"] is not None for b in batch):
        out["mm_token_type_ids"] = torch.stack(
            [_pad_1d(b["mm_token_type_ids"], max_len, 0) for b in batch]
        )

    # ---- labels ----
    labels: dict = {}
    for key in (
        "category_id", "category_mask",
        "violation_prob", "violation_prob_mask",
        "violation_type_id", "violation_type_mask",
        "reason_align", "reason_align_mask",
    ):
        labels[key] = torch.stack([b["labels"][key] for b in batch])

    # token-wise labels need padding to max_len
    labels["attributes_logit_target"] = torch.stack(
        [_pad_1d(b["labels"]["attributes_logit_target"], max_len, 0) for b in batch]
    )
    labels["attributes_token_mask"] = torch.stack(
        [_pad_1d(b["labels"]["attributes_token_mask"], max_len, 0) for b in batch]
    )
    out["labels"] = labels
    return out


def field_wise_pair_collate(batch: List[Dict], pad_token_id: int = 0) -> Dict:
    """Pairwise collate (for sanity evaluation)."""
    out: dict = {}
    for prefix in ("chosen", "rejected"):
        max_len = max(b[f"{prefix}_input_ids"].shape[0] for b in batch)
        out[f"{prefix}_input_ids"] = torch.stack(
            [_pad_1d(b[f"{prefix}_input_ids"], max_len, pad_token_id) for b in batch]
        )
        out[f"{prefix}_attention_mask"] = torch.stack(
            [_pad_1d(b[f"{prefix}_attention_mask"], max_len, 0) for b in batch]
        )
        pix = [b[f"{prefix}_pixel_values"] for b in batch if b[f"{prefix}_pixel_values"] is not None]
        if pix:
            out[f"{prefix}_pixel_values"] = torch.cat(pix, dim=0)
        grid = [b[f"{prefix}_image_grid_thw"] for b in batch if b[f"{prefix}_image_grid_thw"] is not None]
        if grid:
            out[f"{prefix}_image_grid_thw"] = torch.cat(grid, dim=0)
        if all(b[f"{prefix}_mm_token_type_ids"] is not None for b in batch):
            out[f"{prefix}_mm_token_type_ids"] = torch.stack(
                [_pad_1d(b[f"{prefix}_mm_token_type_ids"], max_len, 0) for b in batch]
            )

    # 仅 chosen 有 labels (跟 __getitem__ 对齐)
    chosen_max_len = out["chosen_input_ids"].shape[1]
    labels: dict = {}
    for key in (
        "category_id", "category_mask",
        "violation_prob", "violation_prob_mask",
        "violation_type_id", "violation_type_mask",
        "reason_align", "reason_align_mask",
    ):
        labels[key] = torch.stack([b["labels"][key] for b in batch])
    labels["attributes_logit_target"] = torch.stack(
        [_pad_1d(b["labels"]["attributes_logit_target"], chosen_max_len, 0) for b in batch]
    )
    labels["attributes_token_mask"] = torch.stack(
        [_pad_1d(b["labels"]["attributes_token_mask"], chosen_max_len, 0) for b in batch]
    )
    out["labels"] = labels
    return out
