"""
Stage 2 v3: Field-wise Process Reward Model.

5 heads (all on top of a shared frozen backbone + shared trunk):
    1. category_head        - multi-class CE over 10 COARSE_CATEGORIES
    2. attributes_head      - per-token sigmoid (BCE) on `grounded` label
    3. violation_prob_head  - sigmoid (BCE) on soft target ∈ [0, 1]
    4. violation_type_head  - multi-class CE over 11 VIOLATION_TYPES
    5. reason_align_head    - linear (MSE) on continuous BGE cosine ∈ [0, 1]

All heads share:
    backbone (frozen Qwen-VL) → last_hidden_state (B, T, D)
    trunk = LN + Linear(D → D/2) + GELU + Dropout

Pooling strategy (key design choice):
    - category / violation_prob / violation_type / reason_align:
        EOS-pooled hidden state (last non-pad token)
    - attributes_head:
        per-token logits over the response region (response_mask),
        sequence-aligned BCE against per-token grounded label.

Loss (default weights, all configurable):
    L = 1.0 * CE(category)        \\
      + 0.5 * BCE(attributes)     \\  (mean over response tokens with valid attr labels)
      + 1.0 * BCE(violation_prob) \\
      + 1.0 * CE(violation_type)  \\
      + 0.3 * MSE(reason_align)

Each loss term is masked-out when its label is missing in the batch (mask in
labels_dict[<head>_mask]) so a single dataset can mix samples that have only
some labels (legacy + new + partial).

Backward compatibility:
    For PRM/ORM evaluation pipelines, expose a `score(...)` returning the
    `violation_prob_head` sigmoid as a backwards-compatible scalar reward
    (continuous version of the old binary `violation` head).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from src.schema import COARSE_CATEGORIES, VIOLATION_TYPES

NUM_CATEGORIES = len(COARSE_CATEGORIES)        # 10
NUM_VIOLATION_TYPES = len(VIOLATION_TYPES)     # 11


@dataclass
class FieldWiseLossWeights:
    category: float = 1.0
    attributes: float = 0.5
    violation_prob: float = 1.0
    violation_type: float = 1.0
    reason_align: float = 0.3


@dataclass
class FieldWiseOutput:
    """Forward pass result. All `*_logits` fields are (B, ...)."""
    category_logits: torch.Tensor          # (B, NUM_CATEGORIES)
    attributes_logits: torch.Tensor        # (B, T)  per-token grounded logit
    violation_prob_logit: torch.Tensor     # (B,)    pre-sigmoid
    violation_type_logits: torch.Tensor    # (B, NUM_VIOLATION_TYPES)
    reason_align_pred: torch.Tensor        # (B,)    in [0,1] via sigmoid

    def violation_prob(self) -> torch.Tensor:
        return torch.sigmoid(self.violation_prob_logit)

    def violation_type_probs(self) -> torch.Tensor:
        return F.softmax(self.violation_type_logits, dim=-1)


class FieldWisePRM(nn.Module):
    """Field-wise Process Reward Model with 5 heads on a shared trunk."""

    def __init__(
        self,
        base_model: PreTrainedModel,
        head_dropout: float = 0.1,
        trunk_hidden_ratio: float = 0.5,
    ):
        super().__init__()
        self.backbone = base_model
        cfg = base_model.config
        hidden_size = (
            getattr(cfg, "hidden_size", None)
            or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
        )
        if hidden_size is None:
            raise ValueError(f"Could not infer hidden_size from {type(cfg).__name__}")
        self.hidden_size = int(hidden_size)
        trunk_dim = max(64, int(self.hidden_size * trunk_hidden_ratio))
        self.trunk_dim = trunk_dim

        # 共享 trunk：5 个 head 在它之后分叉
        self.trunk = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, trunk_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
        )

        # 5 个 head（attributes 是 per-token，所以单独构造 token-wise 路径）
        self.category_head = nn.Linear(trunk_dim, NUM_CATEGORIES)
        self.violation_prob_head = nn.Linear(trunk_dim, 1)
        self.violation_type_head = nn.Linear(trunk_dim, NUM_VIOLATION_TYPES)
        self.reason_align_head = nn.Linear(trunk_dim, 1)

        # attributes head 用 token-level，独立 trunk 共享 (LN + Linear) 后单 logit
        # 这里直接基于原 hidden state 做 token-wise 投影（避免在序列上反复跑 trunk）
        self.attributes_token_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, trunk_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(trunk_dim, 1),
        )

        # 冻结 backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

    # --------------------------------------------------------------
    # forward
    # --------------------------------------------------------------
    def _backbone_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )
        if mm_token_type_ids is not None:
            kwargs["mm_token_type_ids"] = mm_token_type_ids
        out = self.backbone(**kwargs)
        return out.hidden_states[-1]  # (B, T, D)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> FieldWiseOutput:
        last_hidden = self._backbone_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )

        # EOS-pool: 取每条最后一个非 pad token 的 hidden
        seq_len = attention_mask.sum(dim=1) - 1
        seq_len = seq_len.clamp(min=0)
        idx = torch.arange(last_hidden.shape[0], device=last_hidden.device)
        eos_hidden = last_hidden[idx, seq_len].float()  # (B, D)

        trunk_eos = self.trunk(eos_hidden)              # (B, trunk_dim)
        category_logits = self.category_head(trunk_eos)              # (B, 10)
        violation_prob_logit = self.violation_prob_head(trunk_eos).squeeze(-1)  # (B,)
        violation_type_logits = self.violation_type_head(trunk_eos)  # (B, 11)
        reason_align_pred = torch.sigmoid(self.reason_align_head(trunk_eos).squeeze(-1))  # (B,) in [0,1]

        # token-wise attributes logits
        attributes_logits = self.attributes_token_head(last_hidden.float()).squeeze(-1)  # (B, T)

        return FieldWiseOutput(
            category_logits=category_logits,
            attributes_logits=attributes_logits,
            violation_prob_logit=violation_prob_logit,
            violation_type_logits=violation_type_logits,
            reason_align_pred=reason_align_pred,
        )

    # --------------------------------------------------------------
    # convenience: backwards-compatible scalar reward
    # --------------------------------------------------------------
    @torch.no_grad()
    def score_scalar(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Backward-compatible scalar reward.

        For Stage 3 (FIPO RL) we keep using a scalar signal: combine the 5 heads
        into a single confidence-weighted score:

            scalar = w_v * (1 - violation_prob)        # 越合规越高 (默认无违规)
                   + w_a * mean_attribute_grounded
                   + w_r * reason_align
                   + w_c * category_confidence

        Default weights mirror reward_fn v2 magnitudes; can be overridden by
        passing through `score_scalar_with_weights` below.
        """
        out = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        # 简单聚合：v3 阶段先给 reward_fn v3 用 raw heads，scalar 只作 sanity
        v_prob = out.violation_prob()                              # (B,)
        cat_conf = F.softmax(out.category_logits, -1).max(-1).values  # (B,)
        # attributes mean grounded over response — 训练时由 dataset 提供 mask；
        # inference 阶段简单平均 attention_mask 内位置
        attr_mean = (
            torch.sigmoid(out.attributes_logits) * attention_mask.float()
        ).sum(-1) / attention_mask.float().sum(-1).clamp(min=1)
        return (
            -2.0 * v_prob          # 违规越高越扣（与 reward_v2 -1.0/+2.0 对齐）
            + 1.5 * out.reason_align_pred
            + 1.0 * attr_mean
            + 0.5 * cat_conf
        )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _safe_mean(loss_per_sample: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(loss_per_sample.dtype)
    denom = mask.sum().clamp(min=1.0)
    return (loss_per_sample * mask).sum() / denom


def field_wise_loss(
    out: FieldWiseOutput,
    labels: dict,
    weights: FieldWiseLossWeights | None = None,
) -> tuple[torch.Tensor, dict]:
    """Compute weighted sum of 5 head losses, masking missing labels.

    `labels` dict must contain (each key optional; if missing/empty, that head
    is skipped):
        category_id        : (B,)        long, ∈ [0, NUM_CATEGORIES)
        category_mask      : (B,)        bool / 0-1 float
        attributes_logit_target : (B, T) float in [0,1] (per-token grounded)
        attributes_token_mask   : (B, T) 0-1 (1 where the position has a label)
        violation_prob     : (B,)        float in [0,1]
        violation_prob_mask: (B,)        0-1
        violation_type_id  : (B,)        long, ∈ [0, NUM_VIOLATION_TYPES)
        violation_type_mask: (B,)        0-1
        reason_align       : (B,)        float in [0,1]
        reason_align_mask  : (B,)        0-1

    Returns:
        total_loss : scalar tensor
        per_head   : dict[str, float] with each head's masked-mean loss + count
    """
    if weights is None:
        weights = FieldWiseLossWeights()

    metrics: dict[str, float] = {}
    total = out.category_logits.new_zeros(())

    # 1. category CE
    if "category_id" in labels and "category_mask" in labels:
        ce = F.cross_entropy(out.category_logits, labels["category_id"], reduction="none")
        loss_cat = _safe_mean(ce, labels["category_mask"])
        total = total + weights.category * loss_cat
        metrics["loss/category"] = loss_cat.item()
        metrics["count/category"] = float(labels["category_mask"].sum().item())

    # 2. attributes token-wise BCE
    if "attributes_logit_target" in labels and "attributes_token_mask" in labels:
        target = labels["attributes_logit_target"].float()
        mask = labels["attributes_token_mask"].float()
        # 按 token 算 BCE，再按 token mask 取均值
        bce = F.binary_cross_entropy_with_logits(
            out.attributes_logits, target, reduction="none"
        )
        denom = mask.sum().clamp(min=1.0)
        loss_attr = (bce * mask).sum() / denom
        total = total + weights.attributes * loss_attr
        metrics["loss/attributes"] = loss_attr.item()
        metrics["count/attributes_tokens"] = float(mask.sum().item())

    # 3. violation_prob BCE
    if "violation_prob" in labels and "violation_prob_mask" in labels:
        bce = F.binary_cross_entropy_with_logits(
            out.violation_prob_logit, labels["violation_prob"].float(), reduction="none",
        )
        loss_vp = _safe_mean(bce, labels["violation_prob_mask"])
        total = total + weights.violation_prob * loss_vp
        metrics["loss/violation_prob"] = loss_vp.item()
        metrics["count/violation_prob"] = float(labels["violation_prob_mask"].sum().item())

    # 4. violation_type CE
    if "violation_type_id" in labels and "violation_type_mask" in labels:
        ce = F.cross_entropy(out.violation_type_logits, labels["violation_type_id"], reduction="none")
        loss_vt = _safe_mean(ce, labels["violation_type_mask"])
        total = total + weights.violation_type * loss_vt
        metrics["loss/violation_type"] = loss_vt.item()
        metrics["count/violation_type"] = float(labels["violation_type_mask"].sum().item())

    # 5. reason_align MSE
    if "reason_align" in labels and "reason_align_mask" in labels:
        mse = (out.reason_align_pred - labels["reason_align"].float()) ** 2
        loss_ra = _safe_mean(mse, labels["reason_align_mask"])
        total = total + weights.reason_align * loss_ra
        metrics["loss/reason_align"] = loss_ra.item()
        metrics["count/reason_align"] = float(labels["reason_align_mask"].sum().item())

    metrics["loss/total"] = total.item()
    return total, metrics


# ---------------------------------------------------------------------------
# Pairwise margin (for sanity checks against v2 RM mean_margin)
# ---------------------------------------------------------------------------

def field_wise_pair_metrics(
    chosen_out: FieldWiseOutput,
    rejected_out: FieldWiseOutput,
) -> dict[str, float]:
    """Compute per-head pair-margin / pair-accuracy on a chosen vs rejected batch.

    Used at eval time when the dataset is in (chosen, rejected) pairs (legacy
    RM holdout) — convert each head's prediction to a scalar score and check
    chosen > rejected.
    """
    out: dict[str, float] = {}

    # 1. violation_prob: chosen 期望低，rejected 期望高 → score = -prob
    cv = -chosen_out.violation_prob()
    rv = -rejected_out.violation_prob()
    out["pair_acc/violation_prob"] = float((cv > rv).float().mean().item())
    out["margin/violation_prob"] = float((cv - rv).mean().item())

    # 2. reason_align: chosen 期望高
    out["pair_acc/reason_align"] = float(
        (chosen_out.reason_align_pred > rejected_out.reason_align_pred).float().mean().item()
    )
    out["margin/reason_align"] = float(
        (chosen_out.reason_align_pred - rejected_out.reason_align_pred).mean().item()
    )

    # 3. category 置信度：chosen 期望高
    cc = F.softmax(chosen_out.category_logits, -1).max(-1).values
    rc = F.softmax(rejected_out.category_logits, -1).max(-1).values
    out["pair_acc/category_confidence"] = float((cc > rc).float().mean().item())
    out["margin/category_confidence"] = float((cc - rc).mean().item())

    return out
