"""
Process Reward Model (PRM) for vision-language reasoning chains.

The model wraps a frozen VLM backbone (e.g. Qwen2.5-VL, Qwen3-VL) with a
lightweight scalar head that emits a reward score at *every* token position
of the response, rather than a single sequence-level score at EOS.

Training uses a Bradley-Terry pairwise objective in which the per-token
rewards are mean-pooled over the response mask before the BT log-sigmoid
is taken — so gradients flow through every response token, not only the
last one. This is the key structural difference from a classic Outcome RM
and is what makes PRMs useful as step-level critics for chain-of-thought
generations.

Head architecture (kept deliberately small to fit alongside a frozen 7B/8B
backbone in <24GB):

    LayerNorm(hidden) → Linear(hidden, hidden//4) → GELU → Dropout(p=0.1)
        → Linear(hidden//4, 1)

The backbone is frozen at construction time; LoRA / full-finetune is the
caller's responsibility (apply *before* wrapping with ProcessRewardModel
if the backbone weights should be trainable).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PreTrainedModel


class ProcessRewardModel(nn.Module):
    """Token-level reward model. Returns (B, T) per-token scores.

    Same frozen-backbone pattern as a sequence-level RewardModel, but the
    head applies to every token position (no last-token pooling). During
    training, token rewards are mean-pooled over the response region to
    produce a sequence-level Bradley-Terry loss; gradients flow through
    *all* response tokens — the key difference from an Outcome RM.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = base_model
        cfg = base_model.config
        # VLM configs (e.g. Qwen3VLConfig) nest hidden_size under text_config
        hidden_size = (
            getattr(cfg, "hidden_size", None)
            or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
        )
        if hidden_size is None:
            raise ValueError(f"Could not infer hidden_size from {type(cfg).__name__}")

        self.reward_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_size // 4, 1),
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns (B, T) per-token reward scores."""
        backbone_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )
        if mm_token_type_ids is not None:
            backbone_kwargs["mm_token_type_ids"] = mm_token_type_ids
        outputs = self.backbone(**backbone_kwargs)
        last_hidden = outputs.hidden_states[-1]  # (B, T, D)
        token_rewards = self.reward_head(last_hidden.float()).squeeze(-1)  # (B, T)
        return token_rewards

    @torch.no_grad()
    def score_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean-pool token rewards over response tokens → (B,) scalar."""
        token_rewards = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        counts = response_mask.sum(dim=1).clamp(min=1)
        return (token_rewards * response_mask).sum(dim=1) / counts


def prm_bt_loss(
    chosen_token_rewards: torch.Tensor,
    rejected_token_rewards: torch.Tensor,
    chosen_response_mask: torch.Tensor,
    rejected_response_mask: torch.Tensor,
) -> torch.Tensor:
    """PRM Bradley-Terry loss: mean-pool token rewards over response, then BT.

    Gradients flow from the single loss scalar back through the mean-pool
    to every response token position — the key advantage over an Outcome RM
    where only the final token contributes.
    """
    chosen_counts = chosen_response_mask.sum(dim=1).clamp(min=1)
    rejected_counts = rejected_response_mask.sum(dim=1).clamp(min=1)

    chosen_seq = (chosen_token_rewards * chosen_response_mask).sum(dim=1) / chosen_counts
    rejected_seq = (rejected_token_rewards * rejected_response_mask).sum(dim=1) / rejected_counts

    return -torch.log(torch.sigmoid(chosen_seq - rejected_seq) + 1e-8).mean()
