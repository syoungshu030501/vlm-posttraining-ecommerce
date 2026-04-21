"""
Stage 2: Reward Model (Bradley-Terry pairwise loss).

Backbone: frozen Qwen VL model (auto-detected family). Only the scalar head is trained.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PreTrainedModel


class RewardModel(nn.Module):
    def __init__(
        self,
        base_model: PreTrainedModel,
        head_bias: bool = False,
        head_layernorm: bool = False,
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

        # Head architecture is configurable for ablation. Defaults preserve the
        # original behaviour (no LN, no bias) so existing checkpoints stay loadable.
        self.head_bias = head_bias
        self.head_layernorm = head_layernorm
        head_layers: list[nn.Module] = []
        if head_layernorm:
            head_layers.append(nn.LayerNorm(hidden_size))
        head_layers.append(nn.Linear(hidden_size, 1, bias=head_bias))
        self.reward_head = nn.Sequential(*head_layers) if len(head_layers) > 1 else head_layers[0]

        # Freeze backbone
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
        # Use last non-padding token's hidden state
        seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
        last_hidden = outputs.hidden_states[-1]  # (B, T, D)
        pooled = last_hidden[range(last_hidden.shape[0]), seq_lengths]  # (B, D)
        reward = self.reward_head(pooled.float()).squeeze(-1)  # (B,)
        return reward

    def score(self, pixel_values, input_ids, attention_mask, image_grid_thw=None) -> float:
        """Convenience for inference."""
        with torch.no_grad():
            return self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            ).item()


def bradley_terry_loss(chosen_reward: torch.Tensor, rejected_reward: torch.Tensor) -> torch.Tensor:
    """Standard Bradley-Terry pairwise ranking loss."""
    return -torch.log(torch.sigmoid(chosen_reward - rejected_reward) + 1e-8).mean()
