"""
Stage 2: Reward Model (Bradley-Terry pairwise loss).

Backbone: frozen Qwen VL model (auto-detected family). Only the scalar head is trained.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PreTrainedModel


class RewardModel(nn.Module):
    def __init__(self, base_model: PreTrainedModel):
        super().__init__()
        self.backbone = base_model
        hidden_size = base_model.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )
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
