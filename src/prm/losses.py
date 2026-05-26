"""Per-step masked-MSE loss for the Process Reward Model.

The VisualPRM400K-v1.1-Raw dataset publishes a Monte Carlo expected-accuracy
score ``mc_i ∈ [0, 1]`` for every reasoning step. We regress the PRM's
per-step mean-pooled token reward against ``mc_i`` (Math-Shepherd recipe).

Compared to the Bradley-Terry pairwise loss in :mod:`src.prm.model`, this
loss gives a denser signal (one scalar per step rather than one scalar per
pair) and avoids the data-side cost of constructing chosen/rejected pairs.
"""
from __future__ import annotations

import torch


def prm_mse_loss(
    token_rewards: torch.Tensor,
    step_spans: torch.Tensor,
    step_targets: torch.Tensor,
    step_valid: torch.Tensor,
) -> torch.Tensor:
    """MSE between per-step mean-pooled token rewards and ``mc_i`` targets.

    Args:
        token_rewards: (B, T) per-token PRM scores from ``ProcessRewardModel``.
        step_spans:    (B, S, 2) int tensor; ``[start, end)`` for each step.
                       Padded entries are zeroed and ignored via ``step_valid``.
        step_targets:  (B, S) float tensor in [0, 1] (``mc_i``); 0 on pad.
        step_valid:    (B, S) {0, 1} float/bool mask; 1 if the step exists.

    Returns:
        Scalar tensor — mean squared error averaged over *valid* steps across
        the batch. Mirrors the reduction in HF Trainer's ignore_index masking.
    """
    B, S, _ = step_spans.shape
    preds = token_rewards.new_zeros(B, S)
    for s in range(S):
        for b in range(B):
            if step_valid[b, s] == 0:
                continue
            start = int(step_spans[b, s, 0].item())
            end = int(step_spans[b, s, 1].item())
            if end <= start:
                continue
            preds[b, s] = token_rewards[b, start:end].mean()

    diff = (preds - step_targets) ** 2
    valid = step_valid.to(diff.dtype)
    denom = valid.sum().clamp(min=1.0)
    return (diff * valid).sum() / denom


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, S = 2, 16, 3
    token_rewards = torch.randn(B, T, requires_grad=True)
    step_spans = torch.tensor(
        [
            [[0, 4], [4, 9], [9, 14]],
            [[0, 3], [3, 7], [0, 0]],
        ],
        dtype=torch.long,
    )
    step_targets = torch.tensor([[0.9, 0.5, 0.1], [0.8, 0.2, 0.0]])
    step_valid = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)

    loss = prm_mse_loss(token_rewards, step_spans, step_targets, step_valid)
    assert torch.isfinite(loss).item()
    loss.backward()
    assert token_rewards.grad is not None
    grad_norm = token_rewards.grad.norm().item()
    assert grad_norm > 0, "no gradient reached token_rewards"
    print(f"prm_mse_loss self-test OK: loss={float(loss):.4f}, |grad|={grad_norm:.4f}")
