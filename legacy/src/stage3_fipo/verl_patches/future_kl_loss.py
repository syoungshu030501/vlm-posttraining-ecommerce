"""
FIPO future-KL policy loss, forward-ported from FIPO-main (old verl 0.5.x)
to verl-latest (>=0.7.x) which uses the (pg_loss, metrics_dict) interface.

Original implementation:
    vendor/FIPO-main/verl/trainer/ppo/core_algos.py
        compute_policy_loss_future_kl  (lines 941-1161)

Adaptation summary
------------------
- Return value: tuple of 29 items  ->  (pg_loss, dict[str, float])
- Added accepted (but currently ignored) arg `rollout_is_weights` to match the
  new PolicyLossFn signature in verl-latest.
- agg_loss now passes `**config.global_batch_info` for global normalisation.
- Uses register_policy_loss("future_kl"): importing this module side-effect
  registers the loss into verl's POLICY_LOSS_REGISTRY.

How to enable in a training run
-------------------------------
1. Set in your verl YAML / CLI:
       actor.policy_loss.loss_mode=future_kl
       actor.policy_loss.decay_rate=12.0          # FIPO default
       actor.policy_loss.chunk_size=128
       actor.policy_loss.future_kl_clip_ratio=0.2
       actor.policy_loss.future_kl_clip_high_only=false
       actor.policy_loss.safety_thresh=4.0
2. In your launcher Python entry, add ONE line BEFORE building the Trainer:
       import src.stage3_fipo.verl_patches.future_kl_loss  # noqa: F401
   This triggers the @register_policy_loss decorator.
3. Done. dp_actor.py / megatron_actor.py need NO patch — verl-latest's unified
   (pg_loss, metrics) contract takes care of metric collection automatically.

Algorithm summary (FIPO paper)
------------------------------
- Compute negative_approx_kl_t = log_prob_t - old_log_prob_t (token KL diff).
- Decay-weighted **future** KL: F_t = sum_{j>=t} gamma^(j-t) * kl_j  with
  gamma = 2^(-1/decay_rate). Uses chunked matmul for O(L * chunk_size).
- influence_weight_t = clip(exp(F_t)) — penalises tokens whose future has
  drifted from the old policy (instability proxy).
- Clip-c safety: tokens whose ratio > clip_ratio_c contribute neither to the
  KL accumulation nor to the gradient; tokens with negative advantage and
  high IS get a hard cap to avoid over-penalisation.
- Final loss: standard PPO clipped surrogate, but each token's advantage is
  multiplied by its influence_weight before clipping.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config.algorithm import AlgoConfig
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss
from verl.workers.config import ActorConfig


@register_policy_loss("future_kl")
def compute_policy_loss_future_kl(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """FIPO future-KL surrogate loss. See module docstring for adaptation notes."""

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)

    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        f"clip_ratio_c for dual-clip PPO must be > 1.0, got {clip_ratio_c}"
    )

    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # ------------------------------------------------------------------ shape checks
    assert log_prob.shape == old_log_prob.shape == advantages.shape, (
        f"log/old/adv shape mismatch: {log_prob.shape}, {old_log_prob.shape}, {advantages.shape}"
    )
    assert response_mask.dim() == 2 and response_mask.size(0) == log_prob.size(0), (
        f"response_mask shape {response_mask.shape} incompatible with batch {log_prob.shape}"
    )

    batch_size, response_len = log_prob.shape
    device = log_prob.device
    dtype = log_prob.dtype
    assert response_mask.size(1) == response_len, (
        f"Time dim mismatch: log_prob length={response_len}, response_mask length={response_mask.size(1)}"
    )

    # ------------------------------------------------------------------ FIPO core
    # verl-latest's PolicyLossConfig is a strict dataclass that does not accept
    # future_kl-specific fields. We read them from env vars to stay zero-intrusion.
    import os as _os
    chunk_size = int(_os.environ.get("FIPO_CHUNK_SIZE", "128"))
    decay_rate = float(_os.environ.get("FIPO_DECAY_RATE", "12.0"))
    gamma = 2.0 ** (-1.0 / decay_rate)

    pos_i = torch.arange(response_len, device=device).unsqueeze(1)  # (L, 1)
    filter_threshold = torch.log(torch.tensor(clip_ratio_c, device=device, dtype=dtype))
    is_negative_adv = (advantages < 0)
    ignore_mask = negative_approx_kl > filter_threshold
    participation_mask = ~ignore_mask

    kl_response_premask = negative_approx_kl * response_mask.to(dtype)
    kl_response = kl_response_premask * participation_mask.to(dtype)

    future_kl = torch.zeros((batch_size, response_len), device=device, dtype=dtype)
    gamma_t = torch.tensor(gamma, dtype=dtype, device=device)
    for j_start in range(0, response_len, chunk_size):
        j_end = min(response_len, j_start + chunk_size)
        j_idx = torch.arange(j_start, j_end, device=device).unsqueeze(0)  # (1, Kb)
        distance = j_idx - pos_i                                          # (L, Kb)
        mask = distance >= 0
        distance_clamped = distance.clamp(min=0)
        decay_block = torch.pow(gamma_t, distance_clamped) * mask.to(dtype)
        kl_block = kl_response[:, j_start:j_end]                          # (B, Kb)
        contrib = torch.matmul(kl_block, decay_block.t())                 # (B, L)
        future_kl += contrib

    # ------------------------------------------------------------------ influence weight + clip
    fkl_clip_ratio = float(_os.environ.get("FIPO_FKL_CLIP_RATIO", "0.2"))
    fkl_clip_high_only = _os.environ.get("FIPO_FKL_CLIP_HIGH_ONLY", "false").lower() == "true"
    if fkl_clip_ratio != 0.0:
        if not fkl_clip_high_only:
            upper_bound = 1.0 + fkl_clip_ratio
            lower_bound = 1.0 - fkl_clip_ratio
        else:
            upper_bound = 1.0 + fkl_clip_ratio
            lower_bound = 1.0
        influence_weights = torch.clamp(torch.exp(future_kl), min=lower_bound, max=upper_bound).detach()
    else:
        upper_bound = 10.0
        lower_bound = 0.0
        influence_weights = torch.clamp(torch.exp(future_kl), max=10.0).detach()

    # safety threshold: cap influence on negative-adv high-IS tokens
    safe_threshold = float(_os.environ.get("FIPO_SAFETY_THRESH", "4.0"))
    mask_neg_high_is = (advantages < 0) & (ratio > safe_threshold)
    influence_weights = torch.where(
        mask_neg_high_is,
        torch.clamp(influence_weights, min=0.8, max=1.0),
        influence_weights,
    )

    # ------------------------------------------------------------------ PPO surrogate
    weighted_advantages = advantages * influence_weights
    pg_losses1 = -weighted_advantages * ratio
    pg_losses2 = -weighted_advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -weighted_advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * is_negative_adv.float(), response_mask
    )

    # sequence-level invalidation: if a sequence has >1 dual-clipped tokens, drop entire sequence
    lower_clip_mask = (
        (advantages < 0)
        & (clip_pg_losses1 > pg_losses3)
        & response_mask.bool()
    )
    seq_has_low_clip = (lower_clip_mask.sum(dim=1) > 1)
    seq_valid_mask = (~seq_has_low_clip).unsqueeze(1)
    final_mask = response_mask.bool() & seq_valid_mask
    final_mask_f = final_mask.to(dtype)

    pg_losses = torch.where(weighted_advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # rollout correction (verl-latest >=0.7 contract)
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=final_mask_f,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    # ------------------------------------------------------------------ metrics
    valid_mask = response_mask.to(dtype=torch.bool)
    iw_raw = torch.exp(future_kl)
    influence_weights_mean_raw = verl_F.masked_mean(iw_raw, response_mask)
    influence_weights_mean = verl_F.masked_mean(influence_weights, response_mask)

    valid_iw = influence_weights[valid_mask]
    valid_iw_raw = iw_raw[valid_mask]

    # Note: keys MUST be identical across every DP rank, otherwise
    # verl/utils/metric/utils.py:Metric.aggregate_dp raises
    # `All Metric instances must have the same number of values`.
    # Conditional fields (no positive/negative samples in this micro batch)
    # default to 0.0 instead of being omitted.
    metrics: dict[str, Any] = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/fipo/influence_weights_mean": influence_weights_mean.detach().item(),
        "actor/fipo/influence_weights_mean_raw": influence_weights_mean_raw.detach().item(),
        "actor/fipo/influence_weights_min": valid_iw.min().item() if valid_iw.numel() > 0 else 0.0,
        "actor/fipo/influence_weights_max": valid_iw.max().item() if valid_iw.numel() > 0 else 0.0,
        "actor/fipo/influence_weights_max_raw": valid_iw_raw.max().item() if valid_iw_raw.numel() > 0 else 0.0,
    }

    neg_valid = ratio[(advantages < 0) & valid_mask]
    pos_valid = ratio[(advantages > 0) & valid_mask]
    metrics["actor/fipo/neg_is_max"] = neg_valid.max().item() if neg_valid.numel() > 0 else 0.0
    metrics["actor/fipo/neg_is_p995"] = (
        torch.quantile(neg_valid, 0.995).item() if neg_valid.numel() > 0 else 0.0
    )
    metrics["actor/fipo/pos_is_min"] = pos_valid.min().item() if pos_valid.numel() > 0 else 0.0
    metrics["actor/fipo/pos_is_p005"] = (
        torch.quantile(pos_valid, 0.005).item() if pos_valid.numel() > 0 else 0.0
    )
    metrics["actor/fipo/pos_mini_frac"] = verl_F.masked_mean(
        ((ratio < 1e-3) & (advantages > 0)).float(), response_mask
    ).item()
    metrics["actor/fipo/clip_frac_upper"] = verl_F.masked_mean(
        (influence_weights >= upper_bound - 1e-7).float(), response_mask
    ).item()
    metrics["actor/fipo/clip_frac_lower"] = verl_F.masked_mean(
        (influence_weights <= lower_bound + 1e-7).float(), response_mask
    ).item()

    return pg_loss, metrics
