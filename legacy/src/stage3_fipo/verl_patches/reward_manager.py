"""
Custom verl reward manager for VLM e-commerce audit FIPO training.

Wires our rule-based reward_fn v2 (semantic alignment via bge-small-zh-v1.5)
into verl-latest's experimental.reward_loop.RewardManagerBase interface.
Loaded via importlib (not register), so this file does not need to be
imported by the driver or pre-registered in any registry. verl reads:

    reward.reward_manager.source=importlib
    reward.reward_manager.name=VLMAuditRewardManager
    reward.reward_manager.module.path=<absolute path to this file>

Design notes
------------
- v1: rule-based only (no RM forward). Reward = JSON + label + length +
  lexicon + semantic alignment. Encoder lazy-loads BAAI/bge-small-zh-v1.5
  on first run_single() per Ray worker.
- HF download mirrored via HF_ENDPOINT=https://hf-mirror.com (set inline if
  not already exported).
- Heavy work (tokenizer.decode, encoder.encode) is dispatched through
  self.loop.run_in_executor() to keep the async event loop responsive.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional

# When verl loads this module via importlib.util.spec_from_file_location in a
# Ray worker subprocess, the project root may not be on sys.path, breaking
# `from src.stage3_fipo.reward_fn import ...` below. Inject it explicitly.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

# Process-global singletons. Each Ray worker process loads the encoder once.
_REWARD_FN = None
_ENCODER = None

# verl's _postprocess builds non_tensor_batch by looking up every key from the
# first sample's reward_extra_info on every other sample. So all rollouts in a
# batch must share the same key set. reward_fn v2 emits keys conditionally
# (e.g. `lexicon` only when reason is non-empty, `parse_failure` only on JSON
# error). Force every rollout's breakdown to this fixed schema, defaulting to 0.0.
_BREAKDOWN_SCHEMA = (
    "parse_failure",
    "missing_fields",
    "format_base",
    "violation_match",
    "reason_length",
    "lexicon",
    "semantic_align_sim",
    "semantic_align",
    "rm",
    "total",
)


def _lazy_imports():
    global _REWARD_FN
    if _REWARD_FN is None:
        from src.stage3_fipo.reward_fn import compute_reward, make_encoder  # noqa: WPS433
        _REWARD_FN = (compute_reward, make_encoder)
    return _REWARD_FN


@register("vlm_audit_v2")
class VLMAuditRewardManager(RewardManagerBase):
    """Rule-based reward manager for VLM compliance audit (FIPO v1).

    Inherits the new verl-latest async interface (`run_single` per rollout).
    """

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,            # ignored — we always use reward_fn v2
        reward_router_address=None,    # accepted for interface compat, unused
        reward_model_tokenizer=None,   # accepted for interface compat, unused
        encoder_model: str = "BAAI/bge-small-zh-v1.5",
        encoder_device: str = "cpu",
        weight_overrides: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, tokenizer, compute_score)
        self.tokenizer = tokenizer
        self.weight_overrides = weight_overrides or {}

        compute_reward, make_encoder = _lazy_imports()
        self._compute_reward = compute_reward

        global _ENCODER
        if _ENCODER is None:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            print(f"[reward_manager] loading encoder {encoder_model} on {encoder_device} ...")
            _ENCODER = make_encoder(model_name=encoder_model, device=encoder_device)
            print("[reward_manager] encoder ready")
        self._encoder = _ENCODER

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "run_single expects exactly one rollout"
        item = data[0]

        response_ids = item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(item.batch["attention_mask"][-response_length:].sum())
        valid_response_ids = response_ids[:valid_response_length]

        # Decode in thread pool — tokenizer is sync and can be slow for long seqs.
        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        rm_meta = item.non_tensor_batch.get("reward_model", {}) or {}
        gt = rm_meta.get("ground_truth", None)
        if isinstance(gt, str):
            try:
                gt = json.loads(gt)
            except json.JSONDecodeError:
                gt = None
        elif not isinstance(gt, dict):
            gt = None

        # Reward computation involves encoder.encode() — also offload.
        score, breakdown = await self.loop.run_in_executor(
            None,
            lambda: self._compute_reward(
                response_str,
                gt_annotation=gt,
                rm_score=None,
                encoder=self._encoder,
                weights=self.weight_overrides,
                return_breakdown=True,
            ),
        )

        # verl expects each entry in reward_extra_info to be picklable scalars
        # AND every rollout in the batch to share an identical key set. Pad
        # missing keys with 0.0 to satisfy verl/_postprocess.
        reward_extra_info = {f"reward_v2/{k}": float(breakdown.get(k, 0.0)) for k in _BREAKDOWN_SCHEMA}
        reward_extra_info["acc"] = float(score)

        return {"reward_score": float(score), "reward_extra_info": reward_extra_info}
