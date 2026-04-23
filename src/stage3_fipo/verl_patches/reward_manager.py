"""
Custom verl reward manager for VLM e-commerce audit FIPO training.

Wires our rule-based reward_fn v2 (semantic alignment via bge-small-zh-v1.5)
into verl's reward_manager interface. Registered as `vlm_audit_v2`.

Use in YAML / CLI:
    reward_model.reward_manager=vlm_audit_v2

Importing this module side-effect registers the manager into
REWARD_MANAGER_REGISTRY (via @register decorator).

Design notes
------------
- v1: rule-based only (no RM forward) — keeps Stage 3 self-contained, avoids
  loading another 17GB merged-SFT backbone alongside the actor.
- v2 (future): set `enable_rm_score=True` via init kwargs and pass a callable
  `rm_score_fn(image, response_text) -> float`.
- Encoder lazy-loads BAAI/bge-small-zh-v1.5 on first __call__ (per worker).
  Set HF_ENDPOINT=https://hf-mirror.com if HF direct is blocked.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Optional

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

# We import the project's reward_fn lazily inside __init__ so that this module
# can be imported even when running outside the project (for tests).
_REWARD_FN = None
_ENCODER = None


def _lazy_imports():
    global _REWARD_FN
    if _REWARD_FN is None:
        from src.stage3_fipo.reward_fn import compute_reward, make_encoder  # noqa: WPS433
        _REWARD_FN = (compute_reward, make_encoder)
    return _REWARD_FN


@register("vlm_audit_v2")
class VLMAuditRewardManager(AbstractRewardManager):
    """Rule-based reward manager for VLM compliance audit.

    Args:
        tokenizer:           HF tokenizer (used to decode response token ids).
        num_examine:         How many rollouts per data_source to print for debug.
        compute_score:       Ignored — we always use reward_fn v2. Kept for interface.
        reward_fn_key:       Key in non_tensor_batch for data source name.
        encoder_model:       Sentence encoder model id (default bge-small-zh-v1.5).
        encoder_device:      "cpu" recommended; GPU is busy with actor.
        weight_overrides:    Optional dict to override DEFAULT_WEIGHTS in reward_fn.
        enable_rm_score:     Reserved for v2; currently False.
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,  # noqa: ARG002 — interface
        reward_fn_key: str = "data_source",
        encoder_model: str = "BAAI/bge-small-zh-v1.5",
        encoder_device: str = "cpu",
        weight_overrides: Optional[dict] = None,
        enable_rm_score: bool = False,
        **kwargs: Any,
    ) -> None:
        compute_reward, make_encoder = _lazy_imports()
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.weight_overrides = weight_overrides or {}
        self.enable_rm_score = enable_rm_score
        if enable_rm_score:
            raise NotImplementedError(
                "RM scoring inside reward_manager not yet wired. "
                "Set enable_rm_score=False for v1."
            )

        # Encoder is heavy; load once per worker.
        global _ENCODER
        if _ENCODER is None:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            print(f"[reward_manager] loading encoder {encoder_model} on {encoder_device} ...")
            _ENCODER = make_encoder(model_name=encoder_model, device=encoder_device)
            print("[reward_manager] encoder ready")
        self._encoder = _ENCODER
        self._compute_reward = compute_reward

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        # If upstream already computed rm_scores, just forward.
        cached = self._extract_reward_from_rm_scores(data, return_dict)
        if cached is not None:
            return cached

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, list] = defaultdict(list)
        already_print: dict[str, int] = {}

        for i in range(len(data)):
            item = data[i]
            prompt_ids = item.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]

            response_ids = item.batch["responses"]
            valid_response_length = int(item.batch["attention_mask"][prompt_len:].sum())
            valid_response_ids = response_ids[:valid_response_length]

            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # ground_truth: we expect data_prep to put the chosen JSON dict here
            rm_meta = item.non_tensor_batch.get("reward_model", {}) or {}
            gt = rm_meta.get("ground_truth", None)
            if isinstance(gt, str):
                try:
                    gt = json.loads(gt)
                except json.JSONDecodeError:
                    gt = None
            elif not isinstance(gt, dict):
                gt = None

            data_source = item.non_tensor_batch.get(self.reward_fn_key, "vlm_audit")

            score, breakdown = self._compute_reward(
                response_str,
                gt_annotation=gt,
                rm_score=None,
                encoder=self._encoder,
                weights=self.weight_overrides,
                return_breakdown=True,
            )

            reward_tensor[i, valid_response_length - 1] = float(score)
            for k, v in breakdown.items():
                reward_extra_info[f"reward_v2/{k}"].append(v)

            already_print.setdefault(data_source, 0)
            if already_print[data_source] < self.num_examine:
                already_print[data_source] += 1
                print(f"[reward_v2] data_source={data_source} score={score:.3f}")
                print(f"           response={response_str[:200]}...")
                print(f"           gt_violation={gt.get('violation') if gt else None}")
                print(f"           breakdown={breakdown}")

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(reward_extra_info)}
        return reward_tensor
