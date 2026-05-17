"""
Model loading utilities — abstracts over Qwen2.5-VL / Qwen3-VL / Qwen3.5-VL
so the rest of the pipeline doesn't hardcode model classes.

Handles:
  - Auto-detecting model family from config.json
  - Disabling Qwen3.5 thinking mode (enable_thinking=False)
  - Applying LoRA config
  - Loading with appropriate dtype and attention impl
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoProcessor, PreTrainedModel

try:
    from transformers import AutoModelForImageTextToText  # transformers >= 4.45
except ImportError:  # pragma: no cover
    AutoModelForImageTextToText = None  # type: ignore[assignment]

# LoRA targets — chosen to cover both attention and MLP in the language tower
# AND the vision→text merger MLP (Qwen3-VL names them linear_fc1/linear_fc2 only
# inside `visual.merger.*` and `visual.deepstack_merger_list.*`, so the suffix
# match is safe — language layers use gate/up/down_proj instead).
# Vision encoder attention (qkv/proj) is intentionally excluded; it is frozen
# by `freeze_vision_encoder` as the vision backbone is generally robust enough
# without adaptation on small SFT datasets.
DEFAULT_LORA_CONFIG = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        # LM tower attention
        "q_proj", "k_proj", "v_proj", "o_proj",
        # LM tower MLP (most parameters; biggest expressivity gain)
        "gate_proj", "up_proj", "down_proj",
        # Vision→text merger (4 layers across main + 3 deepstack mergers)
        "linear_fc1", "linear_fc2",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)


def detect_model_family(model_path: str) -> str:
    """Detect model family from config.json."""
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return "qwen2_5_vl"

    with open(config_path) as f:
        config = json.load(f)

    model_type = config.get("model_type", "").lower()
    architectures = [a.lower() for a in config.get("architectures", [])]

    if "qwen3" in model_type or any("qwen3" in a for a in architectures):
        if "3.5" in config.get("_name_or_path", "") or "3.5" in str(config.get("model_name", "")):
            return "qwen3_5_vl"
        return "qwen3_vl"
    return "qwen2_5_vl"


def load_model_and_processor(
    model_path: str,
    *,
    torch_dtype=torch.bfloat16,
    device_map: str = "auto",
    use_flash_attn: bool = True,
    apply_lora: bool = False,
    lora_config: Optional[LoraConfig] = None,
) -> Tuple[PreTrainedModel, "AutoProcessor"]:
    """
    Load model + processor with correct config for detected family.

    Returns:
        (model, processor) tuple
    """
    family = detect_model_family(model_path)
    print(f"Detected model family: {family} (path: {model_path})")

    attn_impl = _resolve_attn_impl(use_flash_attn)

    # Load processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Load model — use AutoModelForCausalLM which routes correctly for all Qwen VL variants
    model_kwargs = dict(
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    # Qwen*-VL families require AutoModelForImageTextToText in transformers >= 4.45;
    # AutoModelForCausalLM does not register vision-language configs.
    model = None
    last_err: Exception | None = None
    if family in ("qwen2_5_vl", "qwen3_vl", "qwen3_5_vl") and AutoModelForImageTextToText is not None:
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"[WARN] AutoModelForImageTextToText failed ({exc}); trying AutoModelForCausalLM")
    if model is None:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        except Exception as exc:  # noqa: BLE001
            if last_err is not None:
                raise RuntimeError(
                    f"Failed to load model. ImageTextToText error: {last_err}; CausalLM error: {exc}"
                ) from exc
            raise

    # Disable thinking mode for Qwen3.x series
    if family in ("qwen3_vl", "qwen3_5_vl"):
        _disable_thinking_mode(model, processor)

    # Apply LoRA if requested
    if apply_lora:
        config = lora_config or DEFAULT_LORA_CONFIG
        # Verify target modules exist in model
        valid_targets = _validate_lora_targets(model, config.target_modules)
        if set(valid_targets) != set(config.target_modules):
            print(f"[WARN] Adjusted LoRA targets: {config.target_modules} → {valid_targets}")
            config = LoraConfig(
                r=config.r,
                lora_alpha=config.lora_alpha,
                target_modules=valid_targets,
                lora_dropout=config.lora_dropout,
                bias=config.bias,
                task_type=config.task_type,
            )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    return model, processor


def _resolve_attn_impl(use_flash_attn: bool) -> str:
    # Prefer flash_attention_2 → sdpa → eager.
    # SDPA uses PyTorch's memory-efficient attention; vital for VL vision towers
    # where eager softmax over 10k+ vision tokens easily blows past 30 GB.
    if not torch.cuda.is_available():
        return "eager"
    if use_flash_attn:
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except Exception as exc:
            print(f"[WARN] flash-attn unavailable, falling back to SDPA: {exc}")
    return "sdpa"


def _disable_thinking_mode(model: PreTrainedModel, processor) -> None:
    """
    Disable Qwen3.x default thinking/CoT mode.
    Forces direct JSON output without <think>...</think> wrapper.
    """
    if hasattr(model, "config"):
        # Qwen3 uses enable_thinking flag in generation config
        if hasattr(model.config, "enable_thinking"):
            model.config.enable_thinking = False
            print("  Disabled thinking mode (model.config.enable_thinking)")

    if hasattr(model, "generation_config"):
        if hasattr(model.generation_config, "enable_thinking"):
            model.generation_config.enable_thinking = False
            print("  Disabled thinking mode (generation_config.enable_thinking)")

    # For chat template: some Qwen3 models respect enable_thinking in apply_chat_template
    # We handle this at inference time in schema.py / inference.py


def _validate_lora_targets(model: PreTrainedModel, targets: list) -> list:
    """Filter target_modules to only those that exist in the model."""
    all_names = {n for n, _ in model.named_modules()}
    # Also check parameter-level names (for nested modules like visual_merger.mlp.0)
    all_param_prefixes = set()
    for n, _ in model.named_parameters():
        parts = n.rsplit(".", 1)
        if len(parts) == 2:
            all_param_prefixes.add(parts[0])

    valid = []
    for t in targets:
        # LoRA target matching is prefix-based, so check if any module name ends with the target
        if any(n.endswith(t) for n in all_names) or any(n.endswith(t) for n in all_param_prefixes):
            valid.append(t)
        else:
            print(f"  [INFO] LoRA target '{t}' not found in model, skipping")
    return valid


def freeze_vision_encoder(model: PreTrainedModel) -> None:
    """Freeze the vision encoder (default for all stages)."""
    frozen_count = 0
    for name, param in model.named_parameters():
        if "visual" in name.lower() and "merger" not in name.lower():
            param.requires_grad = False
            frozen_count += 1
    print(f"Froze {frozen_count} vision encoder parameters")
