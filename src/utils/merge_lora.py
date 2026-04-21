"""
LoRA merge utility: merge LoRA adapters back into the base model.

Used between Stage 1 (SFT) and Stage 3 (FIPO RL) to avoid
online LoRA compatibility issues with veRL rollout engine.

Usage:
    python -m src.utils.merge_lora \
        --base_model models/pretrained/Qwen2.5-VL-7B-Instruct \
        --lora_path models/sft_ckpt/epoch-3 \
        --out_dir models/sft_merged
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel

from src.utils.model_loader import load_model_and_processor


def merge_and_save(
    base_model_path: str,
    lora_path: str,
    out_dir: str,
    torch_dtype=torch.bfloat16,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {base_model_path}")
    base_model, processor = load_model_and_processor(
        base_model_path,
        device_map="cpu",
        torch_dtype=torch_dtype,
        use_flash_attn=False,
        apply_lora=False,
    )

    print(f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {out_dir}")
    model.save_pretrained(out_dir, safe_serialization=True)

    # Copy processor/tokenizer from base model
    processor.save_pretrained(out_dir)

    # Copy any extra config files (generation_config, etc.)
    for extra in ["generation_config.json", "preprocessor_config.json"]:
        src = Path(base_model_path) / extra
        if src.exists():
            shutil.copy2(src, out / extra)

    print(f"Done. Merged model saved to {out_dir}")
    print(f"  Model size: {sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 1e9:.1f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="Path to base pretrained model")
    parser.add_argument("--lora_path", required=True, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--out_dir", required=True, help="Output directory for merged model")
    args = parser.parse_args()
    merge_and_save(args.base_model, args.lora_path, args.out_dir)
