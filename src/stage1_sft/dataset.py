"""
Stage 1 SFT dataset.

Expects parquet files with columns:
    image       - bytes or local path
    prompt      - product description string
    response    - gold JSON string (AuditOutput)
    violation   - bool (for contrastive loss labels)
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        processor,  # Qwen2.5-VL processor (text + vision)
        max_prompt_len: int = 1024,
        max_response_len: int = 512,
        image_col: str = "image",
        prompt_col: str = "prompt",
        response_col: str = "response",
        violation_col: str = "violation",
    ):
        import pandas as pd
        self.df = pd.read_parquet(parquet_path)
        self.processor = processor
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len
        self.image_col = image_col
        self.prompt_col = prompt_col
        self.response_col = response_col
        self.violation_col = violation_col

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, val) -> Image.Image:
        if isinstance(val, bytes):
            return Image.open(io.BytesIO(val)).convert("RGB")
        return Image.open(str(val)).convert("RGB")

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        image = self._load_image(row[self.image_col])
        prompt_text = str(row[self.prompt_col])
        response_text = str(row[self.response_col])
        violation_label = int(bool(row[self.violation_col]))

        from src.schema import SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {"role": "assistant", "content": response_text},
        ]

        # Apply chat template - get full sequence
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Also get prompt-only for loss masking
        prompt_messages = messages[:-1]
        prompt_text_full = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=False,
        )
        prompt_inputs = self.processor(
            text=[prompt_text_full],
            images=[image],
            return_tensors="pt",
            padding=False,
        )

        input_ids = inputs["input_ids"][0]
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # Labels: -100 for prompt tokens (no loss), actual ids for response
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"][0],
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "labels": labels,
            "violation_label": torch.tensor(violation_label, dtype=torch.long),
        }


def sft_collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict:
    """Left-pad sequences to the same length."""
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids = torch.stack(
        [_pad(b["input_ids"], max_len, pad_token_id) for b in batch]
    )
    attention_mask = torch.stack(
        [_pad(b["attention_mask"], max_len, 0) for b in batch]
    )
    labels = torch.stack(
        [_pad(b["labels"], max_len, -100) for b in batch]
    )
    violation_labels = torch.stack([b["violation_label"] for b in batch])

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "violation_labels": violation_labels,
    }

    # Pixel values may differ per image - keep as list for now
    pixel_values = [b["pixel_values"] for b in batch if b["pixel_values"] is not None]
    if pixel_values:
        out["pixel_values"] = torch.cat(pixel_values, dim=0)
    image_grid_thw = [b["image_grid_thw"] for b in batch if b["image_grid_thw"] is not None]
    if image_grid_thw:
        out["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)

    return out


def _pad(tensor: torch.Tensor, target_len: int, pad_val: int) -> torch.Tensor:
    pad_size = target_len - tensor.shape[0]
    if pad_size <= 0:
        return tensor[:target_len]
    return torch.cat([torch.full((pad_size,), pad_val, dtype=tensor.dtype), tensor])
