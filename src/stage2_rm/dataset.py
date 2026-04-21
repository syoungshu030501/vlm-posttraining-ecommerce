"""
Stage 2 Reward Model dataset.

Expects parquet files with columns:
    image    - bytes or path
    prompt   - product description
    chosen   - preferred response (JSON string)
    rejected - dispreferred response (JSON string)
"""
from __future__ import annotations

import io
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset


class PreferenceDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        processor,
        max_len: int = 1536,
    ):
        import pandas as pd
        self.df = pd.read_parquet(parquet_path)
        self.processor = processor
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, val) -> Image.Image:
        if isinstance(val, bytes):
            return Image.open(io.BytesIO(val)).convert("RGB")
        return Image.open(str(val)).convert("RGB")

    def _encode(self, image: Image.Image, prompt: str, response: str) -> Dict:
        from src.schema import SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": response},
        ]
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_len,
        )

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        image = self._load_image(row["image"])
        prompt = str(row["prompt"])

        chosen_enc = self._encode(image, prompt, str(row["chosen"]))
        rejected_enc = self._encode(image, prompt, str(row["rejected"]))

        return {
            "chosen_input_ids": chosen_enc["input_ids"][0],
            "chosen_attention_mask": chosen_enc["attention_mask"][0],
            "chosen_pixel_values": chosen_enc.get("pixel_values"),
            "chosen_image_grid_thw": chosen_enc.get("image_grid_thw"),
            "rejected_input_ids": rejected_enc["input_ids"][0],
            "rejected_attention_mask": rejected_enc["attention_mask"][0],
            "rejected_pixel_values": rejected_enc.get("pixel_values"),
            "rejected_image_grid_thw": rejected_enc.get("image_grid_thw"),
        }


def preference_collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict:
    chosen_max_len = max(b["chosen_input_ids"].shape[0] for b in batch)
    rejected_max_len = max(b["rejected_input_ids"].shape[0] for b in batch)

    out = {
        "chosen_input_ids": torch.stack(
            [_pad(b["chosen_input_ids"], chosen_max_len, pad_token_id) for b in batch]
        ),
        "chosen_attention_mask": torch.stack(
            [_pad(b["chosen_attention_mask"], chosen_max_len, 0) for b in batch]
        ),
        "rejected_input_ids": torch.stack(
            [_pad(b["rejected_input_ids"], rejected_max_len, pad_token_id) for b in batch]
        ),
        "rejected_attention_mask": torch.stack(
            [_pad(b["rejected_attention_mask"], rejected_max_len, 0) for b in batch]
        ),
    }

    chosen_pixel_values = [b["chosen_pixel_values"] for b in batch if b["chosen_pixel_values"] is not None]
    if chosen_pixel_values:
        out["chosen_pixel_values"] = torch.cat(chosen_pixel_values, dim=0)
    chosen_image_grid_thw = [
        b["chosen_image_grid_thw"] for b in batch if b["chosen_image_grid_thw"] is not None
    ]
    if chosen_image_grid_thw:
        out["chosen_image_grid_thw"] = torch.cat(chosen_image_grid_thw, dim=0)

    rejected_pixel_values = [b["rejected_pixel_values"] for b in batch if b["rejected_pixel_values"] is not None]
    if rejected_pixel_values:
        out["rejected_pixel_values"] = torch.cat(rejected_pixel_values, dim=0)
    rejected_image_grid_thw = [
        b["rejected_image_grid_thw"] for b in batch if b["rejected_image_grid_thw"] is not None
    ]
    if rejected_image_grid_thw:
        out["rejected_image_grid_thw"] = torch.cat(rejected_image_grid_thw, dim=0)

    return out


def _pad(tensor: torch.Tensor, target_len: int, pad_val: int) -> torch.Tensor:
    pad_size = target_len - tensor.shape[0]
    if pad_size <= 0:
        return tensor[:target_len]
    return torch.cat([torch.full((pad_size,), pad_val, dtype=tensor.dtype), tensor])
