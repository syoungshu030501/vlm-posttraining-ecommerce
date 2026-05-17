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


def compute_response_mask(
    input_ids: torch.Tensor,
    im_start_id: int,
    im_end_id: int,
) -> torch.Tensor:
    """Return (T,) mask: 1 for assistant response tokens, 0 elsewhere.

    Finds the last <|im_start|> (assistant turn) and marks from there to
    <|im_end|> as response.  Works with Qwen2.5/3 chat templates.
    """
    ids = input_ids.tolist()

    last_start = -1
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] == im_start_id:
            last_start = i
            break
    if last_start == -1:
        return torch.ones_like(input_ids, dtype=torch.long)

    end_pos = len(ids)
    for i in range(last_start + 1, len(ids)):
        if ids[i] == im_end_id:
            end_pos = i
            break

    mask = torch.zeros_like(input_ids, dtype=torch.long)
    mask[last_start : end_pos + 1] = 1
    return mask


class PreferenceDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        processor,
        max_len: int = 1536,
        response_mask: bool = False,
    ):
        import pandas as pd
        self.df = pd.read_parquet(parquet_path)
        self.processor = processor
        self.max_len = max_len
        self._response_mask = response_mask

        if response_mask:
            tokenizer = getattr(processor, "tokenizer", processor)
            self._im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
            self._im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

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
        # NOTE: Do NOT truncate. Truncation can drop image-placeholder tokens and
        # break the image-token-count check inside Qwen-VL processors (the
        # number of <|image_pad|> tokens must match pixel_values count).
        return self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
            padding=False,
        )

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        image = self._load_image(row["image"])
        prompt = str(row["prompt"])

        chosen_enc = self._encode(image, prompt, str(row["chosen"]))
        rejected_enc = self._encode(image, prompt, str(row["rejected"]))

        item = {
            "chosen_input_ids": chosen_enc["input_ids"][0],
            "chosen_attention_mask": chosen_enc["attention_mask"][0],
            "chosen_pixel_values": chosen_enc.get("pixel_values"),
            "chosen_image_grid_thw": chosen_enc.get("image_grid_thw"),
            "rejected_input_ids": rejected_enc["input_ids"][0],
            "rejected_attention_mask": rejected_enc["attention_mask"][0],
            "rejected_pixel_values": rejected_enc.get("pixel_values"),
            "rejected_image_grid_thw": rejected_enc.get("image_grid_thw"),
        }
        if "mm_token_type_ids" in chosen_enc:
            item["chosen_mm_token_type_ids"] = chosen_enc["mm_token_type_ids"][0]
        if "mm_token_type_ids" in rejected_enc:
            item["rejected_mm_token_type_ids"] = rejected_enc["mm_token_type_ids"][0]

        if self._response_mask:
            item["chosen_response_mask"] = compute_response_mask(
                chosen_enc["input_ids"][0], self._im_start_id, self._im_end_id,
            )
            item["rejected_response_mask"] = compute_response_mask(
                rejected_enc["input_ids"][0], self._im_start_id, self._im_end_id,
            )

        return item


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

    if all("chosen_mm_token_type_ids" in b for b in batch):
        out["chosen_mm_token_type_ids"] = torch.stack(
            [_pad(b["chosen_mm_token_type_ids"], chosen_max_len, 0) for b in batch]
        )
    if all("rejected_mm_token_type_ids" in b for b in batch):
        out["rejected_mm_token_type_ids"] = torch.stack(
            [_pad(b["rejected_mm_token_type_ids"], rejected_max_len, 0) for b in batch]
        )

    if all("chosen_response_mask" in b for b in batch):
        out["chosen_response_mask"] = torch.stack(
            [_pad(b["chosen_response_mask"], chosen_max_len, 0) for b in batch]
        )
        out["rejected_response_mask"] = torch.stack(
            [_pad(b["rejected_response_mask"], rejected_max_len, 0) for b in batch]
        )

    return out


def _pad(tensor: torch.Tensor, target_len: int, pad_val: int) -> torch.Tensor:
    pad_size = target_len - tensor.shape[0]
    if pad_size <= 0:
        return tensor[:target_len]
    return torch.cat([torch.full((pad_size,), pad_val, dtype=tensor.dtype), tensor])
