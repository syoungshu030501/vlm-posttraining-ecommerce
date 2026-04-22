"""
Hallucination triplet dataset for Stage 1 auxiliary loss.

Each row of `data/sft/triplets.parquet` carries:
    image_path        - relative path under image_dir (e.g. data/raw/images/xx.jpg)
    positive_attr     - real attribute description (e.g. "颜色: 深棕色")
    negative_attr     - perturbed attribute description (e.g. "颜色: 银色")

Returned items are PIL images + raw strings; tokenisation happens inside the
training loop because the triplet is consumed alongside the main forward pass
(per-image processor invocation is fine at the cadence of one triplet per
optimizer step).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    def __init__(self, parquet_path: str, image_dir: str = "data/raw/images") -> None:
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.image_dir = Path(image_dir)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image(self, raw_path: str) -> Image.Image:
        # `image_path` already includes "data/raw/images/..." — try as-is first,
        # then fall back to image_dir + basename.
        candidate = Path(raw_path)
        if not candidate.exists():
            candidate = self.image_dir / Path(raw_path).name
        return Image.open(candidate).convert("RGB")

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        return {
            "image": self._resolve_image(str(row["image_path"])),
            "positive_attr": str(row["positive_attr"]),
            "negative_attr": str(row["negative_attr"]),
        }


def triplet_collate_fn(batch: List[Dict]) -> Dict:
    """Triplet loop consumes one sample at a time (batch_size=1 in DataLoader)."""
    return batch[0]
