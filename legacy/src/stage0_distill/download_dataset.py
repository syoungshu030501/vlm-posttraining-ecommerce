"""
Download public e-commerce product image datasets.

Sources (in priority order):
  1. Taobao-LMDB / AliProducts — ModelScope hosted, Chinese e-commerce
  2. Products-10K (Kaggle) — multi-category product images
  3. RPC (Retail Product Checkout) — supermarket product images

Usage:
    # Option A: Download from ModelScope (recommended for Chinese e-commerce)
    python -m src.stage0_distill.download_dataset --source modelscope --out_dir data/raw --max_images 5000

    # Option B: Download from HuggingFace datasets
    python -m src.stage0_distill.download_dataset --source huggingface --out_dir data/raw --max_images 5000

    # Option C: Use local images you already have
    python -m src.stage0_distill.download_dataset --source local --local_dir /path/to/images --out_dir data/raw
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
from pathlib import Path
from typing import List, Optional

from PIL import Image
from tqdm import tqdm

# ── Category metadata for Chinese e-commerce ────────────────────────
# Used to generate realistic product descriptions when source data
# only provides images + category labels.
CATEGORY_TEMPLATES = {
    "clothing": {
        "zh": "服装",
        "attrs": ["颜色", "面料", "款式", "尺码"],
        "desc_template": "这是一款{颜色}{面料}制的{款式}，适合{场景}穿着",
    },
    "shoes": {
        "zh": "鞋靴",
        "attrs": ["颜色", "材质", "鞋型", "尺码"],
        "desc_template": "这是一双{颜色}{材质}{鞋型}，适合{场景}使用",
    },
    "electronics": {
        "zh": "数码电子",
        "attrs": ["品牌", "型号", "颜色", "规格"],
        "desc_template": "这是一款{品牌}{型号}{颜色}电子产品",
    },
    "food": {
        "zh": "食品",
        "attrs": ["口味", "重量", "保质期", "产地"],
        "desc_template": "这是一款{口味}风味的食品，净含量{重量}",
    },
    "cosmetics": {
        "zh": "美妆",
        "attrs": ["品牌", "功效", "规格", "成分"],
        "desc_template": "这是一款{品牌}{功效}护肤/美妆产品",
    },
    "household": {
        "zh": "家居日用",
        "attrs": ["材质", "尺寸", "颜色", "用途"],
        "desc_template": "这是一款{颜色}{材质}家居用品，用于{用途}",
    },
    "other": {
        "zh": "其他",
        "attrs": ["颜色", "材质", "规格"],
        "desc_template": "这是一款商品",
    },
}


def download_from_modelscope(out_dir: Path, max_images: int = 5000) -> List[dict]:
    """
    Download from ModelScope datasets.
    Try AliProducts or similar Chinese e-commerce datasets.
    """
    try:
        from modelscope.msdatasets import MsDataset
    except ImportError:
        print("[ERROR] modelscope not installed. Run: pip install modelscope")
        print("  Falling back to HuggingFace source.")
        return download_from_huggingface(out_dir, max_images)

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(">>> Downloading dataset from ModelScope...")
    # Try multiple dataset options
    datasets_to_try = [
        ("ali_pai/pai_ecommerce_product_images", "train"),
        ("tany0699/product-images-with-text", "train"),
    ]

    ds = None
    for ds_name, split in datasets_to_try:
        try:
            ds = MsDataset.load(ds_name, split=split)
            print(f"  Loaded: {ds_name}")
            break
        except Exception as e:
            print(f"  [SKIP] {ds_name}: {e}")

    if ds is None:
        print("  No ModelScope dataset available. Falling back to synthetic.")
        return generate_synthetic_dataset(out_dir, max_images)

    annotations = []
    for i, sample in enumerate(tqdm(ds, total=min(max_images, len(ds)))):
        if i >= max_images:
            break

        img = sample.get("image")
        if img is None:
            continue
        if isinstance(img, str):
            img = Image.open(img)

        fname = f"product_{i:05d}.jpg"
        img.convert("RGB").save(str(img_dir / fname), quality=90)

        category = sample.get("category", sample.get("label", "other"))
        description = sample.get("description", sample.get("text", ""))

        annotations.append({
            "image_file": fname,
            "description": description if description else f"电商商品图片，品类：{category}",
            "category_hint": str(category),
        })

    return annotations


def download_from_huggingface(out_dir: Path, max_images: int = 5000) -> List[dict]:
    """
    Download product images from HuggingFace datasets.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] datasets not installed.")
        return generate_synthetic_dataset(out_dir, max_images)

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(">>> Downloading dataset from HuggingFace...")
    hf_datasets = [
        ("BestWishYsh/Product-Images", "train", "image", "label"),
        ("thefcraft/ecommerce-product-images", "train", "image", "label"),
    ]

    # Set mirror if available
    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if not hf_endpoint:
        code = os.popen('curl -sk --connect-timeout 3 -o /dev/null -w "%{http_code}" "https://hf-mirror.com" 2>/dev/null').read().strip()
        if code in ("200", "301", "302"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    ds = None
    ds_image_col = "image"
    ds_label_col = "label"
    for ds_name, split, img_col, lbl_col in hf_datasets:
        try:
            ds = load_dataset(ds_name, split=split, streaming=True)
            ds_image_col = img_col
            ds_label_col = lbl_col
            print(f"  Loaded: {ds_name}")
            break
        except Exception as e:
            print(f"  [SKIP] {ds_name}: {e}")

    if ds is None:
        print("  No HuggingFace dataset available. Falling back to synthetic.")
        return generate_synthetic_dataset(out_dir, max_images)

    annotations = []
    for i, sample in enumerate(tqdm(ds, total=max_images)):
        if i >= max_images:
            break

        img = sample.get(ds_image_col)
        if img is None:
            continue
        if not isinstance(img, Image.Image):
            continue

        fname = f"product_{i:05d}.jpg"
        img.convert("RGB").save(str(img_dir / fname), quality=90)

        label = sample.get(ds_label_col, "other")
        annotations.append({
            "image_file": fname,
            "description": f"电商商品图片，品类：{label}",
            "category_hint": str(label),
        })

    return annotations


def generate_synthetic_dataset(out_dir: Path, max_images: int = 2000) -> List[dict]:
    """
    Fallback: generate simple synthetic product-like images.
    These are colored rectangles with category text — enough to test the pipeline.
    Real images should replace these for actual training.
    """
    import random
    import numpy as np

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> Generating {max_images} synthetic product images...")
    rng = random.Random(42)
    categories = list(CATEGORY_TEMPLATES.keys())
    colors_rgb = [
        (255, 0, 0), (0, 128, 0), (0, 0, 255), (255, 165, 0),
        (128, 0, 128), (255, 255, 0), (0, 255, 255), (192, 192, 192),
        (139, 69, 19), (255, 192, 203), (0, 0, 0), (255, 255, 255),
    ]

    annotations = []
    for i in range(max_images):
        cat = rng.choice(categories)
        color = rng.choice(colors_rgb)
        # Create a simple colored image with some noise
        arr = np.full((224, 224, 3), color, dtype=np.uint8)
        noise = np.random.randint(-30, 30, arr.shape, dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        fname = f"product_{i:05d}.jpg"
        Image.fromarray(arr).save(str(img_dir / fname), quality=85)

        template = CATEGORY_TEMPLATES[cat]
        annotations.append({
            "image_file": fname,
            "description": f"这是一款{template['zh']}商品",
            "category_hint": cat,
        })

    return annotations


def copy_local_images(local_dir: str, out_dir: Path, max_images: int = 10000) -> List[dict]:
    """Use existing local images."""
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    src_dir = Path(local_dir)
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in exts)[:max_images]

    annotations = []
    for i, src in enumerate(tqdm(files, desc="Copying images")):
        fname = f"product_{i:05d}.jpg"
        img = Image.open(str(src)).convert("RGB")
        img.save(str(img_dir / fname), quality=90)
        annotations.append({
            "image_file": fname,
            "description": f"电商商品图片: {src.stem}",
            "category_hint": "other",
        })

    return annotations


def save_annotations(annotations: List[dict], out_dir: Path) -> None:
    ann_path = out_dir / "annotations.jsonl"
    with open(ann_path, "w") as f:
        for ann in annotations:
            f.write(json.dumps(ann, ensure_ascii=False) + "\n")
    print(f"Saved {len(annotations)} annotations → {ann_path}")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "modelscope":
        annotations = download_from_modelscope(out_dir, args.max_images)
    elif args.source == "huggingface":
        annotations = download_from_huggingface(out_dir, args.max_images)
    elif args.source == "local":
        if not args.local_dir:
            raise ValueError("--local_dir required when --source=local")
        annotations = copy_local_images(args.local_dir, out_dir, args.max_images)
    elif args.source == "synthetic":
        annotations = generate_synthetic_dataset(out_dir, args.max_images)
    else:
        raise ValueError(f"Unknown source: {args.source}")

    save_annotations(annotations, out_dir)
    print(f"\n  Images: {out_dir / 'images'}")
    print(f"  Annotations: {out_dir / 'annotations.jsonl'}")
    print(f"  Total: {len(annotations)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["modelscope", "huggingface", "local", "synthetic"],
                        default="huggingface")
    parser.add_argument("--out_dir", default="data/raw")
    parser.add_argument("--max_images", type=int, default=5000)
    parser.add_argument("--local_dir", default=None, help="Path to local image directory (for --source=local)")
    run(parser.parse_args())
