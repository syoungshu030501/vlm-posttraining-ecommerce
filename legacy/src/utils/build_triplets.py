"""
Hallucination triplet construction for Stage 1 auxiliary loss.

Generates (anchor_image, positive_attr, negative_attr) triplets:
  - anchor: product image
  - positive: real attribute description extracted from annotation
  - negative: hallucinated attribute (color/material/shape swapped)

Usage:
    python -m src.utils.build_triplets \
        --annotation_file data/raw/annotations.jsonl \
        --image_dir data/raw/images \
        --out_file data/sft/triplets.parquet
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm

# Common attribute swaps for hallucination generation
COLOR_POOL = [
    "红色", "蓝色", "绿色", "黄色", "黑色", "白色", "紫色", "橙色", "棕色", "灰色",
    "粉色", "金色", "银色", "米白", "藏蓝", "酒红",
    "red", "blue", "green", "yellow", "black", "white", "purple", "pink",
]
MATERIAL_POOL = [
    "棉", "涤纶", "丝绸", "羊毛", "亚麻", "皮革", "牛仔", "雪纺", "尼龙", "氨纶",
    "cotton", "polyester", "silk", "wool", "leather", "denim", "nylon",
]
SHAPE_POOL = [
    "圆形", "方形", "长方形", "椭圆", "三角形", "不规则形",
    "round", "square", "rectangular", "oval", "triangular",
]
STYLE_POOL = [
    "修身", "宽松", "直筒", "喇叭", "紧身", "阔腿", "高腰", "低腰", "常规",
    "翻领", "立领", "圆领", "V领", "无领", "连帽",
    "长袖", "短袖", "七分袖", "无袖",
    "手提包", "单肩包", "斜挎包", "双肩包", "托特包",
    "运动款", "商务款", "休闲款", "正装款",
    "slim", "loose", "regular", "casual",
]

SWAP_POOLS = {
    "color": COLOR_POOL,
    "颜色": COLOR_POOL,
    "表盘颜色": COLOR_POOL,
    "表带颜色": COLOR_POOL,
    "material": MATERIAL_POOL,
    "材质": MATERIAL_POOL,
    "面料": MATERIAL_POOL,
    "表带材质": MATERIAL_POOL,
    "shape": SHAPE_POOL,
    "形状": SHAPE_POOL,
    "表壳形状": SHAPE_POOL,
    "款式": STYLE_POOL,
    "style": STYLE_POOL,
    "版型": STYLE_POOL,
    "领型": STYLE_POOL,
    "袖长": STYLE_POOL,
}

# Only generate triplets for these attribute keys (high-signal, stable set)
TRIPLET_KEY_WHITELIST = {
    "颜色", "color", "材质", "material", "面料", "款式", "style", "版型",
    "形状", "shape", "表盘颜色", "表带颜色", "表带材质", "表壳形状",
    "领型", "袖长",
}


def swap_attribute(attr_key: str, attr_value: str, seed: int = 0) -> str:
    """Replace an attribute value with a hallucinated one."""
    rng = random.Random(seed)

    # Find the right swap pool
    pool = None
    for key_pattern, p in SWAP_POOLS.items():
        if key_pattern in attr_key.lower():
            pool = p
            break

    if pool is None:
        # Fallback: shuffle characters or pick from a generic pool
        pool = COLOR_POOL + MATERIAL_POOL

    candidates = [v for v in pool if v.lower() != str(attr_value).lower()]
    if not candidates:
        return attr_value + "_fake"
    return rng.choice(candidates)


def build_triplets_from_annotation(
    ann: dict,
    image_dir: str,
    seed: int = 0,
) -> List[Dict]:
    """Generate triplets from a single annotation."""
    triplets = []

    # Get attributes from response JSON
    response_text = ann.get("response", ann.get("label_json", ""))
    if isinstance(response_text, str):
        try:
            response = json.loads(response_text)
        except json.JSONDecodeError:
            return []
    elif isinstance(response_text, dict):
        response = response_text
    else:
        return []

    attributes = response.get("attributes", {})
    if not attributes:
        return []

    img_file = ann.get("image_file", ann.get("image", ""))
    img_path = os.path.join(image_dir, img_file)
    if not os.path.exists(img_path):
        return []

    # For each attribute, create a triplet (whitelist only, to keep signal stable)
    for i, (key, value) in enumerate(attributes.items()):
        if key not in TRIPLET_KEY_WHITELIST:
            continue
        positive_text = f"{key}: {value}"
        neg_value = swap_attribute(key, str(value), seed=seed + i)
        negative_text = f"{key}: {neg_value}"

        triplets.append({
            "image_path": img_path,
            "positive_attr": positive_text,
            "negative_attr": negative_text,
            "attr_key": key,
            "attr_value_real": str(value),
            "attr_value_fake": neg_value,
        })

    return triplets


def build_all_triplets(
    annotations: List[dict],
    image_dir: str,
    embed_images: bool = False,
) -> pd.DataFrame:
    all_triplets = []
    for i, ann in enumerate(tqdm(annotations, desc="Building triplets")):
        triplets = build_triplets_from_annotation(ann, image_dir, seed=i * 100)
        all_triplets.extend(triplets)

    df = pd.DataFrame(all_triplets)

    if embed_images and len(df) > 0:
        print("Embedding images into parquet...")
        image_bytes = []
        for path in tqdm(df["image_path"], desc="Reading images"):
            try:
                img = Image.open(path).convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                image_bytes.append(buf.getvalue())
            except Exception:
                image_bytes.append(None)
        df["image"] = image_bytes
        df = df.dropna(subset=["image"])

    return df


def run(args: argparse.Namespace) -> None:
    annotations = []
    with open(args.annotation_file) as f:
        for line in f:
            line = line.strip()
            if line:
                annotations.append(json.loads(line))

    print(f"Loaded {len(annotations)} annotations")
    df = build_all_triplets(annotations, args.image_dir, embed_images=args.embed)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out_path), index=False)
    print(f"Saved {len(df)} triplets to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_file", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--out_file", default="data/sft/triplets.parquet")
    parser.add_argument("--embed", action="store_true", help="Embed image bytes into parquet")
    run(parser.parse_args())
