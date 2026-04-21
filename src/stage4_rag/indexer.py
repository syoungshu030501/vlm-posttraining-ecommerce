"""
Stage 4: Build FAISS (visual) and BM25 (textual) retrieval indices.

Usage:
    python -m src.stage4_rag.indexer \
        --image_dir data/raw/images \
        --rule_file data/raw/rules.jsonl \
        --out_dir data/rag_index
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def build_visual_index(
    image_paths: List[str],
    clip_model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 32,
    out_dir: Path = Path("data/rag_index"),
) -> None:
    """Encode images with CLIP and build a FAISS index."""
    import faiss
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model = CLIPModel.from_pretrained(clip_model_name).to(device)
    model.eval()

    all_embeddings = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        all_embeddings.append(emb.cpu().numpy())

    embeddings = np.vstack(all_embeddings).astype("float32")  # (N, D)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)  # Inner product = cosine after normalisation
    index.add(embeddings)

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "visual.faiss"))
    with open(out_dir / "image_paths.pkl", "wb") as f:
        pickle.dump(image_paths, f)

    print(f"FAISS index built: {len(image_paths)} images, dim={dim}")


def build_text_index(
    rule_file: str,
    out_dir: Path = Path("data/rag_index"),
) -> None:
    """Build BM25 index over policy rule documents."""
    from rank_bm25 import BM25Okapi

    rules: List[dict] = []
    with open(rule_file) as f:
        for line in f:
            line = line.strip()
            if line:
                rules.append(json.loads(line))

    corpus = [r.get("text", "") for r in rules]
    tokenized = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump((bm25, rules), f)

    print(f"BM25 index built: {len(rules)} rule documents")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)

    image_paths = sorted(
        str(p)
        for p in Path(args.image_dir).rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    print(f"Found {len(image_paths)} images")
    build_visual_index(image_paths, out_dir=out_dir)

    if args.rule_file and os.path.exists(args.rule_file):
        build_text_index(args.rule_file, out_dir=out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--rule_file", default="data/raw/rules.jsonl")
    parser.add_argument("--out_dir", default="data/rag_index")
    run(parser.parse_args())
