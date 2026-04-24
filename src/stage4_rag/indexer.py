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
    clip_model_name: str = "models/pretrained/clip-vit-base-patch32",
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
        # transformers >= 4.46 may wrap the return in a BaseModelOutput; unwrap defensively
        if hasattr(emb, "image_embeds"):
            emb = emb.image_embeds
        elif hasattr(emb, "last_hidden_state"):
            emb = emb.pooler_output if hasattr(emb, "pooler_output") and emb.pooler_output is not None else emb.last_hidden_state.mean(1)
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


def _tokenize_zh(text: str) -> List[str]:
    """Chinese-aware tokenizer for BM25.

    Falls back to character-level tokenization if jieba is unavailable. This
    matters because the original `text.lower().split()` collapses each rule
    document into ~1 token (Chinese has no whitespace), making BM25 useless.
    """
    text = (text or "").lower()
    try:
        import jieba

        toks = [t for t in jieba.lcut(text) if t.strip()]
        return toks or list(text.replace(" ", ""))
    except Exception:
        return list(text.replace(" ", ""))


def build_text_index(
    rule_file: str,
    out_dir: Path = Path("data/rag_index"),
    extra_case_files: List[str] | None = None,
) -> None:
    """Build BM25 index over policy rules + (optionally) violation cases."""
    from rank_bm25 import BM25Okapi

    rules: List[dict] = []
    with open(rule_file) as f:
        for line in f:
            line = line.strip()
            if line:
                rules.append(json.loads(line))

    for path in extra_case_files or []:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # Normalise to {"text": ..., "source": ...}
                txt = rec.get("text") or rec.get("description") or rec.get("reason") or ""
                if txt:
                    rules.append({"text": txt, "source": path, **rec})

    corpus = [r.get("text", "") for r in rules]
    tokenized = [_tokenize_zh(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump((bm25, rules), f)

    print(f"BM25 index built: {len(rules)} documents (rules + cases)")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)

    image_paths = sorted(
        str(p)
        for p in Path(args.image_dir).rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    print(f"Found {len(image_paths)} images")
    if not args.skip_visual:
        build_visual_index(image_paths, clip_model_name=args.clip_model, out_dir=out_dir)

    if args.rule_file and os.path.exists(args.rule_file):
        extras = []
        if args.case_file and os.path.exists(args.case_file):
            extras.append(args.case_file)
        build_text_index(args.rule_file, out_dir=out_dir, extra_case_files=extras)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--rule_file", default="data/raw/rules.jsonl")
    parser.add_argument("--case_file", default="data/raw/violation_cases.jsonl",
                        help="Extra violation case corpus to merge into BM25 index.")
    parser.add_argument("--out_dir", default="data/rag_index")
    parser.add_argument("--clip_model", default="models/pretrained/clip-vit-base-patch32",
                        help="Local path or HF id. Local path avoids needing the network at index time.")
    parser.add_argument("--skip_visual", action="store_true",
                        help="Build BM25 only (useful when CLIP weights are not yet available).")
    run(parser.parse_args())
