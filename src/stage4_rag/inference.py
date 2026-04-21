"""
Stage 4: RAG-augmented inference pipeline.

Flow:
    image + text
        → policy model direct inference
        → if avg_token_confidence < threshold: trigger RAG
            ├── FAISS: retrieve top-k visually similar violation cases
            ├── BM25:  retrieve top-k policy rule documents
            └── re-infer with retrieved context injected
        → return final AuditOutput

Usage:
    from src.stage4_rag.inference import AuditPipeline

    pipeline = AuditPipeline(
        model_path="models/rl_ckpt/...",
        index_dir="data/rag_index",
    )
    result = pipeline.predict(image, description)
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from src.schema import SYSTEM_PROMPT, AuditOutput, try_parse


class AuditPipeline:
    def __init__(
        self,
        model_path: str,
        index_dir: str = "data/rag_index",
        confidence_threshold: float = 0.85,
        top_k_visual: int = 3,
        top_k_text: int = 3,
        clip_model: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
    ):
        self.threshold = confidence_threshold
        self.top_k_visual = top_k_visual
        self.top_k_text = top_k_text
        self.device = device

        from src.utils.model_loader import load_model_and_processor

        self.model, self.processor = load_model_and_processor(
            model_path,
            apply_lora=False,
            use_flash_attn=True,
        )
        self.model.eval()

        self._load_indices(index_dir, clip_model)

    # ------------------------------------------------------------------
    # Index loading
    # ------------------------------------------------------------------

    def _load_indices(self, index_dir: str, clip_model: str) -> None:
        idx_dir = Path(index_dir)
        self.faiss_index = None
        self.image_paths: List[str] = []
        self.bm25 = None
        self.rules: List[dict] = []

        faiss_path = idx_dir / "visual.faiss"
        if faiss_path.exists():
            import faiss

            self.faiss_index = faiss.read_index(str(faiss_path))
            with open(idx_dir / "image_paths.pkl", "rb") as f:
                self.image_paths = pickle.load(f)

            from transformers import CLIPModel, CLIPProcessor

            self.clip_proc = CLIPProcessor.from_pretrained(clip_model)
            self.clip_model = CLIPModel.from_pretrained(clip_model).to(self.device)
            self.clip_model.eval()

        bm25_path = idx_dir / "bm25.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                self.bm25, self.rules = pickle.load(f)

    # ------------------------------------------------------------------
    # Core inference helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        image: Image.Image,
        description: str,
        rag_context: str = "",
    ) -> list:
        user_content: list = [
            {"type": "image", "image": image},
            {"type": "text", "text": description},
        ]
        if rag_context:
            user_content.append(
                {"type": "text", "text": f"\n\n[参考案例与规则]\n{rag_context}"}
            )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    @torch.inference_mode()
    def _generate(
        self,
        messages: list,
        image: Image.Image,
    ) -> Tuple[str, float]:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.device)

        out = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        generated_ids = out.sequences[0, inputs["input_ids"].shape[1] :]
        response_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        # Average token confidence
        if out.scores:
            log_probs = [F.log_softmax(s[0], dim=-1).max().item() for s in out.scores]
            confidence = float(torch.tensor(log_probs).exp().mean())
        else:
            confidence = 1.0

        return response_text, confidence

    # ------------------------------------------------------------------
    # RAG retrieval
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _embed_image(self, image: Image.Image) -> "np.ndarray":
        import numpy as np

        inputs = self.clip_proc(images=image, return_tensors="pt").to(self.device)
        emb = self.clip_model.get_image_features(**inputs)
        emb = F.normalize(emb, dim=-1)
        return emb.cpu().numpy().astype("float32")

    def _retrieve_visual(self, image: Image.Image) -> List[dict]:
        if self.faiss_index is None:
            return []
        emb = self._embed_image(image)
        distances, indices = self.faiss_index.search(emb, self.top_k_visual)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                results.append({"path": self.image_paths[idx], "score": float(dist)})
        return results

    def _retrieve_text(self, query: str) -> List[dict]:
        if self.bm25 is None:
            return []
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idx = scores.argsort()[-self.top_k_text :][::-1]
        return [{"rule": self.rules[i], "score": float(scores[i])} for i in top_idx]

    def _build_rag_context(self, image: Image.Image, description: str) -> str:
        visual_hits = self._retrieve_visual(image)
        text_hits = self._retrieve_text(description)

        lines = []
        if visual_hits:
            lines.append("相似违规案例（视觉检索）：")
            for i, h in enumerate(visual_hits, 1):
                lines.append(f"  {i}. {h['path']} (相似度 {h['score']:.3f})")

        if text_hits:
            lines.append("相关平台规则（文本检索）：")
            for i, h in enumerate(text_hits, 1):
                rule_text = h["rule"].get("text", "")[:200]
                lines.append(f"  {i}. {rule_text}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(
        self,
        image: Image.Image,
        description: str,
    ) -> AuditOutput:
        messages = self._build_messages(image, description)
        response, confidence = self._generate(messages, image)

        if confidence < self.threshold:
            rag_ctx = self._build_rag_context(image, description)
            rag_messages = self._build_messages(image, description, rag_context=rag_ctx)
            response, _ = self._generate(rag_messages, image)

        result = try_parse(response)
        if result is None:
            # Fallback: return safe default
            result = AuditOutput(
                category="unknown",
                attributes={},
                violation=False,
                reason=f"Parse failed: {response[:100]}",
            )
        return result
