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

from src.schema import SYSTEM_PROMPT, AuditOutput, coarse_category, try_parse


class AuditPipeline:
    def __init__(
        self,
        model_path: str,
        index_dir: str = "data/rag_index",
        confidence_threshold: float = 0.85,
        confidence_method: str = "field_min",
        top_k_visual: int = 3,
        top_k_text: int = 3,
        clip_model: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
    ):
        self.threshold = confidence_threshold
        self.confidence_method = confidence_method
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
    ) -> Tuple[str, Dict[str, float]]:
        """Run greedy decode and return the response + a dict of confidence
        signals that downstream gating can compose:

        * ``mean_max``    : legacy v1 — mean of greedy-token max softmax probs
        * ``min_max``     : min of greedy-token max softmax probs (catches the
                            single most uncertain token, robust to JSON
                            structure tokens that always score ~1.0)
        * ``mean_entropy``: mean per-token entropy (lower = more confident,
                            captures top-1 vs top-2 spread that ``mean_max``
                            misses)
        * ``field_min``   : ``min_max`` restricted to tokens that decode into
                            JSON value characters (skips structural tokens
                            like ``{``, ``}``, ``"``, ``:``). This is the
                            recommended single-value confidence.
        """
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
        confidence = self._compute_confidence(out.scores, generated_ids)
        return response_text, confidence

    @staticmethod
    def _compute_confidence(scores, generated_ids) -> Dict[str, float]:
        """Compute multi-view per-token confidence signals (see _generate)."""
        if not scores:
            return {"mean_max": 1.0, "min_max": 1.0, "mean_entropy": 0.0,
                    "field_min": 1.0, "n_tokens": 0, "n_field_tokens": 0}

        import math

        max_probs: List[float] = []
        entropies: List[float] = []
        for s in scores:
            logits = s[0]
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            max_probs.append(float(probs.max().item()))
            # Numerically stable entropy in nats
            entropies.append(float(-(probs * log_probs).sum().item()))

        # JSON structural tokens: anything whose decoded form is purely
        # punctuation / whitespace / digits-only contributes very little
        # information about the model's audit decision.
        _STRUCT_CHARS = set('{}[]":,\n\t \\')
        field_mask: List[bool] = []
        # generated_ids may be 2D (batch=1, T) or 1D depending on caller
        ids = generated_ids.tolist() if hasattr(generated_ids, "tolist") else list(generated_ids)
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        # Reusing the parent's tokenizer would couple this static method to
        # the instance; callers typically have only ~hundreds of tokens so the
        # overhead of decoding each id once is negligible.
        # However we don't have direct tokenizer access here, so we fall back
        # to a heuristic: tokens whose top-1 prob is *exactly* 1.0 are almost
        # always deterministic structural tokens (e.g. continuation of "true").
        for p in max_probs:
            field_mask.append(not (p > 0.999))

        field_probs = [p for p, keep in zip(max_probs, field_mask) if keep]
        return {
            "mean_max": sum(max_probs) / len(max_probs),
            "min_max": min(max_probs),
            "mean_entropy": sum(entropies) / len(entropies),
            "field_min": min(field_probs) if field_probs else 1.0,
            "n_tokens": len(max_probs),
            "n_field_tokens": len(field_probs),
        }

    # ------------------------------------------------------------------
    # RAG retrieval
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _embed_image(self, image: Image.Image) -> "np.ndarray":
        import numpy as np

        inputs = self.clip_proc(images=image, return_tensors="pt").to(self.device)
        emb = self.clip_model.get_image_features(**inputs)
        # transformers >= 4.46 may wrap the return; mirror indexer.build_visual_index
        if hasattr(emb, "image_embeds"):
            emb = emb.image_embeds
        elif hasattr(emb, "last_hidden_state"):
            emb = (
                emb.pooler_output
                if hasattr(emb, "pooler_output") and emb.pooler_output is not None
                else emb.last_hidden_state.mean(1)
            )
        emb = F.normalize(emb, dim=-1)
        return emb.cpu().numpy().astype("float32")

    def _retrieve_visual(self, image: Image.Image) -> List[dict]:
        # `top_k_visual <= 0` is the explicit "disable visual retrieval"
        # ablation knob (text-only RAG). Skip both CLIP encode and FAISS
        # search so the cost matches what we are actually measuring.
        if self.faiss_index is None or self.top_k_visual <= 0:
            return []
        emb = self._embed_image(image)
        distances, indices = self.faiss_index.search(emb, self.top_k_visual)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                results.append({"path": self.image_paths[idx], "score": float(dist)})
        return results

    def _retrieve_text(self, query: str) -> List[dict]:
        # Symmetric short-circuit for visual-only ablation.
        if self.bm25 is None or self.top_k_text <= 0:
            return []
        from src.stage4_rag.indexer import _tokenize_zh

        tokens = _tokenize_zh(query)
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
        *,
        return_debug: bool = False,
    ) -> AuditOutput:
        messages = self._build_messages(image, description)
        response, conf = self._generate(messages, image)
        triggered, score = self._should_trigger_rag(conf)

        rag_ctx = ""
        if triggered:
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

        if return_debug:
            return result, {
                "confidence": conf,
                "gating_score": score,
                "rag_triggered": triggered,
                "rag_context": rag_ctx,
                "raw_response": response,
            }
        return result

    # ------------------------------------------------------------------
    # Confidence gating
    # ------------------------------------------------------------------

    def _should_trigger_rag(self, conf: Dict[str, float]) -> Tuple[bool, float]:
        """Decide whether to invoke RAG given the multi-view confidence dict.

        Default policy is `field_min < self.threshold` because field-min is
        the most discriminative single signal we observed during calibration
        (`mean_max` is dominated by JSON structural tokens that always score
        ~1.0). Override with the `confidence_method` ctor arg to swap to
        another signal (e.g. `mean_max` for backward-compat).
        """
        method = getattr(self, "confidence_method", "field_min")
        score = float(conf.get(method, conf.get("mean_max", 1.0)))
        return score < self.threshold, score


# ----------------------------------------------------------------------------
# Stage 4.1 — Agentic RAG: Plan-Then-Retrieve + Verify-then-Rewrite
# ----------------------------------------------------------------------------

class AgenticAuditPipeline(AuditPipeline):
    """Two extensions on top of AuditPipeline that turn single-pass RAG into
    a 3-step agentic loop:

    1. **Plan-Then-Retrieve** — first-pass output's ``category`` is mapped to
       a coarse bucket (10 buckets via :func:`schema.coarse_category`); BM25
       text retrieval is then *routed* to docs whose ``category`` matches the
       same coarse bucket OR is "通用" (cross-category platform rules).
    2. **Long-tail forced trigger** — categories in the long-tail set
       (medical / electronics / food / others) bypass the confidence gate and
       always trigger retrieval, since these are the categories where the
       base model is least reliable (training data <2% each).
    3. **Verify-then-Rewrite** — after the second-pass output, ask the *same*
       model whether the audit conclusion is consistent with both the image
       and the retrieved evidence. We read the next-token softmax mass on
       "是" / "否" to obtain ``verify_score ∈ [0,1]``:

       - ``verify_score < vt_low``      → fall back to v1 (reject v2).
       - ``vt_low ≤ score < vt_high``   → rewrite once with feedback prompt.
       - ``verify_score ≥ vt_high``     → accept v2.
    """

    DEFAULT_LONG_TAIL = ("医药", "电子产品", "食品", "其他")

    def __init__(
        self,
        *args,
        long_tail_categories: Optional[List[str]] = None,
        verify_threshold_low: float = 0.5,
        verify_threshold_high: float = 0.7,
        enable_verify: bool = True,
        enable_rewrite: bool = True,
        enable_routing: bool = True,
        enable_long_tail_trigger: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.long_tail_set = set(long_tail_categories or self.DEFAULT_LONG_TAIL)
        self.vt_low = verify_threshold_low
        self.vt_high = verify_threshold_high
        self.enable_verify = enable_verify
        self.enable_rewrite = enable_rewrite
        self.enable_routing = enable_routing
        self.enable_long_tail_trigger = enable_long_tail_trigger

        tok = self.processor.tokenizer
        # Take the first sub-token id of "是"/"否" — Qwen tokenizer encodes
        # each Chinese character as a single token in practice, but we still
        # take [0] to be safe.
        self.yes_id = tok.encode("是", add_special_tokens=False)[0]
        self.no_id = tok.encode("否", add_special_tokens=False)[0]

    # ------------------------------------------------------------------
    # Routed text retrieval (Plan-Then-Retrieve)
    # ------------------------------------------------------------------

    def _retrieve_text_routed(self, query: str, coarse: str) -> List[dict]:
        """Same as ``_retrieve_text`` but only keeps docs whose ``category``
        matches ``coarse`` or is "通用" / unset (platform-wide rules)."""
        if self.bm25 is None or self.top_k_text <= 0:
            return []
        if not self.enable_routing:
            return self._retrieve_text(query)

        from src.stage4_rag.indexer import _tokenize_zh

        tokens = _tokenize_zh(query)
        scores = self.bm25.get_scores(tokens)

        import numpy as np

        keep_mask = np.zeros(len(self.rules), dtype=bool)
        for i, r in enumerate(self.rules):
            cat = r.get("category", "") or ""
            keep_mask[i] = (cat == coarse) or (cat in ("通用", "")) or (
                # cases use free-text but are normalised by coarse_category
                coarse_category(cat) == coarse
            )

        masked = scores.copy().astype("float32")
        masked[~keep_mask] = -1e9
        top_idx = masked.argsort()[-self.top_k_text :][::-1]
        return [
            {"rule": self.rules[i], "score": float(masked[i])}
            for i in top_idx
            if masked[i] > -1e8
        ]

    def _build_routed_context(
        self,
        image: Image.Image,
        description: str,
        coarse: str,
    ) -> Tuple[str, List[dict], List[dict]]:
        visual_hits = self._retrieve_visual(image)
        text_hits = self._retrieve_text_routed(description, coarse)

        lines: List[str] = []
        if visual_hits:
            lines.append("相似违规案例（视觉检索）：")
            for i, h in enumerate(visual_hits, 1):
                lines.append(f"  {i}. {h['path']} (相似度 {h['score']:.3f})")
        if text_hits:
            lines.append(f"相关平台规则（文本检索 · 路由至「{coarse}」品类）：")
            for i, h in enumerate(text_hits, 1):
                rule_text = (h["rule"].get("text", "") or "")[:200]
                cat = h["rule"].get("category", "")
                lines.append(f"  {i}. [{cat}] {rule_text}")
        return "\n".join(lines), visual_hits, text_hits

    # ------------------------------------------------------------------
    # Verifier (self-verification)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _verify(
        self,
        image: Image.Image,
        parsed: AuditOutput,
        rag_context: str,
    ) -> float:
        """Return P(yes) / (P(yes) + P(no)) at the next-token position.

        ``1.0`` = the model fully agrees that the audit conclusion is
        supported by image + retrieved evidence; ``0.0`` = strong disagreement.
        """
        attrs_str = "; ".join(
            f"{k}: {v}" for k, v in (parsed.attributes or {}).items()
        )
        verify_q = (
            "你是审核质检员。请基于商品图片以及下方的检索证据，"
            "判断「待复核审核结论」是否被同时支持。\n\n"
            f"【待复核审核结论】\n"
            f"品类：{parsed.category}\n"
            f"提取属性：{attrs_str}\n"
            f"违规判定：{parsed.violation}\n"
            f"理由：{parsed.reason}\n\n"
            f"【检索证据】\n{rag_context or '（无）'}\n\n"
            "结论的视觉断言是否与图片一致，且违规判定是否与证据一致？"
            "只输出一个汉字：是 或 否。"
        )
        messages = [
            {"role": "system", "content": "你是严谨的审核质检员，只输出「是」或「否」。"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": verify_q},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
        logits = out.scores[0][0]
        probs = F.softmax(logits, dim=-1)
        yes_p = float(probs[self.yes_id].item())
        no_p = float(probs[self.no_id].item())
        return yes_p / (yes_p + no_p + 1e-8)

    # ------------------------------------------------------------------
    # Public interface — overrides
    # ------------------------------------------------------------------

    def predict(  # type: ignore[override]
        self,
        image: Image.Image,
        description: str,
        *,
        return_debug: bool = False,
    ):
        # ---- Step 1 · First pass (no retrieval) ----
        msg_v1 = self._build_messages(image, description)
        resp_v1, conf = self._generate(msg_v1, image)
        parsed_v1 = try_parse(resp_v1)

        # ---- Step 2 · Multi-trigger gating ----
        triggered_by_conf, score = self._should_trigger_rag(conf)
        coarse = coarse_category(parsed_v1.category if parsed_v1 else "")
        triggered_by_long_tail = (
            self.enable_long_tail_trigger and coarse in self.long_tail_set
        )
        triggered = triggered_by_conf or triggered_by_long_tail

        debug: Dict[str, Any] = {
            "confidence": conf,
            "gating_score": score,
            "coarse_category": coarse,
            "triggered_by_conf": triggered_by_conf,
            "triggered_by_long_tail": triggered_by_long_tail,
            "rag_triggered": triggered,
            "verify_score": None,
            "fallback_used": False,
            "rewrite_used": False,
            "raw_response_v1": resp_v1,
        }

        if not triggered:
            result = parsed_v1 or AuditOutput(
                "unknown", {}, False, f"Parse failed: {resp_v1[:100]}"
            )
            return (result, debug) if return_debug else result

        # ---- Step 3 · Plan-Then-Retrieve (routed) ----
        rag_ctx, visual_hits, text_hits = self._build_routed_context(
            image, description, coarse
        )
        debug["rag_context"] = rag_ctx
        debug["retrieved_text_categories"] = [
            h["rule"].get("category", "") for h in text_hits
        ]
        debug["retrieved_visual_count"] = len(visual_hits)

        # ---- Step 4 · Re-generate with augmented context ----
        msg_v2 = self._build_messages(image, description, rag_context=rag_ctx)
        resp_v2, _ = self._generate(msg_v2, image)
        parsed_v2 = try_parse(resp_v2)
        debug["raw_response_v2"] = resp_v2

        if parsed_v2 is None:
            debug["fallback_used"] = True
            result = parsed_v1 or AuditOutput(
                "unknown", {}, False, f"Both passes parse-failed: {resp_v2[:100]}"
            )
            return (result, debug) if return_debug else result

        # ---- Step 5 · Verify ----
        if not self.enable_verify:
            return (parsed_v2, debug) if return_debug else parsed_v2

        verify_score = self._verify(image, parsed_v2, rag_ctx)
        debug["verify_score"] = verify_score

        # ---- Step 6 · Decide: accept / rewrite / fallback ----
        if verify_score < self.vt_low:
            debug["fallback_used"] = True
            result = parsed_v1 or parsed_v2
        elif verify_score < self.vt_high and self.enable_rewrite:
            debug["rewrite_used"] = True
            feedback = (
                "\n\n[质检员反馈] 上一版结论与证据匹配度不足。"
                "请重新生成 JSON：reason 中每条断言都必须能被检索证据或图片直接支持，"
                "不要引入证据中没有的属性。"
            )
            msg_v3 = self._build_messages(
                image, description, rag_context=rag_ctx + feedback
            )
            resp_v3, _ = self._generate(msg_v3, image)
            parsed_v3 = try_parse(resp_v3)
            debug["raw_response_v3"] = resp_v3
            result = parsed_v3 or parsed_v2
        else:
            result = parsed_v2

        return (result, debug) if return_debug else result
