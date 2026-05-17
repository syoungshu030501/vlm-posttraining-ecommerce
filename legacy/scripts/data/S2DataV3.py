"""
Stage 2 v3 — Field-wise preference data producer.

把"已挖好的 hard samples"再升级为 field-wise PRM 训练数据，输出
preference_v3.parquet（每条样本含 5 个 field 的独立 soft 标签）。

数据产出 pipeline（每条 image+prompt+gold_response）:

    1. 用 sft_aux_merged 跑 K=8 rollouts (greedy 1 + sample K-1)
    2. 解析每个 rollout JSON
    3. attributes mc_i:
         对 gold_response 中每个 (key, val)，看其在 K 个 rollouts 中
         作为 attribute value 出现的次数 / K → mc ∈ [0, 1]
    4. violation_prob:
         (a) --teacher_vote: 调 qwen-vl-max + qwen-plus 各 5 次 → 10 votes
         (b) 否则: K 个 rollouts 自身的 violation 一致率
    5. violation_type:
         (a) 教师投票最常见的 type (优先)
         (b) 否则: 用关键词匹配落到 VIOLATION_TYPES 11 类之一
    6. reason_align:
         BGE-zh-small 余弦(reason, "key1: val1; key2: val2; ...")
    7. rejected = K rollouts 中按"与 gold 最不一致"挑出的一条

输出 schema (parquet):
    image           bytes
    image_file      str
    prompt          str
    response        str            # gold = chosen
    chosen          str            # 同 response（兼容旧 PreferenceDataset）
    rejected        str            # picked rollout
    label_category_coarse           str
    label_attributes_json           str (JSON of [{key,val,grounded,mc}, ...])
    label_violation_prob            float
    label_violation_type            str
    label_reason_align              float
    mc_metadata_json                str  (per-sample debug info)
    pair_strategy                   str  (always 'mc_v3' for new data; preserves v2 column)
    policy_ckpt                     str

Modes:
    --dry_run          : 不加载模型/不调 API，用 mock 跑代码全流程
    --no_teacher_vote  : 跳过 API 调用，violation_prob 仅来自 rollouts
    (默认)              : rollout + teacher vote 全开

Usage:
    # smoke test
    python -m scripts.data.S2DataV3 --dry_run --limit 5

    # 真实生产 (无 API)
    python -m scripts.data.S2DataV3 \\
        --policy_path models/sft_aux_merged \\
        --in_jsonl data/sft/sft.jsonl \\
        --image_dir data/raw/images \\
        --out_parquet data/preference_v3/preference_v3.parquet \\
        --no_teacher_vote --K 8 --limit 100

    # 真实生产 (含 API 投票)
    export DASHSCOPE_API_KEY="sk-xxx"
    python -m scripts.data.S2DataV3 \\
        --policy_path models/sft_aux_merged \\
        --in_jsonl data/sft/train.jsonl \\
        --image_dir data/raw/images \\
        --out_parquet data/preference_v3/preference_v3.parquet \\
        --K 8
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

# Allow `python -m scripts.data.S2DataV3` from repo root
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schema import (
    coarse_category,
    normalize_violation_type,
    try_parse,
    SYSTEM_PROMPT,
    VIOLATION_TYPES,
)


# ---------------------------------------------------------------------------
# Violation type heuristic mapping (fallback when no teacher vote)
# ---------------------------------------------------------------------------

VIOLATION_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "极限词":     ("最", "顶级", "唯一", "首选", "国家级", "极致", "100%", "绝对", "完全", "彻底"),
    "材质虚标":   ("材质", "面料", "成分", "纯棉", "真皮", "纤维", "标注", "实际"),
    "功效夸大":   ("功效", "瘦身", "美白", "祛斑", "速效", "见效", "三天", "一夜", "永久", "根治"),
    "品牌侵权":   ("品牌", "logo", "山寨", "盗版", "假冒", "授权", "商标", "Nike", "Adidas"),
    "价格欺诈":   ("价格", "价", "打折", "原价", "买一送", "返现", "充值", "退款"),
    "图文不符":   ("不符", "实物", "色差", "图片", "夸大色彩", "买家秀", "实拍"),
    "涉黄涉政":   ("黄色", "色情", "暗示", "敏感", "境外", "政治", "诱导", "图腾"),
    "医疗夸大":   ("治疗", "治愈", "肿瘤", "糖尿病", "高血压", "医院", "医疗器械", "替代"),
    "虚假代言":   ("代言", "明星", "盗用", "P图", "授权函", "网红", "白大褂"),
    "违禁品":     ("管制", "刀具", "气枪", "电棍", "走私", "破解", "盗版", "受保护"),
}


def keyword_violation_type(reason: str, attributes: dict | None) -> str:
    """Best-effort fallback: pick a VIOLATION_TYPES entry based on reason / attrs."""
    if not reason:
        return "无违规"
    text = reason
    if attributes:
        text += " " + " ".join(f"{k} {v}" for k, v in attributes.items())
    counts: dict[str, int] = {}
    for vt, kws in VIOLATION_TYPE_KEYWORDS.items():
        c = sum(1 for kw in kws if kw in text)
        if c > 0:
            counts[vt] = c
    if not counts:
        return "无违规"
    return max(counts.items(), key=lambda x: x[1])[0]


# ---------------------------------------------------------------------------
# Mock/Real policy
# ---------------------------------------------------------------------------

class MockPolicy:
    """Generates plausible but fake JSON rollouts. For dry-run / unit tests."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def rollout(
        self,
        image_path: str,
        prompt: str,
        gold_response: str,
        K: int,
    ) -> list[str]:
        """Return K rollouts. K-1 are perturbations of gold; 1 is wildly wrong."""
        gold = try_parse(gold_response)
        if gold is None:
            return [gold_response] * K

        outs: list[str] = []
        for i in range(K):
            attrs = dict(gold.attributes)
            # 每条以一定概率扰动一个属性值
            if i > 0 and attrs and self.rng.random() < 0.4:
                k = self.rng.choice(list(attrs.keys()))
                attrs[k] = attrs[k] + self.rng.choice(["（变化）", "*", "_"])
            # 1/K 概率给一个完全错的 violation
            v = gold.violation
            if i == K - 1:
                v = not gold.violation
            r = json.dumps({
                "category": gold.category,
                "attributes": attrs,
                "violation": v,
                "reason": gold.reason + (" rollout-mock" if i > 0 else ""),
            }, ensure_ascii=False)
            outs.append(r)
        return outs


class RealPolicy:
    """Wraps load_model_and_processor for K rollouts."""

    def __init__(self, model_path: str, flash_attn: bool = True):
        from PIL import Image
        import torch

        from src.utils.model_loader import load_model_and_processor

        self.Image = Image
        self.torch = torch
        print(f"[v3] loading policy from {model_path}")
        self.model, self.processor = load_model_and_processor(
            model_path, apply_lora=False, use_flash_attn=flash_attn,
        )
        self.model.eval()
        # left-padding for batched generate
        self.processor.tokenizer.padding_side = "left"

    def _build_inputs(self, image, prompt: str):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return self.processor(
            text=[text], images=[image], return_tensors="pt", padding=True,
        ).to(self.model.device)

    def rollout(self, image_path: str, prompt: str, gold_response: str, K: int) -> list[str]:
        image = self.Image.open(image_path).convert("RGB")
        outs: list[str] = []
        # 1 greedy + (K-1) sampled
        for i in range(K):
            inputs = self._build_inputs(image, prompt)
            with self.torch.inference_mode():
                gen = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=(i > 0),
                    temperature=0.7 if i > 0 else 1.0,
                    top_p=0.95,
                    pad_token_id=(
                        self.processor.tokenizer.pad_token_id
                        or self.processor.tokenizer.eos_token_id
                    ),
                )
            prompt_len = inputs["input_ids"].shape[1]
            text = self.processor.tokenizer.decode(gen[0, prompt_len:], skip_special_tokens=True)
            outs.append(text.strip())
        return outs


# ---------------------------------------------------------------------------
# Mock/Real teacher
# ---------------------------------------------------------------------------

class MockTeacher:
    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)

    def vote(self, image_path: str, prompt: str, gold_violation: bool) -> list[dict]:
        """10 votes, biased toward gold but with noise."""
        votes = []
        for _ in range(10):
            v = gold_violation if self.rng.random() < 0.8 else (not gold_violation)
            t = self.rng.choice(VIOLATION_TYPES) if v else "无违规"
            votes.append({"violation": v, "violation_type": t})
        return votes


class RealTeacher:
    """Calls DashScope qwen-vl-max + qwen-plus, 5 temps each."""

    TEMPS = (0.3, 0.5, 0.7, 0.9, 1.0)

    def __init__(self, models: tuple[str, ...] = ("qwen-vl-max", "qwen-plus")):
        from openai import OpenAI

        api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Set DASHSCOPE_API_KEY")
        base_url = os.environ.get(
            "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.models = models

    def _encode_img(self, image_path: str) -> str:
        import base64

        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def _call_one(self, model: str, image_path: str, prompt: str, temperature: float) -> Optional[dict]:
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{self._encode_img(image_path)}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                temperature=temperature,
                max_tokens=512,
                extra_body={"enable_thinking": False},
            )
            text = resp.choices[0].message.content
        except Exception as e:
            print(f"  [WARN] teacher call failed: {e}")
            return None
        parsed = try_parse(text)
        if parsed is None:
            return None
        return {
            "violation": bool(parsed.violation),
            "violation_type": getattr(parsed, "violation_type", None) or (
                "无违规" if not parsed.violation else None
            ),
            "reason": parsed.reason,
            "model": model,
            "temperature": temperature,
        }

    def vote(self, image_path: str, prompt: str, gold_violation: bool) -> list[dict]:
        votes = []
        for m in self.models:
            for t in self.TEMPS:
                v = self._call_one(m, image_path, prompt, t)
                if v is not None:
                    votes.append(v)
                time.sleep(0.25)  # rate limit ~4 QPS
        return votes


# ---------------------------------------------------------------------------
# Encoder (BGE) for reason_align — optional
# ---------------------------------------------------------------------------

class _IdentityEncoder:
    """No-op encoder used in dry-run; reason_align falls back to a heuristic."""

    def encode(self, texts: list[str], **_) -> list:
        return [0.0] * len(texts)


def make_encoder(model_name: str, device: str = "cpu", use_dummy: bool = False):
    if use_dummy:
        return _IdentityEncoder()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[WARN] sentence_transformers not installed, using identity encoder")
        return _IdentityEncoder()
    return SentenceTransformer(model_name, device=device)


def cosine(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim == 0 or b.ndim == 0 or a.size == 0 or b.size == 0:
        return 0.0
    na = float((a ** 2).sum() ** 0.5)
    nb = float((b ** 2).sum() ** 0.5)
    if na == 0 or nb == 0:
        return 0.0
    return float((a * b).sum() / (na * nb))


# ---------------------------------------------------------------------------
# Field-label computation
# ---------------------------------------------------------------------------

def attributes_string(attrs: dict | None) -> str:
    if not attrs:
        return ""
    return "; ".join(f"{k}: {v}" for k, v in attrs.items())


def compute_attribute_mc(
    gold_attrs: dict[str, Any] | None, rollout_attrs: list[dict | None],
) -> list[dict]:
    """For each (key, val) in gold_attrs, compute mc = freq across K rollouts."""
    if not gold_attrs:
        return []
    K = len(rollout_attrs)
    out = []
    for key, val in gold_attrs.items():
        val_str = str(val).strip().lower()
        match = 0
        for r in rollout_attrs:
            if not r:
                continue
            r_val = r.get(key)
            if r_val is None:
                continue
            if str(r_val).strip().lower() == val_str:
                match += 1
        mc = match / max(K, 1)
        out.append({
            "key": str(key),
            "val": str(val),
            "grounded": float(mc),  # 用 mc 直接当 soft label
            "mc": float(mc),
        })
    return out


def compute_violation_prob(
    rollout_violations: list[bool | None],
    teacher_votes: list[dict] | None,
) -> tuple[float, str]:
    """
    Returns (violation_prob, violation_type).

    Priority:
      1. teacher_votes > 0 → 用教师投票频率 + 多数 type
      2. else → 用 rollout 一致率 + keyword fallback type
    """
    if teacher_votes:
        n_pos = sum(1 for v in teacher_votes if v.get("violation"))
        prob = n_pos / max(len(teacher_votes), 1)
        type_counts = Counter(
            normalize_violation_type(v.get("violation_type"))
            for v in teacher_votes if v.get("violation_type") is not None
        )
        if not type_counts:
            vt = "无违规" if prob < 0.5 else "极限词"
        else:
            vt = type_counts.most_common(1)[0][0]
        return prob, vt

    valid_rollouts = [v for v in rollout_violations if v is not None]
    if not valid_rollouts:
        return 0.5, "无违规"
    prob = sum(1 for v in valid_rollouts if v) / len(valid_rollouts)
    return prob, ""  # type 由 caller 用 keyword fallback 填


def pick_rejected(
    gold_response: str,
    rollouts: list[str],
    rollout_parses: list,
    gold_parsed,
) -> str:
    """挑一条最 'rejected' 的 rollout：优先选 violation 翻转的；否则选属性偏离最多的。"""
    if gold_parsed is None:
        return rollouts[-1] if rollouts else ""
    flipped = [
        (i, r) for i, (r, p) in enumerate(zip(rollouts, rollout_parses))
        if p is not None and p.violation != gold_parsed.violation
    ]
    if flipped:
        return flipped[0][1]
    # else 选 attributes 重合度最低的
    gold_attr_set = {(k, str(v).strip().lower()) for k, v in (gold_parsed.attributes or {}).items()}
    if not gold_attr_set:
        return rollouts[-1]
    best_idx, best_overlap = 0, 1.0
    for i, p in enumerate(rollout_parses):
        if p is None:
            continue
        ra_set = {(k, str(v).strip().lower()) for k, v in (p.attributes or {}).items()}
        overlap = len(gold_attr_set & ra_set) / max(len(gold_attr_set), 1)
        if overlap < best_overlap:
            best_overlap, best_idx = overlap, i
    return rollouts[best_idx]


# ---------------------------------------------------------------------------
# Main producer
# ---------------------------------------------------------------------------

def produce_one(
    sample: dict,
    image_dir: Path,
    policy,
    teacher,
    encoder,
    K: int = 8,
) -> Optional[dict]:
    image_file = sample.get("image_file") or os.path.basename(sample.get("image", ""))
    image_path = image_dir / image_file
    if not image_path.exists():
        print(f"  [SKIP] image not found: {image_path}")
        return None

    prompt = str(sample.get("prompt") or sample.get("description") or "")
    gold_response = str(sample.get("response") or "")
    gold_parsed = try_parse(gold_response)
    if gold_parsed is None:
        print(f"  [SKIP] gold not parseable: {image_file}")
        return None

    # 1. K rollouts
    rollouts = policy.rollout(str(image_path), prompt, gold_response, K=K)
    rollout_parses = [try_parse(r) for r in rollouts]

    # 2. attributes mc
    rollout_attrs = [p.attributes if p else None for p in rollout_parses]
    attr_labels = compute_attribute_mc(gold_parsed.attributes, rollout_attrs)

    # 3. violation_prob + type
    rollout_violations = [p.violation if p else None for p in rollout_parses]
    teacher_votes = teacher.vote(str(image_path), prompt, gold_parsed.violation) if teacher else None
    vio_prob, vio_type = compute_violation_prob(rollout_violations, teacher_votes)
    if not vio_type:
        vio_type = keyword_violation_type(gold_parsed.reason, gold_parsed.attributes)
        if not gold_parsed.violation:
            vio_type = "无违规"

    # 4. reason_align via BGE
    reason_align = 0.0
    if gold_parsed.reason and gold_parsed.attributes:
        attrs_str = attributes_string(gold_parsed.attributes)
        if isinstance(encoder, _IdentityEncoder):
            # crude fallback: token overlap rate
            tokens = set(re.split(r"[，。；:：\s]+", gold_parsed.reason))
            attr_tokens = set(re.split(r"[，。；:：\s]+", attrs_str))
            if attr_tokens:
                reason_align = len(tokens & attr_tokens) / max(len(attr_tokens), 1)
        else:
            embs = encoder.encode([gold_parsed.reason, attrs_str])
            reason_align = (cosine(embs[0], embs[1]) + 1.0) / 2.0  # rescale [-1,1] → [0,1]

    # 5. pick rejected
    rejected = pick_rejected(gold_response, rollouts, rollout_parses, gold_parsed)

    # 6. assemble row
    label_attributes_json = json.dumps(attr_labels, ensure_ascii=False)
    mc_meta = {
        "K": K,
        "rollout_violations_pos_rate": (
            sum(1 for v in rollout_violations if v) / max(len([v for v in rollout_violations if v is not None]), 1)
        ),
        "n_teacher_votes": len(teacher_votes) if teacher_votes else 0,
        "teacher_violations_pos_rate": (
            sum(1 for v in teacher_votes if v.get("violation")) / max(len(teacher_votes), 1)
            if teacher_votes else None
        ),
        "n_parseable_rollouts": sum(1 for p in rollout_parses if p is not None),
    }

    image_bytes = image_path.read_bytes()

    return {
        "image": image_bytes,
        "image_file": image_file,
        "prompt": prompt,
        "response": gold_response,
        "chosen": gold_response,
        "rejected": rejected,
        "category": gold_parsed.category,
        "label_category_coarse": coarse_category(gold_parsed.category),
        "label_attributes_json": label_attributes_json,
        "label_violation_prob": float(vio_prob),
        "label_violation_type": vio_type,
        "label_reason_align": float(reason_align),
        "mc_metadata_json": json.dumps(mc_meta, ensure_ascii=False),
        "pair_strategy": "mc_v3",
        "policy_ckpt": "sft_aux_merged",  # filled by caller arg below if not dry_run
    }


def run(args: argparse.Namespace) -> None:
    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir)

    samples: list[dict] = []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if args.limit > 0:
        samples = samples[: args.limit]
    print(f"[v3] {len(samples)} input samples")

    # Resume support
    seen: set[str] = set()
    if args.resume and out_path.exists():
        try:
            import pandas as pd
            df_old = pd.read_parquet(out_path)
            seen = set(df_old["image_file"].astype(str).tolist())
            print(f"[v3] resume: {len(seen)} samples already done")
        except Exception:
            pass

    # Initialize components
    if args.dry_run:
        print("[v3] DRY RUN — using MockPolicy + MockTeacher + identity encoder")
        policy = MockPolicy()
        teacher = MockTeacher() if not args.no_teacher_vote else None
        encoder = make_encoder("", use_dummy=True)
    else:
        policy = RealPolicy(args.policy_path, flash_attn=args.flash_attn)
        teacher = None if args.no_teacher_vote else RealTeacher(
            models=tuple(args.teacher_models.split(","))
        )
        encoder = make_encoder(args.encoder_model, device=args.encoder_device)

    rows: list[dict] = []
    t0 = time.time()
    for i, s in enumerate(samples):
        img_file = s.get("image_file") or os.path.basename(s.get("image", ""))
        if img_file in seen:
            continue
        try:
            row = produce_one(
                s, image_dir, policy, teacher, encoder, K=args.K,
            )
        except Exception as e:
            print(f"  [WARN] sample {i} ({img_file}): {e}")
            continue
        if row is None:
            continue
        row["policy_ckpt"] = args.policy_path if not args.dry_run else "mock"
        rows.append(row)

        if (i + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(samples)}] kept={len(rows)} elapsed={elapsed:.0f}s "
                  f"rate={(len(rows)/max(elapsed,1e-6)):.2f}/s")

        # incremental dump every N (resume safety)
        if args.dump_every > 0 and len(rows) >= args.dump_every:
            _dump(rows, out_path, append=True)
            rows = []

    if rows:
        _dump(rows, out_path, append=out_path.exists())

    # Print final stats
    import pandas as pd
    df = pd.read_parquet(out_path)
    print(f"\n[v3] DONE. wrote {len(df)} rows → {out_path}")
    if "label_violation_type" in df.columns:
        print("\nviolation_type distribution:")
        for t, c in df["label_violation_type"].value_counts().items():
            print(f"  {t}: {c}")
    if "label_violation_prob" in df.columns:
        print(f"\nviolation_prob: mean={df['label_violation_prob'].mean():.3f}, "
              f"std={df['label_violation_prob'].std():.3f}")
    if "label_reason_align" in df.columns:
        print(f"reason_align:   mean={df['label_reason_align'].mean():.3f}")


def _dump(rows: list[dict], out_path: Path, append: bool) -> None:
    import pandas as pd

    df_new = pd.DataFrame(rows)
    if append and out_path.exists():
        df_old = pd.read_parquet(out_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(out_path, index=False)
    print(f"  → wrote checkpoint: {len(df)} total rows in {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_jsonl", default="data/sft/sft.jsonl")
    parser.add_argument("--image_dir", default="data/raw/images")
    parser.add_argument("--out_parquet", default="data/preference_v3/preference_v3.parquet")
    parser.add_argument("--policy_path", default="models/sft_aux_merged",
                        help="Frozen policy used for K rollouts")
    parser.add_argument("--K", type=int, default=8, help="MC rollout count")
    parser.add_argument("--teacher_models", default="qwen-vl-max,qwen-plus",
                        help="Comma-separated DashScope model ids for teacher voting")
    parser.add_argument("--no_teacher_vote", action="store_true",
                        help="Skip API calls; violation_prob from rollouts only")
    parser.add_argument("--encoder_model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--encoder_device", default="cpu")
    parser.add_argument("--flash_attn", action="store_true", default=True)
    parser.add_argument("--no_flash_attn", dest="flash_attn", action="store_false")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--dump_every", type=int, default=50,
                        help="Incremental parquet dump every N kept rows (0 = only at end)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    run(parser.parse_args())
