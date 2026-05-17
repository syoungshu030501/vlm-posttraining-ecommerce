"""
Stage 0: Teacher-model distillation data construction via DashScope API.

Uses Qwen3.5-VL-Plus (or other Qwen VL models) through DashScope's
OpenAI-compatible endpoint to generate:
  - SFT samples   (image, prompt, response)
  - Preference pairs (image, prompt, chosen, rejected)

DashScope setup:
    export DASHSCOPE_API_KEY="sk-..."
    # Or: export OPENAI_API_KEY="sk-..."  OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

Usage:
    # Generate SFT data
    python -m src.stage0_distill.distill \
        --image_dir data/raw/images \
        --annotation_file data/raw/annotations.jsonl \
        --out_dir data/sft \
        --mode sft \
        --model qwen-vl-max

    # Generate preference pairs
    python -m src.stage0_distill.distill \
        --image_dir data/raw/images \
        --annotation_file data/raw/annotations.jsonl \
        --out_dir data/preference \
        --mode preference \
        --model qwen-vl-max

Available DashScope VL models (as of 2026-04):
    qwen-vl-max             — Qwen3.5-VL 旗舰（最强，推荐做教师）
    qwen-vl-max-latest      — 最新版本
    qwen-vl-plus            — Qwen3.5-VL 高性价比
    qwen2.5-vl-72b-instruct — Qwen2.5-VL 72B
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from src.schema import SYSTEM_PROMPT, try_parse

# ── DashScope client setup ──────────────────────────────────────────

def make_client() -> OpenAI:
    """Create OpenAI client pointing to DashScope."""
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set DASHSCOPE_API_KEY or OPENAI_API_KEY environment variable.\n"
            "Get your key at: https://dashscope.console.aliyun.com/apiKey"
        )

    base_url = os.environ.get(
        "OPENAI_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    return OpenAI(api_key=api_key, base_url=base_url)


# ── Image encoding ──────────────────────────────────────────────────

def encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_messages(image_path: str, description: str, system_prompt: str = SYSTEM_PROMPT) -> list:
    """Build chat messages with image for VL model."""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image_base64(image_path)}",
                    },
                },
                {
                    "type": "text",
                    "text": f"商品描述：{description}\n\n请分析这个商品图片并输出审核结果JSON。",
                },
            ],
        },
    ]


# ── Thinking mode handling ──────────────────────────────────────────

def strip_thinking(text: str) -> str:
    """
    Qwen3.5 默认开启 thinking mode，输出可能包含 <think>...</think>。
    蒸馏时我们需要纯 JSON，去掉 thinking 部分。
    """
    # Remove <think>...</think> blocks
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


# ── API call with retry ─────────────────────────────────────────────

def call_teacher(
    client: OpenAI,
    model: str,
    image_path: str,
    description: str,
    temperature: float = 0.7,
    max_retries: int = 3,
    enable_thinking: bool = False,
) -> Optional[str]:
    """
    Call DashScope VL model and return response text.

    Args:
        enable_thinking: If False (default), passes extra_body to disable thinking.
                        Some models may not support this parameter, so we also
                        strip <think> tags as fallback.
    """
    messages = build_messages(image_path, description)

    extra_body = {}
    if not enable_thinking:
        # DashScope Qwen3.5 supports enable_thinking parameter
        extra_body["enable_thinking"] = False

    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1024,
            )
            if extra_body:
                kwargs["extra_body"] = extra_body

            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            # Strip thinking tags as fallback
            content = strip_thinking(content)
            return content

        except Exception as e:
            err_msg = str(e)
            if "rate_limit" in err_msg.lower() or "429" in err_msg:
                wait = min(2 ** attempt * 2, 30)
                print(f"  [rate limit] waiting {wait}s...")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [WARN] Failed after {max_retries} attempts: {e}")
                return None

    return None


# ── SFT sample generation ───────────────────────────────────────────

SFT_SYSTEM_PROMPT = (
    "你是一位专业的电商商品合规审核员。\n"
    "分析给定的商品图片和描述，输出一个JSON对象，包含以下字段：\n"
    '- "category" (str): 商品品类\n'
    '- "attributes" (dict): 从图片中提取的关键视觉属性（颜色、材质、款式等）\n'
    '- "violation" (bool): 该商品是否违反平台规则\n'
    '- "reason" (str): 审核理由，必须引用具体的视觉属性作为证据\n'
    "只输出合法JSON，不要markdown代码块或其他文本。"
)

# Separate prompt to generate violation scenarios
VIOLATION_SYSTEM_PROMPT = (
    "你是一位专业的电商商品合规审核员。\n"
    "分析给定的商品图片和描述。这个商品存在违规问题。\n"
    "请指出违规之处，输出一个JSON对象，包含以下字段：\n"
    '- "category" (str): 商品品类\n'
    '- "attributes" (dict): 从图片中提取的关键视觉属性\n'
    '- "violation" (bool): 必须为 true\n'
    '- "reason" (str): 详细的违规原因，引用具体视觉属性\n'
    "只输出合法JSON，不要markdown代码块或其他文本。"
)


def generate_sft_sample(
    client: OpenAI,
    model: str,
    image_path: str,
    description: str,
    force_violation: bool = False,
) -> Optional[dict]:
    """Generate one SFT sample."""
    system = VIOLATION_SYSTEM_PROMPT if force_violation else SFT_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image_base64(image_path)}"},
                },
                {"type": "text", "text": f"商品描述：{description}"},
            ],
        },
    ]

    extra_body = {"enable_thinking": False}
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3 if not force_violation else 0.7,
            max_tokens=1024,
            extra_body=extra_body,
        )
        content = strip_thinking(resp.choices[0].message.content)
    except Exception as e:
        print(f"  [WARN] API error: {e}")
        return None

    parsed = try_parse(content)
    if parsed is None or not parsed.is_valid():
        return None

    return {
        "image_file": os.path.basename(image_path),
        "image": image_path,
        "prompt": description,
        "response": content,
        "violation": parsed.violation,
        "category": parsed.category,
    }


# ── Preference pair generation ──────────────────────────────────────

def generate_preference_pair(
    client: OpenAI,
    model: str,
    image_path: str,
    description: str,
) -> Optional[dict]:
    """Generate chosen/rejected pair for RM training."""
    # chosen: low temperature, deterministic
    chosen = call_teacher(client, model, image_path, description, temperature=0.1)
    if chosen is None or not try_parse(chosen):
        return None

    # rejected: high temperature to induce hallucination/format errors
    rejected = call_teacher(client, model, image_path, description, temperature=1.2)
    if rejected is None:
        return None

    # Ensure rejected is actually worse (parseable but lower quality, or unparseable)
    chosen_parsed = try_parse(chosen)
    rejected_parsed = try_parse(rejected)

    # If both parse and are identical, skip
    if chosen_parsed and rejected_parsed:
        if chosen_parsed.violation == rejected_parsed.violation and \
           chosen_parsed.category == rejected_parsed.category:
            # Not enough contrast; try once more with even higher temp
            rejected = call_teacher(client, model, image_path, description, temperature=1.5)
            if rejected is None:
                return None

    return {
        "image_file": os.path.basename(image_path),
        "image": image_path,
        "prompt": description,
        "chosen": chosen,
        "rejected": rejected,
    }


# ── Main pipeline ───────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    client = make_client()

    ann_path = Path(args.annotation_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotations: List[dict] = []
    with ann_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                annotations.append(json.loads(line))

    if args.max_samples:
        annotations = annotations[: args.max_samples]

    print(f"Mode: {args.mode} | Model: {args.model} | Samples: {len(annotations)}")

    out_file = out_dir / f"{args.mode}.jsonl"
    written = 0
    failed = 0

    # Resume support: skip already-processed entries
    existing = set()
    if out_file.exists() and args.resume:
        with out_file.open() as f:
            for line in f:
                d = json.loads(line)
                existing.add(d.get("image_file", d.get("image", "")))
        print(f"  Resuming: {len(existing)} already processed")

    mode_open = "a" if (args.resume and out_file.exists()) else "w"

    with out_file.open(mode_open) as fout:
        for i, ann in enumerate(annotations):
            img_file = ann.get("image_file", "")
            if img_file in existing:
                continue

            image_path = os.path.join(args.image_dir, img_file)
            if not os.path.exists(image_path):
                print(f"  [SKIP] Image not found: {image_path}")
                continue

            description = ann.get("description", "")

            if args.mode == "sft":
                # 70% normal, 30% forced violation for balanced data
                force_viol = (i % 10) < 3
                sample = generate_sft_sample(
                    client, args.model, image_path, description,
                    force_violation=force_viol,
                )
            else:
                sample = generate_preference_pair(
                    client, args.model, image_path, description,
                )

            if sample is not None:
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                fout.flush()
                written += 1
            else:
                failed += 1

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(annotations)}] written={written}, failed={failed}")

            # Rate limit: ~10 QPS for DashScope free tier
            if args.rate_limit > 0:
                time.sleep(1.0 / args.rate_limit)

    print(f"\nDone. Wrote {written} samples to {out_file} (failed: {failed})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, help="Directory containing product images")
    parser.add_argument("--annotation_file", required=True, help="JSONL with image_file + description")
    parser.add_argument("--out_dir", default="data/sft")
    parser.add_argument("--mode", choices=["sft", "preference"], default="sft")
    parser.add_argument("--model", default="qwen-vl-max",
                        help="DashScope model: qwen-vl-max / qwen-vl-plus / qwen2.5-vl-72b-instruct")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--rate_limit", type=float, default=5.0, help="Max requests per second (0=unlimited)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    run(parser.parse_args())
