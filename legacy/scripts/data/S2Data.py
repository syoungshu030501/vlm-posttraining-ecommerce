"""
Stage 2 偏好数据 API 蒸馏：生成同图同品类但推理质量更差的 rejected。

动机
----
现有 preference.jsonl (4001 对) 80% 的 rejected 是其它图片的 SFT gold（描述另一件商品）,
RM 只会学到 image-text matching 的浅层 shortcut，无法学到推理质量的偏好。
本脚本针对每张训练图，用 qwen-vl-max 基于同一张图生成一个"审核质量更差的" rejected，
与 SFT gold (chosen) 组成 hard negative pair。

四种退化模式（每条样本随机采一种）
-----------------------------------
- weak_evidence  : 保留 violation 结论，但 reason 空泛不引证具体属性
- missed_cue     : violation=True 的样本改判 False（漏判违规），给宽松理由
- over_strict    : violation=False 的样本改判 True（误判违规），给过严理由
- wrong_attribute: 把 1-2 个视觉可辨属性改为错值，reason 用错误属性作证据

用法
----
    export DASHSCOPE_API_KEY=sk-xxx
    python scripts/data/S2Data.py \
        --sft_file data/sft/sft.jsonl \
        --pool_file data/preference/preference.jsonl \
        --out_file data/preference/preference_distilled.jsonl \
        --n_pairs 1500 --workers 6 --rate_limit 10
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Optional

from openai import OpenAI


# ── API client ──────────────────────────────────────────────
def make_client() -> OpenAI:
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("set DASHSCOPE_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get(
            "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
    )


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def safe_load_json(text: str) -> Optional[dict]:
    text = str(text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        )
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return None


# ── 退化模式定义 ────────────────────────────────────────────
DEGRADE_MODES = {
    "weak_evidence": {
        "desc": "保留原 violation 结论，但 reason 变得空泛，不引用任何具体属性",
        "instruction": (
            "保持 violation 字段与原审核相同，保持 category 不变，"
            "attributes 字段原样拷贝。只改写 reason。"
            "新 reason 长度 25-55 字，要求：\n"
            "- 不得引用任何具体颜色/材质/款式/品类/属性值或商品描述原文\n"
            "- 不得出现属性键名（如'材质''款式''颜色''品牌''规格'）\n"
            "- 每条必须用全新的措辞，禁止复用任何示例句式，禁止以'该商品'开头\n"
            "- 写成一个真实审核员在赶时间时敷衍写下的结论，语气自然，可以使用主观判断词"
            "（例如'看起来''基本上''大致''综合评估''初步判断'等），"
            "可以提及'整体''常规情况''平台惯例''以往审核经验'等抽象理由，但不得触及具体属性\n"
            "风格参考（仅风格，不得照抄任何短语）：\n"
            "  · 从整体观感评估，此条送审内容属常见处理范围，未触发需深入核查的条件。\n"
            "  · 按以往同类审核惯例，本次结论基本稳妥，后续无需额外跟进。\n"
            "  · 综合各方因素来看，当前内容处于可接受区间，没有特别突出的异常点。\n"
            "  · 初步浏览后给出本判断，整体情况与日常标准差别不大。"
        ),
        "require_violation_flip": False,
    },
    "missed_cue": {
        "desc": "把 violation=True 改为 False（漏判违规），理由宽松",
        "instruction": (
            "将 violation 字段由 true 改为 false（必须翻转），category 保持不变，"
            "attributes 拷贝原值但可删除其中最能暴露违规的 1 个键。"
            "reason 改写为 30-60 字，解释为什么这个商品看起来'不违规'，"
            "刻意忽略原审核中指出的违规证据。例如把极限词解释成'常见营销表达'、"
            "把图文不符解释成'商品细节展示差异'。"
        ),
        "require_violation_flip": True,
    },
    "over_strict": {
        "desc": "把 violation=False 改为 True（误判违规），理由牵强",
        "instruction": (
            "将 violation 字段由 false 改为 true（必须翻转），category 保持不变，"
            "attributes 拷贝原值。reason 改写为 30-60 字，牵强地认定违规，"
            "例如把普通描述词当作极限词、把常规属性当作虚假宣传、过度解读普通图片细节。"
            "理由要听起来像审核员过度敏感，但实际证据不充分。"
        ),
        "require_violation_flip": True,
    },
    "wrong_attribute": {
        "desc": "把 1-2 个属性改为视觉错值，reason 引用错误属性",
        "instruction": (
            "保持 violation 结论与 category 不变。attributes 字段：选 1-2 个有具体值的键"
            "（如 颜色/材质/款式），把值改为与图片明显不符的错值（颜色改为对比色、材质改为不同类别、"
            "款式改为另一种类型）。reason 改写为 30-60 字，引用这些**错误**属性值作证据。"
            "最终输出看起来像审核员把另一件商品的属性套到了这张图上。"
        ),
        "require_violation_flip": False,
    },
}


SYSTEM_PROMPT = """你是审核质量扰动工程师，任务是为 RM 训练生成 hard negative 样本。
用户会给你一张商品图、原始商品描述、以及一份当前合格的审核 JSON（chosen）。
你需要按照指定的「退化模式」生成一份**同一张图、同一个品类**、但审核质量更差的新 JSON（rejected）。

严格要求：
1. 只输出新的审核 JSON，不要 markdown 代码块，不要解释文字
2. JSON 必须包含字段：category, attributes, violation, reason
3. category 必须与 chosen 完全相同
4. 严格遵守退化模式的指令，该翻 violation 就必须翻，该保持就必须保持
5. 输出看起来要像一个真实但质量较差的审核员所写，不要过于夸张或明显机器生成"""


def build_user_message(
    image_b64: str,
    prompt: str,
    chosen_obj: dict,
    mode_name: str,
) -> list:
    mode = DEGRADE_MODES[mode_name]
    chosen_json = json.dumps(chosen_obj, ensure_ascii=False, indent=2)
    text = f"""【商品描述】
{prompt}

【当前合格审核 (chosen)】
{chosen_json}

【退化模式】{mode_name}
{mode["instruction"]}

请严格按照退化模式，生成一个更差但看起来真实的审核 JSON。"""
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
        },
        {"type": "text", "text": text},
    ]


def pick_mode(chosen_violation: bool, only_mode: Optional[str] = None) -> Optional[str]:
    """Pick a degrade mode compatible with the chosen label."""
    if only_mode is not None:
        mode = DEGRADE_MODES[only_mode]
        if mode["require_violation_flip"]:
            if only_mode == "missed_cue" and not chosen_violation:
                return None
            if only_mode == "over_strict" and chosen_violation:
                return None
        return only_mode
    if chosen_violation:
        pool = ["weak_evidence", "missed_cue", "wrong_attribute"]
        weights = [0.35, 0.35, 0.30]
    else:
        pool = ["weak_evidence", "over_strict", "wrong_attribute"]
        weights = [0.35, 0.35, 0.30]
    return random.choices(pool, weights=weights, k=1)[0]


def gen_rejected(
    client: OpenAI,
    model: str,
    image_path: str,
    prompt: str,
    chosen_obj: dict,
    mode_name: str,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> Optional[dict]:
    try:
        image_b64 = encode_image(image_path)
    except Exception:
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(image_b64, prompt, chosen_obj, mode_name)},
    ]

    mode = DEGRADE_MODES[mode_name]
    # Phrases that we saw overused in v1 (prompt examples being echoed).
    weak_ev_banned = [
        "该商品审核结论基于一般性判断",
        "整体情况符合平台规则",
        "整体情况不符合平台规则",
        "符合平台规则要求",
        "不符合平台规则要求",
    ]
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=700,
                extra_body={"enable_thinking": False},
            )
            content = strip_thinking(resp.choices[0].message.content)
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                time.sleep(min(2 ** attempt * 2, 30))
                continue
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None

        data = safe_load_json(content)
        if not data:
            continue
        if not all(k in data for k in ("category", "attributes", "violation", "reason")):
            continue
        # Validate category preserved
        if data.get("category") != chosen_obj.get("category"):
            continue
        # Validate violation flip contract
        if mode["require_violation_flip"]:
            if bool(data["violation"]) == bool(chosen_obj.get("violation")):
                continue  # didn't flip, retry
        else:
            if bool(data["violation"]) != bool(chosen_obj.get("violation")):
                continue  # shouldn't flip, retry
        # Banned-phrase filter: avoid echoing prompt examples verbatim
        if mode_name == "weak_evidence":
            reason_text = str(data.get("reason", "") or "")
            if any(p in reason_text for p in weak_ev_banned):
                continue
            if reason_text.startswith("该商品"):
                continue
        return {
            "category": data["category"],
            "attributes": data["attributes"],
            "violation": bool(data["violation"]),
            "reason": data["reason"],
        }
    return None


# ── 主流程 ───────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_file", default="data/sft/sft.jsonl")
    ap.add_argument(
        "--pool_file",
        default="data/preference/preference.jsonl",
        help="用于限定 train-only 图片池（复用现有 preference 的图片集合，已保证不泄漏）",
    )
    ap.add_argument("--image_dir", default="data/raw/images")
    ap.add_argument("--out_file", default="data/preference/preference_distilled.jsonl")
    ap.add_argument("--n_pairs", type=int, default=1500)
    ap.add_argument("--model", default="qwen-vl-max-latest")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rate_limit", type=float, default=10, help="QPS")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_per_image", type=int, default=1, help="每张图最多产多少个 pair")
    ap.add_argument(
        "--only_mode",
        default=None,
        choices=list(DEGRADE_MODES.keys()) + [None],
        help="若指定，则强制只生成该退化模式（用于定向重生成）",
    )
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    random.seed(args.seed)
    client = make_client()
    project_root = Path(__file__).resolve().parents[2]

    sft_path = project_root / args.sft_file
    pool_path = project_root / args.pool_file
    out_path = project_root / args.out_file
    image_dir = project_root / args.image_dir

    # Load SFT index: image_file -> row
    sft_rows = {}
    with sft_path.open() as f:
        for line in f:
            r = json.loads(line)
            sft_rows[r["image_file"]] = r
    print(f"[sft] loaded {len(sft_rows)} rows")

    # Image pool: intersection of existing preference.jsonl images (train-safe) and SFT
    pool_images = set()
    with pool_path.open() as f:
        for line in f:
            pool_images.add(json.loads(line).get("image_file"))
    pool_images &= set(sft_rows.keys())
    pool_images = sorted(pool_images)
    print(f"[pool] {len(pool_images)} train-safe images available")

    # When only_mode is specified, pre-filter images whose SFT-gold violation
    # is compatible with the mode (otherwise many tasks would no-op).
    if args.only_mode in ("missed_cue", "over_strict"):
        need_violation = args.only_mode == "missed_cue"
        filtered = []
        for img in pool_images:
            sft_row = sft_rows.get(img)
            if not sft_row:
                continue
            chosen = safe_load_json(sft_row["response"])
            if not chosen:
                continue
            if bool(chosen.get("violation")) == need_violation:
                filtered.append(img)
        print(f"[only_mode={args.only_mode}] pre-filtered to {len(filtered)} eligible images")
        pool_images = filtered

    # Resume-safe: skip already-done images
    done_images: set[str] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done_images.add(json.loads(line).get("image_file"))
                except Exception:
                    pass
    print(f"[resume] {len(done_images)} already done, will skip")

    candidates = [img for img in pool_images if img not in done_images]
    random.shuffle(candidates)
    need = max(0, args.n_pairs - len(done_images))
    candidates = candidates[:need]
    print(f"[task] will generate {len(candidates)} new pairs (target total {args.n_pairs})")

    if not candidates:
        print("Nothing to do.")
        return

    est_cost = len(candidates) * 0.04
    print(f"[cost] estimated ~¥{est_cost:.1f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rate_delay = 1.0 / args.rate_limit if args.rate_limit > 0 else 0
    t0 = time.time()
    written = 0
    failed = 0

    def work(img_file: str) -> Optional[dict]:
        sft_row = sft_rows[img_file]
        chosen_obj = safe_load_json(sft_row["response"])
        if not chosen_obj:
            return None
        img_path = str(image_dir / img_file)
        if not os.path.exists(img_path):
            return None
        mode = pick_mode(bool(chosen_obj.get("violation")), only_mode=args.only_mode)
        if mode is None:
            return None
        rejected = gen_rejected(
            client=client,
            model=args.model,
            image_path=img_path,
            prompt=sft_row.get("prompt", ""),
            chosen_obj=chosen_obj,
            mode_name=mode,
            temperature=args.temperature,
        )
        if not rejected:
            return None
        return {
            "image_file": img_file,
            "image": f"data/raw/images/{img_file}",
            "prompt": sft_row.get("prompt", ""),
            "chosen": json.dumps(chosen_obj, ensure_ascii=False),
            "rejected": json.dumps(rejected, ensure_ascii=False),
            "pair_strategy": f"api_weaker_{mode}",
        }

    with out_path.open("a", encoding="utf-8") as fout, ThreadPoolExecutor(
        max_workers=args.workers
    ) as ex:
        window = args.workers * 3
        pending = set()
        idx = 0

        def submit_next():
            nonlocal idx
            if idx >= len(candidates):
                return None
            img = candidates[idx]
            idx += 1
            return ex.submit(work, img)

        for _ in range(window):
            f = submit_next()
            if f is None:
                break
            pending.add(f)

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                try:
                    sample = fut.result()
                except Exception as e:
                    print(f"[err] {e}", flush=True)
                    sample = None
                if sample:
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    fout.flush()
                    written += 1
                else:
                    failed += 1
                if (written + failed) % 20 == 0:
                    rate = written / max(1, time.time() - t0) * 60
                    eta = (len(candidates) - written) / max(1, rate) if rate > 0 else 0
                    print(
                        f"  [{written+failed}/{len(candidates)}] ok={written} fail={failed} "
                        f"rate={rate:.1f}/min ETA={eta:.0f}min",
                        flush=True,
                    )
                f = submit_next()
                if f is not None:
                    pending.add(f)
                time.sleep(rate_delay)

    dt = time.time() - t0
    print(f"\n>>> Done: written={written}, failed={failed}, time={dt/60:.1f} min")
    print(f">>> Estimated cost: ¥{written * 0.04:.1f}")


if __name__ == "__main__":
    main()
