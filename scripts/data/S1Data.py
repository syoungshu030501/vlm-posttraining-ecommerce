"""
Stage 0 补充蒸馏：看图写真实描述 + 可控合规/违规比例。

与原版 distill.py 的区别：
  - 不依赖 annotations.jsonl 的预设描述（那些是随机模板配图，导致图文不符）
  - Qwen-max 先看图真实描述，再生成 JSON 审核结果
  - 合规样本：忠实描述 + violation=False
  - 违规样本：忠实描述上注入极限词/材质虚标/夸大功效/品牌侵权，violation=True

用法：
    export DASHSCOPE_API_KEY=sk-xxx
    python scripts/data/S1Data.py \
        --image_dir data/raw/images \
        --out_file data/sft/sft.jsonl \
        --target_total 5000 --compliant_ratio 0.6 \
        --model qwen-vl-max-latest --rate_limit 8 --workers 4
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
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED


def _wait_first(pending):
    done, still_pending = wait(pending, return_when=FIRST_COMPLETED)
    return done, still_pending
from pathlib import Path
from typing import Optional

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.schema import try_parse

# ── API client ──────────────────────────────────────────────
def make_client() -> OpenAI:
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("set DASHSCOPE_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL",
                                "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


# ── Prompt 库 ────────────────────────────────────────────────
COMPLIANT_SYSTEM = """你是电商商品合规审核员。步骤：
1. 仔细观察图片，识别真实的商品品类、颜色、材质、款式等属性
2. 基于真实观察写一段 15-30 字的自然商品描述（模拟卖家的商品标题/详情）
3. 这是合规商品，输出审核 JSON

严格按以下 JSON 格式输出，不要markdown不要其它文字：
{
  "description": "<真实合规描述>",
  "category": "<品类>",
  "attributes": {"颜色":"<>", "材质":"<>", "款式":"<>"},
  "violation": false,
  "reason": "<30-80字的具体合规说明：必须引用本条商品的具体品类/材质/颜色/款式等属性，说明描述如何与图片一致、哪些合规要点被满足（如未使用极限词、未夸大功效、未虚标材质、未傍名牌等）。禁止使用空洞模板或照抄本提示词中的句式，每条必须基于当前图文独立表达。>"
}

示例（仅示范风格，不要照抄）：
- 合规鞋类："描述'深棕色绒面革一脚蹬休闲鞋'与图片完全吻合，材质、颜色、款式均可从图中核实，未使用'最/顶级/100%'等绝对化或虚标材质用语。"
- 合规食品："描述'经典原味曲奇饼干300g'对应图中包装信息，配料与规格标注真实，未作保健/功效承诺，不涉及广告法第28条禁区。"
- 合规服装："图中为米白色针织开衫，描述仅陈述颜色与版型，未蹭'小香风原单''大牌平替'等侵权表达，属性一致。"""

# 违规类型（注入策略）
VIOLATION_TYPES = [
    {
        "name": "极限词",
        "inject": "在描述中加入『最新款』『超低价』『全网最低』『绝对顶级』等绝对化用语",
        "reason_hint": "使用最/第一/顶级等绝对化用语，违反广告法第九条",
    },
    {
        "name": "材质虚标",
        "inject": "在描述中把图片里明显非纯棉/真丝的材质标为『100%纯棉』『100%真丝』",
        "reason_hint": "图片所示材质与描述的100%材质明显不符，属虚假宣传",
    },
    {
        "name": "功效夸大",
        "inject": "在描述中加入『包治百病』『7天瘦10斤』『一抹白』等不可能的功效承诺",
        "reason_hint": "对商品功效作不可能承诺，违反广告法第二十八条",
    },
    {
        "name": "品牌侵权",
        "inject": "在描述中蹭大牌，用『NIKE同款』『LV平替』『小香风原单』等傍名牌用语",
        "reason_hint": "擅自使用他人品牌作宣传,涉嫌侵犯注册商标专用权",
    },
    {
        "name": "价格欺诈",
        "inject": "在描述中加入『原价999直降到99』『限时清仓甩卖』等虚构原价/打折",
        "reason_hint": "虚标原价制造打折假象，违反价格法",
    },
    {
        "name": "图文不符",
        "inject": "在描述中编造与图片明显不符的商品类型（比如图是鞋子，描述写成连衣裙）",
        "reason_hint": "商品描述与主图展示内容明显不符，误导消费者",
    },
]


def violation_system_prompt(vt: dict) -> str:
    return f"""你是电商商品合规审核员。步骤：
1. 仔细观察图片，识别真实的商品品类、颜色、材质、款式等属性
2. {vt['inject']}，生成一段 15-35 字的违规商品描述
3. 这是违规商品（违规类型：{vt['name']}），输出审核 JSON

严格按以下 JSON 格式输出，不要markdown不要其它文字：
{{
  "description": "<包含违规的描述>",
  "category": "<真实品类>",
  "attributes": {{"颜色":"<真实>", "材质":"<真实>", "款式":"<真实>"}},
  "violation": true,
  "violation_type": "{vt['name']}",
  "reason": "<具体理由,必须引用描述中的违规词和图片真实属性> — 参考：{vt['reason_hint']}"
}}"""


# ── 单次调用 ────────────────────────────────────────────────
def gen_sample(
    client: OpenAI,
    model: str,
    image_path: str,
    compliant: bool,
    temperature: float,
    max_retries: int = 3,
) -> Optional[dict]:
    if compliant:
        system = COMPLIANT_SYSTEM
        v_type = None
    else:
        vt = random.choice(VIOLATION_TYPES)
        system = violation_system_prompt(vt)
        v_type = vt["name"]

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}},
                {"type": "text", "text": "请分析这张商品图片，生成要求的 JSON。"},
            ],
        },
    ]

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=700,
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

        # 解析完整 JSON（我们需要 description 字段）
        try:
            content_clean = content.strip()
            if content_clean.startswith("```"):
                content_clean = "\n".join(content_clean.splitlines()[1:-1]
                                          if content_clean.splitlines()[-1].strip() == "```"
                                          else content_clean.splitlines()[1:])
            data = json.loads(content_clean)
        except (json.JSONDecodeError, IndexError):
            continue

        # 必需字段校验
        if not all(k in data for k in ("description", "category", "attributes", "violation", "reason")):
            continue
        if bool(data["violation"]) != (not compliant):
            continue  # 模型没听话，重试

        # 构造 response（不含 description，供后续 SFT 训练用）
        response_obj = {
            "category": data["category"],
            "attributes": data["attributes"],
            "violation": bool(data["violation"]),
            "reason": data["reason"],
        }

        return {
            "image_file": os.path.basename(image_path),
            "image": image_path,
            "prompt": data["description"],
            "response": json.dumps(response_obj, ensure_ascii=False),
            "violation": bool(data["violation"]),
            "category": data["category"],
            "violation_type": v_type,
            "gen_strategy": "balanced_v3",
        }

    return None


# ── 主流程 ───────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", default="data/raw/images")
    ap.add_argument("--out_file", default="data/sft/sft.jsonl")
    ap.add_argument("--target_total", type=int, default=5000, help="最终总 SFT 样本数")
    ap.add_argument("--compliant_ratio", type=float, default=0.6)
    ap.add_argument("--model", default="qwen-vl-max-latest")
    ap.add_argument("--rate_limit", type=float, default=8, help="QPS")
    ap.add_argument("--workers", type=int, default=4, help="并发线程数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_per_image", type=int, default=3, help="同张图最多复用次数")
    args = ap.parse_args()

    random.seed(args.seed)
    client = make_client()

    # 读现有数据，判断当前合规/违规数
    out_path = Path(args.out_file)
    existing = []
    if out_path.exists():
        with out_path.open() as f:
            existing = [json.loads(l) for l in f if l.strip()]
    n_exist_compliant = sum(1 for s in existing if not s.get("violation"))
    n_exist_violation = sum(1 for s in existing if s.get("violation"))
    print(f"[exist] {len(existing)} total: {n_exist_compliant} compliant + {n_exist_violation} violation")

    # 计算目标增量
    target_compliant = int(args.target_total * args.compliant_ratio)
    target_violation = args.target_total - target_compliant
    need_compliant = max(0, target_compliant - n_exist_compliant)
    need_violation = max(0, target_violation - n_exist_violation)
    total_need = need_compliant + need_violation
    print(f"[target] {args.target_total} = {target_compliant} compliant + {target_violation} violation")
    print(f"[need]   +{need_compliant} compliant + {need_violation} violation = {total_need}")
    est_cost = total_need * 0.033
    print(f"[cost]   估算 ~¥{est_cost:.1f}")

    if total_need <= 0:
        print("已达目标，无需补充。")
        return

    # 构建图片池
    images = sorted(str(p) for p in Path(args.image_dir).glob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    print(f"[images] {len(images)} available")

    # 每张图使用次数计数
    image_used: dict[str, int] = {}
    for s in existing:
        fn = s.get("image_file", "")
        p = os.path.join(args.image_dir, fn)
        image_used[p] = image_used.get(p, 0) + 1

    # 任务列表
    tasks = []  # (image_path, is_compliant)
    for _ in range(need_compliant):
        tasks.append(True)
    for _ in range(need_violation):
        tasks.append(False)
    random.shuffle(tasks)

    # 分配图：优先用得少的
    def pick_image() -> Optional[str]:
        candidates = [(image_used.get(p, 0), p) for p in images]
        candidates.sort()
        usage_min = candidates[0][0]
        if usage_min >= args.max_per_image:
            return None  # 用完了
        pool = [p for cnt, p in candidates if cnt == usage_min]
        return random.choice(pool)

    # 预分配图片到任务
    assignments = []
    for is_compliant in tasks:
        img = pick_image()
        if img is None:
            print("[warn] 所有图片都已用满上限，停止分配", flush=True)
            break
        image_used[img] = image_used.get(img, 0) + 1
        assignments.append((img, is_compliant))
    print(f"[assigned] {len(assignments)} tasks", flush=True)

    rate_delay = 1.0 / args.rate_limit if args.rate_limit > 0 else 0
    written = 0
    failed = 0
    t0 = time.time()

    with out_path.open("a", encoding="utf-8") as fout, \
         ThreadPoolExecutor(max_workers=args.workers) as ex:

        # 滑动窗口：保持 workers × 3 的 pending 队列，边提交边消费
        window = args.workers * 3
        pending = set()
        idx = 0

        def submit_next():
            nonlocal idx
            if idx >= len(assignments):
                return None
            img, is_compliant = assignments[idx]
            idx += 1
            temp = 0.3 if is_compliant else 0.7
            return ex.submit(gen_sample, client, args.model, img, is_compliant, temp)

        # 预热窗口
        for _ in range(window):
            f = submit_next()
            if f is None:
                break
            pending.add(f)

        while pending:
            done, pending = _wait_first(pending)
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
                if (written + failed) % 25 == 0:
                    rate = written / max(1, time.time() - t0) * 60
                    eta = (len(assignments) - written) / max(1, rate) if rate > 0 else 0
                    print(f"  [{written+failed}/{len(assignments)}] ok={written} fail={failed} "
                          f"rate={rate:.1f}/min ETA={eta:.0f}min", flush=True)
                # 补一个新任务进窗口
                f = submit_next()
                if f is not None:
                    pending.add(f)
                time.sleep(rate_delay)

    dt = time.time() - t0
    print(f"\n>>> 完成: 写入 {written}, 失败 {failed}, 用时 {dt/60:.1f} min")
    print(f">>> 实际花费估算: ¥{written * 0.033:.1f}")


if __name__ == "__main__":
    main()
