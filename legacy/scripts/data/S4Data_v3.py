"""
Stage 4 / v3 — 合成 4 类新违规案例追加到 violation_cases 库。

新增类别（VIOLATION_TYPES idx 7-10）:
    涉黄涉政 / 医疗夸大 / 虚假代言 / 违禁品

为什么不爬政府站：
    - 涉黄涉政内容有合规风险；
    - 政府公开案例对"虚假代言"覆盖弱（多在民事判决而非行政处罚）；
    - LLM 合成 + banned-phrase + temperature 1.0 对模板聚集足够防御。

输入:
    - data/raw/violation_cases.jsonl (老 150 条，仅读 case_id 计数)
    - DASHSCOPE_API_KEY (--dry_run 不需要)

输出:
    - data/raw/violation_cases_v3.jsonl (老 150 条 + 新合成的 ~280 条)
    - 仅追加，不破坏老条目；case_id 续编号 VC0151+

Schema 与 v1 一致:
    {category, violation_type, description, evidence, penalty, case_id, text}

Usage:
    # dry-run, 不调 API, 验证代码
    python -m scripts.data.S4Data_v3 --dry_run

    # 真实合成
    export DASHSCOPE_API_KEY="sk-xxx"
    python -m scripts.data.S4Data_v3 --per_type 60 --model qwen-plus
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Optional

# 4 类新增违规的 seed 设计：
#   coarse_categories: 这类违规高发的电商粗品类（用于 RAG 检索时 category 路由）
#   seed_keywords: 用于 prompt 模板的具体场景关键词
#   penalty_template: 处罚措辞模板（让合成数据看起来像真实处罚书）
NEW_VIOLATION_SPECS: dict[str, dict] = {
    "涉黄涉政": {
        "coarse_categories": ("服装", "配饰", "其他"),
        "seed_keywords": (
            "暗示性图案", "敏感符号", "境外政治标识", "不雅文字",
            "成人用品擦边宣传", "诱导性图片", "禁用图腾",
        ),
        "penalty_template": "立即下架并永久封店，移交属地公安/网信部门",
    },
    "医疗夸大": {
        "coarse_categories": ("化妆品", "食品", "医药", "其他"),
        "seed_keywords": (
            "宣称治疗肿瘤", "宣称根治糖尿病", "保健品当药品",
            "宣称替代医院治疗", "三天见效根除", "宣称医疗器械功效",
            "压片糖果暗示降血压",
        ),
        "penalty_template": "下架整改，按《广告法》第十七条罚款，违规金额×3 倍",
    },
    "虚假代言": {
        "coarse_categories": ("化妆品", "服装", "电子产品", "包"),
        "seed_keywords": (
            "盗用明星头像", "P图明星合影", "伪造代言文字",
            "未授权使用网红 IP", "假冒品牌方代言函", "盗用医生白大褂背书",
        ),
        "penalty_template": "下架并赔偿肖像权人，扣分 12 分，承担连带民事责任",
    },
    "违禁品": {
        "coarse_categories": ("其他", "电子产品"),
        "seed_keywords": (
            "管制刀具仿真", "气枪/电棍变体", "易燃易爆物品",
            "走私烟草", "未审批保健药品", "受保护动物制品",
            "盗版光盘/教材", "破解版软件激活码",
        ),
        "penalty_template": "立即下架，封禁卖家账户，移交市场监管/海关",
    },
}


# 防模板聚集的 banned phrases（reason 复读率高的句式）
BANNED_PHRASES = (
    "本商品涉嫌", "存在严重违规", "经查证发现", "属于平台严令禁止",
    "依据相关法律法规",
)


def build_synth_prompt(violation_type: str, batch_size: int, seed_kws: tuple[str, ...]) -> str:
    """合成 prompt：让 LLM 一次产 batch_size 条该违规类型的案例。"""
    kws_sample = random.sample(seed_kws, k=min(3, len(seed_kws)))
    return (
        f"你是一位资深电商合规审核员。请生成 {batch_size} 条「{violation_type}」类型的"
        f"电商违规案例样本，用于训练审核模型的检索库。\n\n"
        f"要求：\n"
        f"1. 每条 JSON 一行，包含字段：category（电商粗品类如 服装/食品/化妆品/电子产品/医药/其他）、"
        f"description（30-60 字描述违规手法）、evidence（20-40 字证据，引用图片或描述中的具体痕迹）、"
        f"penalty（10-20 字处罚结论）。\n"
        f"2. 多样化场景，可参考关键词：{', '.join(kws_sample)}，但**不要直接复述这些词**。\n"
        f"3. description 必须包含一个具体动作（修改/隐藏/伪造/混入/盗用 等），不要只说"
        f"\"涉嫌违规\"\"违反规定\"等空话。\n"
        f"4. 严禁使用以下套话：{', '.join(BANNED_PHRASES)}。\n"
        f"5. 每条必须独立、风格不同（口吻、句式、证据切入角度都要变）。\n"
        f"6. 直接输出 {batch_size} 行 JSON，不要 markdown，不要编号，不要解释。每行一个独立 JSON。"
    )


def call_llm(client, model: str, prompt: str, temperature: float = 1.0,
             max_retries: int = 3) -> Optional[str]:
    """调 OpenAI-compatible 文本接口。复用 stage0_distill 的 client 风格。"""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一位严谨的电商合规专家，输出风格多样化的违规案例样本。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=2048,
                extra_body={"enable_thinking": False},
            )
            return resp.choices[0].message.content
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "429" in err:
                wait = min(2 ** attempt * 2, 30)
                print(f"  [rate limit] sleep {wait}s")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [WARN] failed: {e}")
                return None
    return None


_JSON_LINE = re.compile(r"\{[^{}]*\}")


def parse_response(text: str) -> list[dict]:
    """从模型输出抽出 JSON 行。容忍 markdown fence、编号前缀、混排。"""
    if not text:
        return []
    text = re.sub(r"```json|```", "", text)
    out: list[dict] = []
    for m in _JSON_LINE.finditer(text):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if {"category", "description", "evidence"}.issubset(d.keys()):
            out.append(d)
    return out


def make_record(d: dict, vtype: str, case_id: str, penalty_template: str) -> dict:
    """补齐 schema，对齐老 violation_cases 字段顺序。"""
    cat = str(d.get("category", "其他")).strip()
    desc = str(d.get("description", "")).strip()
    ev = str(d.get("evidence", "")).strip()
    penalty = str(d.get("penalty") or penalty_template).strip()
    text = f"品类:{cat} 违规类型:{vtype} {desc} 证据:{ev}"
    return {
        "category": cat,
        "violation_type": vtype,
        "description": desc,
        "evidence": ev,
        "penalty": penalty,
        "case_id": case_id,
        "text": text,
    }


def synthesize(args: argparse.Namespace) -> None:
    raw_dir = Path("data/raw")
    in_path = raw_dir / "violation_cases.jsonl"
    out_path = raw_dir / "violation_cases_v3.jsonl"

    # 1. 读老库 → 找最大 case_id 续编
    existing: list[dict] = []
    max_id = 0
    if in_path.exists():
        with in_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                existing.append(d)
                m = re.match(r"VC(\d+)", str(d.get("case_id", "")))
                if m:
                    max_id = max(max_id, int(m.group(1)))
    print(f"loaded {len(existing)} existing cases; next id = VC{max_id+1:04d}")

    # 2. 真实/dry-run 调 LLM 生 4 类
    new_cases: list[dict] = []
    next_id = max_id + 1
    rng = random.Random(args.seed)

    if args.dry_run:
        print("DRY RUN: 用模板生 mock 数据（不调 API）")
        for vtype, spec in NEW_VIOLATION_SPECS.items():
            for k in range(args.per_type):
                kws = spec["seed_keywords"]
                desc = (
                    f"商品主图存在 {rng.choice(kws)} 现象，"
                    f"卖家试图通过该手法规避平台审核（mock 样本 #{k+1}）"
                )
                ev = f"图片局部可见 {rng.choice(kws)} 的证据特征"
                rec = make_record(
                    {
                        "category": rng.choice(spec["coarse_categories"]),
                        "description": desc,
                        "evidence": ev,
                    },
                    vtype, f"VC{next_id:04d}", spec["penalty_template"],
                )
                new_cases.append(rec)
                next_id += 1
    else:
        from openai import OpenAI
        api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Set DASHSCOPE_API_KEY or OPENAI_API_KEY")
        base_url = os.environ.get(
            "OPENAI_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        client = OpenAI(api_key=api_key, base_url=base_url)

        for vtype, spec in NEW_VIOLATION_SPECS.items():
            target = args.per_type
            collected: list[dict] = []
            attempt = 0
            while len(collected) < target and attempt < args.max_batches_per_type:
                batch_size = min(args.batch_size, target - len(collected) + 2)
                prompt = build_synth_prompt(vtype, batch_size, spec["seed_keywords"])
                text = call_llm(client, args.model, prompt, temperature=args.temperature)
                if text is None:
                    attempt += 1
                    continue
                parsed = parse_response(text)
                # banned-phrase filter
                parsed = [
                    p for p in parsed
                    if not any(b in (p.get("description", "") + p.get("evidence", "")) for b in BANNED_PHRASES)
                ]
                # de-dup within type by description hash
                seen_desc = {p["description"] for p in collected}
                parsed = [p for p in parsed if p.get("description") not in seen_desc]
                collected.extend(parsed)
                attempt += 1
                print(f"  [{vtype}] batch {attempt}: +{len(parsed)} (total {len(collected)}/{target})")
                if args.rate_limit > 0:
                    time.sleep(1.0 / args.rate_limit)

            for d in collected[:target]:
                new_cases.append(make_record(
                    d, vtype, f"VC{next_id:04d}", spec["penalty_template"],
                ))
                next_id += 1

    # 3. 写文件：老条目原样 + 新条目追加
    with out_path.open("w", encoding="utf-8") as f:
        for d in existing + new_cases:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    by_type = {}
    for d in new_cases:
        by_type[d["violation_type"]] = by_type.get(d["violation_type"], 0) + 1
    print(f"\nDone. Wrote {len(existing)} old + {len(new_cases)} new = "
          f"{len(existing) + len(new_cases)} cases → {out_path}")
    print("New by type:")
    for t, c in by_type.items():
        print(f"  {t}: {c}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_type", type=int, default=60,
                        help="每个新类合成多少条 (默认 60；4 类共 ~240)")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="单次 API 调用产出条数 (默认 10)")
    parser.add_argument("--max_batches_per_type", type=int, default=12,
                        help="单类最多重试多少次 batch (容错)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--model", default="qwen-plus",
                        help="DashScope 文本模型 (qwen-plus/qwen-max/qwen-vl-max 都行)")
    parser.add_argument("--rate_limit", type=float, default=4.0,
                        help="QPS 上限 (默认 4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true",
                        help="不调 API，用模板生 mock 数据 (代码 sanity check)")
    synthesize(parser.parse_args())
