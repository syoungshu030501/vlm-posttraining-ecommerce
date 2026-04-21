"""
Output JSON schema and validation for the e-commerce audit system.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

REQUIRED_FIELDS = ("category", "attributes", "violation", "reason")

SYSTEM_PROMPT = (
    "你是一位专业的电商商品合规审核员。"
    "分析给定的商品图片和描述，输出一个JSON对象，包含以下字段：\n"
    '- "category" (str): 商品品类\n'
    '- "attributes" (dict): 从图片中提取的关键视觉属性（如颜色、材质、款式等）\n'
    '- "violation" (bool): 该商品是否违反平台规则\n'
    '- "reason" (str): 简明审核理由，必须引用具体的视觉属性作为证据\n'
    "只输出合法JSON，不要用markdown代码块或其他格式包裹。"
)

# English version for models that work better in English
SYSTEM_PROMPT_EN = (
    "You are a professional e-commerce product compliance auditor. "
    "Analyze the given product image and description, then output a JSON object with the "
    "following fields:\n"
    "- category (str): product category\n"
    "- attributes (dict): key visual attributes extracted from the image\n"
    "- violation (bool): whether the product violates platform policies\n"
    "- reason (str): concise reasoning that references specific visual attributes\n"
    "Output ONLY valid JSON, no markdown fences or extra text."
)


@dataclass
class AuditOutput:
    category: str
    attributes: Dict[str, Any]
    violation: bool
    reason: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "category": self.category,
                "attributes": self.attributes,
                "violation": self.violation,
                "reason": self.reason,
            },
            ensure_ascii=False,
            indent=None,
        )

    @classmethod
    def from_json(cls, text: str) -> "AuditOutput":
        """Parse model output, stripping markdown fences if present."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        data = json.loads(text)
        return cls(
            category=data["category"],
            attributes=data["attributes"],
            violation=bool(data["violation"]),
            reason=data["reason"],
        )

    def is_valid(self) -> bool:
        return all(
            getattr(self, f, None) is not None for f in REQUIRED_FIELDS
        )


def try_parse(text: str) -> Optional[AuditOutput]:
    """Return AuditOutput or None if parsing fails."""
    try:
        return AuditOutput.from_json(text)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Coarse-category normalization
# ---------------------------------------------------------------------------
# SFT 蒸馏时 qwen-vl-max 返回的 `category` 是自由文本（如 "男装衬衫"、"handbag"、
# "女装-连衣裙"、"运动短裤" 等），jsonl 中累计出现 125+ 种互不归并的字符串。
# 这对 Stage 2 RM 的「同品类」契约与分层评估都是噪声源。
#
# 不改原始 jsonl，而是在读取侧提供一个确定性映射到 10 个粗粒度桶，供：
#   - guard.py 校验同粗桶契约（chosen ⇄ rejected）
#   - Stage 2/3 的分层评估分组
#   - Stage 4 RAG 按粗类目路由检索
#
# 规则：按关键词优先级匹配（越具体越靠前），首命中即返回。

COARSE_CATEGORIES = (
    "食品", "化妆品", "电子产品", "医药",
    "鞋", "手表", "包", "服装", "配饰",
    "其他",
)

_COARSE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # 高优先级（独立领域，易被误匹配）
    ("食品",   ("食品", "饮料", "零食", "酒水", "餐饮", "保健食品", "糖果", "茶叶")),
    ("化妆品", ("化妆品", "护肤", "口红", "粉底", "面膜", "香水", "彩妆", "精华", "乳液", "美妆")),
    # 「手表」置于「电子产品」之前，避免「电子手表」被「电子」误抢
    ("手表",   ("手表", "腕表", "电子手表", "智能手表")),
    ("电子产品", ("电子", "电器", "手机", "耳机", "相机", "电脑", "笔记本", "平板", "充电",
                 "音响", "蓝牙", "智能设备", "家电")),
    ("医药",   ("药品", "药店", "医疗器械", "保健品")),
    # 中优先级（容易和服装混淆的边缘）
    ("鞋",     ("鞋", "靴", "拖", "凉拖", "帆布")),
    ("包",     ("包袋", "箱包", "背包", "手袋", "手提包", "单肩包", "女包", "钱包", "包包",
                 "包",  # 放在最后，仅作兜底
                 "handbag")),
    # 配饰（帽子、腰带、围巾、领带、袜子、眼镜 等）
    ("配饰",   ("帽", "皮带", "腰带", "围巾", "领带", "袜", "眼镜", "手套", "发饰", "首饰",
                 "项链", "耳环", "戒指", "配件", "服饰配件", "belt")),
    # 默认：服装（涵盖 衣/裤/裙/衫/POLO/T恤/...）
    ("服装",   ("服装", "服饰", "男装", "女装", "童装", "婴儿", "儿童", "运动服",
                 "衣", "裤", "裙", "衫", "Polo", "POLO", "polo", "T恤", "毛衣",
                 "卫衣", "背心", "夹克", "外套", "羽绒", "针织", "家居服", "内衣", "球衣",
                 "shoes",  # 兜底：少量英文异常值
                 )),
)


def coarse_category(category: str | None) -> str:
    """Map a free-text SFT `category` field to one of COARSE_CATEGORIES.

    Deterministic, order-dependent keyword match. Unmatched → "其他".
    """
    if not category:
        return "其他"
    s = str(category)
    for bucket, keywords in _COARSE_RULES:
        for kw in keywords:
            if kw in s:
                return bucket
    return "其他"


def same_coarse(a: str | None, b: str | None) -> bool:
    """Whether two free-text categories land in the same coarse bucket."""
    return coarse_category(a) == coarse_category(b)

