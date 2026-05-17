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


# ---------------------------------------------------------------------------
# Violation types (Stage 2 v3: field-wise PRM multi-class head)
# ---------------------------------------------------------------------------
# v3 把原 binary `violation` 拆为两个信号：
#   1. violation_prob ∈ [0, 1]   (continuous，由双教师投票得到 soft target)
#   2. violation_type             (multi-class 11 类，含 "无违规" 兜底)
#
# 类别可扩展：新增类型只需追加到 VIOLATION_TYPES 末尾并重训 type_head 最后
# 一层；前面的 logits index 保持向后兼容。
#
# 顺序约定：index 0 永远是 "无违规"。

VIOLATION_TYPES: tuple[str, ...] = (
    "无违规",       # 0  fallback / negative class
    # ↓ v2 已存在的 6 类
    "极限词",       # 1
    "材质虚标",     # 2
    "功效夸大",     # 3
    "品牌侵权",     # 4
    "价格欺诈",     # 5
    "图文不符",     # 6
    # ↓ v3 新增 4 类
    "涉黄涉政",     # 7
    "医疗夸大",     # 8
    "虚假代言",     # 9
    "违禁品",       # 10
)

VIOLATION_TYPE_TO_ID: dict[str, int] = {t: i for i, t in enumerate(VIOLATION_TYPES)}

# v2 → v3 老类名兼容映射（旧偏好数据里用的字段值）
_VIOLATION_ALIASES: dict[str, str] = {
    "false": "无违规", "False": "无违规", "0": "无违规", "": "无违规",
    "true": "极限词",  # 老 binary 数据若只标 True 而无 type，回退到最常见类
    "no_violation": "无违规",
    "limit_word": "极限词",
    "fake_material": "材质虚标",
    "exaggeration": "功效夸大",
    "brand_infringement": "品牌侵权",
    "price_fraud": "价格欺诈",
    "image_text_mismatch": "图文不符",
}


def normalize_violation_type(t: str | bool | None) -> str:
    """Map free-text / legacy bool to canonical VIOLATION_TYPES entry.

    Unknown strings → '无违规' (safer default; never picks a wrong specific type).
    """
    if t is None or t is False:
        return "无违规"
    if t is True:
        return "极限词"
    s = str(t).strip()
    if s in VIOLATION_TYPES:
        return s
    if s in _VIOLATION_ALIASES:
        return _VIOLATION_ALIASES[s]
    return "无违规"


def violation_type_id(t: str | bool | None) -> int:
    return VIOLATION_TYPE_TO_ID[normalize_violation_type(t)]


# ---------------------------------------------------------------------------
# Field-wise label container (Stage 2 v3 preference data)
# ---------------------------------------------------------------------------
# 每条样本除了原 (image, prompt, response, chosen, rejected) 还有这块 labels：

@dataclass
class FieldLabels:
    """Field-wise PRM training targets for one response.

    All fields are optional — missing fields are masked out in the loss
    so legacy data that only has `violation_prob` still works.
    """
    category_coarse: str | None = None        # ∈ COARSE_CATEGORIES
    attributes: list[dict] | None = None       # [{key, val, grounded: bool, mc: float}]
    violation_prob: float | None = None        # ∈ [0, 1]
    violation_type: str | None = None          # ∈ VIOLATION_TYPES
    reason_align: float | None = None          # BGE 余弦 ∈ [0, 1]

    @classmethod
    def from_dict(cls, d: dict | None) -> "FieldLabels":
        if not d:
            return cls()
        vt = d.get("violation_type")
        return cls(
            category_coarse=d.get("category_coarse") or (
                coarse_category(d.get("category")) if d.get("category") else None
            ),
            attributes=d.get("attributes"),
            violation_prob=d.get("violation_prob"),
            violation_type=normalize_violation_type(vt) if vt is not None else None,
            reason_align=d.get("reason_align"),
        )

    def has(self, name: str) -> bool:
        v = getattr(self, name, None)
        return v is not None and (not isinstance(v, list) or len(v) > 0)

