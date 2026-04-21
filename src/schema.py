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
