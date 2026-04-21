"""JSON parsing helpers shared across all stages."""
import json
import re
from typing import Any, Optional


def extract_json(text: str) -> Optional[str]:
    """Extract the first JSON object from text, handling markdown fences."""
    text = text.strip()
    # Strip ```json ... ``` fences
    fence = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if fence:
        return fence.group(1).strip()
    # Try to find bare JSON object
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        return match.group(0)
    return None


def safe_loads(text: str) -> Optional[Any]:
    raw = extract_json(text)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
