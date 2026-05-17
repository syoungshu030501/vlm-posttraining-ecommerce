"""Sanity test for reward_fn v2.

Run:
    cd /home/young/VLM-posttraining
    /home/young/miniconda3/envs/VLM/bin/python -m src.stage3_fipo.test_reward_fn
"""
import json
from src.stage3_fipo.reward_fn import compute_reward, make_encoder


def _wrap(d):
    return json.dumps(d, ensure_ascii=False)


def main():
    print("Loading sentence encoder (first run will download ~100MB)...")
    encoder = make_encoder(device="cpu")
    print("Encoder ready.\n")

    cases = [
        (
            "good_violation",
            _wrap({
                "category": "美妆",
                "attributes": {"功效宣称": "美白祛斑100%有效", "保质期": "12个月"},
                "violation": True,
                "reason": "宣称美白祛斑100%有效，使用了极限词且夸大效果，违反广告法。",
            }),
            {"violation": True},
        ),
        (
            "good_non_violation",
            _wrap({
                "category": "美妆",
                "attributes": {"功效宣称": "温和保湿", "保质期": "12个月"},
                "violation": False,
                "reason": "宣称温和保湿，描述合理，标注完整，符合规范。",
            }),
            {"violation": False},
        ),
        (
            "hallucinated_reason",
            _wrap({
                "category": "美妆",
                "attributes": {"功效宣称": "温和保湿"},
                "violation": True,
                "reason": "图片中含有明星代言，且产品包装颜色违规，使用了未授权的奥运标志。",
            }),
            {"violation": True},
        ),
        (
            "label_wrong",
            _wrap({
                "category": "美妆",
                "attributes": {"功效宣称": "美白祛斑100%有效"},
                "violation": False,
                "reason": "无任何违规内容。",
            }),
            {"violation": True},
        ),
        (
            "lexicon_contradict",
            _wrap({
                "category": "美妆",
                "attributes": {"功效宣称": "保湿"},
                "violation": True,
                "reason": "产品标注完整，符合规范，未发现违规。",
            }),
            {"violation": True},
        ),
        (
            "too_short",
            _wrap({
                "category": "美妆",
                "attributes": {"功效宣称": "保湿"},
                "violation": False,
                "reason": "OK",
            }),
            {"violation": False},
        ),
        (
            "parse_failure",
            "this is not json at all",
            {"violation": False},
        ),
        (
            "missing_field",
            _wrap({"category": "美妆", "violation": True}),
            {"violation": True},
        ),
    ]

    print(f"{'case':<22}  {'reward':>7}  breakdown")
    print("-" * 100)
    for name, text, gt in cases:
        r, bd = compute_reward(text, gt_annotation=gt, encoder=encoder, return_breakdown=True)
        print(f"{name:<22}  {r:>7.3f}  {bd}")


if __name__ == "__main__":
    main()
