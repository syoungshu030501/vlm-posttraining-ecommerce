"""
估算 DashScope API 调用成本（基于本地蒸馏产物计数）。
官方真实用量请以 https://dashscope.console.aliyun.com/overview 为准。
"""
import os

PRICE_IN_PER_K = 0.02
PRICE_OUT_PER_K = 0.06
TOK_IN = 1200
TOK_OUT = 150

def count(p):
    return sum(1 for _ in open(p)) if os.path.exists(p) else 0

sft = count("data/sft/sft.jsonl")
pref = count("data/preference/preference.jsonl")
calls = sft + pref * 2

cost_in = calls * TOK_IN / 1000 * PRICE_IN_PER_K
cost_out = calls * TOK_OUT / 1000 * PRICE_OUT_PER_K

print(f"SFT 样本数:    {sft}")
print(f"偏好对 (×2):   {pref} × 2 = {pref*2}")
print(f"总调用次数:    {calls}")
print(f"  输入成本:    ¥{cost_in:.2f} ({calls*TOK_IN:,} token)")
print(f"  输出成本:    ¥{cost_out:.2f} ({calls*TOK_OUT:,} token)")
print(f"  合计:        ¥{cost_in+cost_out:.2f}")
print()
print("精确用量请看: https://dashscope.console.aliyun.com/overview")
