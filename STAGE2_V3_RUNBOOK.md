# Stage 2 v3 RUNBOOK — Field-wise PRM 数据 + 模型上线流程

> 配套设计文档：[reference/data-redesign-2026.md](reference/data-redesign-2026.md)
>
> 本 RUNBOOK 覆盖**仅 v3 新增**的命令；不动 v1/v2 ckpt 与脚本。

## 0. 前置检查

```bash
# 进入 VLM env
source /home/young/miniconda3/etc/profile.d/conda.sh && conda activate VLM
cd /home/young/VLM-posttraining

# 确认 sft_aux_merged 存在（PRM v3 backbone）
ls -lh models/sft_aux_merged/ | head -3

# DashScope key（若要走双教师投票路径）
export DASHSCOPE_API_KEY="sk-xxx"   # https://dashscope.console.aliyun.com/apiKey

# 全部 dry-run 验证（不调任何 GPU/API，2 分钟）
python -m scripts.data.S4Data_v3 --dry_run --per_type 5
python -m scripts.data.S2DataV3  --dry_run --limit 5 --K 6 --out_parquet /tmp/preference_v3_dry.parquet
```

---

## 1. 扩 violation_type 案例库（4 类，约 ¥3）

```bash
# 真实合成（4 类各 60 条 ≈ 240 条新案例追加到原 150 条）
python -m scripts.data.S4Data_v3 \
    --per_type 60 \
    --batch_size 10 \
    --temperature 1.0 \
    --model qwen-plus \
    --rate_limit 4

# 输出：data/raw/violation_cases_v3.jsonl  ≈ 390 条
# 校验：
wc -l data/raw/violation_cases_v3.jsonl
python -c "
import json
from collections import Counter
c = Counter()
with open('data/raw/violation_cases_v3.jsonl') as f:
    for line in f:
        c[json.loads(line)['violation_type']] += 1
for t, n in c.most_common(): print(f'  {t}: {n}')"
```

预期产出（4 个新类各约 50-60 条，老 6 类不变）：
```
极限词: 27 / 材质虚假标注: 18 / ... / 涉黄涉政: 60 / 医疗夸大: 60 / 虚假代言: 60 / 违禁品: 60
```

---

## 2. 生成 v3 field-wise preference 数据（核心，~8h GPU + ~¥80 API）

### 2.1 仅 rollout 不投票（更快，~5h，免费）

```bash
mkdir -p data/preference_v3 logs/v3_data
python -m scripts.data.S2DataV3 \
    --policy_path models/sft_aux_merged \
    --in_jsonl data/sft/sft.jsonl \
    --image_dir data/raw/images \
    --out_parquet data/preference_v3/preference_v3.parquet \
    --K 8 \
    --no_teacher_vote \
    --dump_every 100 \
    --resume \
    2>&1 | tee logs/v3_data/s2_v3_no_vote.log
```

### 2.2 含双教师投票（推荐，violation_prob 质量显著提升）

```bash
python -m scripts.data.S2DataV3 \
    --policy_path models/sft_aux_merged \
    --in_jsonl data/sft/sft.jsonl \
    --image_dir data/raw/images \
    --out_parquet data/preference_v3/preference_v3.parquet \
    --K 8 \
    --teacher_models qwen-vl-max,qwen-plus \
    --dump_every 100 \
    --resume \
    2>&1 | tee logs/v3_data/s2_v3_full.log
```

成本估算（6685 SFT 样本）：
- K=8 rollouts × 6685 = 53,480 generations × 0.5s/gen 单卡 ≈ **7-8h GPU**
- 双教师 10 votes × 6685 = 66,850 API calls ≈ ¥0.0015 × 66850 ≈ **¥100**
- 4 QPS rate limit → API 部分约 **5h**（与 GPU 并行）

### 2.3 切 train / holdout（保留 v2 切分逻辑）

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

df = pd.read_parquet("data/preference_v3/preference_v3.parquet")
print(f"total rows: {len(df)}")

# 按 image_file 分组切（与 v2 同逻辑），seed=42
import numpy as np
rng = np.random.RandomState(42)
groups = df["image_file"].drop_duplicates().tolist()
rng.shuffle(groups)
n_holdout = max(200, int(len(groups) * 0.1))
holdout_imgs = set(groups[:n_holdout])

is_holdout = df["image_file"].isin(holdout_imgs)
df[~is_holdout].to_parquet("data/preference_v3/preference_v3_train.parquet", index=False)
df[ is_holdout].to_parquet("data/preference_v3/preference_v3_holdout.parquet", index=False)
print("train:", (~is_holdout).sum(), "  holdout:", is_holdout.sum())
PY
```

### 2.4 数据 sanity 检查

```bash
python - <<'PY'
import pandas as pd
df = pd.read_parquet("data/preference_v3/preference_v3.parquet")
print(df["label_violation_type"].value_counts())
print("violation_prob   mean ± std:",
      df["label_violation_prob"].mean(), "±", df["label_violation_prob"].std())
print("reason_align     mean ± std:",
      df["label_reason_align"].mean(), "±", df["label_reason_align"].std())
PY
```

期望：
- violation_type 11 类都有非零计数（可能"虚假代言"等少；需要时按 §1 扩 case 平衡）
- violation_prob 双峰分布（mean ≈ 0.4-0.6，std ≥ 0.3）
- reason_align mean ≈ 0.7-0.85（dry-run 是 0.0 因为 _IdentityEncoder）

---

## 3. 训练 PRM v3（5-loss 联合）

### 3.1 单卡训练（推荐先做）

```bash
mkdir -p models/rm_v3_aux_ckpt logs/v3_train
python -m src.stage2_rm.v3.train \
    --model_path models/sft_aux_merged \
    --train_parquet  data/preference_v3/preference_v3_train.parquet \
    --holdout_parquet data/preference_v3/preference_v3_holdout.parquet \
    --out_dir models/rm_v3_aux_ckpt \
    --epochs 3 \
    --batch_size 1 \
    --lr 1e-4 \
    --w_category 1.0 --w_attributes 0.5 \
    --w_violation_prob 1.0 --w_violation_type 1.0 --w_reason_align 0.3 \
    --flash_attn \
    --use_swanlab \
    2>&1 | tee logs/v3_train/prm_v3.log
```

时间预算：~12-18h on 1×L20 (sft_aux_merged 17GB 单卡刚好)。

### 3.2 训练监控（关键指标）

```
swanlog 查看 stage2-prm-v3 实验：
  loss/category             目标 < 1.5（CE over 10 类，random ~ ln10 = 2.3）
  loss/violation_prob       目标 < 0.4（BCE soft target）
  loss/violation_type       目标 < 1.5（CE over 11 类，random ~ 2.4）
  loss/reason_align         目标 < 0.05（MSE 已被 sigmoid 压在 [0,1]）
  loss/attributes           目标 < 0.5（per-token BCE）
  loss/total                单调下降，无 NaN
```

⚠️ 监控警示：
- 若 `loss/violation_type` 不降但其他降 → violation_type 数据噪声（教师投票分歧大），考虑 mask 掉 mc_metadata 显示 `n_teacher_votes < 6` 的样本
- 若 `loss/attributes` 不降 → 检查 attributes_token_mask 是否非全零（dataset.py 中 `_find_token_spans`）

---

## 4. 评估 PRM v3

### 4.1 pointwise（每 head 独立 metric）

```bash
python -m src.stage2_rm.v3.evaluate \
    --model_path models/sft_aux_merged \
    --ckpt models/rm_v3_aux_ckpt/prm_v3_best.pt \
    --eval_parquet data/preference_v3/preference_v3_holdout.parquet \
    --mode pointwise \
    --flash_attn \
    --out_json results/eval_prm_v3_pointwise.json
```

期望（≥ 200 holdout）：
| 指标 | 目标 |
|---|---|
| `category_accuracy` | ≥ 0.85 |
| `violation_prob_ece` | ≤ 0.10 |
| `violation_prob_binary_acc` | ≥ 0.90 |
| `violation_type_macro_f1` | ≥ 0.65（11 类难度）|
| `reason_align_mse` | ≤ 0.03 |
| `attributes_token_best_acc` | ≥ 0.75 |

### 4.2 pairwise（与 v2-aux RM 直接对比）

```bash
python -m src.stage2_rm.v3.evaluate \
    --model_path models/sft_aux_merged \
    --ckpt models/rm_v3_aux_ckpt/prm_v3_best.pt \
    --eval_parquet data/preference_v3/preference_v3_holdout.parquet \
    --mode pairwise \
    --flash_attn \
    --out_json results/eval_prm_v3_pairwise.json
```

期望（每 head 单独看）：
| Head | pair_acc 目标 | margin 目标 |
|---|---|---|
| violation_prob | ≥ 0.85 | ≥ 0.5（连续 prob，不与 v2 mean_margin 直接可比）|
| reason_align | ≥ 0.80 | ≥ 0.2 |
| category_confidence | ≥ 0.70 | ≥ 0.1 |

> v2-aux RM mean_margin = 11.21（scalar Bradley-Terry）。v3 用 sigmoid 输出 margin 不再直接可比，但 pair-acc 应不低于 v2 的 0.825。

---

## 5. 上线到 Stage 3 (FIPO RL)

`src/stage2_rm/v3/model.FieldWisePRM.score_scalar()` 已经把 5 head 聚合为单 scalar，可作为 reward_v2 的替代品挂入 reward_fn v3：

```python
# src/stage3_fipo/reward_fn_v3.py (TODO，本 RUNBOOK 之后再写)
prm = FieldWisePRM(load_backbone(), ...)
load_ckpt(prm, "models/rm_v3_aux_ckpt/prm_v3_best.pt")
scalar = prm.score_scalar(input_ids, attention_mask, pixel_values, ...)
# OR: reward = w_v * (1 - violation_prob) + w_a * mean_attr_grounded + w_r * reason_align + ...
```

---

## 6. 决策回顾（如要回滚）

| 决策 | 默认值 | 修改入口 |
|---|---|---|
| violation 是否硬分类 | 否（continuous BCE） | `src/stage2_rm/v3/model.py: violation_prob_head` |
| violation_type 类别数 | N=11（含"无违规"） | `src/schema.py: VIOLATION_TYPES` |
| MC rollout K | 8 | `scripts/data/S2DataV3.py: --K` |
| 教师投票 | qwen-vl-max + qwen-plus | `--teacher_models` |
| MC rollout policy | sft_aux_merged | `--policy_path` |
| Loss 权重 | (1.0, 0.5, 1.0, 1.0, 0.3) | `train.py: --w_*` |
| backbone 是否冻结 | 是 | `model.py: FieldWisePRM.__init__` 末尾 freeze loop |

---

## 7. 已落地代码清单

```
src/schema.py                            # +VIOLATION_TYPES (11) +FieldLabels +helpers
src/stage2_rm/v3/__init__.py             # 新建子包
src/stage2_rm/v3/model.py                # FieldWisePRM + field_wise_loss + pair_metrics
src/stage2_rm/v3/dataset.py              # FieldWisePreferenceDataset + 2 collates
src/stage2_rm/v3/train.py                # 5-loss 联合训练 + per-head metrics
src/stage2_rm/v3/evaluate.py             # pointwise + pairwise eval
scripts/data/S4Data_v3.py                # 4 类违规案例 LLM 合成
scripts/data/S2DataV3.py                 # MC rollout + 双教师投票数据生产
STAGE2_V3_RUNBOOK.md                     # 本文档
```

---

> **下一步（用户决定）**：
> 1. 启动 §1 真实合成（¥3，10 分钟）
> 2. 启动 §2.1 或 §2.2 真实数据生产（5-13h，¥0 或 ¥100）
> 3. 启动 §3 训练（12-18h GPU）
> 4. 完成后回这份 RUNBOOK 看 §4 评估 — 若 violation_type macro-F1 < 0.5 则回 §1 平衡数据
