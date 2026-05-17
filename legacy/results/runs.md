# legacy/results/runs.md — 旧版运行记录与调试史

> 本文件按**单次运行 (run)** 维度记录旧版（e-commerce VLM 合规审核）
> 流水线的训练 / 评估实验，包含：超参、产物路径、收敛曲线、关键观察、
> 踩坑修复。
> 全部结果 JSON 见同目录 [./](./)，方法论详解见
> [../README.md](../README.md)。
>
> 时间锚：所有日期使用本地时区；项目冻结日为 **2026-04-28**。
> 硬件：8×NVIDIA L20 (45 GB)，**GPU 0 持续 ECC 错误，全部禁用**。

---

## 目录

- [Stage 1 SFT 运行](#stage-1-sft-运行)
  - [run-S1-A · sft_baseline（CE only）](#run-s1-a--sft_baselinece-only)
  - [run-S1-B · sft_aux（CE + 0.05·SupCon + 0.03·Triplet）](#run-s1-b--sft_auxce--005supcon--003triplet)
- [Stage 2 Reward Model 运行](#stage-2-reward-model-运行)
  - [run-S2-v0 · Linear head, base backbone](#run-s2-v0--linear-head-base-backbone)
  - [run-S2-v1 · LN+bias head, base backbone](#run-s2-v1--lnbias-head-base-backbone)
  - [run-S2-v2-baseline · MLP head, sft_baseline backbone](#run-s2-v2-baseline--mlp-head-sft_baseline-backbone)
  - [run-S2-v2-aux · MLP head, sft_aux backbone（最终采用）](#run-s2-v2-aux--mlp-head-sft_aux-backbone最终采用)
  - [run-S2-PRM · token-level PRM holdout](#run-s2-prm--token-level-prm-holdout)
- [Stage 3 FIPO-RL 运行](#stage-3-fipo-rl-运行)
  - [run-S3-v1 · 易样本基线（reward saturation）](#run-s3-v1--易样本基线reward-saturation)
  - [run-S3-v2 · hard-mining 70/30 mix（最终采用 step160）](#run-s3-v2--hard-mining-7030-mix最终采用-step160)
- [Stage 4 RAG 评估](#stage-4-rag-评估)
  - [run-S4-CAL · field-aware confidence calibration](#run-s4-cal--field-aware-confidence-calibration)
  - [run-S4-T030 / T035 / T040 / T045 / T050 · 阈值扫描](#run-s4-t030--t035--t040--t045--t050--阈值扫描)
  - [run-S4-Vis / run-S4-Text · 模态消融](#run-s4-vis--run-s4-text--模态消融)
  - [run-S4-Agentic-n200 · Plan-Then-Retrieve + Verify-Then-Rewrite](#run-s4-agentic-n200--plan-then-retrieve--verify-then-rewrite)
- [Stage 2 v3（field-wise PRM）暂停时状态](#stage-2-v3field-wise-prm暂停时状态)
- [关键踩坑速查](#关键踩坑速查)

---

## Stage 1 SFT 运行

### run-S1-A · sft_baseline（CE only）

| 项 | 值 |
|---|---|
| **入口** | `src/stage1_sft/train.py` |
| **数据** | `data/sft/train.parquet` (5353)，按 image_file 分组切 |
| **超参** | LoRA r=32, alpha=64, dropout=0.05；lr=2e-4；batch_size=1；grad_accum=8；3 epoch |
| **辅助损失** | 无（仅 CE） |
| **flash_attn / grad-ckpt** | 是 / 是 |
| **硬件** | 单卡 L20（GPU 1-7 任一） |
| **峰值显存** | ~36 GB |
| **耗时** | ~6 h |
| **产物** | `models/sft_ckpt/` → merge → `models/sft_baseline_merged/` (~17.5 GB) |
| **评测 ckpt** | `results/eval_sft_baseline.json` |

收敛要点：
- CE 三 epoch 收敛到 ~0.18（接近饱和）；val loss 与 train loss 同步下行
- attribute 字段精度上升，到 epoch 2 后基本不动

测试集表现（664 条 test）：

| 指标 | 值 |
|---|---:|
| json_format_accuracy | 1.000 |
| violation_precision | 0.9806 |
| violation_recall | **0.9961** |
| violation_f1 | **0.9883** |
| hallucination_rate | 0.3072 |

来源：[`eval_sft_baseline.json`](eval_sft_baseline.json)。

---

### run-S1-B · sft_aux（CE + 0.05·SupCon + 0.03·Triplet）

| 项 | 值 |
|---|---|
| **入口** | `src/stage1_sft/train.py`（开启 SupCon + Triplet） |
| **数据** | 同 run-S1-A；额外 `data/sft/triplets.parquet` (16061，仅 train 图) |
| **超参** | 同 run-S1-A |
| **辅助损失** | 0.05 · SupCon（EOS embedding，memory bank=64）+ 0.03 · Triplet（cosine, margin=0.3） |
| **硬件 / 耗时 / 显存** | 单卡 L20 / ~6.5 h / ~37 GB |
| **产物** | `models/sft_aux_ckpt/` → merge → `models/sft_aux_merged/` (~17.5 GB) |
| **评测 ckpt** | `results/eval_sft_aux.json` |

测试集表现（664 条 test）：

| 指标 | 值 | vs run-S1-A |
|---|---:|---:|
| violation_f1 | 0.9844 | −0.39 pp |
| hallucination_rate | **0.3027** | **−0.45 pp** |
| violation_recall | 0.9921 | −0.40 pp |

**结论**：辅助损失在 SFT 任务上**几乎中性**（F1 差异 < 1 pp），但
sft_aux 是后续 Stage 2 / Stage 3 的 backbone —— SupCon 的 embedding
几何让 RM mean_margin 从 9.62 升到 11.21（见 run-S2-v2-aux）。

来源：[`eval_sft_aux.json`](eval_sft_aux.json)。

---

## Stage 2 Reward Model 运行

200-pair holdout 全部来自 `data/preference/preference_holdout.parquet`
（4 策略分布：weak_evidence 64 / wrong_attribute 55 / over_strict 49 /
missed_cue 32）。

### run-S2-v0 · Linear head, base backbone

| 项 | 值 |
|---|---|
| **入口** | `src/stage2_rm/train.py` |
| **backbone** | Qwen3-VL-8B-Instruct（**未 SFT 的 base 权重**），冻结 |
| **head** | `Linear(4096, 1)`，无 LN / 无 bias |
| **loss / lr / epochs** | Bradley-Terry / 1e-4 / 2 |
| **产物 / 评测** | — / [`rm_holdout_baseline.json`](rm_holdout_baseline.json) |

| 指标 | 值 |
|---|---:|
| pair_accuracy | 0.825 |
| mean_margin | **4.00** |
| len_shortcut（chosen − rejected token 数） | +31.2 |

分策略：wrong_attribute 0.855 / over_strict 0.837 / missed_cue 0.813 /
weak_evidence 0.797。

---

### run-S2-v1 · LN+bias head, base backbone

| 项 | 值 |
|---|---|
| **head** | `LN → Linear(4096, 1, bias=True)` |
| **其余** | 同 run-S2-v0 |
| **评测** | [`rm_holdout_headv1.json`](rm_holdout_headv1.json) |

| 指标 | 值 |
|---|---:|
| pair_accuracy | 0.810 |
| mean_margin | 3.30 |

**观察**：单纯加 LN+bias 反而下降，与 head 容量不够吃满 BT 信号
有关。决策：换更深的 MLP head。

---

### run-S2-v2-baseline · MLP head, sft_baseline backbone

| 项 | 值 |
|---|---|
| **backbone** | `models/sft_baseline_merged`，冻结 |
| **head** | `LN → Linear(4096,2048) → GELU → Dropout(0.1) → Linear(2048,1)` (~10 M) |
| **其余** | 同上 |

| 指标 | 值 |
|---|---:|
| pair_accuracy | 0.825 |
| train_acc | 0.863 |
| mean_margin | **9.62** |

**观察**：MLP head + SFT backbone 把 mean_margin 从 4.00 拉到 9.62
（+140%），这是 RM 升级的主要增量；pair_acc 卡在 0.825（200 holdout
的 ceiling）。

---

### run-S2-v2-aux · MLP head, sft_aux backbone（最终采用）

| 项 | 值 |
|---|---|
| **backbone** | `models/sft_aux_merged`，冻结 |
| **head** | 同 run-S2-v2-baseline |
| **产物** | `models/rm_ckpt_v2_aux/reward_head_best.pt` (~10 MB) |

| 指标 | 值 | vs v2-baseline |
|---|---:|---:|
| pair_accuracy | 0.825 | — |
| train_acc | **0.884** | +2.1 pp |
| mean_margin | **11.21** | +16.5% |

**决策**：RL 阶段的 reward signal 选 v2-aux。pair_acc 三者都 0.825 →
200 holdout 已到 ceiling；margin 维度 v2-aux 显著胜出。

⚠️ `len_shortcut +31` 需要在 RL 训练时关注：chosen 平均比 rejected
长 31 token，需确认 RM 不是靠"长度"作弊。

---

### run-S2-PRM · token-level PRM holdout

> Process Reward Model 的早期尝试：在 sft_aux backbone 上把 reward
> 头变成 token-level，mean-pool 后做 Bradley-Terry。这是新版仓库
> [`src/prm/`](../../src/prm/) 的前身。

| 项 | 值 |
|---|---|
| **backbone** | `models/sft_aux_merged`，冻结 |
| **head** | LN → Linear(4096, 1)（**简化版**，无 MLP） |
| **目标** | token-level mean-pool → Bradley-Terry |
| **评测** | [`prm_holdout.json`](prm_holdout.json) |

| 指标 | 值 |
|---|---:|
| pair_accuracy | **1.000** |
| mean_margin | **16.70** |
| mean_token_reward | 10.20 |

**观察**：把 reward 信号铺到每个 response token 后，200 holdout 上
pair-acc 直接顶到天花板。这一发现支撑了 2026-04 之后**新版仓库**把
PRM 当作主线的方向（见 [`../README.md`](../README.md) §十一）。

---

## Stage 3 FIPO-RL 运行

### run-S3-v1 · 易样本基线（reward saturation）

| 项 | 值 |
|---|---|
| **入口** | `src/stage3_fipo/run_fipo_v1.sh` |
| **数据** | `data/fipo/train.parquet`（5353 条均匀采样） |
| **算法** | GRPO + future-KL；reward_v2（5 组件） |
| **超参** | actor.lr=1e-6；rollout n=8 / temp=1.0 / top_p=0.95；train_batch=6；max_prompt=8192；max_response=1024；total_epochs=2 |
| **分布式** | FSDP2 + param_offload + optimizer_offload=False；6 卡 L20（GPU 1-6） |
| **硬件 / 耗时** | 6×L20 ~25 GB/卡 / 240 → 309 共 69 步耗 ~2.3 h |

**观察（reward saturation）**：
- val/total reward 几乎不动：4.7122 → 4.7097
- `critic/score/min` 19/19 step 全是 4.5；`critic/score/max` 19/19 step 全是 5.0
- **89% step `actor/loss = 0` 且 `actor/grad_norm = 0`**

**根因**：GRPO 的 advantage `A = (r − mean) / std`。所有 rollout 都
≈ 满分 → std → 0 → A → 0 → policy gradient = 0。

**修复路径**：进入 run-S3-v2（hard sample mining 70/30 mix）。

---

### run-S3-v2 · hard-mining 70/30 mix（最终采用 step160）

| 项 | 值 |
|---|---|
| **数据准备** | `mine_hard_samples.py`（sft_aux_merged 全量打分 2000 条 4651s on 1×L20）→ 6 规则筛 351 hard → `build_rl_train.py` 70/30 mix + 2x 上采样 → 1749 条 `data/fipo/rl_train_hard.parquet` |
| **超参** | 同 run-S3-v1，仅训练数据替换 |
| **环境变量**（必备） | `FIPO_DECAY_RATE=12.0 FIPO_CHUNK_SIZE=128 FIPO_FKL_CLIP_RATIO=0.2 FIPO_SAFETY_THRESH=4.0 RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com` |
| **硬件 / 时延** | 6×L20 ~25 GB/卡；timing/step ~117s（gen 13s / old_log_prob 27s / update_actor 70s / update_weights 7s）；throughput ~180 token/s |
| **总训练步** | 2 epoch × 333 step = 666 step，**采用 step 160 ckpt（早停）** |
| **产物** | `models/rl_ckpt/`（FSDP shards）→ `merge_fipo_ckpt.py` → `models/fipo_v2_step160_merged/` (~17.5 GB) |

**训练动态**（214 step 截取）：

| 指标 | simple data (1-120) | hard data (121-200) |
|---|---|---|
| influence_weights_min | 0.851 | 0.861 |
| influence_weights_max | 1.155 | 1.148 |
| ppo_kl | ~3e-6 | ~-1e-4 |
| entropy | 0.167 | 0.127 (**−24%**) |
| response_length | 113 | 110 |

**关键证据**：
1. `influence_weights ∈ [0.85, 1.15]` 全程 → future-KL 每 step 都在工作
2. `ppo_kl ≈ 0` 全程 → KL 被压在 1e-4 量级（GRPO 典型 5e-3 ~ 5e-2）
3. entropy −24% 但 response_length 不变 → confidence 抬升而**无 mode collapse**

**测试集表现**（664 条 test）：

| 指标 | 值 | vs sft_baseline | vs sft_aux |
|---|---:|---:|---:|
| violation_f1 | **0.9883** | 0 | +0.39 pp |
| violation_recall | **0.9961** | 0 | +0.40 pp |
| hallucination_rate | **0.2349** | **−7.23 pp** | **−6.78 pp** |

来源：[`eval_fipo_step160.json`](eval_fipo_step160.json)。

**核心发现**：F1 在 SFT 阶段已经 98% 饱和，**RL 真正打的是
hallucination**：30.27% → 23.49%，相对降 22%。

---

## Stage 4 RAG 评估

所有 RAG 实验默认使用 `field_min` confidence 信号（屏蔽
max_prob > 0.999 的结构 token），CLIP-ViT-B/32 visual top-3 + BM25
(jieba) textual top-3，索引 `data/rag_index/` 共 3093 条视觉 + 170
条文本（20 规则 + 150 案例）。

### run-S4-CAL · field-aware confidence calibration

| 项 | 值 |
|---|---|
| **入口** | `scripts/calibrate_confidence.py` |
| **数据** | 30 条 val 样本 |
| **目标** | 在 4 信号（mean_max / min_max / field_min / mean_entropy）上扫描阈值，找 target_recall=0.8 对应阈值 |
| **结果** | [`stage4_confidence_calibration.json`](stage4_confidence_calibration.json) |

实测分位数（4 信号）：

| 信号 | min | p25 | median | p75 | max | 默认 0.85 触发率 |
|---|---|---|---|---|---|---|
| `mean_max` | 0.902 | 0.935 | 0.950 | 0.960 | 0.979 | **0%（永不触发）** |
| `min_max` / `field_min` | 0.245 | 0.358 | 0.413 | 0.482 | 0.616 | **100%（每条都触发）** |

**校准决策**：选 `field_min < 0.40`（约 val p35 分位），实测 test 上
触发约 32%。

⚠️ **历史踩坑**：旧版 `AuditPipeline` 的默认阈值 `0.85` 是
`mean_max` 时代留下来的；切到 `field_min` 后必须重新校准，否则
要么不触发，要么 100% 触发。

---

### run-S4-T030 / T035 / T040 / T045 / T050 · 阈值扫描

锁定 `fipo_v2_step160_merged` backbone，扫描 `field_min` 阈值（664 test）：

| 阈值 | RAG 触发率 | hallucination | F1 | Precision | Recall | Δhallu vs no-RAG | JSON 文件 |
|---|---:|---:|---:|---:|---:|---:|---|
| no-RAG | 0% | 0.2349 | **0.9883** | 0.9806 | **0.9961** | – | [`eval_fipo_step160.json`](eval_fipo_step160.json) |
| 0.30 | 7.7% | 0.2364 | 0.9863 | 0.9805 | 0.9921 | +0.15 pp | [`eval_fipo_rag_t030.json`](eval_fipo_rag_t030.json) |
| 0.35 | 17.3% | 0.2364 | 0.9842 | 0.9881 | 0.9803 | +0.15 pp | [`eval_fipo_rag_t035.json`](eval_fipo_rag_t035.json) |
| **0.40** ✓ | **32.2%** | **0.2304** | 0.9802 | 0.9841 | 0.9764 | **−0.45 pp** | [`eval_fipo_step160_rag.json`](eval_fipo_step160_rag.json) |
| 0.45 | 50.8% | 0.2440 | 0.9821 | **0.9920** | 0.9724 | +0.91 pp | [`eval_fipo_rag_t045.json`](eval_fipo_rag_t045.json) |
| 0.50 | 75.3% | 0.2380 | 0.9801 | 0.9919 | 0.9685 | +0.31 pp | [`eval_fipo_rag_t050.json`](eval_fipo_rag_t050.json) |

**三个清晰现象**：
1. **hallucination 是 U 型曲线，0.40 是 sweet spot**
2. **F1 单调下降，Recall 是受害者**（认知性收紧：少误报但漏真违规）
3. **hallucination 与 F1 是 trade-off**：0.40 给最低 hallucination
   但 F1 不是最高

**生产采用**：`field_min < 0.40`。

---

### run-S4-Vis / run-S4-Text · 模态消融

锁定 fipo_v2 + 阈值 0.40，触发样本完全一致，唯一变量是检索模态。

| 配置 | hallucination | F1 | Precision | Recall | JSON 文件 |
|---|---:|---:|---:|---:|---|
| no-RAG（参照） | 0.2349 | **0.9883** | 0.9806 | **0.9961** | [`eval_fipo_step160.json`](eval_fipo_step160.json) |
| **visual-only** | **0.2334** | 0.9825 | 0.9693 | **0.9961** | [`eval_fipo_rag_visual_only.json`](eval_fipo_rag_visual_only.json) |
| text-only | 0.2380 ↑ | 0.9761 | **0.9879** | 0.9646 | [`eval_fipo_rag_text_only.json`](eval_fipo_rag_text_only.json) |
| both（参照） | **0.2304** | 0.9802 | 0.9841 | 0.9764 | [`eval_fipo_step160_rag.json`](eval_fipo_step160_rag.json) |

**关键发现**：
1. visual-only 已贡献几乎全部减幻觉效果（0.2334 vs both 0.2304，
   只差 0.30 pp）
2. text-only **不降反升**到 0.2380；text 真正用途是 precision
   tightening（0.9879 最高，但 recall 跌到 0.9646）

**机制解释**：幻觉 = "reason 提到了图里没有的属性"。视觉检索召回
3 张相似真实样本提供视觉锚点；文本检索给业务规则但不告诉模型
"图里有没有这个东西"，本来看错时只会被规则强化。

---

### run-S4-Agentic-n200 · Plan-Then-Retrieve + Verify-Then-Rewrite

> 一面之后的改进尝试：把单跳 confidence-gated RAG 升级为三阶段
> Agentic Loop。**未拉满到 664 完整 test**，仅在 200 条
> `seed=42` 子集上做苹果对苹果对比。

| 项 | 值 |
|---|---|
| **数据** | `data/sft/test.parquet` 中 200 条，`seed=42` |
| **basebone** | `models/fipo_v2_step160_merged` |
| **新增机制** | (1) 长尾品类强制触发（食品 / 其他 / 医药）；(2) verify 评分 < 0.5 fallback 到 v1；0.5 ≤ score < 0.7 rewrite 一次 |
| **入口** | `scripts/evaluate_agentic.py`（仍在 `legacy/scripts/` 外的新版分支） |
| **耗时** | 5.6 s/sample (Agentic) vs 5.14 s/sample (old RAG)；多 0.46 s 来自 verify 的 1-token forward |
| **总结报告** | [`AGENTIC_RESULTS.md`](AGENTIC_RESULTS.md) |

主对比表（同 200 条 seed=42）：

| 指标 | Old RAG (v1) | **Agentic (v2)** | Δ | JSON |
|---|---:|---:|---:|---|
| violation_f1 | 0.9711 | **0.9767** | +0.56 pp | [`eval_fipo_step160_agentic_n200.json`](eval_fipo_step160_agentic_n200.json) |
| violation_precision | 0.9655 | **0.9767** | +1.12 pp | (同上) |
| violation_recall | 0.9767 | 0.9767 | 0 | (同上) |
| hallucination_rate | **0.275** | 0.290 | +1.5 pp ↑ | (同上) |
| 总触发率 | 35.0% | 39.5% | +4.5 pp | (同上) |

老 RAG baseline 数字来自
[`eval_fipo_step160_oldrag_n200.json`](eval_fipo_step160_oldrag_n200.json)，
冒烟健康检查文件
[`_smoke_agentic.json`](_smoke_agentic.json) / [`_smoke_agentic_debug.json`](_smoke_agentic_debug.json)
来自 3 样本 dry run（n_samples=3 时 F1=0 是预期：阴性样本太少，
仅验证管道可跑通）。

Agentic 内部信号分解：

| 信号 | 数值 |
|---|---:|
| 触发率（总） | 39.5% |
| ├─ confidence 触发 | 35.0% |
| └─ 长尾品类强制触发 | 7.5%*（食品 5 + 其他 3 + 医药 7 = 15 条） |
| **fallback 到 v1 比例** | **31.65%** |
| rewrite 比例 | 2.53% |
| verify_score 中位数 | **0.996** |
| verify_score 均值 | 0.715 |

（\*confidence 触发与长尾触发存在重叠，所以两者之和大于总触发率）

**一句话总结**：Agentic 在 F1 / Precision 上有正收益（verify 把
不可信 v2 拒掉），hallu 整体小幅 ↑ 是因为强制触发把医药这种**案例
库覆盖度差**的品类拖累。verify 不是摆设——31.65% 的触发样本被
verify 拒掉，把模型从 v2 的"过度修改"拉回 v1 的稳健输出。

---

## Stage 2 v3（field-wise PRM）暂停时状态

Stage 2 v3 的训练流程见 [`../README.md` §11](../README.md#十一stage-2-v3field-wise-prm暂停时状态)。
冻结时**未跑出最终评估 JSON**；data 生产路径有过 dry-run，权重未训。

冻结时的待办：
1. `python -m scripts.data.S4Data_v3 --per_type 60 --batch_size 10 --temperature 1.0 --model qwen-plus --rate_limit 4`（约 ¥3）
2. `python -m scripts.data.S2DataV3 --policy_path models/sft_aux_merged ... --K 8 --teacher_models qwen-vl-max,qwen-plus`（~13 h GPU + ~¥100 API）
3. `python -m src.stage2_rm.v3.train --epochs 3 --lr 1e-4 ...`（12-18 h on 1×L20）
4. pointwise / pairwise 评估两套

冻结时已落地的代码清单见 README §11.1。

---

## 关键踩坑速查

### Stage 1 SFT

| 现象 | 根因 | 修复 |
|---|---|---|
| `device_map="auto"` 4 GPU pipeline 慢 6-7×（30s → 200s/step） | batch=1 + grad_accum=16 时大量 GPU stall | 单卡 + grad-ckpt + flash-attn（除非显存装不下） |
| flash-attn 装不上 | torch / CUDA wheel 不匹配 | 用 `flash_attention_2 → sdpa → eager` 三级自动回退（见 `src/utils/model_loader.py`） |

### Stage 2 RM

| 现象 | 根因 | 修复 |
|---|---|---|
| `chosen` 平均比 `rejected` 长 31 token | 4 策略中 weak_evidence + over_strict 倾向于让 rejected 更短 | 评估时记录 `len_shortcut`；RL 阶段在 reward_v2 加 `8 ≤ len ≤ 250` 长度卫生 |
| pair_acc 三个 v2 都卡在 0.825 | 200 holdout ceiling | 看 mean_margin（v2-aux 11.21 > v2-baseline 9.62）做决策 |

### Stage 3 FIPO

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 4 | `FileNotFoundError: module_path='src.stage3_fipo...'` | verl `load_extern_object` 期望文件路径 | 传绝对文件路径 + `sys.path` 注入 |
| 5 | `Prompt length 4339 > model max 2048` | vLLM KV cache 没考虑 image tokens | `MAX_PROMPT_LEN=8192` |
| 12 | actor worker `Unsupported loss mode: future_kl` | `POLICY_LOSS_REGISTRY` 是 per-process 状态，Ray actor 是 fresh interpreter | 项目根加 `sitecustomize.py`，site 机制让每个 Python 子进程启动时 auto-import |
| 13 | `NCCL ncclInvalidUsage: Duplicate GPU` | Ray 默认不覆盖 actor 的 CUDA env，6 worker 都默认绑 device 0 | `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` |
| 14 | `All Metric instances must have same number of values` | future_kl_loss 的 metrics 字段是条件性的，6 DP rank 字段集不一致 | 固定 metric schema，缺失字段填 0.0 |
| — | reward saturation：89% step grad_norm=0 | GRPO advantage std → 0 | hard mining + 70/30 mix（见 run-S3-v2） |

### Stage 4 RAG

| 现象 | 根因 | 修复 |
|---|---|---|
| `confidence_threshold=0.85` 用 `field_min` 时**永不触发** | 0.85 是 `mean_max` 时代历史值（mean_max 分布 [0.90, 0.98]，field_min 分布 [0.25, 0.62]） | 切信号必须重新校准，见 run-S4-CAL |
| BM25 召回退化为全检索 | `text.lower().split()` 把每条中文规则收缩成 1 token | 抽公共 `_tokenize_zh()`：jieba 优先，缺失字符级回退 |
| 150 条违规案例未召回 | indexer 只索引规则文件 | `indexer.py` 新增 `--case_file`：规则 + 案例统一进 BM25 |
| text-only RAG **升幻觉** | BM25 召回的规则在缺视觉锚点时把模型带偏 | 生产用 visual-only 或 visual+text；不要单跑 text-only |

### GPU 约束

| 现象 | 根因 | 修复 |
|---|---|---|
| 训练突然 NaN / loss 异常 | GPU 0 持续 ECC error（硬件） | **所有脚本必须 `CUDA_VISIBLE_DEVICES=1,..,7`**；RL 用 6 卡（GPU 1-6），评估用 GPU 7 |

---

> **最后更新**：2026-04-28（项目冻结）。本文件按"按运行追踪"维度
> 整理了旧版可复现性所需的全部超参 / 产物 / 收敛信号 / 踩坑修复，
> 与同目录 JSON 报告 + [`../README.md`](../README.md) 互为索引。
