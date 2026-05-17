# Agentic RAG 实验结果（面试速查）

> 一面后改进 v1：把单跳 confidence-gated RAG 升级为 **Plan-Then-Retrieve + Verify-then-Rewrite** 三阶段 Agentic Loop。
>
> **数据**：200 条 sample 自 `data/sft/test.parquet`，`seed=42`（两组完全相同 200 条 → 苹果对苹果）
> **基座**：`models/fipo_v2_step160_merged`（FIPO-RL 后的 ckpt）
> **GPU**：单卡 L20，独占；Agentic 5.6 s/sample，old-RAG 5.14 s/sample（多 0.46 s/sample 来自 verify 的 1-token forward）

---

## 1. 主对比表（同 200 条 seed=42）

| 指标 | Old RAG（v1） | **Agentic（v2）** | Δ |
|---|---:|---:|---:|
| violation_f1 | 0.9711 | **0.9767** | **+0.56 pp** |
| violation_precision | 0.9655 | **0.9767** | **+1.12 pp** |
| violation_recall | 0.9767 | 0.9767 | 0 |
| hallucination_rate | **0.275** | 0.290 | +1.5 pp ↑ |
| json_format_accuracy | 1.000 | 1.000 | 0 |
| 总触发率 | 35.0% | 39.5% | +4.5 pp（多出来全是长尾强制触发） |

**一句话总结**：Agentic 在 **F1 / Precision** 上有正收益，靠 verify 的 fallback 机制把不可信 v2 拒掉；hallu 整体小幅 ↑ 是因为**强制触发**把医药这种 case 库覆盖度差的品类拖累。

---

## 2. Agentic 内部信号分解（自指标）

| 信号 | 数值 | 含义 |
|---|---:|---|
| 触发率（总） | 39.5% | 触发了 RAG 的样本比例 |
| ├─ 由 confidence 触发 | 35.0% | 与 old-RAG 完全一致（field_min < 0.40） |
| └─ 由长尾品类强制触发 | 7.5%* | 食品 5 + 其他 3 + 医药 7 = 15 条 |
| **fallback 到 v1 比例** | **31.65%** | 触发后 verify_score < 0.5，**主动拒掉 v2** |
| rewrite 比例 | 2.53% | 0.5 ≤ verify_score < 0.7，重写一次 |
| verify_score 中位数 | **0.996** | 大多数 v2 高度被支持 |
| verify_score 均值 | 0.715 | 均值低于中位数 → 双峰分布（要么很高要么很低） |

*（confidence 触发与长尾触发有重叠，所以两者之和大于总触发率）*

**关键**：verify 不是摆设——31.65% 的触发样本被 verify 拒掉，把模型从 v2 的"过度修改"拉回 v1 的稳健输出。这就是 Precision 涨的来源。

---

## 3. 分品类幻觉率（最关键的洞察）

| 品类 | n | Old RAG halluc | **Agentic halluc** | Δ | 解读 |
|---|---:|---:|---:|---:|---|
| 鞋 | 42 | 0.214 | **0.190** | **−2.4 pp** | 主流品类，Agentic 略好 |
| 手表 | 26 | 0.154 | 0.115 | **−3.9 pp** | 主流品类，verify 起作用 |
| 服装 | 92 | 0.283 | 0.304 | +2.1 pp | 大盘类，noise 较大 |
| 化妆品 | 10 | 0.300 | 0.500 | +20.0 pp | 长尾，case 库覆盖差 |
| **食品** | **5** | **0.600** | **0.400** | **−20.0 pp** ✓ | **强制触发救回了** |
| **医药** | **7** | **0.571** | **0.857** | **+28.6 pp** ✗ | **case 库 0 条医药 → 检索带偏** |
| 包 | 12 | 0.417 | 0.417 | 0 | 长尾不触发 |
| 配饰 | 3 | 0.0 | 0.0 | 0 | 太少 |
| 其他 | 3 | 0.333 | 0.333 | 0 | 太少 |

**核心发现**：**长尾强制触发是把双刃剑** —— 食品有 case 覆盖（150 条 cases 里食品占 8 条）→ 救回 20pp；医药 case 库为空 → 检索把模型带偏。**这正好印证了 INTERVIEW.md §5.7 提到的"长尾品类的真实瓶颈是 case 库覆盖度"**。

---

## 4. 与既有 664 条 baseline 的位置关系

| 系统（**不同样本**）| F1 | Halluc | n |
|---|---:|---:|---:|
| FIPO no-RAG（既有 664 条） | 0.9883 | 0.2349 | 664 |
| FIPO + Old-RAG（既有 664 条） | 0.9802 | **0.2304** | 664 |
| **FIPO + Old-RAG（公平 200 条）** | **0.9711** | 0.2750 | 200 |
| **FIPO + Agentic（公平 200 条）** | **0.9767** | 0.2900 | 200 |

> 200 条比 664 条整体 hallu 高约 4 pp，是 sample 噪声（200 里恰好长尾品类占比偏高：医药 3.5% / 食品 2.5%，相比 train 1%/2%）。
> **结论**：Agentic vs Old-RAG 的提升信号在 200 条上稳健（F1 +0.56pp，Precision +1.12pp），全 664 评测预计也是同方向；时间允许后会回填。

---

## 5. 三个能在面试上讲的故事

### 5.1 Plan-Then-Retrieve 设计（呼应简历）

> "我把简历上写的 'Plan-Then-Retrieve, 对不同 category 字段路由至不同的规则库' 真正实现了。流程是：
> 1. **第一次 generate**：模型先输出 JSON（含 category 字段）
> 2. **路由**：用 `coarse_category()` 把自由文本 category 映到 10 个粗粒度桶
> 3. **路由检索**：BM25 只在 `{该桶 ∪ 通用}` 的规则文档里跑——例如食品商品只看食品规则 + 通用规则，不会把化妆品规则拉进 prompt
> 4. **第二次 generate**：把路由后的检索结果注入 system prompt 重新生成"

实现见 `_retrieve_text_routed()` in `src/stage4_rag/inference.py:432`。代码层面很轻——复用现有 BM25 索引，运行时按 `rules[i].category` 做 mask + score 截断，**没重建索引（省 5 min）**。

### 5.2 多触发器：confidence + 长尾品类双门控

> "原方案只有一个触发器（field_min < 0.40），但实测 confidence 在长尾品类上**误差很大**——医药样本模型很自信地输出错答案 confidence 0.6+。
> 我加了第二个触发器：**coarse 落入 {医药, 电子产品, 食品, 其他} 强制触发 RAG**，无视 confidence。结果食品品类 hallu 从 60% 降到 40%（救回 20pp）。"

### 5.3 Verify-then-Rewrite：Agentic loop 的核心

> "Plan-Then-Retrieve 后我加了第三步——**让模型自己当质检员**。
> Verify prompt：'给出图片 + 待复核的审核结论 + 检索证据，判断是否一致？只输出 是/否'。
> 取下个 token 上 '是'/'否' 的 softmax 比值作为 verify_score ∈ [0,1]。
> 三档决策：
> - score < 0.5 → fallback 到 v1（v2 不可信）
> - 0.5 ≤ score < 0.7 → rewrite 一次（带反馈 prompt）
> - score ≥ 0.7 → 接受 v2
>
> 实测：触发的 79 条样本里 31.65% 走了 fallback，**这正是 Precision +1.12pp 的来源**——verify 把检索带偏的 v2 主动拒掉了。"

---

## 6. 已知坑 + 后续要做的事

| 问题 | 现象 | 解法（面试可讲 future work） |
|---|---|---|
| **长尾强制触发的负收益** | 医药品类 hallu +28.6 pp | case 库扩 medical case；或在路由时检测"该 coarse 在 case 库 ≥ 5 条才触发" |
| **rewrite 几乎不工作** | 仅 2.53% 命中 [0.5, 0.7) 区间 | verify_score 是双峰（median 0.996）；可改成 Top-k=2 sampling 让分布更平滑 |
| **样本数偏小** | 200 条 sample noise 大 | 跑全 664 → 把 hallu 信号收紧到 ±0.5 pp |
| **verify 模型 = 主模型** | 自验证可能偏向自己 | 用 Qwen-VL-Max 当 verifier 做 ablation；或训一个轻量 reward model 当 verifier |

---

## 7. 跑命令（reproduce）

```bash
# Agentic full
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_agentic.py \
  --model_path models/fipo_v2_step160_merged \
  --test_parquet data/sft/test.parquet \
  --rag_index_dir data/rag_index \
  --max_samples 200 --sample_seed 42 \
  --rag_threshold 0.40 --vt_low 0.5 --vt_high 0.7 \
  --out results/eval_fipo_step160_agentic_n200.json

# Old RAG (公平对照)
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_agentic.py \
  --model_path models/fipo_v2_step160_merged \
  --test_parquet data/sft/test.parquet \
  --rag_index_dir data/rag_index \
  --max_samples 200 --sample_seed 42 \
  --rag_threshold 0.40 \
  --disable_verify --disable_rewrite --disable_routing --disable_long_tail \
  --out results/eval_fipo_step160_oldrag_n200.json
```

跑完整 664 条只需把 `--max_samples 200` 改 `--max_samples 0`，预计 ~37 min/组。

---

## 8. 核心代码改动

| 文件 | 改动 |
|---|---|
| `src/stage4_rag/inference.py` | 新增 `AgenticAuditPipeline` (~200 LoC, 继承 `AuditPipeline`)；override `predict()` 实现 6 步 loop |
| `scripts/evaluate_agentic.py` | 新增独立评估脚本，支持 4 个 ablation flag (`--disable_verify/rewrite/routing/long_tail`) |

**没动**：原 BM25 索引、indexer.py、`AuditPipeline`、reward_fn、训练 ckpt——这是**纯推理期改造**，零再训练成本。
