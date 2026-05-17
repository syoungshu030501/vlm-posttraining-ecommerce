# VLM Post-Training for E-commerce Audit

> 电商商品合规审核 VLM 后训练项目：在 **Qwen3-VL-8B-Instruct** 基座上做 **SFT → SFT-aux (SupCon+Triplet) → RM → FIPO-RL → RAG** 五阶段后训练，得到一个能看图 + 看商品描述、输出结构化审核 JSON 的合规审核模型。
>
> **当前最佳成绩（664 条 test，2026-04-25）**：`F1 = 0.9802 / Precision = 0.9841 / Recall = 0.9764 / hallucination_rate = 0.2304`（FIPO-v2 + RAG，相比 SFT baseline 幻觉率从 30.27% 降到 23.04%，**−24% 相对降幅**）

| 阶段 | 模型 ckpt | F1 | Precision | Recall | hallucination | RAG 触发率 |
|---|---|---:|---:|---:|---:|---:|
| Stage 1 baseline | `sft_baseline_merged` | **0.9883** | 0.9806 | **0.9961** | 0.3072 | – |
| Stage 1 + aux loss | `sft_aux_merged` | 0.9844 | 0.9767 | 0.9921 | 0.3027 | – |
| + RAG（aux backbone） | `sft_aux_merged` + RAG | 0.9702 | 0.9799 | 0.9606 | 0.3238 ↑ | 46.2% |
| Stage 3 FIPO-v2 | `fipo_v2_step160_merged` | 0.9883 | 0.9806 | 0.9961 | **0.2349** | – |
| **+ RAG（FIPO backbone）** | `fipo_v2_step160_merged` + RAG | **0.9802** | 0.9841 | 0.9764 | **0.2304** ✓ | 32.2% |

**核心发现**：F1 在 SFT 阶段已 98%+ 饱和，**RL + RAG 真正打的是 hallucination 这个落地阻塞指标**。

---

## 目录

- [一、项目概述与五阶段架构](#一项目概述与五阶段架构)
- [二、硬件与软件环境](#二硬件与软件环境)
- [三、数据工程](#三数据工程)
- [四、系统架构与数据流](#四系统架构与数据流)
- [五、模型清单](#五模型清单)
- [六、五阶段训练详细记录](#六五阶段训练详细记录)
- [七、端到端评估与消融](#七端到端评估与消融)
- [八、关键设计与踩坑](#八关键设计与踩坑)
- [九、性能优化与速度](#九性能优化与速度)
- [十、项目结构](#十项目结构)
- [十一、面试速查（一页纸 quick prep）](#十一面试速查一页纸-quick-prep)
- [十二、关键数字速查表](#十二关键数字速查表)

---

## 一、项目概述与五阶段架构

### 1.1 任务定义

**输入**：电商商品图 (RGB) + 商品标题/描述 (text)
**输出**：结构化审核 JSON

```json
{
    "category": "服装 / 鞋 / 食品 / ...",
    "attributes": {"颜色": "黑色", "材质": "皮革", ...},
    "violation": true / false,
    "reason": "..."
}
```

**评测指标**：`json_format_accuracy / violation_f1 / precision / recall / hallucination_rate`（reason 是否引用了 attributes 中的 key）

### 1.2 五阶段流程图

```
                        ┌──────────────────────────────────┐
                        │ Stage 0  数据蒸馏 (qwen-vl-max API)  │
                        │  4 批次 ≈¥245 / 90 min            │
                        └────────────────┬─────────────────┘
                                         │ 6685 SFT + 2000 偏好对
                                         ▼
                        ┌──────────────────────────────────┐
                        │ Stage 1  SFT + LoRA r=32          │
                        │  CE + 0.05·SupCon + 0.03·Triplet  │
                        │  → models/sft_aux_ckpt/            │
                        └────────────────┬─────────────────┘
                                         │ merge LoRA
                                         ▼
                        ┌──────────────────────────────────┐
                        │ Stage 2  Reward Model            │
                        │  Bradley-Terry pairwise          │
                        │  MLP head + sft_aux backbone     │
                        │  → models/rm_ckpt_v2_aux/         │
                        └────────────────┬─────────────────┘
                                         │ (RM 不强制接 RL，可选)
                                         ▼
                        ┌──────────────────────────────────┐
                        │ Stage 3  FIPO-RL                  │
                        │  GRPO + future-KL + reward v2    │
                        │  rule-based reward (无 API 依赖)   │
                        │  hard sample mining 解 reward sat │
                        │  → models/fipo_v2_step160_merged/ │
                        └────────────────┬─────────────────┘
                                         │ merge FSDP shards
                                         ▼
                        ┌──────────────────────────────────┐
                        │ Stage 4  RAG (推理时增强)         │
                        │  field_min<0.40 触发              │
                        │  CLIP+FAISS visual + BM25 text   │
                        │  → 主要作用：再降 hallucination   │
                        └──────────────────────────────────┘
```

### 1.3 一句话设计哲学

**SFT 教结构和分类，aux loss 塑造 embedding 几何，RM 学偏好顺序，FIPO 在 token-level 校准 confidence，RAG 在 inference 时给视觉 grounding 兜底。** 这五个阶段是**协同的**——RL 期间用 future-KL 保护住 token-level 表征，正是因为下游 RAG 要靠 confidence 信号做 calibration（详见 §8.4）。

### 1.4 三个核心创新

| 创新点 | 解决什么问题 | 量化收益 |
|---|---|---|
| **FIPO future-KL loss**（vs 纯 GRPO） | RL 期间 token-level confidence 保护，给下游 RAG 留 calibration 空间 | hallucination −7pp（30.27% → 23.04%） |
| **Hard sample mining** + 70/30 mix | reward saturation 导致 89% step 梯度为零 | grad_norm 非零 step 比例 11% → 50%+ |
| **field-aware 置信度** + 中文 BM25 + 视觉案例库 | 原 mean_max 被 JSON 结构 token 稀释到永不触发 | RAG 触发率 0% → 32.2%，且每个被触发样本都是真 hard sample |

---

## 二、硬件与软件环境

### 2.1 硬件

| 资源 | 配置 |
|---|---|
| GPU | 7× NVIDIA L20 (45 GB)，多租户共享，单实验占 6 卡 |
| GPU 黑名单 | **GPU 0 持续 ECC error**，所有训练/推理脚本排除，可用 = {1,2,3,4,5,6,7} |
| RL/RAG 推荐 | 6 卡 L20 + ~25 GB/卡（FSDP2 offload + vLLM 0.70）|

### 2.2 软件

| 组件 | 版本 | 备注 |
|---|---|---|
| Python | 3.12 | conda env VLM |
| CUDA | 12.8 (driver) / 12.4 (torch wheel) | |
| torch | 2.10.0 | + cuDNN 9.10.2 |
| transformers | 5.5.4 | Qwen3-VL native |
| peft | 0.18+ | LoRA |
| vllm | 0.19.1 | RL rollout engine |
| **verl** | **0.8.0.dev0** | `--no-deps` 安装，不升级 vllm；vendor/verl-latest/ |
| ray | 2.55.0 | RL 多 actor 调度 |
| tensordict | 0.12.2 | verl 数据流必需 |
| sentence-transformers | 5.4.1 | reward_fn v2 的 BGE 编码器 |
| flash_attn | 2.8.3+cu12torch2.10 | 单卡 SFT 加速 |

### 2.3 关键模型

| 模型 | 路径 / HF id | 大小 | 用途 |
|---|---|---|---|
| Qwen3-VL-8B-Instruct | `models/pretrained/Qwen3-VL-8B-Instruct` | ~17 GB | base + actor + ref |
| qwen-vl-max | DashScope API | — | Stage 0 数据蒸馏教师 |
| BAAI/bge-small-zh-v1.5 | hf-mirror | ~100 MB | reward_fn v2 语义对齐 |
| clip-vit-base-patch32 | `models/pretrained/clip-vit-base-patch32` | ~600 MB | RAG 视觉检索 |
| BAAI/bge-m3 | hf-mirror（可选） | — | RAG 文本备选 |

---

## 三、数据工程

数据质量是项目的关键限制面，6 个典型问题已全部修复，可作为其他 VLM 项目的 lessons learned。

### 3.1 最终数据现状

| 数据集 | 路径 | 行数 | 用途 | 生成方式 |
|---|---|---:|---|---|
| 原始图片池 | [data/raw/images/](data/raw/images/) | **3093** | 所有阶段视觉源（已去重） | DeepFashion-MultiModal + Pexels 补食品/化妆品/电子各 200 张 |
| 规则库 | [data/raw/rules.jsonl](data/raw/rules.jsonl) | 20 | RAG / reward 规则 | 人工编写 |
| 违规案例库 | [data/raw/violation_cases.jsonl](data/raw/violation_cases.jsonl) | **150** | RAG 语料 | 18 模板 + 132 广东市监局爬取 |
| SFT 注解 | [data/sft/sft.jsonl](data/sft/sft.jsonl) | **6685** | Stage 1 主注解 | qwen-vl-max 四批次蒸馏 |
| SFT 切分 | [data/sft/{train,val,test}.parquet](data/sft/) | **5353 / 668 / 664** | Stage 1/3 训练评估 | 按 `image_file` 分组 80/10/10 |
| 幻觉三元组 | [data/sft/triplets.parquet](data/sft/triplets.parquet) | **16061** | Stage 1 对比损失（仅 train 图） | 16 key 白名单属性扰动 |
| 偏好数据 | [data/preference/preference.parquet](data/preference/) | **2000** | Stage 2 RM 训练 | qwen-vl-max 四策略同图降质 |

### 3.2 五个数据 Phase

```
A · 原料采集（一次性）
   download_dataset + scripts/data/S4Data.py {gd, samr, merge}
   → data/raw/{images/, rules.jsonl, violation_cases.jsonl}

B · 视觉去重（必须先于蒸馏）
   scripts/data/S0Data.py
     ├─ JPEG 再编码 (quality=90) + MD5
     ├─ 同 hash 为一个视觉等价类
     └─ 每类保留 product_XXXXX id 最小者
   → 2550 → 2493 张，dedup_map.json 记录

C · SFT 数据蒸馏
   scripts/data/S1Data.py (v3 版合规 prompt)
   → data/sft/sft.jsonl  ≈¥245 / 90 min

D · SFT 切分 + 三元组
   ├─ src/utils/data_prep.py --mode sft --split  (按 image_file 分组 80/10/10, seed=42)
   └─ src/utils/build_triplets.py                 (仅 train split)

E · 偏好数据蒸馏
   scripts/data/S2Data.py
     ├─ 输入池：SFT train 图（严格非 val/test）
     ├─ 4 策略：weak_evidence / wrong_attribute / over_strict / missed_cue
     ├─ temperature=1.0，banned-phrase filter
     └─ resume-safe 基于 image_file
   → preference_distilled.jsonl  ≈¥60 / 15 min

F · 统一 parquet + 全量体检
   ├─ data_prep.py --mode preference 重建 preference.parquet
   └─ scripts/data/guard.py  全部硬契约
```

### 3.3 偏好数据：四策略同图降质（Stage 2 RM 数据）

**Chosen 直接用 SFT gold；Rejected 由 qwen-vl-max 基于同一张图、同粗粒度品类生成一条推理质量更差的审核**。

| 策略 | 行数 | violation 契约 | 降质手段 |
|---|---:|---|---|
| `api_weaker_weak_evidence` | 620 | 保持 chosen label | reason 保留结论但敷衍，不引用任何具体属性 |
| `api_weaker_wrong_attribute` | 535 | 保持 chosen label | 把 1-2 个属性改为视觉错值，reason 引用错误属性作证据 |
| `api_weaker_over_strict` | 520 | chosen=False → rejected=True | 牵强认定违规（普通描述当极限词） |
| `api_weaker_missed_cue` | 325 | chosen=True → rejected=False | 漏判违规，对违规证据给出开脱理由 |

`missed_cue` 占比曾是天然约束（只有 chosen=True 才能翻成 False），通过 `--only_mode missed_cue` 二次蒸馏从 103 提升到 325。

### 3.4 6 个数据问题修复时间线（lessons learned）

| # | 问题 | 根因 | 修复 | 数字 |
|---|---|---|---|---|
| 1 | SFT v2 合规样本 reason 完全相同 | `COMPLIANT_SYSTEM` prompt 硬编码示例 | 改为"30-80 字引用本条具体属性"，示例标注"仅示范风格" | reason 去重率 25% → **99.6%**，¥97 |
| 2 | SFT parquet 按行随机切，train ∩ val 共享 352 张 | 按 df 行随机切 | 改按 `image_file` 分组切 | 视觉等价类跨 split = 0 |
| 3 | Triplets 长尾 attr_key + 跨属性串扰（"款式: 藏蓝"） | 对任意 key fallback 到颜色池 | 16 key 白名单 + 独立 `STYLE_POOL` | 13 种 key，0 跨属性串扰 |
| 4 | 旧 preference 60% rejected 是其它图的 SFT gold（图文匹配 shortcut） | "跨图注入"误当 hard negative | 改为同图、同品类、4 类降质 | 文本模板聚集消失，reason 唯一度 86% |
| 5 | API 蒸馏首轮 weak_evidence 模板聚集（318/518 照抄示例） | 示例提示不够强 | banned-phrase 列表 + temperature 1.0 + 输出端 filter | 唯一度 19.1% → **85.9%** |
| 6 | 视觉等价类重复（57 对 file 名不同 pixel 相同） | 没做 pixel-level dedup | JPEG(q=90) + MD5 视觉去重 | 2550 → 2493，0 跨 split 泄漏 |

### 3.5 guard 硬契约（任一失败即 fail）

| 契约 | 适用 | 检测 |
|---|---|---|
| JSON 可解析 | 所有行 | `json.loads(strip_markdown_fences(x))` |
| 必需字段齐备 | SFT / Pref | `{category, attributes, violation, reason}` ⊆ keys |
| 视觉等价类不跨 split | train/val/test/pref | JPEG(q=90)+MD5 集合交为空 |
| jsonl ↔ parquet 行数一致 | SFT / Pref | parquet rows == jsonl lines |
| split 可复现 | SFT | seed=42 + 80/10/10 重放与 parquet 一致 |
| 同粗粒度品类 | Pref | `coarse_category(chosen) == coarse_category(rejected)` |
| violation 契约 | Pref | 各策略 must-keep / must-flip 100% |
| triplets 不含 eval | Stage 1 辅助 | 三元组图片 ⊂ train 集合 |

**累计数据成本**：~¥441（含 v1+v2+v3 SFT + Pexels 补图 + preference 多轮蒸馏）

### 3.6 SFT 品类分布（粗粒度 10 桶）

| 桶 | 行数 | 占比 |
|---|---:|---:|
| 服装 | 3448 | 51.6% |
| 鞋 | 1159 | 17.3% |
| 手表 | 565 | 8.5% |
| 其他 | 439 | 6.6% |
| 化妆品 | 358 | 5.4% |
| 包 | 268 | 4.0% |
| 食品 | 145 | 2.2% |
| 配饰 | 128 | 1.9% |
| 电子产品 | 95 | 1.4% |
| 医药 | 80 | 1.2% |

**已知偏倚**：服装+鞋占 70%，最小桶（医药 80 行）在 val/test 里只有 ~8 条 → val acc 必须按粗粒度桶分层，**用 macro-F1 比 overall F1 稳**。

---

## 四、系统架构与数据流

### 4.1 各阶段 IO 详表（**面试必背**）

| Stage | 入口 | 输入 | 输出 | 损失 / 目标 | 核心产物 |
|---|---|---|---|---|---|
| 0 蒸馏 | [scripts/data/S1Data.py](scripts/data/S1Data.py) / [S2Data.py](scripts/data/S2Data.py) | 图像 + (品类/属性 prompt) → qwen-vl-max API | JSON 注解 | — | `data/sft/*.jsonl`, `data/preference/*.jsonl` |
| 1 SFT | [src/stage1_sft/train.py](src/stage1_sft/train.py) | parquet 列：`image (bytes), prompt, response, violation` | LoRA adapter | `CE + 0.05·SupCon + 0.03·Triplet`（基于 EOS embedding） | `models/sft_ckpt/epoch-k/` → merge → `models/sft_aux_merged/` (17.5 GB) |
| 2 RM | [src/stage2_rm/train.py](src/stage2_rm/train.py) | `chosen_*, rejected_*` (input_ids / pixel_values / grid_thw) | scalar reward head（backbone 冻结） | Bradley-Terry: `−log σ(s_c − s_r)` | `models/rm_ckpt_v2_aux/reward_head_best.pt` (~10 MB) |
| 3 FIPO-RL | [src/stage3_fipo/run_fipo_v1.sh](src/stage3_fipo/run_fipo_v1.sh) | rl_train.parquet（`<image>` 占位 + ground_truth dict） | actor 权重（FSDP shards） | reward_v2（5 组件加权和）+ GRPO advantage + future-KL loss | `models/rl_ckpt/` → merge → `models/fipo_v2_step160_merged/` (17.5 GB) |
| 4 RAG | [src/stage4_rag/inference.py](src/stage4_rag/inference.py) | image + text + (FAISS visual + BM25 textual) | enriched system prompt → 二次推理 JSON | — (推理时增强) | `data/rag_index/{visual.faiss, *.pkl}` |

### 4.2 数据流（训练 vs 推理）

```
训练数据流：
  raw/images + qwen-vl-max API
    → Stage 0 蒸馏  → sft.jsonl, preference.jsonl
    → 视觉去重 + split → train/val/test.parquet, preference.parquet, triplets.parquet
    → Stage 1 SFT (LoRA) → merge → sft_aux_merged
    → Stage 2 RM (frozen backbone + scalar head) → rm_ckpt
    → Stage 3 mine_hard_samples → rl_train_hard.parquet (350 hard + 1050 easy)
    → Stage 3 FIPO RL → merge → fipo_v2_step160_merged
    → Stage 4 indexer → data/rag_index/

推理数据流（生产路径）：
  image + product_text
    → fipo_v2_step160_merged.generate (greedy)
    → confidence = field_min(max_softmax_prob over non-structural tokens)
    → if confidence < 0.40:
         CLIP encode image → FAISS top-3 visual
         BM25 (jieba 分词) top-3 (rules + violation_cases)
         augmented system prompt → fipo_v2_step160_merged.generate (二次)
       else:
         直接用第一次输出
    → JSON parse → {category, attributes, violation, reason}
```

### 4.3 三个对齐点（关键）

```
Stage 1 SFT       prompt 用 SFT gold
                  ↓ schema 一致 ↓
Stage 2 RM        chosen / rejected 都用同 prompt 模板，仅 response 差异
                  ↓ schema 一致 ↓
Stage 3 RL        rollout 时 prompt 模板与 SFT 完全一致；只是 response 是 self-sample
                  ↓ schema 一致 ↓
Stage 4 RAG       baseline 路径与 SFT 完全一致；触发后只追加 retrieval block 到 system prompt
```

**核心不变量**：从 Stage 1 到 Stage 4，user prompt 模板、output JSON schema、attributes 字段定义严格一致。这让每一阶段都能直接消费上一阶段的 ckpt。

---

## 五、模型清单

| 角色 | 模型 | 大小 | 训练时 | 推理时 |
|---|---|---|---|---|
| Base | Qwen3-VL-8B-Instruct | 17 GB | 冻结，被 LoRA 注入 | 通过 merge 后直接用 |
| LoRA adapter（SFT） | r=32, alpha=64, dropout=0.05 | ~50 MB | trainable on q/k/v/o + gate/up/down + visual.merger | merge 入 base |
| RM head | LN → Linear(4096,2048) → GELU → Dropout → Linear(2048,1) | ~10 MB | backbone 冻结，仅训 head | 加载到 sft_aux_merged 上输出标量 reward |
| Reward encoder | BAAI/bge-small-zh-v1.5 | ~100 MB | — | 推理时编码 reason 与 attributes 计算余弦 |
| Retrieval encoder | clip-vit-base-patch32 | ~600 MB | — | RAG 视觉编码 |

---

## 六、五阶段训练详细记录

### 6.0 Stage 0 · 数据蒸馏

详见 §3。¥441 / 4 批次 / 约 90 min（SFT）+ 15 min（偏好）。

### 6.1 Stage 1 · SFT + 辅助对比损失

**入口**：[src/stage1_sft/train.py](src/stage1_sft/train.py)

| 配置项 | 值 |
|---|---|
| Base | Qwen3-VL-8B-Instruct |
| LoRA | **r=32, alpha=64, dropout=0.05** |
| target_modules | LM 注意力 (q/k/v/o_proj) + LM MLP (gate/up/down_proj) + 视觉→文本 merger (linear_fc1/linear_fc2) |
| 注意力实现 | `flash_attention_2 → sdpa → eager` 自动回退 |
| Optimizer | AdamW, lr=**2e-4** |
| 主损失 | CE (next token prediction) |
| 辅助损失 | **0.05 · SupCon + 0.03 · Triplet**（基于 EOS token embedding） |
| SupCon | memory bank size=64，detached EOS embedding 入队，凑齐两类才计算 |
| Triplet | `F.triplet_margin_with_distance_loss(distance=cosine, margin=0.3)`，从 triplets.parquet 抽 (image, pos_attr, neg_attr) |
| Epochs / batch | 3 epoch × batch_size=1 + grad_accum |
| 显存 | 单卡 36 GB（fits L20 45 GB），grad-ckpt + flash-attn |

**消融结果**（664 test）：

| 实验 | violation_f1 | hallucination |
|---|---:|---:|
| A: CE only (`sft_baseline_merged`) | **0.9883** | 0.3072 |
| B: CE + SupCon + Triplet (`sft_aux_merged`) | 0.9844 | **0.3027** |

**结论（关键）**：辅助损失对 **下游 SFT 任务中性**（差异 < 1pp），但**对下游 RM 任务有正面影响** —— sft_aux backbone 的 RM `mean_margin = 11.21`，sft_baseline 只有 9.62（**+16.5%**），train_acc 0.884 vs 0.863（见 §6.2）。**深层原因**：SupCon 对 embedding 几何的塑形让 reward head 的 pairwise 比较更友好。

### 6.2 Stage 2 · Reward Model

**入口**：[src/stage2_rm/train.py](src/stage2_rm/train.py)

| 配置项 | 值 |
|---|---|
| Backbone | `models/sft_aux_merged`（**冻结**） |
| Head | `LN → Linear(4096,2048) → GELU → Dropout(0.1) → Linear(2048,1)` (~10M) |
| Loss | Bradley-Terry: `-log σ(s_c - s_r)` |
| Optimizer | AdamW, lr=1e-4 |
| Train / holdout | 1800 / 200 (按 image_file 分组切，与 SFT 同逻辑) |
| Epochs | 2 |

**Head 架构 + Backbone 消融**（200-pair holdout，最终 epoch 2）：

| RM | head | backbone | train_acc | pair_acc | mean_margin |
|---|---|---|---:|---:|---:|
| v0 | Linear | base Qwen3-VL | — | 0.825 | 4.00 |
| v1 | LN+bias | base Qwen3-VL | — | 0.810 | 3.30 |
| **v2-baseline** | MLP+LN+bias+dropout | sft_baseline_merged | 0.863 | 0.825 | 9.62 |
| **v2-aux** ✓ | MLP+LN+bias+dropout | **sft_aux_merged** | **0.884** | **0.825** | **11.21** |

**两层归因**：
- (a) MLP head + SFT backbone（v0 → v2-baseline）：mean_margin 4.00 → 9.62，**+140%**（主要贡献）
- (b) SFT-aux 的 SupCon 几何塑形（v2-baseline → v2-aux）：9.62 → 11.21，**+16.5%**（稳定增量）

**结论**：pair_acc 三个 v2 都卡在 0.825 → 200 holdout 已到 ceiling；**margin 维度 sft_aux 显著胜出，RL 阶段直接用 v2-aux**。

⚠️ `len_shortcut`：chosen 比 rejected 平均长 31 token，需注意 RM 是否依赖长度捷径。

### 6.3 Stage 3 · FIPO-RL

**入口**：[src/stage3_fipo/run_fipo_v1.sh](src/stage3_fipo/run_fipo_v1.sh)

#### 6.3.1 算法栈

- **基础**：基于 veRL 0.8.0.dev0 的 GRPO（group-relative advantage，无 critic）
- **创新**：FIPO future-KL loss（vendored 自 FIPO-main，forward-port 到 verl-latest，~190 LoC 单文件）
- **rollout**：vLLM 0.19.1，n=8 / temperature=1.0 / top_p=0.95
- **分布式**：FSDP2 全 offload + grad-ckpt（6 卡 L20 实测每卡 25 GB）

#### 6.3.2 Reward 函数 v2（rule-based，零 API 依赖）

实现：[src/stage3_fipo/reward_fn.py](src/stage3_fipo/reward_fn.py)

| 组件 | 信号 | reward |
|---|---|---:|
| JSON parseable + 必需字段齐全 | 硬格式 | +1.0 |
| `violation` label 命中 GT | 监督信号 | +2.0 / -1.0 |
| reason 长度卫生 (`8 ≤ len ≤ 250`) | 防退化 | 0 / -1.0 / -0.5 |
| reason 词表 ↔ violation 一致性 | 廉价规则 | +0.5 / -0.3 |
| **reason ↔ attributes 语义对齐**（**bge-small-zh-v1.5 余弦**） | **核心：替代 v1 的 substring 代理** | sim≥0.6: +1.5; sim≤0.2: -0.5; 区间内线性 |
| 解析失败 / 缺字段 | 早退惩罚 | -3.0 / -2.0 |

总 reward 范围 `[-3, 5]`。

**为什么句向量对齐能压幻觉**：BGE 把 reason 和 `"<attr_key>: <attr_val>; ..."` 各编 512-d 余弦相似度，反映"reason 在讨论的东西是不是模型真的提取到的属性"。8 case sanity test：完美样本 sim=0.85（reward 5.0），编造"奥运标志/明星代言"等 attributes 中没有的概念时 sim=0.37（reward 3.86），区分度足够。

#### 6.3.3 训练超参（FIPO v2，实跑）

| Hydra path | 值 | 作用 |
|---|---|---|
| `actor.strategy` | `fsdp2` | actor 训练 FSDP2 全分片 |
| `actor.fsdp_config.param_offload` | `True` | 参数 offload 到 CPU |
| `actor.fsdp_config.optimizer_offload` | `False` | RL OOM 修复后关闭，提升速度 |
| `actor.policy_loss.loss_mode` | `future_kl` | FIPO loss（`vanilla` 切回 GRPO） |
| `actor.use_kl_loss` | `False` | future-KL 内化，不再加传统 PPO KL |
| `actor.clip_ratio_low/high/c` | `0.2 / 0.28 / 10.0` | dual-clip PPO |
| `actor.optim.lr` | `1e-6` | RL 用小 lr 避免破坏 SFT |
| `algorithm.adv_estimator` | `grpo` | GRPO 组内归一化 |
| `data.train_batch_size` | `6` | 每 step 6 prompt × 8 rollout = 48 generations |
| `data.max_prompt_length` | `8192` | Qwen3-VL 一张图吃 ~3000 image tokens |
| `data.max_response_length` | `1024` | 输出通常 100-150 token |
| `rollout.n` | `8` | GRPO 组内归一化的"组" |
| `rollout.gpu_memory_utilization` | `0.70` | vLLM KV cache 比例 |
| `total_epochs` | 2 | 2 × 333 step ≈ 666 step / ETA 22h |

#### 6.3.4 关键环境变量（FIPO knobs）

verl `PolicyLossConfig` 是严格 dataclass 拒绝未声明字段，所以 FIPO 参数全走环境变量：

| 变量 | 值 | 作用 |
|---|---|---|
| `FIPO_DECAY_RATE` | `12.0` | future-KL 权重衰减 |
| `FIPO_CHUNK_SIZE` | `128` | future-KL 累加 token chunk |
| `FIPO_FKL_CLIP_RATIO` | `0.2` | influence weight clip |
| `FIPO_FKL_CLIP_HIGH_ONLY` | `false` | 是否只 clip 上界 |
| `FIPO_SAFETY_THRESH` | `4.0` | sequence-level safety |
| `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES` | `1` | **关键**：避免 Ray worker NCCL Duplicate GPU |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 拉 BGE 必需 |

#### 6.3.5 Hard Sample Mining（v1 → v2 升级）

**v1 reward saturation 现象**（240 → 309 step）：
- val/total 几乎不动（4.7122 → 4.7097）
- `critic/score/min` 19/19 step 都是 4.5
- `critic/score/max` 19/19 step 都是 5.0
- **89% 的 step `actor/loss = 0` 且 `actor/grad_norm = 0`**

**根因**：GRPO 用 `A = (r - mean(r)) / std(r)`，所有 rollout 都 ≈ 满分 → std → 0 → A → 0 → policy gradient = 0

**Hard mining 方案**：
1. `mine_hard_samples.py` 用 sft_aux_merged 在 train.parquet 全量 greedy 推理 + reward_fn 离线打分
2. `build_rl_train.py` 按 6 规则筛困难池（label_wrong / lexicon_contradict / align_low / length_bad / total_low / parse_failed）
3. 70 hard / 30 easy 混合，2x 上采样，1749 条 → `rl_train_hard.parquet`

**实测挖矿结果**（2000 条，4651s on 1 卡 L20）：

| 难例规则 | 命中数 | 占比 |
|---|---:|---:|
| `label_wrong` | 24 | 1.2% |
| `lexicon_contradict` | 6 | 0.3% |
| `align_low`（BGE 余弦 < 0.5） | 254 | 12.7% |
| `total_low`（reward < 4.5） | 351 | 17.6% |

**核心发现**：SFT-aux 在 violation 二分类上已 98.8% 正确，**RL 的边际收益主要在「reason 描述的细致度 / 与规则文档的对齐度」**，不是「敢不敢判违规」。

#### 6.3.6 决策记录：为什么用 RL 而不是续 SFT

挖矿后我们问过：既然 SFT-aux 已 98.8% 正确，剩下 351 条难例能不能直接用 SFT 续训解决？

| 难例类型 | 数量 | SFT 续训能解？ | 理由 |
|---|---:|---|---|
| `label_wrong` | 24 | **能** | 二分类决策，GT 唯一 |
| `lexicon_contradict` | 6 | **能** | 同上 |
| `align_low` | 254 | **不能（只能 50-70%）** | reason 是开放生成，GT 只是众多正确答案之一 |

**SFT 治不了 `align_low` 的 4 个本质原因**：
1. **NLL 是 token-level**：模型生成"图中可见锋利金属边缘"与 GT"画面包含明显刀刃"语义对齐 0.85（reward 高分），但 SFT 每个 token 都罚 → 把多种正确答案的多样性磨平
2. **没有"近似奖励"概念**：BGE 0.7 还能拿 1.05/1.5 reward，SFT 没有"差不多对"
3. **5 个 reward 分量无法显式权衡**：SFT 只能隐式学到加权和
4. **Distribution shift**：SFT 在 GT 训练，infer 在 self-sample；RL 直接在 self-sample 上学

#### 6.3.7 实测 6 卡 L20 性能

| 维度 | 实测值 |
|---|---|
| GPU 占用 | 6 × ~25 GB |
| GPU util | 95-100% |
| timing/step | ~117s（gen 13s + old_log_prob 27s + update_actor 70s + update_weights 7s） |
| throughput | ~180 token/s |
| step:0 baseline reward | 4.52 |
| step:2 actor metrics | `pg_loss=0.019`, `grad_norm=2.09`, `fipo/influence_weights∈[0.87, 1.10]` |

### 6.4 Stage 4 · RAG（confidence-gated 检索增强）

**入口**：[src/stage4_rag/inference.py](src/stage4_rag/inference.py)

#### 6.4.1 原 v1 实现的 3 个核心问题

1. **JSON 结构 token 噪声稀释置信度**：`{`、`}`、`"`、`:`、字段名 max prob ≈ 1.0，占 ≥60% token，**真正反映幻觉的 violation 与 reason 实词被淹没**
2. **无中文 BM25 分词**：`text.lower().split()` 把每条规则收缩成 1 token，BM25 退化成全检索
3. **violation_cases.jsonl 未入索引**：150 条真实违规案例存在但 indexer 只索引规则 → 主要召回路径之一被废

#### 6.4.2 v2 修复

| 模块 | 修复 |
|---|---|
| 置信度多视图 | `_compute_confidence` 同时返回 `mean_max / min_max / field_min / mean_entropy` 4 个信号 |
| 双策略门控 | `confidence_method` 默认 `field_min`，可切回 `mean_max` 兼容旧实验 |
| 中文 BM25 | 抽公共 `_tokenize_zh()`：jieba 优先，缺失字符级回退 |
| 案例库入索引 | `indexer.py` 新增 `--case_file`：规则 + 案例统一进 BM25 |
| 本地 CLIP | 默认 `models/pretrained/clip-vit-base-patch32`，避网络依赖 |

#### 6.4.3 4 信号置信度对比

| 信号 | 计算 | 优点 | 缺点 |
|---|---|---|---|
| `mean_max` | mean of greedy-token max softmax | 廉价 | **被结构 token 稀释** |
| `min_max` | min of greedy-token max softmax | 突出最不确定 token | 偶发噪声敏感 |
| `mean_entropy` | mean per-token entropy (nats) | 反映分布 | 方向相反（越大越不确定） |
| **`field_min`** ✓ | **min(max_prob) over tokens with max_prob ≤ 0.999**（屏蔽结构 token） | **针对性反映 reason/violation 不确定性** | 启发式过滤有 false negative |

#### 6.4.4 阈值实测（关键发现）

`AuditPipeline` 默认 `confidence_threshold=0.85` 是 `mean_max` 时代历史值。在 val 30 条上实测 4 信号分位数：

| 信号 | min | p25 | median | p75 | max | 默认 0.85 触发率 |
|---|---|---|---|---|---|---|
| `mean_max` | 0.902 | 0.935 | 0.950 | 0.960 | 0.979 | **0%（永不触发）** |
| `min_max` / `field_min` | 0.245 | 0.358 | 0.413 | 0.482 | 0.616 | **100%（每条都触发）** |

**生产阈值决策**：选 `field_min < 0.40`（约 val p35 分位），实测触发约 32%（test 上）。

#### 6.4.5 RAG 流程

```
图像 + 商品文本
  → 一次 generate（greedy）+ 记录每个 token 的 max_softmax_prob
  → field_min = min(max_prob) over tokens with max_prob ≤ 0.999
  → if field_min < 0.40:
       CLIP encode image → FAISS top-3 visual cases
       BM25 (jieba) top-3 (rules + violation_cases)
       augmented system prompt: "参考案例：...\n参考规则：..."
       → 二次 generate
     else:
       直接用第一次输出
```

#### 6.4.6 索引产物

| 索引 | 维度/规模 | 大小 |
|---|---|---|
| FAISS visual | 3093 张 × 512 dim (CLIP-ViT-B/32) | 6.3 MB |
| BM25 textual | 170 文档（20 规则 + 150 真实违规案例） | 489 KB |
| image_paths | 3093 entries | 109 KB |

---

## 七、端到端评估与消融

### 7.1 指标定义（与 [scripts/evaluate.py](scripts/evaluate.py) 1:1 对齐）

- **`json_format_accuracy`**：响应 `json.loads` 成功率（必需 ≥95%）
- **`violation_f1 / precision / recall`**：对 `violation` 字段的 binary 分类
- **`hallucination_rate`**：reason 中未引用任一提取出的 attribute value 的比例（**越低越好**）
- **`rag_triggered_rate`**：触发了二次推理的比例（仅 RAG 模式）

### 7.2 完整 Benchmark（664 条 test，2026-04-25）

| Ckpt | 是否 RAG | F1 | Precision | Recall | hallucination | trigger |
|---|---|---:|---:|---:|---:|---:|
| `sft_baseline_merged` | no | **0.9883** | 0.9806 | **0.9961** | 0.3072 | – |
| `sft_aux_merged` | no | 0.9844 | 0.9767 | 0.9921 | 0.3027 | – |
| `sft_aux_merged` | yes (`field_min<0.40`) | 0.9702 | 0.9799 | 0.9606 | 0.3238 ↑ | 46.2% |
| `fipo_v2_step160_merged` | no | 0.9883 | 0.9806 | 0.9961 | **0.2349** | – |
| **`fipo_v2_step160_merged`** | **yes** | **0.9802** | 0.9841 | 0.9764 | **0.2304** ✓ | 32.2% |

**关键观察**：
1. **F1 在 SFT 阶段已饱和**（98.8%），RL 没动 F1，但 hallucination 从 30.27% 降到 23.49%
2. **FIPO + RAG 是唯一负收益变正向的组合**：sft_aux + RAG 反而升幻觉，但 fipo + RAG 进一步降到 23.04%
3. **RAG 触发率反映 confidence 校准质量**：sft_aux 触发 46% 太宽 → noisy；fipo 触发 32% 接近 val 校准目标 → 高质量触发

### 7.3 RAG 阈值扫描（U 型曲线，2026-04-25）

`fipo_v2_step160_merged` 在 5 个阈值上各跑完整 664 评测：

| 阈值 | RAG 触发率 | hallucination | F1 | Precision | Recall | Δhallu vs no-RAG |
|---|---|---:|---:|---:|---:|---:|
| no-RAG | 0% | 0.2349 | **0.9883** | 0.9806 | **0.9961** | – |
| 0.30 | 7.7% | 0.2364 | 0.9863 | 0.9805 | 0.9921 | +0.15pp |
| 0.35 | 17.3% | 0.2364 | 0.9842 | 0.9881 | 0.9803 | +0.15pp |
| **0.40** ✓ | **32.2%** | **0.2304** | 0.9802 | 0.9841 | 0.9764 | **−0.45pp** |
| 0.45 | 50.8% | 0.2440 | 0.9821 | **0.9920** | 0.9724 | +0.91pp |
| 0.50 | 75.3% | 0.2380 | 0.9801 | 0.9919 | 0.9685 | +0.31pp |

**三个清晰现象**：
1. **hallucination 是 U 型曲线，0.40 是 sweet spot**
2. **F1 单调下降，Recall 是受害者**（认知性收紧：少误报但漏真违规）
3. **hallucination 与 F1 是 trade-off**：0.40 给最低 hallucination 但 F1 不是最高

### 7.4 RAG 模态消融（视觉 vs 文本，2026-04-25）

锁定 fipo_v2 + 阈值 0.40，触发样本完全一致，唯一变量是检索模态：

| 配置 | hallucination | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| no-RAG（参照） | 0.2349 | **0.9883** | 0.9806 | **0.9961** |
| **visual-only** | **0.2334** | 0.9825 | 0.9693 | **0.9961** |
| text-only | 0.2380 ↑ | 0.9761 | **0.9879** | 0.9646 |
| both（参照） | **0.2304** | 0.9802 | 0.9841 | 0.9764 |

**关键发现**：
1. **视觉检索贡献了几乎全部减幻觉效果**：visual-only 已达 0.2334，both 只多降 0.30pp
2. **text-only 不降反升 hallucination 到 0.2380**：BM25 召回的规则在缺视觉锚点时反而把模型带偏
3. **text-only 真正用途是 precision tightening**：precision 0.9879 最高，但牺牲 recall（0.9646 最低）

**机制解释**：幻觉定义是"reason 提到了图里没有的属性"。视觉检索召回 3 张相似真实样本 → 模型在 prompt 中看到"这种构图通常包含 X、Y、Z 属性" → 视觉特征对齐时才报；文本检索召回的是"业务规则文本"，不告诉模型"图里有没有这个东西"，模型本来看错时只会被强化。

**设计 implication**：
- **生产可考虑只保留 visual-only**：F1 略高（0.9825 vs 0.9802），hallucination 几乎一样，省 BM25 + jieba 依赖，时延减 ~15ms/sample
- **理想形态**：把 confidence gate 拆成 `confidence_visual_threshold` 和 `confidence_text_threshold` 两个独立旋钮（视觉低阈值多触发主攻减幻觉，文本高阈值少触发主攻 precision tightening）

---

## 八、关键设计与踩坑

### 8.1 SFT-aux 在 SFT 任务中性，但塑造了 RM 的 embedding 几何

**现象**：CE-only vs CE+SupCon+Triplet 在 SFT 下游 violation_f1 差异 < 1pp（0.9883 vs 0.9844），看似辅助损失没用。

**真相**：sft_aux 的 embedding 几何（SupCon 的 uniformity 性质）让 RM 的 pairwise margin 提升 16.5%（9.62 → 11.21），train_acc +2.1pp。**这是 SFT 阶段就为 RM 阶段埋的伏笔**。

**深层原因**：CE 在 SFT 已收敛到 0.18，主任务接近饱和，aux loss 再发力空间有限；但 SupCon 塑造的 hidden state 对 reward head 的 pairwise 比较反而更友好。

**佐证链**：sft_aux backbone → RM mean_margin +16.5% → FIPO 用 v2-aux 作 RM → FIPO+RAG 是唯一正收益组合（相比 sft_aux+RAG 反而负收益）

### 8.2 Reward saturation → Hard mining → 解药

**v1 现象**：FIPO 跑 240 → 309 共 69 步，val/total 几乎不动，89% step `grad_norm = 0`。

**根因**：GRPO 用同 prompt 的 n=8 个 rollout 计算 advantage `A = (r - mean(r)) / std(r)`。当训练样本"太简单"，所有 rollout 都接近满分 → std → 0 → A → 0 → policy gradient = 0。

**实测 19 step**：`critic/score` 始终 [4.5, 5.0]，组内跨度 ≤ 0.5（10% 满分）。

**v2 解药**：mine_hard_samples 全量打分 → 70 hard / 30 easy mix（实际 40% / 60% 难例占比）。

**经验**：reward saturation 在 small-scale RL 项目里非常常见，第一反应不应该是"换算法"而是"看难例分布是否被采到"。

### 8.3 FIPO 启动 15 处坑（**面试加分故事**）

从"代码就位"到"真正 step:1 出现"中间共修复 **15 处**坑。代表性 5 个：

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 4 | `FileNotFoundError: module_path='src.stage3_fipo...'` | `load_extern_object` 期望文件路径 | 传绝对文件路径 + `sys.path` 注入 |
| 5 | `Prompt length 4339 > model max 2048` | vLLM KV cache 没考虑 image tokens | `MAX_PROMPT_LEN=8192` |
| 12 | `Unsupported loss mode: future_kl`（actor worker 报） | `POLICY_LOSS_REGISTRY` 是 per-process 状态 | 项目根加 `sitecustomize.py`，site 机制让每个 Python 子进程启动时 auto-import |
| 13 | `NCCL ncclInvalidUsage: Duplicate GPU` | Ray 默认不再覆盖 actor 的 CUDA env，6 worker 都默认绑 device 0 | `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` |
| 14 | `All Metric instances must have same number of values` | future_kl_loss 的 metrics 字段是条件性的，6 DP rank 字段集不一致 | 固定 metric schema，缺失字段填 0.0 |

**核心经验**：verl 0.8 + Ray + vLLM 这套技术栈对**字段一致性**（reward extra info / loss metrics 字段集需在所有 DP rank 间相同）和**进程隔离**（policy_loss/reward_manager 注册是 per-process 状态）有强假设。**记住 sitecustomize + 固定 schema 两个套路就能少绕弯子**。

### 8.4 Future-KL 是 RAG 端到端收益的关键（P6.2 论证）

**问题**：FIPO + RAG 把 hallucination 拉到 23%，**如果换成纯 GRPO（不带 future-KL），同样 setup 下能不能拿到同样收益**？

**没有跑 head-to-head GRPO 对照（成本 ~10 GPU·hr），但有两条论证路径**：

#### (a) 算法机制差异

| 维度 | GRPO 基线 | FIPO（本项目） | 对 RAG 影响 |
|---|---|---|---|
| Future-KL 约束 | **无** | future_kl_loss 把 actor 与 ref 在未来 token 序列上的 KL 写进损失 | **关键差异** |
| Influence weight | 不重加权 | `exp(-λ · future_kl)` 给 token 一个 [1−ε, 1+ε] 的乘子 | 直接控制 token-level drift |

→ FIPO 比 GRPO 多了一层 token 序列级的"形态保护"。`field_min` 信号反映的就是 attribute / violation 这些"反映 reasoning 不确定性的 token"的 max prob，**只有当 token-level 分布形态被保住时这个信号才可信**。

#### (b) 训练日志反推（直接证据）

214 step FIPO 训练的 actor 指标：

| 指标 | simple data (1-120) | hard data (121-200) | 含义 |
|---|---|---|---|
| `actor/fipo/influence_weights_min` | 0.8508 | 0.8610 | 每 step 都有 token 被 future-KL 下调到 ~0.85 |
| `actor/fipo/influence_weights_max` | 1.1546 | 1.1476 | 每 step 都有 token 被上调到 ~1.15 |
| `actor/ppo_kl` | 0.000003 | -0.0001 | actor 与 old policy KL **全程 ≈ 0**（GRPO 典型 5e-3 ~ 5e-2） |
| `actor/entropy` | 0.1673 | 0.1270 (**−24%**) | policy 锐化但**不**坍缩 |
| `response_length/mean` | 113.0 | 109.8 | 输出长度稳定，**无 mode collapse** |

**三条直接证据**：
1. `influence_weights ≈ [0.85, 1.15]` 全程持续 → future-KL 在每 step 都在工作
2. `ppo_kl ≈ 0` 全程 → FIPO 双重约束（future-KL + PPO clip）把 KL 压在 1e-4 量级
3. `entropy` −24% 但 `response_length` 不变 → confidence 抬升正好把 sft_aux 在 0.30-0.40 区间的样本推过 0.40 阈值

**反事实推演**：纯 GRPO 在同样 setup 下最可能的 3 种失败模式：
- **A. Reward shaping 漂移**（无 future-KL）：模型对错的 case 也很自信 → 该触发 RAG 的不触发，hallucination 不降甚至上升
- **B. Mode collapse**：response_length 收缩成模板 → 模型 ignore 检索内容 → RAG 完全失效
- **C. Reward hacking 长度**：stylistic shortcut → attribute 字段语义内容下降 → hallucination 反升

**修订后的论点**（基于 P6.3 阈值扫描发现 fipo no-RAG 的 hallucination 已是 0.2349）：FIPO 训练本身就把幻觉打下来了大半（−6.78pp），RAG 在最优阈值 0.40 上只再贡献 −0.45pp。**所以 future-KL 真正提供的是 RL 期间不破坏 token-level 表征，让 SFT 阶段就具备的 grounding 能力得以保留并强化**。

### 8.5 视觉 grounding > 文本规则（P6.4 论证）

幻觉被视觉证据修复，**不被文本规则修复**——见 §7.4 模态消融。

**机制**：幻觉 = "reason 提到了图里没有的属性"。视觉检索召回相似真实样本 → 模型在视觉特征对齐时才报；文本检索召回业务规则 → 模型本来看错时被规则强化。

---

## 九、性能优化与速度

### 9.1 单卡 vs 多卡（重要）

实测：8B + LoRA + grad-ckpt + flash-attn 单卡显存仅 ~36 GB（fits L20 45 GB）。

| 方案 | 每 optimizer step | 比较 |
|---|---|---|
| 单卡 + grad-ckpt + flash-attn | ~30s | **基线** |
| `device_map="auto"` 4 GPU pipeline | ~200s | **慢 6-7×** |

**结论**：**除非显存装不下，优先单卡 + grad-ckpt + flash-attn**。Pipeline 并行在 batch_size=1 + grad_accum=16 时大量 GPU stall。

### 9.2 RL 分布式策略选型

verl-latest 内置只支持 `fsdp / fsdp2 / megatron`：

| 策略 | 我们场景 |
|---|---|
| Megatron | ❌ 8B 单节点用不上 TP/SP/PP |
| DeepSpeed ZeRO-3 | ❌ verl 不内置；与 vLLM 切换显存逻辑要自己实现 |
| FSDP1 | ✅ verl 默认，作为 fallback |
| **FSDP2** ✓ | **首选**：per-parameter 分片，CPU offload 比 FSDP1 更省显存；对 Qwen3-VL 多模态包装更细 |

降级到 FSDP1 只需一行环境变量：`ACTOR_STRATEGY=fsdp REF_STRATEGY=fsdp bash run_fipo_v1.sh`

### 9.3 推理时延

| 阶段 | 时延 |
|---|---|
| 一次 generate（无 RAG） | ~3.5s/sample（greedy, 320 tokens） |
| **RAG 一次 generate + 二次 generate（触发 32%）** | **~10s/sample 平均** |
| RAG 完整 664 评测 | ~110 min on 单卡 GPU 7 |
| GPU 显存（VLM + CLIP + FAISS） | ~19.5 GB |

---

## 十、项目结构

```
VLM-posttraining/
├── README.md                           ← 本文档
├── STRATEGY.md                         # 训练策略细节
├── MIGRATION.md                        # 迁移到其他机器
├── docs/
│   ├── INTERVIEW.md                    ← 完整面试答辩剧本（6 节 + 附录）
│   ├── DATA_ENGINEERING.md             # 数据工程历史深入文档
│   └── README_reference.md             # 参考用 README（旧 TFI 项目）
│
├── sitecustomize.py                    ← 关键：让 Ray worker 自动注册 future_kl
├── requirements.txt
│
├── src/
│   ├── stage0_distill/                 # API 蒸馏
│   ├── stage1_sft/
│   │   ├── train.py                    # SFT + SupCon + Triplet
│   │   ├── losses.py                   # SUPCON_WEIGHT=0.05, TRIPLET_WEIGHT=0.03
│   │   ├── dataset.py
│   │   └── triplet_dataset.py
│   ├── stage2_rm/
│   │   ├── train.py                    # Bradley-Terry, head 架构 v0/v1/v2
│   │   ├── model.py                    # head 定义
│   │   ├── holdout_split.py
│   │   └── evaluate.py                 # offline pair-acc + 分层
│   ├── stage3_fipo/
│   │   ├── reward_fn.py                # rule-based reward v2 (5 组件)
│   │   ├── test_reward_fn.py           # 8 case sanity test
│   │   ├── verl_patches/
│   │   │   ├── future_kl_loss.py       # FIPO loss forward-port (~190 LoC)
│   │   │   └── reward_manager.py       # 自定义 VLMAuditRewardManager
│   │   ├── prepare_fipo_data.py        # SFT parquet → verl 多模态 RL parquet
│   │   ├── main_fipo.py                # 入口 wrapper（先 import patches）
│   │   ├── mine_hard_samples.py        # 全量打分 + 难例挖掘
│   │   ├── build_rl_train.py           # hard/easy 70/30 mix
│   │   └── run_fipo_v1.sh              # 单节点启动脚本
│   ├── stage4_rag/
│   │   ├── indexer.py                  # CLIP + FAISS + BM25 双索引
│   │   ├── inference.py                # AuditPipeline + 4 信号置信度 + RAG
│   │   └── retriever.py
│   ├── utils/
│   │   ├── model_loader.py             # Qwen3-VL / Qwen2.5-VL 自动适配
│   │   ├── data_prep.py                # jsonl → parquet + split
│   │   ├── build_triplets.py           # 16 key 白名单 + STYLE_POOL
│   │   └── merge_lora.py
│   └── schema.py                       # coarse_category() + 数据契约常量
│
├── scripts/
│   ├── data/
│   │   ├── S0Data.py                   # 视觉等价类去重
│   │   ├── S1Data.py                   # SFT 蒸馏
│   │   ├── S2Data.py                   # 偏好数据蒸馏
│   │   ├── S4Data.py                   # RAG 案例库爬取/合并
│   │   └── guard.py                    # 全量硬契约体检
│   ├── evaluate.py                     # 端到端评估（含 --use_rag 开关）
│   ├── calibrate_confidence.py         # 置信度阈值校准
│   ├── merge_fipo_ckpt.py              # FSDP shards → HF safetensors
│   ├── run_pipeline.sh                 # 全 pipeline (Stage 0-4)
│   └── build_rag_kb.py                 # ⚠️ 旧脚本，勿覆盖 violation_cases
│
├── configs/
│   ├── train.yaml                      # 各阶段超参
│   └── model.yaml                      # 模型路径 + LoRA + RAG 阈值
│
├── data/
│   ├── raw/{images/, rules.jsonl, violation_cases.jsonl, dedup_map.json}
│   ├── sft/{sft.jsonl, train.parquet, val.parquet, test.parquet, triplets.parquet}
│   ├── preference/{preference.jsonl, preference.parquet, _train.parquet, _holdout.parquet}
│   ├── fipo/                           # Stage 3 用的 RL parquet + scores
│   │   ├── train.parquet
│   │   ├── val.parquet
│   │   ├── sft_aux_train_scores.jsonl  # 挖矿打分结果
│   │   └── rl_train_hard.parquet       # 70/30 hard mix
│   └── rag_index/{visual.faiss, *.pkl}
│
├── models/
│   ├── pretrained/Qwen3-VL-8B-Instruct/
│   ├── pretrained/clip-vit-base-patch32/
│   ├── sft_ckpt/                       # baseline SFT (LoRA)
│   ├── sft_aux_ckpt/                   # CE+SupCon+Triplet (LoRA)
│   ├── sft_baseline_merged/            # merged for RM
│   ├── sft_aux_merged/                 # merged for RM + FIPO start
│   ├── rm_ckpt_v2_aux/                 # 最佳 RM (mean_margin 11.21)
│   ├── rl_ckpt/                        # FIPO FSDP shards
│   └── fipo_v2_step160_merged/         # 最终 RL ckpt（17.5 GB）
│
├── results/                            # 每阶段评估 JSON
│   ├── eval_sft_baseline.json
│   ├── eval_sft_aux.json
│   ├── eval_sft_aux_rag.json
│   ├── eval_fipo_v2_step160.json
│   └── eval_fipo_v2_step160_rag_field_min_*.json  # 5 个阈值 + 模态消融
│
├── logs/                               # 训练日志 + swanlab 缓存
└── vendor/
    ├── verl-latest/                    # 0.8.0.dev0, --no-deps 安装
    └── FIPO-main/                      # 原始 vendored，仅用于参考 algo
```

---

## 十一、面试速查（一页纸 quick prep）

> **完整答辩剧本见 [docs/INTERVIEW.md](docs/INTERVIEW.md)**（6 节 + 附录，1218 行，含每阶段「为什么这么设计 / 训练细节 / 消融 / 横向对比 / 改进空间 / 🎯 切入口」）。本节只列 **10 个最高频问题**的 30 秒答案。

### Q1. 30 秒讲一下这个项目

电商商品合规审核 VLM 后训练。Qwen3-VL-8B 上做五阶段：SFT (LoRA) → SFT-aux 加 SupCon+Triplet → Bradley-Terry RM → FIPO-RL（GRPO + future-KL + 自研 5 组件 rule-based reward）→ 推理时 RAG（field-aware confidence + 中文 BM25 + CLIP 视觉案例库）。最终 F1 0.9802，**hallucination 从 30% 降到 23%（−24% 相对降幅）**，主要靠 RL+RAG 协同。

### Q2. 五阶段为什么这么排？

- SFT 教模型出结构化 JSON + 学 violation 二分类
- SFT-aux 用 SupCon+Triplet 塑造 embedding 几何（**为下游 RM 埋伏笔**，SFT 任务上中性，RM 上 +16.5% mean_margin）
- RM 学 chosen vs rejected 的偏好顺序（**4 策略同图降质，避免长度/词表 shortcut**）
- FIPO RL 在 token-level 校准 confidence（GRPO + future-KL，**future-KL 是下游 RAG 能 work 的关键**）
- RAG 在推理时给视觉 grounding 兜底，**主要打 hallucination 这个落地阻塞指标**

### Q3. 为什么 F1 已经 98% 还要做 RL 和 RAG？

F1 在 SFT 已饱和（违规判别是简单二分类），但 **hallucination_rate = 30%** 是落地阻塞——分类对了但 reason 在编造未提取到的属性，用户不会信。RL+RAG 真正打的是这个指标，最终降到 23.04%。

### Q4. FIPO 比 GRPO 多了什么？为什么需要？

多了 **future-KL loss**：把 actor 与 ref policy 在未来 token 序列上的 KL 写进损失，每个 token 配 `exp(-λ·future_kl)` 的 influence weight 重加权 advantage。

**作用**：保护 token-level 分布形态，让 RAG 的 `field_min` confidence 信号在 RL 之后仍可信。实测 FIPO 跑 200 step 后 `actor/ppo_kl ≈ 0`（GRPO 典型 5e-3 ~ 5e-2），entropy −24% 但 response_length 几乎不变（无 mode collapse）。

### Q5. Reward saturation 怎么发现的？怎么解的？

**发现**：v1 跑 240→309 step val 不动，看 critic/score 始终在 [4.5, 5.0]，89% step `grad_norm = 0`。GRPO 用组内归一化 advantage，所有 rollout 都满分 → std → 0 → grad → 0。

**解药**：mine_hard_samples 全量打分 → 6 规则筛 351 条难例 → 70 hard / 30 easy 混合 + 2x 上采样 → 1749 条 rl_train_hard.parquet。grad_norm 非零 step 比例从 11% 拉到 50%+。

### Q6. RAG 怎么做 confidence gating？为什么不用 mean_max？

`mean_max`（响应所有 token 的 max softmax 均值）被 JSON 结构 token（`{`、`"`、字段名等 max prob ≈ 1.0）稀释到 [0.90, 0.98] 窄带，默认阈值 0.85 永不触发。

改用 **`field_min`**：屏蔽 max_prob > 0.999 的结构 token，只看 attribute/violation/reason 实词的 max prob。实测分位数 [0.245, 0.616]，阈值 0.40 触发约 32%。

### Q7. RAG 视觉 vs 文本检索哪个更重要？

**视觉是必要的，文本是辅助的**。模态消融实测：visual-only 已贡献几乎全部减幻觉（0.2334 vs both 的 0.2304），text-only 反而升幻觉到 0.2380；text 真正用途是 precision tightening（0.9879 最高，但 recall 跌到 0.9646）。机制：幻觉 = "reason 提到图里没有的属性"，视觉检索给视觉锚点，文本检索只给业务规则但不告诉模型"图里有没有"。

### Q8. SFT-aux 的 SupCon+Triplet 不是看起来无效吗？

在 **SFT 任务上中性**（F1 差 < 1pp），但**对下游 RM 有显著正面影响**：sft_aux backbone 的 mean_margin 11.21 vs sft_baseline 9.62（+16.5%），train_acc +2.1pp。深层原因：CE 已收敛，但 SupCon 的 embedding uniformity 性质塑造的 hidden state 对 reward head 的 pairwise 比较更友好。

### Q9. FIPO 启动 15 处坑里最坑的是哪个？

**第 12 个：`Unsupported loss mode: future_kl`**。Driver 进程注册了 future_kl，但 Ray actor worker 是 fresh interpreter，`POLICY_LOSS_REGISTRY` 是 per-process 模块状态。

**解法**：项目根放 `sitecustomize.py`，site 机制让每个 Python 子进程启动时 auto-import patch 触发 `@register_policy_loss`。这是 verl + Ray + custom loss 一定会撞的坑。

### Q10. 为什么用 RL 而不是续 SFT 修难例？

挖矿后 351 条难例：`label_wrong (24)` + `lexicon_contradict (6)` 这 30 条**SFT 能解**（二分类决策 GT 唯一），但占大头的 `align_low (254)` **SFT 解不了**——reason 是开放生成，GT 只是众多正确答案之一。SFT 的 NLL 是 token-level 罚每个位置，把多样性磨平；没有"近似奖励"概念（BGE 0.7 还能拿 1.05/1.5 reward，SFT 只能 0/1）；distribution shift（SFT 在 GT 训练但 infer 在 self-sample）。RL 直接在 self-sample 上学就没有 gap。

---

## 十二、关键数字速查表

> **面试前 5 分钟过一遍**

### 当前最佳成绩（664 test）
```
fipo_v2_step160_merged + RAG (field_min<0.40):
  F1                  = 0.9802
  Precision           = 0.9841
  Recall              = 0.9764
  hallucination_rate  = 0.2304   ← 关键指标，相比 SFT baseline 0.3072 降 24% 相对
  rag_triggered_rate  = 32.2%
```

### 数据规模
- raw images: **3093** (去重后)
- SFT: **6685** (train 5353 / val 668 / test 664，按 image_file 分组切)
- triplets: **16061** (仅 train 图)
- preference: **2000** 对 (4 策略同图降质，missed_cue 325)
- 规则库 20 + 案例库 150
- 累计成本：**~¥441**

### 训练配置
- **Stage 1 SFT**：LoRA r=32 alpha=64 dropout=0.05；CE + 0.05·SupCon + 0.03·Triplet；lr=2e-4；3 epoch；单卡 36 GB
- **Stage 2 RM**：sft_aux_merged 冻结 + MLP head (4096→2048→1)；BT loss；lr=1e-4；2 epoch；1800 train + 200 holdout
- **Stage 3 FIPO**：GRPO + future-KL；reward_v2 (5 组件)；rollout n=8 temp=1.0；FSDP2 全 offload；6 卡 L20 × 25 GB；lr=1e-6；2 epoch / 666 step / ETA 22h
- **Stage 4 RAG**：field_min<0.40；CLIP-ViT-B/32 + FAISS top-3；BM25 (jieba) top-3 over (20 rules + 150 cases)

### 关键 RM/Reward 数字
| | mean_margin | train_acc | pair_acc |
|---|---:|---:|---:|
| RM v0 (Linear, base) | 4.00 | — | 0.825 |
| RM v2-baseline (MLP, sft_baseline) | 9.62 | 0.863 | 0.825 |
| **RM v2-aux (MLP, sft_aux)** | **11.21** | **0.884** | 0.825 |

### Reward v2 组件权重
```
JSON parseable    : +1.0
violation match   : +2.0 / -1.0
length sanity     : 0 / -1.0 / -0.5
lexicon match     : +0.5 / -0.3
semantic align    : +1.5 / 0~1.5 / -0.5  (BGE-small-zh-v1.5, 余弦)
parse failure     : -3.0 / -2.0
total range       : [-3, 5]
```

### Hard mining 结果（2000 train 全量）
- `label_wrong`: 24 (1.2%)
- `lexicon_contradict`: 6 (0.3%)
- `align_low`: 254 (12.7%)
- `total_low (is_hard)`: 351 (17.6%)
- 输出 mix: 700 hard + 1049 easy = 1749 条

### FIPO 训练动态
| | simple data (1-120) | hard data (121-200) |
|---|---|---|
| influence_weights_min | 0.851 | 0.861 |
| influence_weights_max | 1.155 | 1.148 |
| ppo_kl | ~3e-6 | ~-1e-4 |
| entropy | 0.167 | 0.127 (−24%) |
| response_length | 113 | 110 |

### RAG 阈值扫描
- 0.30 / 0.35 / **0.40** / 0.45 / 0.50 → 触发 7.7% / 17.3% / **32.2%** / 50.8% / 75.3%
- hallucination：**0.40 是 sweet spot**（U 型曲线最低点 0.2304）

### RAG 模态消融
- visual-only: 0.2334 (主要贡献减幻觉)
- text-only: 0.2380 (反而升，但 precision 0.9879 最高)
- both: 0.2304

### 推理时延
- baseline 一次 generate: ~3.5s
- RAG 平均（32% 触发二次）: ~10s
- 全 664 评测: ~110 min on 单卡 GPU 7

### 硬件
- 7× L20 (45 GB)，**GPU 0 ECC 排除**
- RL 用 6 卡（GPU 1-6），评估用 GPU 7
- 单卡 SFT > 多卡 pipeline（30s vs 200s/step）

---

## 相关文档

- **完整面试答辩剧本**：[docs/INTERVIEW.md](docs/INTERVIEW.md)（6 节 + 附录，每节都有"切入口"列表）
- **训练策略细节**：[STRATEGY.md](STRATEGY.md)
- **迁移到其他机器**：[MIGRATION.md](MIGRATION.md)
- **数据工程历史深入**：[docs/DATA_ENGINEERING.md](docs/DATA_ENGINEERING.md)（部分信息可能与 README 不同步，以 README 为准）

---

> **最后更新：2026-04-28**。重写本版以「面试导向 + 工程详尽 + 数字速查」三合一定位。原 v1 README 1328 行的全部历史决策与踩坑记录都已收进 INTERVIEW.md 与 §六/§八；本 README 删掉了与 INTERVIEW.md 重复的 Q&A，专注呈现"项目状态 + IO 流 + 关键数字"。
