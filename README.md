# VLM Post-Training for E-commerce Audit

电商商品合规审核 VLM 后训练项目：在 **Qwen3-VL-8B-Instruct** 基座上做 **SFT → 辅助对比 → RM → FIPO-RL → RAG** 五阶段
后训练，得到一个能看图 + 看商品描述、输出结构化审核 JSON 的合规审核模型。

本 README 按以下顺序展开：

- **Part 1 — 数据工程**：原料采集、蒸馏、质量诊断、修复、最终数据现状（重点）
- **Part 2 — 训练阶段总览**：每个阶段的输入/损失/输出
- **Part 3 — 评估与指标**
- **Part 4 — 项目结构与配置**
- **Part 5 — 快速开始**
- **Part 6 — 已知限制与数据成本**


NOTE：本项目模型使用Qwen3-VL-8B，用于Agentic RAG的模型为bge-m3和clip-vit-base-patch32，请自行下载；数据集已经放在清华大学云盘https://cloud.tsinghua.edu.cn/library/77bf2259-4f1b-4ac9-8a3c-19ef63f4d30f/VLM-posttraining/
---

## Part 1 · 数据工程

数据质量是本项目的关键限制面。下面先给最终状态，再讲怎么走到这里。

### 1.1 最终数据现状

| 数据集 | 产物 | 行数 | 用途 | 生成方式 |
|---|---|---|---|---|
| 原始图片池 | [data/raw/images/](data/raw/images/) | **3093** | 所有阶段视觉源（已去重） | DeepFashion-MultiModal 衍生下载 → 视觉等价类去重；Pexels 补食品/化妆品/电子各 200 张 |
| 规则库 | [data/raw/rules.jsonl](data/raw/rules.jsonl) | 20 | Stage 4 RAG / reward 规则 | 人工编写 |
| 违规案例库 | [data/raw/violation_cases.jsonl](data/raw/violation_cases.jsonl) | **150** | Stage 4 RAG 语料 | 18 模板（含服装/鞋补录） + 132 广东市监局爬取 |
| SFT 注解 | [data/sft/sft.jsonl](data/sft/sft.jsonl) | **6685** | Stage 1 SFT 主注解 | qwen-vl-max 四批次 API 蒸馏（基础 4889 + Pexels 补充 1796） |
| SFT 切分 | [data/sft/{train,val,test}.parquet](data/sft/) | **5353 / 668 / 664** | Stage 1 训练/评估 | 按 `image_file` 分组 80/10/10 切 |
| 幻觉三元组 | [data/sft/triplets.parquet](data/sft/triplets.parquet) | **16061** | Stage 1 对比损失辅助（仅 train 图） | 属性白名单扰动 |
| 偏好数据 | [data/preference/preference.{jsonl,parquet}](data/preference/) | **2000** | Stage 2 RM 训练 | qwen-vl-max 四策略同图降质蒸馏 |
| 粗粒度品类映射 | [src/schema.py](src/schema.py) (`coarse_category`) | 10 桶 | Stage 2/3/4 同品类契约 + 分层评估 | 自由文本 `category` → 食品/化妆品/电子产品/医药/鞋/手表/包/服装/配饰/其他 |

**核心硬契约**（guard 通过，见 [scripts/data/guard.py](scripts/data/guard.py)）：

- 图片视觉等价类去重：hash 碰撞 0，train ∩ val、train ∩ test、val ∩ test、preference ∩ val、preference ∩ test 均为 0
- `preference.image_file ⊂ SFT_train.image_file`：100%（2000/2000）
- chosen/rejected JSON 解析：100% / 100%
- **同粗粒度品类契约**：100%（基于 [src/schema.py](src/schema.py) `coarse_category()` 而非原始 425 种自由文本）；violation-flip 契约（各策略）：100%
- triplets 仅取自 SFT train 图：0 eval 泄漏

**Stage 1 品类分布**（SFT 6685 行，粗粒度桶，见 §6.1 的现状与限制）：

| 桶 | 行数 | 占比 |
|---|---|---|
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

### 1.2 五阶段数据流

```
┌──────────────────────────────────────────────────────────────────────┐
│  Phase A · 原料采集（一次性）                                         │
│  download_dataset + scripts/data/S4Data.py {gd,samr,merge}           │
│  → data/raw/{images/, rules.jsonl, violation_cases.jsonl}            │
└──────────────────────────────────────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Phase B · 视觉去重（必须先于蒸馏）                                   │
│  scripts/data/S0Data.py                                              │
│    ├─ JPEG 再编码 (quality=90) + MD5                                 │
│    ├─ 同 hash 为一个视觉等价类                                        │
│    └─ 每类保留 product_XXXXX id 最小者，删除其余                      │
│  → 2550 → 2493 张，dedup_map.json 记录映射                           │
└──────────────────────────────────────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Phase C · SFT 数据蒸馏                                              │
│  scripts/data/S1Data.py (v3 版合规 prompt)                           │
│  → data/sft/sft.jsonl  ≈¥245, 90 min                                 │
└──────────────────────────────────────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Phase D · SFT 切分 + 三元组                                         │
│  ├─ src/utils/data_prep.py --mode sft --split                        │
│  │   按 image_file 分组 80/10/10，seed=42                            │
│  └─ src/utils/build_triplets.py  仅用 train split                    │
│  → train/val/test.parquet, triplets.parquet                          │
└──────────────────────────────────────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Phase E · 偏好数据蒸馏                                              │
│  scripts/data/S2Data.py                                              │
│    ├─ 输入池: SFT train 图片（严格非 val/test）                       │
│    ├─ 四策略: weak_evidence / wrong_attribute / over_strict /        │
│    │         missed_cue                                              │
│    ├─ temperature=1.0, banned-phrase filter                          │
│    └─ resume-safe 基于 image_file                                    │
│  → preference_distilled.jsonl  ≈¥60, 15 min                          │
└──────────────────────────────────────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Phase F · 统一 parquet + 全量体检                                    │
│  ├─ data_prep.py --mode preference 重建 preference.parquet           │
│  └─ scripts/data/guard.py  运行全部硬契约检查                        │
│  guard 全绿 → 可进入训练                                              │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.3 Stage 2 偏好数据：四策略同图降质

偏好对的 chosen 直接用 SFT gold；rejected 由 qwen-vl-max 基于**同一张图、同一品类**生成一条推理质量更差的审核：

| 策略 | 行数 | violation 契约 | 降质手段 |
|---|---|---|---|
| `api_weaker_weak_evidence` | 620 | 保持 chosen label | reason 保留结论但变敷衍，不引用任何具体属性 |
| `api_weaker_wrong_attribute` | 535 | 保持 chosen label | 把 1–2 个属性改为视觉错值，reason 引用错误属性作证据 |
| `api_weaker_over_strict` | 520 | chosen=False → rejected=True | 牵强认定违规（把普通描述当极限词） |
| `api_weaker_missed_cue` | 325 | chosen=True → rejected=False | 漏判违规，对违规证据给出开脱理由 |

`missed_cue` 占比曾是天然约束——只有 chosen=True 的样本才能翻成 False。当前通过 `--only_mode missed_cue` 二次蒸馏（见 [scripts/data/S2Data.py](scripts/data/S2Data.py)）从 103 提升到 325，覆盖了所有 violation=True 候选图的 ~50%，已足以做分层 RM 评估。

### 1.4 数据质量问题发现与修复时间线

项目在数据工程阶段暴露了 6 个典型问题，全部已修复。可作为其他 VLM 项目的 lessons learned。

**问题 1：SFT v2 合规样本 reason 完全相同**
- 现象：balanced_v2 的 2946 条合规样本 reason 去重率仅 25%
- 根因：`COMPLIANT_SYSTEM` prompt 硬编码了"审核理由"示例，模型全部照抄
- 修复：重写 prompt 为"生成 30-80 字引用本条商品具体属性的合规说明"，示例标注"仅示范风格"
- 结果：重跑 balanced_v3 后 reason 去重率 **99.6%**，¥97

**问题 2：SFT parquet 按行随机切导致图片泄漏**
- 现象：train ∩ val 共享 352 张图
- 根因：`data_prep.py` 按 df 行随机切 80/10/10
- 修复：改为按 `image_file` 分组切，同一张图的所有注解绑定在同一 split
- 结果：val/test 完全独立（文件级）

**问题 3：Triplets 长尾 attr_key + 跨属性串扰**
- 现象：171 种 attr_key，"款式: 藏蓝"（颜色值套到款式键）
- 根因：替换池对任意 key 都 fallback 到颜色/材质
- 修复：16 key 白名单 + 独立 `STYLE_POOL`（[src/utils/build_triplets.py:29-76](src/utils/build_triplets.py#L29-L76)）
- 结果：13 种 key，0 跨属性串扰

**问题 4：旧 preference 数据的双重 shortcut**
- 现象：旧 `preference.jsonl` 4001 对中 60% rejected 是**其它图的 SFT gold**（图文匹配 shortcut）+ 20% 的 rejected reason 用固定 3 条模板（文本模板 shortcut）
- 根因：早期"跨图注入"策略把图文匹配误当作 hard negative
- 修复：API 蒸馏方案——同图、同品类、4 类降质（见 §1.3）
- 结果：文本模板聚集消失，reason 唯一度 86%

**问题 5：API 蒸馏首轮 weak_evidence 模板聚集**
- 现象：518 条 weak_evidence rejected 中 318 条照抄 prompt 示例原句
- 根因：示例"仅示范风格"提示不够强
- 修复：prompt 加 banned-phrase 列表 + temperature 1.0 + 输出侧 banned-phrase filter 作安全网（见 [scripts/data/S2Data.py](scripts/data/S2Data.py)）
- 结果：唯一度 19.1% → **85.9%**

**问题 6：视觉等价类重复导致事实泄漏**
- 现象：2550 张图中有 57 对视觉完全一样（JPEG 再编码后 hash 相同），file 级不重复但 pixel 级重复。SFT eval 与 train 共享 14 个视觉等价类，preference 与 val/test 共享 10 个
- 根因：原始采集没做 pixel-level dedup；file 名不同让 file-based 的 split 感知不到
- 修复：[scripts/data/S0Data.py](scripts/data/S0Data.py)：JPEG(q=90) + MD5 → 同 hash 视为一类，每类保留最小 id，其余删除；后续 sft/preference jsonl 按 dedup_map 剔除引用
- 结果：2550→2493 图片，0 视觉等价类跨 split 泄漏，preference 相应 1255→1014 再补蒸馏至 1500

### 1.5 数据 guard 契约

[scripts/data/guard.py](scripts/data/guard.py) 的全部硬契约（任何一条失败即 fail）：

| 契约 | 适用 | 检测方式 |
|---|---|---|
| JSON 可解析 | SFT / Pref 所有行 | `json.loads(strip_markdown_fences(x))` |
| 必需字段齐备 | SFT / Pref | `{category, attributes, violation, reason}` ⊆ keys |
| 图片存在 | 所有 | `Path(image_dir / image_file).exists()` |
| 视觉等价类不跨 split | train/val/test/pref | JPEG(q=90)+MD5 → 集合交为空 |
| jsonl ↔ parquet 行数一致 | SFT / Pref | parquet rows == jsonl lines |
| split 可复现 | SFT | 按 `image_file` 分组 + seed=42 + 80/10/10 重放后与 parquet 一致 |
| 同品类 | Pref | `chosen.category == rejected.category` |
| violation 契约 | Pref | 各策略 must-keep / must-flip 遵守率 100% |
| triplets 不含 eval | Stage 1 辅助 | 三元组图片 ⊂ train 集合 |

---

## Part 2 · 训练阶段总览

| 阶段 | 入口 | 输入 parquet 列 | 损失 / 奖励 | 产物 |
|---|---|---|---|---|
| 0 蒸馏 | [src/stage0_distill/distill.py](src/stage0_distill/distill.py) | — | — | `data/sft/*.jsonl`, `data/preference/*.jsonl` |
| 1 SFT | [src/stage1_sft/train.py](src/stage1_sft/train.py) | `image, prompt, response, violation` | CE + 0.05·SupCon + 0.03·Triplet | `models/sft_ckpt/epoch-k/` + `best/` |
| 2 RM | [src/stage2_rm/train.py](src/stage2_rm/train.py) | `chosen_*, rejected_*` (input_ids / pixel_values / grid) | Bradley-Terry pairwise | `models/rm_ckpt/reward_head_best.pt`（仅标量头，backbone 冻结） |
| 3 FIPO-RL | [src/stage3_fipo/run_fipo.sh](src/stage3_fipo/run_fipo.sh) | train.parquet（prompt 池） | **reward v2**：JSON +1 / label ±2 / 长度卫生 -1\|-0.5 / 词表一致性 ±0.5/-0.3 / **reason↔attr 语义对齐 ±1.5/-0.5** / RM ×1（详见 §P4） | `models/rl_ckpt/`（需要 merge） |
| 4 RAG | [src/stage4_rag/indexer.py](src/stage4_rag/indexer.py) | 索引：`raw/images/` + `violation_cases.jsonl` | — | `data/rag_index/{visual.faiss, *.pkl}` |

**Stage 1 细节**
- 主损失：`losses.py` 中 `SUPCON_WEIGHT=0.05, TRIPLET_WEIGHT=0.03`，基于 EOS token embedding 做对比
- LoRA 注入：LM 注意力 (`q/k/v/o_proj`) + LM MLP (`gate/up/down_proj`) + 视觉→文本 merger (`linear_fc1/linear_fc2`，仅命中 `visual.merger.*` 和 `visual.deepstack_merger_list.*`)；详见 [configs/model.yaml](configs/model.yaml)
- 注意力实现优先级：`flash_attention_2 → sdpa → eager`，自动回退（[src/utils/model_loader.py](src/utils/model_loader.py)）
- 视觉编码器默认冻结（merger 不冻结，是 LoRA 注入点之一）
- 多卡训练：`device_map="auto"` 触发 accelerate 的 model parallelism；不支持 DDP/FSDP

**Stage 2 细节**
- Bradley-Terry：对 (chosen, rejected) 仅训练一个 scalar head 输出标量 reward
- Loss: `-log(sigmoid(s_c - s_r))`，见 [src/stage2_rm/model.py:22](src/stage2_rm/model.py#L22)
- 期望指标：held-out pair accuracy > 80%，分层按 pair_strategy 监控（wrong_attribute 最难）

**Stage 3 细节**
- 基于 veRL 的 GRPO + FIPO future-KL 损失
- 每 prompt 采 8 response（`n_resp_per_prompt=8`），组内 normalise advantage
- 奖励组装：**规则奖励 v2（无 API 依赖）+ RM 奖励**
  - 规则奖励 v2 在 [src/stage3_fipo/reward_fn.py](src/stage3_fipo/reward_fn.py)，把 v1 的 substring 幻觉代理换成**句向量语义对齐**（BAAI/bge-small-zh-v1.5），详见 §P4
  - RM 用 §P1.5 选定的 `v2-aux`（sft_aux backbone + MLP head）
- Loss 模式：`LOSS_MODE=future_kl` (FIPO) 或 `vanilla` (标准 GRPO)
- **只在 train 集 rollout**，不触碰 val/test

**Stage 4 细节**
- 推理管线：直出 → confidence 检查 → 低于 0.85 则触发 FAISS (CLIP 视觉) + BM25 (规则文本) 检索 → 注入 system prompt 再次推理
- 类：`AuditPipeline`（[src/stage4_rag/inference.py:35](src/stage4_rag/inference.py#L35)）
- RAG 仅覆盖训练分布外的类目（食品/化妆品/电子产品等），见 §6

---

## Part 3 · 评估与指标

入口：[scripts/evaluate.py](scripts/evaluate.py)，输入 `--test_parquet data/sft/test.parquet`。

**指标集合**：
- **`json_format_accuracy`**：响应 `json.loads` 成功率（必需 ≥95%，低于此合规下游任务无法跑）
- **`violation_f1 / precision / recall`**：对 `violation` 字段的 binary 分类（test GT 来自 parquet `violation` 列）
- **`hallucination_rate`**：reason 文本中未引用任一提取出的 attribute value 的比例（越低越好）

**pipeline 内位置**：[scripts/run_pipeline.sh](scripts/run_pipeline.sh) 在 Stage 1 SFT 后（merge LoRA 后）、Stage 3 RL 后各跑一次评估，结果落 `results/` 目录。

**分层监控（建议）**：
- Stage 2 RM：按 `pair_strategy` 分层报 pair accuracy，比较 weak_evidence vs wrong_attribute 的提升
- Stage 3 RL：对比 SFT baseline 的 violation_f1 和 hallucination_rate，确认 FIPO 带来的推理质量提升
- 长度 shortcut 监控：`mean(len(chosen) - len(rejected))` 不应显著增长

---

## Part 4 · 项目结构与配置

```
VLM-posttraining/
├── src/
│   ├── stage0_distill/      # API 蒸馏生成 SFT/Pref 数据
│   ├── stage1_sft/          # LoRA SFT + 辅助对比损失
│   │   ├── train.py
│   │   ├── losses.py        # SupCon + Triplet
│   │   └── dataset.py
│   ├── stage2_rm/           # Bradley-Terry RM
│   ├── stage3_fipo/         # FIPO RL (veRL/GRPO wrapper)
│   │   ├── run_fipo.sh
│   │   ├── reward_fn.py     # rule-based reward v2 (语义对齐, 无 API)
│   │   └── test_reward_fn.py# 8 case sanity test
│   ├── stage4_rag/          # FAISS 视觉 + BM25 文本 RAG
│   ├── utils/
│   │   ├── model_loader.py  # Qwen3-VL / Qwen2.5-VL 自动适配
│   │   ├── data_prep.py     # jsonl → parquet + split
│   │   ├── build_triplets.py
│   │   └── merge_lora.py
│   └── schema.py            # 数据契约常量
├── scripts/
│   ├── data/                     # 数据工程脚本（Part 1 用到的全部）
│   │   ├── S0Data.py             # Phase B：视觉等价类去重
│   │   ├── S1Data.py             # Phase C：SFT 蒸馏
│   │   ├── S2Data.py             # Phase E：偏好数据蒸馏
│   │   ├── S4Data.py             # Phase A：RAG 案例库爬取/合并 (gd|samr|merge)
│   │   └── guard.py              # Phase F：全量硬契约体检
│   ├── evaluate.py
│   ├── run_pipeline.sh
│   └── build_rag_kb.py
├── configs/
│   ├── train.yaml   # 各阶段超参
│   └── model.yaml   # 模型路径 + LoRA + RAG 阈值
├── data/
│   ├── raw/{images/, rules.jsonl, violation_cases.jsonl, dedup_map.json}
│   ├── sft/{sft.jsonl, train.parquet, val.parquet, test.parquet, triplets.parquet}
│   ├── preference/{preference.jsonl, preference.parquet, preference_distilled.jsonl}
│   └── rag_index/             # Stage 4 构建产物
├── models/                    # checkpoints（pretrained/sft_ckpt/rm_ckpt/rl_ckpt/merged）
├── results/                   # 每阶段评估 JSON
└── docs/DATA_ENGINEERING.md   # 旧版数据工程深入文档（历史参考）
```

**关键超参**（[configs/train.yaml](configs/train.yaml)）：
- SFT: `epochs, batch_size, grad_accum, lr=2e-4, supcon_weight=0.05, triplet_weight=0.03`
- RM: `epochs, batch_size, lr=1e-4`
- FIPO: `loss_mode, decay_rate=12.0, future_kl_*, clip_ratio_*, n_resp_per_prompt=8, epochs=8`

**模型配置**（[configs/model.yaml](configs/model.yaml)）：
- LoRA: `r=32, alpha=64, target_modules=[q_proj,k_proj,v_proj,o_proj,visual_merger.mlp.{0,2}]`
- Generation: `max_new_tokens=512, do_sample=False`（eval），`temperature=1.0`（RL rollout）
- RAG: `confidence_threshold=0.85, top_k_visual=3, top_k_text=3`

---

## Part 5 · 快速开始

### 5.1 环境

```bash
# 用 uv + 阿里云镜像（避免 pytorch 源下载卡顿）
uv pip install --python $(which python) \
    https://mirrors.aliyun.com/pytorch-wheels/cu124/torch-2.6.0%2Bcu124-cp312-cp312-linux_x86_64.whl \
    https://mirrors.aliyun.com/pytorch-wheels/cu124/torchvision-0.21.0%2Bcu124-cp312-cp312-linux_x86_64.whl \
    https://mirrors.aliyun.com/pytorch-wheels/cu124/torchaudio-2.6.0%2Bcu124-cp312-cp312-linux_x86_64.whl
uv pip install -r requirements.txt --index-url https://mirrors.aliyun.com/pypi/simple/
```

### 5.2 重建数据（已有原料时只跑体检）

```bash
# 全量体检（~30s）
python scripts/data/guard.py

# 重新蒸馏偏好（如 RM 收敛差时补充）
export DASHSCOPE_API_KEY=sk-xxx
python scripts/data/S2Data.py \
    --sft_file data/sft/sft.jsonl \
    --pool_file data/preference/preference.jsonl \
    --out_file data/preference/preference_distilled.jsonl \
    --n_pairs 2000 --workers 6

# 从零重建数据（完整流程）
python scripts/data/S0Data.py                                    # 视觉等价类去重
python scripts/data/S1Data.py --image_dir data/raw/images \
    --out_file data/sft/sft.jsonl --target_total 5000            # SFT 蒸馏
python -m src.utils.data_prep --annotation_file data/sft/sft.jsonl \
    --image_dir data/raw/images --out_dir data/sft --mode sft --split
python scripts/data/S2Data.py --sft_file data/sft/sft.jsonl \
    --out_file data/preference/preference.jsonl --n_pairs 1500   # 偏好蒸馏
python scripts/data/S4Data.py merge                              # RAG KB（可选）
python scripts/data/guard.py                                     # 全量硬契约
```

### 5.3 训练

```bash
# 全 pipeline (Stage 0-4)
bash scripts/run_pipeline.sh

# 或分阶段
STAGE_START=1 STAGE_END=1 bash scripts/run_pipeline.sh   # 仅 SFT
STAGE_START=2 STAGE_END=3 bash scripts/run_pipeline.sh   # RM + FIPO
```

### 5.4 单阶段手动调用

```bash
# Stage 1 SFT
python -m src.stage1_sft.train \
    --model_path models/pretrained/Qwen3-VL-8B-Instruct \
    --train_parquet data/sft/train.parquet \
    --triplet_parquet data/sft/triplets.parquet \
    --out_dir models/sft_ckpt \
    --epochs 3 --flash_attn

# Stage 2 RM
python -m src.stage2_rm.train \
    --model_path models/merged/sft \
    --train_parquet data/preference/preference.parquet \
    --out_dir models/rm_ckpt \
    --epochs 2

# Stage 3 FIPO
bash src/stage3_fipo/run_fipo.sh \
    models/merged/sft  models/rm_ckpt  models/rl_ckpt  data/sft/train.parquet

# Stage 4 RAG 索引
python -m src.stage4_rag.indexer \
    --image_dir data/raw/images \
    --rule_file data/raw/rules.jsonl \
    --case_file data/raw/violation_cases.jsonl \
    --out_dir data/rag_index
```

---

## Part 6 · 数据可训练性、剩余限制、累计成本

### 6.1 可训练性判定：当前数据可直接进入 Stage 1 SFT

Guard 全绿 + 下列可量化指标达到训练准入线：

| 指标 | 当前值 | 训练准入线 | 判定 |
|---|---|---|---|
| SFT 总行数 | 6685 | ≥ 4000 | ✓ |
| 合规 : 违规 | 4009 : 2676 (60 : 40) | 50 : 50 ~ 70 : 30 | ✓ |
| JSON 可解析率 | 100% | ≥ 99% | ✓ |
| train / val / test 切分 | 5353 / 668 / 664 | 按 image_file 分组、可复现 | ✓ |
| 视觉等价类跨 split 泄漏 | 0（train/val/test/pref 两两交集） | 0 | ✓ |
| 幻觉三元组 | 16061（仅 train 图） | 0 eval 泄漏 | ✓ |
| 粗粒度品类覆盖 | 10 桶全非空，最小桶 80 行（医药） | 每桶 ≥ 50 行 | ✓ |
| 同粗粒度契约（Pref） | 100% | ≥ 95% | ✓ |
| violation-flip 契约（Pref） | 100%（各策略） | 100% | ✓ |
| Preference 规模 | 2000 对（missed_cue 325） | ≥ 1500、missed_cue ≥ 200 | ✓ |

直接运行 `bash src/stage1_sft/run.sh` 即可进入 SFT 训练；RM 与 FIPO 阶段亦无数据侧阻塞。

### 6.2 已修复的历史限制

| 问题 | 当前状态 |
|---|---|
| 食品/化妆品/电子产品 SFT 样本数为 0 | Pexels 补 600 图后 → 145 / 358 / 95 行 |
| Preference 总量 1500，`missed_cue` 仅 103 | 扩至 2000，`missed_cue` 325（`--only_mode` 二次蒸馏） |
| SFT `category` 425 种自由文本噪声 | [src/schema.py](src/schema.py) `coarse_category()` → 10 桶；guard 与 Stage 2/3/4 均基于粗粒度 |
| Preference 同 `category` 契约噪声 | 改为同粗粒度桶契约，100% 达标 |
| 数据 split 切换后 triplets 残留 eval 图 | `build_triplets.py` 入参固定为 train 子集，0 泄漏 |
| Parquet 无 `image_file` 列导致 guard 读不到文件名 | `data_prep.py` 保留该列（不影响训练侧 `image` bytes 字段） |

### 6.3 剩余限制（训练侧可接受，但评估需注意）

1. **SFT 分布仍倾向服装/鞋**：服装 51.6% + 鞋 17.3% 占近 70%。Stage 1 验证集准确率不能只看 overall，必须按粗粒度桶分层。最小桶（医药 80 行）在 val/test 里可能只有 ~8 条，分层 accuracy 方差较大，用 macro-F1 更稳。
2. **qwen-vl-max 对 Pexels 图的类目误判残余**：盒装食品会被标成"礼盒服装"、护肤套装被标成"化妆品收纳包"。粗粒度桶吸收了大部分，但 `其他` 桶 6.6% 里混有这类边缘样本。不影响 SFT 学"结构化输出 + 违规识别"的主目标。
3. **`build_rag_kb.py` 会覆盖 `violation_cases.jsonl`**：[scripts/build_rag_kb.py](scripts/build_rag_kb.py) 是旧的合成模板生成器。当前 150 条案例（含 132 条真实处罚）来自 [scripts/data/S4Data.py](scripts/data/S4Data.py) 的 samr/gd 爬取 + 模板补录，**勿再跑 `build_rag_kb.py`** 覆盖掉。Stage 4 RAG 索引构建入口仍是 `src/stage4_rag/indexer.py`。
4. **`missed_cue` 覆盖上限**：只能从 violation=True 的样本翻转，SFT 里 2676 条 violation 中在 train split 的 ~2140 条是候选池，已用 325（~15%）。若后续 RM 需要更多，可继续跑到 ~1000，但边际收益递减。

### 6.4 累计数据成本

| 阶段 | ¥ |
|---|---|
| SFT v1 + balanced_v2 | 163 |
| SFT balanced_v3 重跑 | 97 |
| Preference 首轮蒸馏 + weak_evidence 重跑 | 80 |
| 视觉去重后 preference 补蒸馏（486 条） | 20 |
| Pexels 补图 SFT 蒸馏（+1796 行） | 59 |
| Preference 扩展（missed_cue +200 + 总量至 2000） | 22 |
| **合计** | **~¥441** |

---

## Part 7 · 待办（已发现但未修，按优先级）

### P0 · Stage 1 辅助损失（**已修复，2026-04-22**）

历史问题：原 SFT 跑的实际是 CE-only（SupCon 被 `batch_size>1` 守卫挡掉、Triplet 是占位 0）。

**修复**（commit `576a3b8`）：
- **SupCon**：在 [src/stage1_sft/train.py](src/stage1_sft/train.py) 增加 memory bank（默认 64 条），每个 micro-batch 把 detached EOS embedding 入队，凑齐两个类别后才计算对比损失，**无需调大 batch_size**
- **Triplet**：新增 [src/stage1_sft/triplet_dataset.py](src/stage1_sft/triplet_dataset.py) + 在每个 optimizer step 末从 `triplets.parquet` 抽 (image, pos_attr, neg_attr)，用 `F.triplet_margin_with_distance_loss(distance=cosine, margin=0.3)` 算 anchor-pos vs anchor-neg 的边际，乘 0.03 加权

实测（`vlm-posttraining-ecommerce-SFT-aux-resume`，2026-04-22 完成）：训练 2 epoch 收敛于 `avg_total=0.3545 (ce=0.1785, supcon=3.5203, triplet=0.0074)`，`total = ce + 0.05·supcon + 0.03·triplet` 算式吻合 ✓

**下游 eval（664 样本 test set）**：

| 指标 | A: CE only (baseline) | B: CE + SupCon + Triplet | Δ |
|---|---:|---:|---:|
| violation_f1 | **0.9883** | 0.9844 | -0.39 pp |
| precision | 0.9806 | 0.9767 | -0.39 pp |
| recall | 0.9961 | 0.9921 | -0.40 pp |
| hallucination_rate | 0.3072 | **0.3027** | -0.45 pp ✓ |
| json_format_acc | 1.000 | 1.000 | 0 |

**结论**：辅助损失基本 **中性**（差异都在噪声范围内 ~2.6 个样本）。原因：(1) CE 已收敛到 0.18，主任务接近饱和；(2) Triplet loss 早期 collapse 到 0（margin=0.3 太松）；(3) SupCon 用 EOS hidden state 做 anchor，被 JSON 格式收尾 token 主导，语义信号不足。**但 SupCon 对 embedding 几何的塑形对下游 RM 反而有正面作用（见 P1.5 v2 结果）**。

下次实验改进方向：anchor 换成 `"violation"` value token、bank size 降到 16、Triplet margin 提到 0.7+、或加 momentum encoder。

### P0.5 · RM backbone 用 SFT-merged（**已修复，2026-04-23**）

历史问题：之前两个 RM 用的都是裸 base Qwen3-VL，head 表达力上限被 backbone 限制。

**修复**：
- 用 `src/utils/merge_lora.py` 把 LoRA 合并到 base，得到 `models/sft_baseline_merged` 和 `models/sft_aux_merged`（各 17.5GB）
- RM 训练 `--model_path` 指向 merged ckpt
- 同时跑两个 backbone × 同 head 的对照（见 §P1.5）

### P1 · RM held-out 验证集（**已修复，2026-04-22**）

**实现**（commit `576a3b8`）：
- 切分脚本 [src/stage2_rm/holdout_split.py](src/stage2_rm/holdout_split.py)：按 `image_file` 分组（与 SFT split 同逻辑）切 200 对到 `data/preference/preference_holdout.parquet`，1800 对到 `_train.parquet`，从 `preference.jsonl` 恢复 `pair_strategy` 用于分层
- 评估脚本 [src/stage2_rm/evaluate.py](src/stage2_rm/evaluate.py)：offline pair-accuracy + 按 `pair_strategy` 分层指标 + 长度 shortcut（chosen vs rejected token 长度差）
- [src/stage2_rm/train.py](src/stage2_rm/train.py) 增加 `--holdout_parquet`，每 epoch 末跑 holdout 评估并写入 swanlab（`holdout_pair_accuracy/by-strategy`）

**已有 ckpt 在 holdout 上的成绩**（results/）：

| 模型 | head | pair_acc | 备注 |
|---|---|---|---|
| `models/rm_ckpt` (v0) | Linear | 80.5% | base backbone |
| `models/rm_ckpt_headv1` (v1) | LN+bias | 80.5% | base backbone |

**已知偏差**：这两个 ckpt 是在 2000 对全集（含 holdout 200 对）上训出来的，holdout 评估是乐观估计；新一轮（v2）会先训再评，干净对比。

### P1.5 · RM head 架构消融（**v0/v1/v2 已跑**）

**架构选项**（[src/stage2_rm/model.py](src/stage2_rm/model.py)）：

| head | 参数量（额外） | flag |
|---|---|---|
| v0 `Linear(4096,1)` | 0 | （默认）|
| v1 `LN → Linear(4096,1,bias=True)` | ~8K | `--head_layernorm --head_bias` |
| **v2 `LN → Linear(4096,2048) → GELU → Dropout → Linear(2048,1)`** | **~10M** | `--head_mlp [--head_dropout 0.1]` |

v2 增加非线性表达力，同时通过 `--head_dropout` 控制小数据集（1800 对）过拟合风险。

**holdout pair-acc + mean_margin 对比**（200 pair, 最终 epoch 2）：

| RM | head | backbone | train_acc | pair_acc | mean_margin | swanlab |
|---|---|---|---:|---:|---:|---|
| v0 | Linear | base Qwen3-VL | — | 0.825 | 4.00 | — |
| v1 | LN+bias | base Qwen3-VL | — | 0.810 | 3.30 | — |
| **v2-baseline** | MLP+LN+bias+dropout | `sft_baseline_merged` | 0.863 | **0.825** | 9.62 | [`97ynxnsj07bdhqks6j468`](https://swanlab.cn/@killua/vlm-posttraining-ecommerce/runs/97ynxnsj07bdhqks6j468) |
| **v2-aux** | MLP+LN+bias+dropout | **`sft_aux_merged`** | **0.884** | **0.825** | **11.21** ✓ | [`seo506w0x25krfs0luypf`](https://swanlab.cn/@killua/vlm-posttraining-ecommerce/runs/seo506w0x25krfs0luypf) |

> v0/v1 的 pair_acc 包含 holdout 在训练集中的乐观偏差；v2 是干净 train/holdout 切分后的真实指标。

**两层归因**：

| 升级 | mean_margin 变化 | 增量 |
|---|---|---:|
| (a) MLP head + SFT backbone（v0 → v2-baseline） | 4.00 → 9.62 | **+140 %**（主要贡献） |
| (b) SFT-aux 的 SupCon 几何塑形（v2-baseline → v2-aux） | 9.62 → 11.21 | **+16.5 %**（稳定增量） |

**结论**：
- holdout `pair_acc` 三个 v2 都卡在 0.825 → **200 样本 holdout 已到 ceiling**，acc 维度区分不出胜负
- **margin 维度 sft_aux backbone 显著胜出**（+16.5 %），且 train_acc 也更高（0.884 vs 0.863）→ 验证 SupCon 对 embedding 几何的塑形对 RM 的 pairwise 比较有真实正面作用，即使在 SFT 主任务上看起来中性。**RL 阶段建议直接用 v2-aux 这个 RM**

⚠️ `len_shortcut`：chosen 比 rejected 平均长 31 token，需注意 RM 是否依赖长度捷径——可看 `by_strategy.api_weaker_missed_cue` 这种长度接近的子集判断。

### P2 · Stage 1 辅助损失消融实验（**已完成，2026-04-23**）

**目标**：验证 0.05·SupCon + 0.03·Triplet 是否真带来 ≥ 1 pp 的下游收益。

| 实验 | 损失组合 | 实验名 | ckpt | violation_f1 | hallucination_rate |
|---|---|---|---|---:|---:|
| A | CE only | `vlm-posttraining-ecommerce-SFT` | `models/sft_ckpt/epoch-2` | **0.9883** | 0.3072 |
| B | CE + SupCon + Triplet | `vlm-posttraining-ecommerce-SFT-aux-resume` | `models/sft_aux_ckpt/epoch-2` | 0.9844 | **0.3027** |

**结论**：辅助损失对 **下游 SFT 任务** 表现中性（差异 < 1 pp，单 seed 不显著）。但对 **下游 RM 任务** 有正面影响——同 head 同数据下，sft_aux backbone 的 RM mean_margin 11.21 vs sft_baseline 9.62（**+16.5 %**），train_acc 0.884 vs 0.863（见 §P1.5）。

**深层原因**：CE loss 在 SFT 阶段已经收敛（0.18），主任务接近饱和，aux 损失再发力空间有限；但 SupCon 的 embedding-uniformity 性质塑造的 hidden state 对 reward head 的 pairwise 比较反而更友好。

### P3 · 速度优化（已应用）

- **flash-attention 已可用**：`flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`（社区构建），attention 加速但**不是 SFT 主要瓶颈**
- **单卡训练 vs `device_map="auto"` 多卡 pipeline**：8B BF16 + LoRA + grad-ckpt + flash-attn 单卡显存仅需 ~36GB（fits L20 45GB）。单卡每 optimizer step ~30s，多卡 4-GPU pipeline ~200s/step，**单卡快 6-7×**——pipeline 并行在 batch_size=1 + grad_accum=16 时大量 GPU stall
- **结论**：除非显存装不下，**优先单卡 + grad-ckpt + flash-attn**

### P4 · Stage 3 reward 函数 v2 + FIPO 环境准备（**进行中，2026-04-23**）

**动机**：
- 离线 SFT 评估 `violation_f1=0.988` 已接近天花板，但 `hallucination_rate ≈ 0.30`——意味着 ~30% 的样本里 reason 字段在编造未提取到的属性／说不在图里的内容。**这才是落地阻塞**：分类对了但理由是幻觉，用户不会信。
- v1 reward 用 substring 匹配（reason 里有没有出现某个 attribute 子串）来当幻觉代理，对换种说法的合理 reason 误伤大、对编造但用词重合的 reason 漏检也大。
- 用户决定走**方案 B（纯本地规则强化，零 API 成本）**，不用 VLM API 当 critic（Qwen3-VL-8B 本身够强，更常见的失败是"reason 与提取的 attribute 不对齐"而不是"完全凭空"）。

**reward v2 组成**（[src/stage3_fipo/reward_fn.py](src/stage3_fipo/reward_fn.py)）：

| 组件 | 信号 | reward |
|---|---|---:|
| JSON parseable + 必需字段齐全 | 硬格式 | +1.0（基础分） |
| `violation` label 命中 GT | 监督信号 | +2.0 / -1.0（错） |
| reason 长度卫生（`8 ≤ len ≤ 250`） | 防退化 | 0 / -1.0（太短）/ -0.5（太长） |
| reason 词表 ↔ violation label 一致性 | 廉价规则（违规词 vs 合规词集合） | +0.5 / -0.3（矛盾） |
| **reason ↔ attributes 语义对齐**（bge-small-zh-v1.5 余弦） | **核心升级，替代 v1 substring 代理** | sim≥0.6: +1.5；sim≤0.2: -0.5；区间内线性插值 |
| RM 标量分（外部传入，不再耦合 model 对象） | 学到的偏好信号 | ×1.0 |
| 解析失败 / 缺字段 | 早退惩罚 | -3.0 / -2.0 |

总 reward 大致在 `[-3, 5]` 区间。

**为什么句向量对齐能压幻觉**：
- bge-small-zh-v1.5 把 reason 和 `"<attr_key>: <attr_val>; ..."` 拼接的属性串各编成 512-d 向量，余弦相似度反映的是"reason 在讨论的东西是不是模型真的提取到了的属性"
- 实测 8 case sanity test（[src/stage3_fipo/test_reward_fn.py](src/stage3_fipo/test_reward_fn.py)）：完美样本 sim=0.85（reward 5.0）、reason 编造"奥运标志/明星代言"等 attributes 里没有的概念时 sim 掉到 0.37（reward 3.86），区分度足够把"label 对但 reason 编造"的样本排在"label 对且 reason 扎实"之后

**实现要点**：
- encoder 走依赖注入（`compute_reward(..., encoder=...)`），VeRL 自定义 `reward_manager` 中只实例化一次，避免每次 rollout 都重新加载 ~100MB 模型
- 所有权重集中暴露在 `DEFAULT_WEIGHTS` dict，调权时不用动函数体
- 第一次跑要拉模型，必须设 `HF_ENDPOINT=https://hf-mirror.com`（本地国内镜像可访问，HF 直连超时）
- `batch_compute_reward` 提供批量接口，方便 VeRL custom reward manager 调用

**FIPO 环境策略与 forward-port**（**已完成 2026-04-23**）：
- vendored `vendor/FIPO-main/` 的 verl 是 0.5.0.dev，**不支持 Qwen3-VL**
- 拉取 verl-latest（`vendor/verl-latest/`，0.8.0.dev，已 gitignore）：原生支持 Qwen3-VL（`verl/models/transformers/qwen3_vl.py`）+ 已经把 policy_loss 接口统一为 `(loss, metrics_dict)`，意味着 **dp_actor.py / megatron_actor.py 完全不用 patch**——只需移植 `core_algos.py` 里的 `compute_policy_loss_future_kl`
- 实际 forward-port 工作量从原估 ~300 LoC 缩减到 **~190 LoC 单文件**：[src/stage3_fipo/verl_patches/future_kl_loss.py](src/stage3_fipo/verl_patches/future_kl_loss.py)
  - 接口适配：返回值从 29 项 tuple → `(pg_loss, metrics_dict)`，加 `rollout_is_weights` 形参（接受不用），加 `**config.global_batch_info` 给 `agg_loss`
  - 算法逻辑（chunked future-KL 累加 / influence weights / dual-clip / sequence-level invalidation）原样保留
  - **不修改 vendor/ 任何文件**，靠 `import` 时的 `@register_policy_loss("future_kl")` 装饰器副作用注入
- **不开新 conda env**：用户决定 "verl 是版本问题，先看 verl，别动其他包"。最终方案：直接在现有 VLM env 用 `pip install --no-deps verl-latest`，再补 `tensordict / codetiming / torchdata / pybind11 / pylatexenc`，不升级现存 vllm 0.19.1。verl-latest 的 vllm 兼容范围声明 `>=0.8.5,<=0.12.0` 实测在 vllm 0.19.1 上仍可跑通（接口变化全在 LoRA hot-swap 等高阶 feature，v1 不用）

**工具栈版本**（实测可跑，VLM conda env）：

| 组件 | 版本 | 备注 |
|---|---|---|
| Python | 3.12 | conda env VLM |
| CUDA | 12.8 | driver 自带 |
| torch | 2.10.0 | + cuDNN 9.10.2 |
| transformers | 5.5.4 | Qwen3-VL native |
| vllm | 0.19.1 | rollout engine |
| verl | 0.8.0.dev0 | `--no-deps` 安装到 vendor/verl-latest，PYTHONPATH 注入 |
| ray | 2.55.0 | 单节点多 actor 调度 |
| tensordict | 0.12.2 | verl 数据流必需 |
| hydra-core | 1.3.2 | verl 配置入口 |
| sentence-transformers | 5.4.1 | reward_fn v2 的 bge-small-zh-v1.5 编码器 |
| GPU | 6×NVIDIA L20 (45GB) | 多租户共享，本实验占用 6 卡 |

**Stage 3 全套 FIPO 资产**：

| 文件 | 作用 |
|---|---|
| [src/stage3_fipo/reward_fn.py](src/stage3_fipo/reward_fn.py) | rule-based reward v2（七项打分，依赖注入式 encoder） |
| [src/stage3_fipo/test_reward_fn.py](src/stage3_fipo/test_reward_fn.py) | 8 case sanity test |
| [src/stage3_fipo/verl_patches/future_kl_loss.py](src/stage3_fipo/verl_patches/future_kl_loss.py) | 注册 `policy_loss=future_kl` 到 verl 全局 registry |
| [src/stage3_fipo/verl_patches/reward_manager.py](src/stage3_fipo/verl_patches/reward_manager.py) | 注册 `reward_manager=vlm_audit_v2`（rollout 时调 reward_fn v2） |
| [src/stage3_fipo/prepare_fipo_data.py](src/stage3_fipo/prepare_fipo_data.py) | SFT parquet → verl 多模态 RL parquet（prompt 用 `<image>` 占位 + ground_truth dict） |
| [src/stage3_fipo/main_fipo.py](src/stage3_fipo/main_fipo.py) | 入口 wrapper：先 import patches 触发注册，再 hand-off 给 `verl.trainer.main_ppo` |
| [sitecustomize.py](sitecustomize.py) | **关键**：项目根的 site hook，让每个 Ray worker 子进程启动时自动 import `future_kl_loss` 完成注册（policy_loss registry 是 per-process 状态） |
| [src/stage3_fipo/run_fipo_v1.sh](src/stage3_fipo/run_fipo_v1.sh) | 单节点启动脚本（FSDP2 offload + vLLM rollout + GRPO + future_kl） |

**用户操作两步**：

```bash
# 一次性（每次数据更新后重跑）
python -m src.stage3_fipo.prepare_fipo_data \
    --in_train data/sft/train.parquet --in_val data/sft/val.parquet \
    --out_dir data/fipo --max_train 2000 --max_val 200

# 训练（可重复，默认 6 GPU）
N_GPUS=6 BATCH_SIZE=6 MINI_BSZ=3 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
    bash src/stage3_fipo/run_fipo_v1.sh
```

**v1 设计取舍**：
- **不在 reward_manager 里调 RM**：避免 actor + ref + 17GB RM 同时占显存；rule-based reward 已经覆盖五个核心信号（格式 / label / 长度 / 词表 / 语义对齐），先看效果。如果 rule reward 有顶到天花板的迹象再加 RM（reward_manager 已留 `rm_score` 形参接口）
- **`reward_model.enable=False`**：用 reward_manager 而不是 verl 内置的 RM 模型，不加载额外的 reward backbone
- **FSDP2 全 offload + grad-ckpt + n_resp=8**：6 卡 L20 实测每卡 ~25GB 占用，95-100% util
- **`LOSS_MODE=vanilla`** 一行环境变量切回标准 GRPO 用于 ablation
- **FIPO knobs 走环境变量**：verl-latest 的 `PolicyLossConfig` 是严格 dataclass、拒绝未声明字段，hydra `+actor.policy_loss.decay_rate=...` 会被 reject。改成在 `run_fipo_v1.sh` 里 `export FIPO_DECAY_RATE / FIPO_CHUNK_SIZE / FIPO_FKL_CLIP_RATIO / FIPO_FKL_CLIP_HIGH_ONLY / FIPO_SAFETY_THRESH`，由 `future_kl_loss.py` 内部 `os.environ.get()` 读取——零侵入 vendor

**verl 关键 hydra override 详解**（[run_fipo_v1.sh](src/stage3_fipo/run_fipo_v1.sh) 默认值）：

| Hydra path | 默认值 | 作用 |
|---|---|---|
| `actor_rollout_ref.actor.strategy` | `fsdp2` | actor 训练用 FSDP2 全分片 |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | `True` | 参数 offload 到 CPU，腾 GPU 给 vLLM rollout |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | `True` | optimizer state 全部到 CPU（最省显存） |
| `actor_rollout_ref.actor.policy_loss.loss_mode` | `future_kl` | 我们 patch 进去的 FIPO loss；`vanilla` 切回 GRPO |
| `actor_rollout_ref.actor.use_kl_loss` | `False` | FIPO 用 future-KL 做 IS 权重，**不**再加传统 PPO 的 KL 正则 |
| `actor_rollout_ref.actor.clip_ratio_low/high/c` | `0.2 / 0.28 / 10.0` | dual-clip PPO 的下/上/极端 clip 阈值 |
| `actor_rollout_ref.actor.optim.lr` | `1e-6` | RL 微调常用的小 lr，避免破坏 SFT 知识 |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | `3` | 每次 PPO update 的 prompt 数 |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | `1` | 每 GPU 一次 forward 的 prompt 数（控显存） |
| `algorithm.adv_estimator` | `grpo` | GRPO 组内归一化（无 critic） |
| `algorithm.use_kl_in_reward` | `False` | reward 里不混 KL 项，KL 由 future_kl loss 内部消化 |
| `algorithm.kl_ctrl.kl_coef` | `0.0` | 同上，KL 系数置零 |
| `data.train_batch_size` | `6` | 每 step 采样的 prompt 数（× n_resp=8 = 48 generations） |
| `data.max_prompt_length` | `8192` | 必须远大于纯文本长度——Qwen3-VL 一张图就吃 ~3000 image tokens |
| `data.max_response_length` | `1024` | 输出 reason+JSON 通常 100-150 tokens，留 buffer |

**vLLM rollout 关键配置**：

| Hydra path | 默认值 | 作用 |
|---|---|---|
| `rollout.name` | `vllm` | rollout engine（替代 sglang/hf） |
| `rollout.n` | `8` | 每个 prompt 采 8 条 → GRPO 组内归一化的"组" |
| `rollout.temperature` | `1.0` | 充分探索 |
| `rollout.top_p` | `0.95` | 截断长尾噪声 |
| `rollout.gpu_memory_utilization` | `0.70` | vLLM 自管 KV cache 的 GPU 比例（剩 30% 给 actor swap-in） |
| `rollout.tensor_model_parallel_size` | `1` | 8B 单卡可装；vLLM TP 与 verl FSDP world 解耦 |
| `rollout.enable_chunked_prefill` | `True` | 长 prompt（图像 token）分块 prefill，避免单次 OOM |
| `rollout.max_num_batched_tokens` | `9216` (= prompt+resp) | 必须 ≥ `prompt_length+response_length`，否则 vLLM KV cache 申请失败 |
| `rollout.max_model_len` | `9216` | 同上，决定 vLLM engine 的 context window |
| `rollout.prompt_length` / `response_length` | `8192 / 1024` | 显式声明，让 verl 的 batch padding 长度一致（不一致会触发 `agent_loop._postprocess` 的 cat 维度报错） |

**reward / 自定义 reward_manager 配置**（关键修正点）：

| Hydra path | 值 | 说明 |
|---|---|---|
| `reward.reward_manager.source` | `importlib` | 走文件路径加载（不走 verl 内置 register 表） |
| `reward.reward_manager.name` | `VLMAuditRewardManager` | 我们的类名 |
| `reward.reward_manager.module.path` | `${PWD}/src/stage3_fipo/verl_patches/reward_manager.py` | **必须是绝对文件路径**（verl 内部 `load_extern_object` 调用 `importlib.util.spec_from_file_location`，传 module path 字符串会 FileNotFound） |
| `reward.reward_model.enable` | `False` | 关闭 verl 内置 RM 模型加载 |

**关键环境变量**（[run_fipo_v1.sh](src/stage3_fipo/run_fipo_v1.sh) 顶部 export）：

| 环境变量 | 值 | 作用 |
|---|---|---|
| `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES` | `1` | 用户显式 `CUDA_VISIBLE_DEVICES=0..5` 时，让 verl 的 `worker._setup_env_cuda_visible_devices` 走 `ray.get_accelerator_ids()` 主动 `set_device(local_rank)`，否则 6 个 actor 都看到全部 6 卡且默认绑 device 0 → NCCL `Duplicate GPU detected` |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 国内拉 bge-small-zh-v1.5 必需 |
| `PYTHONPATH` | `${PWD}` | 让 Ray worker 子进程能 import `sitecustomize.py` + `src.stage3_fipo.*` |
| `TOKENIZERS_PARALLELISM` | `false` | 抑制 fork 后 tokenizer 警告 |
| `FIPO_DECAY_RATE` | `12.0` | future-KL 权重衰减系数 |
| `FIPO_CHUNK_SIZE` | `128` | future-KL 累加的 token chunk 长度 |
| `FIPO_FKL_CLIP_RATIO` | `0.2` | influence weight 上下 clip 阈值 |
| `FIPO_FKL_CLIP_HIGH_ONLY` | `false` | 是否只 clip 上界 |
| `FIPO_SAFETY_THRESH` | `4.0` | sequence-level safety threshold（超阈值整条 invalidate） |

**分布式策略：FSDP2 vs FSDP1 vs Megatron vs DeepSpeed**：

verl-latest 内置只支持 `fsdp / fsdp2 / megatron`（搜 `verl/workers/config/{actor,critic,engine}.py`）；**DeepSpeed ZeRO-3 不在 verl 一等公民列表**——要自己接 hybrid-engine 与 vLLM 的显存切换逻辑，工程量大且没有官方测试，本项目不考虑。

| 策略 | 我们场景下的判断 |
|---|---|
| Megatron | ❌ 8B 单节点用不上 TP/SP/PP；要装额外的 megatron-core，配置复杂 |
| DeepSpeed ZeRO-3 | ❌ verl 不内置；与 vLLM rollout 切换显存的逻辑要自己实现 |
| FSDP1 | ✅ verl 默认、最稳，作为 fallback |
| **FSDP2** ✅ | **首选**。per-parameter 分片，CPU offload 比 FSDP1 更省显存；对 Qwen3-VL 这类含 vision tower 的多模态模型包装更细（不会重复包 vision encoder）；我们 `ulysses_sequence_parallel_size=1`，不会触发 verl 内部的 FSDP2→FSDP1 强制回退 |

启动脚本默认配置：
```bash
actor_rollout_ref.actor.strategy=fsdp2          # actor 训练
actor_rollout_ref.ref.strategy=fsdp2            # ref forward
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
actor_rollout_ref.ref.fsdp_config.param_offload=True
```

如果 FSDP2 在 vLLM 显存切换或 Qwen3-VL DTensor 上踩坑，**降级到 FSDP1** 只需一行环境变量：

```bash
ACTOR_STRATEGY=fsdp REF_STRATEGY=fsdp bash src/stage3_fipo/run_fipo_v1.sh
```

**实测 6 卡 L20 跑 FIPO（已 step 通过 2026-04-23）**：

| 维度 | 实测值 |
|---|---|
| GPU 占用 | 6 卡 × ~25 GB（FSDP2 offload + grad-ckpt + vLLM 0.70） |
| GPU util | 95-100%（actor + ref forward + vLLM rollout 共享同卡） |
| timing/step | ~117s（gen 13s + old_log_prob 27s + update_actor 70s + update_weights 7s） |
| throughput | ~180 token/s |
| 总训练步数 | 666 steps（2 epochs × 333 steps/epoch），ETA ≈ 22 小时 |
| step:0 baseline reward | 4.52（format 1.0 + violation 2.0 + lexicon 0.19 + semantic 1.33） |
| step:2 actor metrics | `pg_loss=0.019`、`grad_norm=2.09`、`fipo/influence_weights∈[0.87, 1.10]`、`clip_frac_upper=1.0%` |

→ 实测 6 卡可正常推进，4 卡也跑得动（按需把 `N_GPUS / BATCH_SIZE / MINI_BSZ` 同步缩到能整除）。

### P4.5 · FIPO 训练打通历程（**已完成，2026-04-23**）

从"代码就位"到"真正 step:1 出现"中间共修复 **15 处**坑，按出现顺序：

| # | 错误现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | `actor.ppo_micro_batch_size_per_gpu` 缺失断言 | 必填字段未传 | 加 `MICRO_BSZ_PER_GPU=1` 显式传 |
| 2 | `Unknown reward manager: vlm_audit_v2` | verl 有两个 reward manager registry，我们注册到 experimental 那个但 driver 查的是另一个 | 改用 `reward.reward_manager.source=importlib` 文件路径加载 |
| 3 | hydra path 旧 `reward_model.reward_manager` 被弃用 | verl 0.8 改 schema 到 `reward.reward_manager` | 全量改 path |
| 4 | `FileNotFoundError: module_path='src.stage3_fipo...'` | `load_extern_object` 期望文件路径，不是 module 路径 | 传绝对文件路径 `${PWD}/.../reward_manager.py`，文件顶部注入 `sys.path` |
| 5 | `ValueError: Prompt length 4339 > model max 2048` | vLLM KV cache 申请按 model max len，没考虑 image tokens | `MAX_PROMPT_LEN=8192`，`max_model_len=prompt+resp` |
| 6 | `AssertionError: image_offset 0 != len(images) 1` | prompt 用 list-of-dicts 格式，verl 的 `_build_messages` 要 string + `<image>` 占位 | `prepare_fipo_data.py` 改成 `"<image>商品描述：{title}"` 字符串 |
| 7 | `TypeError: __init__() missing num_examine` | reward_manager 旧接口（4 位置参数） vs 新基类 `RewardManagerBase`（`config, tokenizer, compute_score, **kwargs`） | 改继承 `RewardManagerBase`，实现 `async run_single` |
| 8 | OOM CUDA on GPU 0（31GB 残留） | 上次 vLLM worker 没清，setsid 派生进程也漏杀 | 加 `trap cleanup EXIT INT TERM` 兜底 pkill |
| 9 | `Sizes of tensors must match: 4096 vs 4349 in cat dim 0` | `agent_loop._postprocess` 按第一个样本 padding，image tokens 长度不齐 | 显式传 `rollout.prompt_length`、`rollout.response_length` |
| 10 | `real_train_batch_size 32 not divisible by 6` | `BATCH_SIZE=4 MINI_BSZ=2` 与 6 GPU 不整除 | `BATCH_SIZE=6 MINI_BSZ=3` |
| 11 | `KeyError: 'reward_v2/lexicon'` 在 `_postprocess` | reward_fn breakdown 字段是条件性的（解析失败只有 parse_failure，正常才有 lexicon），verl 用第一个 sample 的 keys 查所有 sample | reward_manager 把 breakdown 标准化到固定 schema，缺失填 0.0 |
| 12 | `Unsupported loss mode: future_kl` | driver 进程注册了，Ray actor worker 进程 fresh interpreter，`POLICY_LOSS_REGISTRY` 是 per-process 模块状态 | 项目根加 `sitecustomize.py`，site 机制让每个 Python 进程启动时 auto-import patch 触发 `@register_policy_loss` |
| 13 | `NCCL ncclInvalidUsage: Duplicate GPU detected: rank N and rank 0 both on CUDA device 65010` | 用户显式 `CUDA_VISIBLE_DEVICES=0..5`，Ray 默认不再覆盖每个 actor 的 CUDA env，6 个 worker 都看 6 卡且都默认绑 device 0 | 加 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`，让 verl `worker._setup_env_cuda_visible_devices` 走 `ray.get_accelerator_ids()` 主动 `set_device(local_rank)` |
| 14 | `ValueError: All Metric instances must have the same number of values for dp aggregation: [2, 2, 1, 1, 2, 1]` | future_kl_loss 的 metrics 字段也是条件性的（pos/neg samples 不存在时跳过 key），6 个 DP rank 字段集不一致触发 `Metric.aggregate_dp` 报错 | 同 #11 思路：固定 metric schema，缺失字段填 0.0 |
| 15 | SSH 中断后主进程残留但 actor 全死、log 未更新 | `setsid + disown` 让 driver 在 hangup 后存活，但 trap cleanup 只在 driver 自己退出时触发 | `pkill -9 -f "main_fipo|ray::|raylet|gcs_server"` 后再启 |

**核心经验**：verl 0.8 + Ray + vLLM 这套技术栈对**字段一致性**（reward extra info / loss metrics 字段集需在所有 DP rank 间相同）和**进程隔离**（policy_loss/reward_manager 注册是 per-process 状态）有强假设；自定义 reward / loss 上手时几乎一定会撞上一次 `KeyError` 和一次 "Unsupported loss"，记住 sitecustomize + 固定 schema 两个套路就能少绕弯子。

### P5 · FIPO v1 跑通后的 Reward Saturation 诊断（**2026-04-24**）

#### 现象

打通 FIPO 后稳定跑了 240 → 309 共 **69 步**（中间因 vLLM 多模态 `masked_scatter` 偶发 bug 崩溃，与 RL 算法无关）。但 **val 曲线几乎不动**：

| step | val/total | Δ vs step 240 | 备注 |
|---|---|---|---|
| 240 | 4.7122 | — | resume 起点 |
| 250 | 4.7123 | +0.0001 | |
| 270 | 4.7130 | +0.0008 | |
| 290 | 4.7071 | -0.0051 | |
| 300 | 4.7097 | -0.0025 | |

#### 根因：组内 reward 方差崩塌

GRPO 用同 prompt 的 n=8 个 rollout 计算 advantage：`A = (r - mean(r)) / std(r)`。当训练样本"太简单"，所有 rollout 都拿到接近满分的 reward → `std → 0 → A → 0 → policy gradient = 0`。

实测 19 个观察 step：

| 指标 | 数值 |
|---|---|
| `critic/score/min` | 19/19 step 都是 4.5 |
| `critic/score/max` | 19/19 step 都是 5.0 |
| 组内 reward 跨度 | **始终 ≤ 0.5**（满分 5.0 的 10%） |
| `actor/loss = 0` 的 step 比例 | **8/9 = 89%** |
| `actor/grad_norm = 0` 的 step 比例 | **8/9 = 89%** |

也就是说 `sft_aux_merged` 已经把当前 `train.parquet` 上的 reward 拉满，**RL 没有 informative sample 可学**。

#### Reward 各分量饱和度（val 集）

| 分量 | 当前 | 满分 | 饱和度 | 是否仍有 RL 学习空间 |
|---|---|---|---|---|
| `format_base` | 1.000 | 1.0 | 100% | 无 |
| `violation_match` | 1.985 | 2.0 | 99.25% | 极小 |
| `semantic_align` (cos sim 0.774) | 1.499 | 1.5 | 99.9% | 无 |
| `lexicon` | 0.225 | 1.0 | 22.5% | **有空间但权重小（10%）** |
| `reason_length` | 0.000 | 中性 | — | — |

#### Reward Hacking 分析：**没发生**

判定方法：观察 hacking 的 4 个典型信号是否在 step 240→309 出现漂移。

| 信号 | step 240 | step 309 | 漂移 | 解读 |
|---|---|---|---|---|
| `response_length/mean` | 108 | 104 | -3.7% | 无堆词 / 无截断 |
| `val/lexicon` | 0.2275 | 0.225 | -0.001 | 词表用法稳定 |
| `val/violation_match` | 1.985 | 1.985 | 0 | 二分类无漂移 |
| `val/semantic_align` | 1.4997 | 1.4997 | 0 | reason 与属性对齐稳定 |

但 hacking 没发生的真正原因是 **策略权重几乎没动**（`grad_norm=0` 占 89%），不是模型"自觉守规矩"。

#### 解决方案：困难样本重采样

唯一可行路径是改变训练分布，让 GRPO 组内方差恢复。脚本：

| 脚本 | 作用 |
|---|---|
| `src/stage3_fipo/mine_hard_samples.py` | 用 `sft_aux_merged` 在 `train.parquet` 全量 greedy 推理，调 `reward_fn v2` 离线打分，输出 `data/fipo/sft_aux_train_scores.jsonl`（含 reward + breakdown） |
| `src/stage3_fipo/build_rl_train.py` | 从 jsonl 按 6 条规则（label_wrong / lexicon_contradict / align_low / length_bad / total_low / parse_failed）筛困难池，与原始 easy 池按 70/30 混合，输出 `rl_train.parquet` |

困难样本判定规则（任一触发即归入 hard pool）：

| 规则 | 触发条件 | 物理含义 |
|---|---|---|
| `label_wrong` | `breakdown.violation_match < 0` | 二分类预测与 GT 不符 |
| `lexicon_contradict` | `breakdown.lexicon < 0` | reason 用词与 violation 标签自相矛盾 |
| `align_low` | `breakdown.semantic_align_sim < 0.5` | reason 与商品属性语义偏离（潜在幻觉） |
| `length_bad` | `breakdown.reason_length < 0` | reason 太短或太长 |
| `total_low` | `reward < 4.5` | 综合得分明显低于 batch 平均 |
| `parse_failed` | breakdown 中出现 `parse_failure` 或 `missing_fields` | JSON 解析失败 |

#### 启动命令（等 GPU 空闲）

```bash
# 1) 离线挖矿（约 30-60 min on 1×A100，~2.5 GB VRAM）
python -m src.stage3_fipo.mine_hard_samples \
    --model_path models/sft_aux_merged \
    --train_parquet data/fipo/train.parquet \
    --out_jsonl data/fipo/sft_aux_train_scores.jsonl \
    --batch_size 4 --max_new_tokens 512 --resume

# 2) 看一眼困难样本分布（不写文件）
python -m src.stage3_fipo.build_rl_train \
    --scores_jsonl data/fipo/sft_aux_train_scores.jsonl \
    --report_only

# 3) 构造 RL 训练集（实跑用 50% hard + easy，共 1400 条；实际产 1749 条因 hard 池小于请求会触发上采样）
#    注意：不要直接用默认 hard_frac=0.7 + total_size=1500 —— 我们实测困难池只有 351
#    条 (17.6%)，0.7×1500 = 1050 → 触发 3x 上采样，难例会过度重复
python -m src.stage3_fipo.build_rl_train \
    --scores_jsonl data/fipo/sft_aux_train_scores.jsonl \
    --train_parquet data/fipo/train.parquet \
    --out_parquet data/fipo/rl_train_hard.parquet \
    --hard_frac 0.5 --total_size 1400 \
    --align_thresh 0.5 --total_thresh 4.5

# 4) FIPO v2：换 train_files + 开 swanlab、其它与 v1 严格对齐
export SWANLAB_API_KEY=$(awk '/password/ {print $2; exit}' "$HOME/.swanlab/.netrc")
CUDA_VISIBLE_DEVICES="1,2,3,4,5,6" N_GPUS=6 \
BATCH_SIZE=6 N_RESP=8 MINI_BSZ=3 MICRO_BSZ_PER_GPU=1 \
LOGGERS=swanlab,console EXP_NAME=FIPO-v2-hard-mined \
TRAIN_FILE=$PWD/data/fipo/rl_train_hard.parquet \
setsid bash src/stage3_fipo/run_fipo_v1.sh > logs/fipo_v2_hard.log 2>&1 < /dev/null &
disown
```

#### 预期效果对比

| 指标 | v1（全 easy） | v2（混困难） |
|---|---|---|
| `critic/score` 分布 | [4.5, 5.0] | [1.5, 5.0] |
| 组内 std | ≈0.1 | ≈1.0 |
| `actor/grad_norm` 非零 step 比例 | 11% | **>80%**（理论） |
| val 提升潜力 | 0 | +0.05 ~ +0.15 |
| Reward hacking 风险 | 低（无更新） | **中**，需监控 length 与 lexicon 漂移 |

#### 实测挖矿结果（2026-04-24，2000/2000 条）

挖矿耗时：**4651 s（≈78 min）**，单卡 GPU 2 (L20 48GB)，BGE 编码器 + Qwen3-VL-8B，峰值显存 25.9 GB / 48 GB。

| 难例规则 | 命中数 | 占比 | 说明 |
|---|---:|---:|---|
| `label_wrong`（violation 判错） | **24** | **1.2%** | SFT-aux 真错判，最硬骨头 |
| `lexicon_contradict`（关键词矛盾） | 6 | 0.3% | reason 含违规词但判 false（或反之） |
| `align_low`（BGE 语义对齐 < 0.5） | 254 | 12.7% | reason 措辞与 GT 偏离 |
| `total_low`（总 reward < 4.5） | 351 | 17.6% | 上面三类的并集，等同 `is_hard` |

> 核心启示：SFT-aux 在 violation 二分类上已经 **98.8% 正确**，下一阶段 RL 的边际收益主要在「reason 描述的细致度 / 与规则文档的对齐度」，而不是「敢不敢判违规」。

#### v2 实际启动参数（2026-04-24）

| 参数 | v1 (saturation) | **v2 (hard-mined)** | 决策依据 |
|---|---|---|---|
| `TRAIN_FILE` | `data/fipo/train.parquet` (2000) | `data/fipo/rl_train_hard.parquet` (1749) | 注入难例 |
| `--hard_frac` | — | **0.5** | 1.0 全难例会丢失"easy 兜底"的稳定信号；0.7 上采样 3x 太激进 |
| `--total_size` | — | **1400**（实际产 1749） | 接近 v1 量级，可比 |
| `--align_thresh` | — | **0.5** | BGE 余弦 < 0.5 在中文电商 reason 上是「明显偏离」的经验线 |
| `--total_thresh` | — | **4.5** | 与 v1 `critic/score` 上限 5.0 的 90% 分位对齐 |
| 上采样比例 | — | **2.0x**（hard pool 351 → 700） | 等比例 + 对应 GRPO 8 rollouts，每条难例期望被采 16 次 |
| 最终 mix | — | **700 hard + 1049 easy = 1749** | 难例占比 17.6% → **40%** |
| `N_GPUS` | 6 | **6**（GPU 1-6） | 避开 GPU 0 ECC，留 GPU 7 给 calibration |
| `BATCH_SIZE / N_RESP / MINI_BSZ / MICRO_BSZ_PER_GPU` | 6 / 8 / 3 / 1 | 6 / 8 / 3 / 1 | 保持不变，单变量对比 |
| `MAX_PROMPT_LEN / MAX_RESP_LEN` | 8192 / 1024 | 8192 / 1024 | 保持不变 |
| `optimizer_offload` | True → False（修复后） | **False** | 保留 v1 OOM 修复 |
| `RAY_memory_usage_threshold` | 0.95 → 0.97（修复后） | **0.97** | 保留 v1 OOM 修复 |
| `ACTOR_STRATEGY / REF_STRATEGY` | fsdp2 / fsdp2 | fsdp2 / fsdp2 | 保留 |
| `GEN_TP` | 1 | 1 | 单卡 vLLM 实例，6 卡即 6 实例 |
| `LOGGERS` | console（v1 没启 swanlab） | **swanlab,console** | 这次开 swanlab，方便 diff |
| `CKPTS_DIR` | `/mnt/nfs/.../FIPO-v1-rule-reward` | `/mnt/nfs/.../FIPO-v2-hard-mined` | NFS，避免占满本地 SSD |
| `total_epochs / save_freq / test_freq` | 2 / 40 / 10 | 2 / 40 / 10 | 保留 |
| `val_before_train` | True | True | 拿 baseline 比较 |

完整启动命令（已实跑，pid 2573644）：

```bash
export SWANLAB_API_KEY=$(awk '/password/ {print $2; exit}' "$HOME/.swanlab/.netrc")

CUDA_VISIBLE_DEVICES="1,2,3,4,5,6" N_GPUS=6 \
BATCH_SIZE=6 N_RESP=8 MINI_BSZ=3 MICRO_BSZ_PER_GPU=1 \
LOGGERS=swanlab,console EXP_NAME=FIPO-v2-hard-mined \
TRAIN_FILE=$PWD/data/fipo/rl_train_hard.parquet \
setsid bash src/stage3_fipo/run_fipo_v1.sh > logs/fipo_v2_hard.log 2>&1 < /dev/null &
disown
```

#### v2 关键观察指标（vs v1，按重要性排序）

1. **`actor/grad_norm` 非零 step 占比**：v1 = 11%，v2 目标 ≥50%。这是判断 hard mining 解了 reward saturation 的最直接信号。
2. **`critic/score` 标准差 / 直方图**：v1 = std≈0.1，几乎只在 [4.5, 5.0]；v2 期望 std≥0.5，分布拉宽到 [1.5, 5.0]。
3. **`reward/lexicon`、`reward/semantic_align`** 各分量：监控是否出现 reward hacking（length 暴涨 / lexicon 全堆关键词）。
4. **val_acc**（每 10 step）：v1 训完几乎不动；v2 期望 +0.05 ~ +0.15。
5. **CPU RAM 占用峰值**：A+B 修复后稳定在 ~80%；超过 95% 即 Ray OOM。

#### v2 启动后的初始化耗时实测

| 阶段 | 耗时 | 状态信号 |
|---|---|---|
| Hydra config 解析 + Ray 启动 | ~30 s | `[TaskRunner]` 第一行日志 |
| 6× RewardLoopWorker（含 BGE 加载） | ~60 s | `[RewardLoopWorker pid=...] BertModel LOAD REPORT` 出现 |
| 6× vLLMHttpServer 初始化 | ~3-5 min | `vLLM ready` / 端口 listen |
| FSDP shard + actor/ref load | ~3-5 min | GPU mem 从 ~500MB 涨到 ~30GB |
| `val_before_train`（200 条） | ~5-8 min | `[validate]` log 块结束 |
| 第一个 `train_step` | **总 ~12-18 min** 后开始 | `actor/loss` 第一次出现非零 |

#### 决策记录：为什么用 RL 而不是继续 SFT 修难例（**2026-04-24**）

挖矿后我们问过这个问题：既然 SFT-aux 已 98.8% 正确，剩下 351 条难例能不能直接用 SFT 续训解决？答：**`label_wrong` 那一类能，但占大头的 `align_low` 不能**——这恰好是为什么 v2 选 RL。

**按难例类型分别分析**

| 难例类型 | 数量 | SFT 续训能解？ | 关键约束 |
|---|---:|---|---|
| `label_wrong` | 24 (1.2%) | **能** | 二分类决策，GT 唯一，加进 SFT 数据 1 epoch 即可 |
| `lexicon_contradict` | 6 (0.3%) | **能** | 同上 |
| `align_low` | 254 (12.7%) | **不能（只能 50-70%）** | reason 是开放生成，GT 只是众多正确答案之一 |

**SFT 治不了 `align_low` 的 4 个本质原因**

1. **NLL 是 token-level**：模型生成「图中可见锋利金属边缘」与 GT「画面包含明显刀刃」语义对齐 0.85（reward_fn v2 给高分），但 SFT loss **每个 token 都罚**——把多种正确答案的多样性磨平
2. **没有"近似奖励"概念**：BGE 语义对齐分数 = 0.7 还能拿 1.05/1.5 reward，SFT 没有"差不多对"这一档
3. **5 个 reward 分量无法显式权衡**：v2 weights 是 `format=0.2, violation_match=2.0, semantic_align=1.5, lexicon_match=0.5, lexicon_contradict=-0.3`。SFT 只能隐式学到加权和，**没有调节旋钮**
4. **Distribution shift**：SFT 在 GT 训练，infer 在 self-sample。RL 直接在 self-sample 上学，**不存在这个 gap**

**佐证**：P2 段 Stage 1 消融已经验证过这个思路——`baseline (NLL)` vs `aux (SupCon+Triplet)` RM acc `+0.018`，证明 token-level NLL 不足以充分利用 reason 文本质量信号。当前问题同构。

**反直觉的观察**：v1 reward saturation 的真正原因，**恰好是 SFT-aux 已经把所有"SFT 范式能解决的"都解决了**。剩下的难例正好就是「SFT 解不了，RL 才有空间」的那一类。SFT-aux 太强反而成了 RL 的前提。

**建议的完整流水线（v3 之后再考虑）**

```
SFT baseline → SFT-aux → [可选 SFT v3：补 24 + 6 = 30 条 label_wrong/contradict 修死硬错] → FIPO v2 (hard-mined RL)
```

即「先 SFT 把硬错纠完，再 RL 把开放生成的文本质量推上去」。当前 v2 直接跳过 SFT v3，因为 30 条样本对 SFT 优化来说太少（容易过拟合），且 RL 的 `violation_match` 权重 2.0 也能间接解决判错；只在 v2 收敛后若 `label_wrong` 仍未消失才回补 SFT v3。

### P6 · Stage 4 RAG：置信度门控重设计 + 索引补强（**2026-04-24**）

#### 原实现的 3 个核心问题

`AuditPipeline._generate` (v1) 用 `mean(max_softmax_prob)` 做置信度，阈值 0.85 拍脑袋。问题：

1. **JSON 结构 token 噪声稀释**：`{`、`}`、`"`、`:`、字段名等 token 的 max prob ≈1.0，占 ≥60% token，**真正反映幻觉的 `violation` true/false 与 `reason` 实词被淹没**
2. **无中文 BM25 分词**：`indexer.py` 与 `inference.py` 都用 `text.lower().split()`，对中文规则文档每条收缩为 1 个 token，BM25 几乎退化成全检索
3. **violation_cases.jsonl 未入索引**：150 条真实违规案例存在 `data/raw/`，但 `indexer` 只索引规则库 → RAG 主要召回路径之一被废

#### 修复

| 模块 | 修复 | 文件 |
|---|---|---|
| 置信度多视图 | `_compute_confidence` 同时返回 `mean_max`/`min_max`/`field_min`/`mean_entropy` 4 个信号；新增 `field_min`（屏蔽 max_prob>0.999 的结构 token，只看真正"思考"的 token） | `src/stage4_rag/inference.py` |
| 双策略门控 | `confidence_method` ctor 参数（默认 `field_min`，可切回 `mean_max` 兼容旧实验）；`predict(return_debug=True)` 暴露所有信号 + RAG 触发原因，便于校准 | 同上 |
| 中文 BM25 | 抽公共 `_tokenize_zh()`：jieba 优先，缺失时回退字符级；`indexer.py` 与 `inference.py._retrieve_text` 共用 | `indexer.py`、`inference.py` |
| 案例库入索引 | `indexer.py` 新增 `--case_file`；规则 + 案例统一进 BM25 | 同上 |
| 本地 CLIP | `--clip_model` 默认指 `models/pretrained/clip-vit-base-patch32`，避开网络依赖 | `indexer.py` |
| 离线 build | 已支持 `HF_HUB_OFFLINE=1` 全离线索引构建 | 同上 |

#### 索引产物（已构建，2026-04-24）

| 索引 | 维度/规模 | 大小 |
|---|---|---|
| FAISS visual | 3093 张 × 512 dim (CLIP-ViT-B/32) | 6.3 MB |
| BM25 textual | 170 文档（20 规则 + 150 真实违规案例） | 489 KB |
| image_paths | 3093 entries | 109 KB |

#### 置信度 4 信号对比

| 信号 | 计算 | 优点 | 缺点 |
|---|---|---|---|
| `mean_max` | mean of greedy-token max softmax prob | 廉价、单 forward | **被结构 token 稀释** |
| `min_max` | min of greedy-token max softmax prob | 突出最不确定 token | 偶发噪声敏感 |
| `mean_entropy` | mean per-token entropy（nats） | 反映分布而非贪心 | HIGHER = 更不确定（与其它三个相反） |
| **`field_min`** | min(max_prob) over tokens with `max_prob ≤ 0.999`（自动屏蔽结构 token） | **针对性反映 reason/violation 不确定性** | 启发式过滤，对极端 case 有 false negative |

> 推荐主选 `field_min`，搭配 RM 二次打分（待实现）做双门控；`mean_max` 仅保留供与旧实验对比用。

#### 校准方法（已脚本化）

`scripts/calibrate_confidence.py`：

1. 用 `sft_aux_merged` greedy 跑 val.parquet 200 条，记录 4 个信号 + `is_correct`
2. 对每个信号扫所有阈值，画 (recall_errors, precision, frac_triggered) 曲线
3. 在 `target_recall=0.8` 约束下挑 precision 最高的阈值，写入 `results/stage4_confidence_calibration.json`
4. 同时把每条 sample 的 `(index, conf, is_correct)` 都序列化，可后续回放/diff

启动命令（约 10 min on 1 卡）：

```bash
python -m scripts.calibrate_confidence \
    --model_path models/sft_aux_merged \
    --val_parquet data/fipo/val.parquet \
    --out_json results/stage4_confidence_calibration.json
```

#### 索引构建命令

```bash
# 完整：图像 (CLIP) + 文本 (BM25)，需 ~40s on 1 卡
python -m src.stage4_rag.indexer \
    --image_dir data/raw/images \
    --rule_file data/raw/rules.jsonl \
    --case_file data/raw/violation_cases.jsonl \
    --out_dir data/rag_index

# 只建文本索引（无 GPU 也可），用于无 CLIP 时
python -m src.stage4_rag.indexer \
    --image_dir data/raw/images \
    --skip_visual --rule_file data/raw/rules.jsonl --case_file data/raw/violation_cases.jsonl \
    --out_dir data/rag_index
```

### P6.1 · 阈值实测、Bug 修复与端到端评测装配（**2026-04-25**）

P6 完成了置信度多视图与索引重建，本节记录把 RAG 真正"接进 evaluation pipeline"过程中实测出来的几个关键发现，**所有结论都来自当天在 `models/sft_aux_merged` + `data/rag_index` 上的实测，不是设计推断**。

#### 1. 阈值实测：默认 `0.85` 在两个方向都错

`AuditPipeline` 默认 `confidence_threshold=0.85` 是历史延续（`mean_max` 时代的经验值）。在 `data/fipo/val.parquet` 上跑 `sft_aux_merged` greedy decode（30 行 sanity，全部 `is_correct=True`，仅看 confidence 分布），4 个信号的实测分位数：

| 信号 | min | p10 | p25 | median | p75 | p90 | max | 默认 0.85 触发率 |
|---|---|---|---|---|---|---|---|---|
| `mean_max` | 0.902 | 0.924 | 0.935 | 0.950 | 0.960 | 0.963 | 0.979 | **0%**（全大于 0.85，永不触发） |
| `min_max` | 0.245 | 0.339 | 0.358 | 0.413 | 0.482 | 0.562 | 0.616 | **100%**（全小于 0.85，每条都触发） |
| `field_min` | 0.245 | 0.339 | 0.358 | 0.413 | 0.482 | 0.562 | 0.616 | **100%** |
| `mean_entropy`* | 0.056 | 0.106 | 0.117 | 0.142 | 0.172 | 0.195 | 0.281 | n/a（方向相反，越大越不确定） |

\* `mean_entropy` 单位是 nats，`AuditPipeline._should_trigger_rag` 走 `< threshold` 路径时方向相反，需在调用方手动反转或换 method。

**两个方向的失败模式**

- `mean_max` 被 JSON 结构 token（max prob ≈ 1.0）稀释到 [0.90, 0.98] 这种"几乎全选满"的窄带，**用 0.85 作阈值意味着 RAG 永远不被触发**。这等同于退化成"baseline + 检索器只在内存里待着"。
- `field_min` 屏蔽了 `max_prob > 0.999` 的结构 token 后，剩下都是 attribute 名 / `violation` / reason 实词的 max prob，本身就显著低（中位数 0.413）。**用 0.85 意味着每条都触发**，把 RAG 退化成"无脑两遍生成"，推理成本翻倍但选择性归零。

**生产阈值决策**

→ 选 `field_min < 0.40`（约 val 实测 p35 分位），**预期触发约 35% 样本**。理由：

| 方案 | field_min 阈值 | 预期触发率 | 推理成本 | 选择性 | 评价 |
|---|---|---|---|---|---|
| 默认 | 0.85 | 100% | 2× | 0 | 无意义 |
| 保守 | 0.36 (p25) | 25% | 1.25× | 高 | RAG 信号被裁剪掉一半，可能漏救 |
| **选用** | **0.40 (~p35)** | **~35%** | **1.35×** | **较高** | **甜蜜点** |
| 激进 | 0.48 (p75) | 75% | 1.75× | 低 | 接近"全触发"，得不偿失 |

**为什么不直接用 `_recommend()` 自动选**：`scripts/calibrate_confidence.py:_recommend()` 依赖 `is_correct=False` 的样本来求 (recall, precision)。当前 `sft_aux_merged` 在 val 上 acc=100%（30/30），**找不到错误样本可标定**，自动选阈值退化成"recall_errors=0 时挑最大阈值"，会推回 0.85 这种无意义值。当 val acc 接近饱和时，更可靠的做法是改用**信号本身的分位数**作为选阈值的依据（本节的做法）。如未来用更难的 val 集（例如把 top-K hard-mined 样本灌进 val），可重新跑 `_recommend()` 闭环。

#### 2. Bug：`_embed_image` 没解包 `BaseModelOutputWithPooling`

`AuditPipeline._embed_image` 最初直接 `F.normalize(self.clip_model.get_image_features(...))`。在 `transformers ≥ 4.46` 上 `CLIPModel.get_image_features` 的返回不再是裸 tensor 而是 `BaseModelOutputWithPooling` 包装：

```
AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'
  at src/stage4_rag/inference.py:215  emb = F.normalize(emb, dim=-1)
```

`indexer.build_visual_index` 早就有 `image_embeds → pooler_output → last_hidden_state.mean(1)` 的 fallback 链（P6 修过），**但 `inference.py` 这条路径漏写**。修复方法是把 indexer 的 fallback 完整搬到 `_embed_image` 中，同时在两边解包逻辑保持一致避免再分叉。

#### 3. `evaluate.py` 加 `--use_rag` 端到端开关

之前 `scripts/evaluate.py` 只走 baseline 推理（`run_inference()` 直接 `model.generate`），**没法测 RAG 的净收益**。本次给它装上 RAG 流水线开关，新参数：

```bash
--use_rag                       # 走 AuditPipeline，否则 baseline
--rag_index_dir data/rag_index
--rag_signal field_min          # mean_max / min_max / field_min / mean_entropy
--rag_threshold 0.40
--rag_top_k_visual 3
--rag_top_k_text 3
--rag_clip_model models/pretrained/clip-vit-base-patch32
```

实现要点：

- `--use_rag` 走 `AuditPipeline.predict(return_debug=True)`，把每条样本的 `confidence`、`gating_score`、`rag_triggered`、`rag_context` 都拿回来；baseline 路径完全不变（向后兼容已有 `results/eval_sft_*.json`）
- 复用现有 `compute_metrics()`，**额外把 `rag_triggered_rate / rag_signal / rag_threshold` 写进 `out` JSON**，方便后面横评不同阈值的成本-收益曲线
- `Path(args.out).parent.mkdir(parents=True, exist_ok=True)`，写到 `results/` 等子目录不再 `FileNotFoundError`

#### 4. 端到端评测：基线、设计与运行中现状

**Baseline（无 RAG，已沉淀）**：

| Ckpt | F1 | Precision | Recall | hallucination_rate | JSON ok |
|---|---|---|---|---|---|
| `sft_baseline_merged` | 0.9883 | 0.9806 | 0.9961 | **30.72%** | 100% |
| `sft_aux_merged` | 0.9844 | 0.9767 | 0.9921 | **30.27%** | 100% |

**关键诊断**：violation F1 已经在 98%+ 饱和（与 P5 RM 的 mean_margin 饱和、P6 RL 的 reward 饱和呼应）。**RAG 真正要打的指标是 `hallucination_rate`（30%）**——它定义为「`reason` 没引用任何 `attributes` key」，正是检索增强的目标场景：被检案例 / 规则文本作为 reason 的"锚点"喂回模型，理论上能直接降低这个比例。

**RAG 评测命令**（**当前运行中**，单卡 GPU 7）：

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 \
python -u scripts/evaluate.py \
    --model_path models/sft_aux_merged \
    --test_parquet data/sft/test.parquet \
    --use_rag --rag_index_dir data/rag_index \
    --rag_signal field_min --rag_threshold 0.40 \
    --rag_top_k_visual 3 --rag_top_k_text 3 \
    --out results/eval_sft_aux_rag.json
```

**实测吞吐**（前 35 条稳态后）：

| 项 | 值 |
|---|---|
| 单样本平均时延 | ~10 s/it（含 ~35% 触发的二次生成 + CLIP encode + FAISS search + BM25 query） |
| 全量 ETA | 664 × 10s ≈ **110 min** |
| GPU 7 显存 | ~19.5 GB（VLM ckpt + CLIP-ViT-B/32 + FAISS in-memory） |
| GPU 7 util | 80% – 100%（单卡推理瓶颈） |

**实测结果（2026-04-25 凌晨跑完）**

| 配置 | F1 | Precision | Recall | hallucination_rate | rag_triggered_rate |
|---|---:|---:|---:|---:|---:|
| `sft_baseline_merged` (no RAG) | 0.9883 | 0.9806 | 0.9961 | 0.3072 | – |
| `sft_aux_merged` (no RAG) | 0.9844 | 0.9767 | 0.9921 | 0.3027 | – |
| `sft_aux_merged` **+ RAG** (`field_min<0.40`) | 0.9702 | 0.9799 | 0.9606 | **0.3238** ↑ | 46.23% |
| `fipo_v2_step160_merged` **+ RAG** (`field_min<0.40`) | 0.9802 | 0.9841 | 0.9764 | **0.2304** ↓ | 32.23% |

**反直觉的结论**

1. **SFT-aux + RAG 是负优化**：F1 跌 1.4 个点（recall 跌 3.2 个点），hallucination 反升 2.1 个点。RAG 触发率 46% 远高于 val 校准的 35%——sft_aux 在 test 上 confidence 比 val 低，0.40 阈值太宽，且模型还没学会"区分图像证据 vs 检索文档"，把检索文本抄进 reason 但不引用 attribute key，**反而触发更多 hallucination 判定**。
2. **FIPO step160 + RAG 是正向收益**：F1 持平（−0.4 个点），**hallucination 大降 7.2 个点（30.27% → 23.04%，相对降 24%）**，precision 反涨 0.7 个点。RAG 触发率 32%，最接近 val 校准目标。
3. **关键改写**：之前 v2 hard-mining 阶段 val_reward 几乎不动（4.71 → 4.71）让我们以为"RL 没用、可以停训"。**端到端 RAG 评测推翻了这个判断**——RL 真正学到的不是"直接降 reward"，而是"在置信度低时更善于使用检索证据"的元能力，这正是 reward function 没显式 reward、被动力学忽略的 capacity。详细机制见下方 P6.2。

#### 5. Calibration 实操经验（踩坑记）

- **stdout buffer 黑洞**：`nohup python scripts/calibrate_confidence.py > log 2>&1 &` 在没有 `PYTHONUNBUFFERED=1` 或 `python -u` 时，**`print` 全被 fully-buffered 吃掉**。第一次 200 行跑了 12+ min 看不到任何 `[calib] 20/200` 进度，看上去像卡死但 `nvidia-smi` SM=100% 在算。生产命令必须 `PYTHONUNBUFFERED=1 python -u ...`。
- **`max_new_tokens=512` 容易撑满**：模型对个别 long-tail 样本会一直生成不出 EOS，跑满 512 token 单样本就要 30s+。对 calibration 这种"只要读 confidence 信号"的任务，`--max_new_tokens 320` 已经足够覆盖目标 JSON 结构（实测最长 reason 也就 200 token），把单样本时延从 ~10s 压到 ~3.7s。
- **`limit=30` 快速 sanity**：用 30 行先跑通流水线（< 2 min），确认 `processor.tokenizer` 没 missing key、`out.scores` shape 一致、JSON parse 成功率 100%，再决定要不要 200 全量。这是被第一次 12 min 卡死后总结的预防做法。

#### 6. 已知 limitations / 下一步

- **没有 errors 可标定**：val 30 行 sft_aux_merged 全对 → 自动 `_recommend()` 失效。下一步可以把 `mine_hard_samples.py` 输出里 `total < 4.0` 的 50-100 条加进 val，构造"困难 val"再跑 `calibrate_confidence.py`，让 (recall_errors, precision) 曲线真正能用
- **RM 二次打分门控未接**：当前只用 confidence 单门控。理想架构是 `confidence < τ_c → 检索 → RM 对 (baseline_resp, rag_resp) 打分 → 取高分`，避免 RAG 把对的样本带偏。`models/rm_ckpt_*` 已经训好可以直接调用，待端到端跑通后再补
- **~~FIPO ckpt 还没合并~~**：✅ 已完成（2026-04-25）。FSDP 6-shard → HF safetensors 合并产物在 `models/fipo_v2_step160_merged/`（17.5 GB）。合并工具 `scripts/merge_fipo_ckpt.py` 用 deepspeed-stub 包装绕开"无 nvcc 但 transformers/accelerate 仍要 import deepspeed"的死循环。
- **阈值在 sft_aux 上偏宽**：实测触发率 46% 而非校准目标 35%，说明 test 比 val 难。后续可以做 `field_min ∈ {0.30, 0.35, 0.40, 0.45}` 的扫描曲线，看 sft_aux 能否找回正向收益区间。

### P6.2 · GRPO 反事实分析：future-KL 是 RAG 端到端收益的关键（**2026-04-25**）

P6.1 的实测结论里有一个**算法层面的关键问题**：FIPO + RAG 把 hallucination 从 30% 拉到 23%（相对降 24%），但同样训练 setup 下**如果换成纯 GRPO（不带 future-KL），能不能拿到同样的端到端收益**？

我们没有跑 head-to-head 的 GRPO 对照实验（成本另起 ~10 GPU·hr），但可以从两条路径论证：(a) **算法机制差异**；(b) **从 FIPO 训练日志反推 future-KL 的实际作用**。

#### 1. FIPO vs GRPO 算法差异（与 RAG calibration 相关的部分）

| 维度 | GRPO（基线） | FIPO（本项目） | 对 RAG calibration 的影响 |
|---|---|---|---|
| Advantage 估计 | group-relative `(r − mean) / std` | 同 GRPO | 一致 |
| Token-level PPO clip | 单边 `min(ratio·A, clip(ratio)·A)` | 同 GRPO | 一致 |
| Future-KL 约束 | **无** | `future_kl_loss` 把 actor 与 ref 在 *未来 token 序列* 上的 KL 也写进损失 | **关键差异** |
| 显式 KL penalty 项 | 可选（默认关） | 默认开 | 二阶约束 |
| Influence weight | 不重加权 | `exp(-λ · future_kl)` 给每个 token 的 advantage 一个 [1−ε, 1+ε] 的乘子 | 直接控制 token-level drift |

→ **FIPO 比 GRPO 多了一层 token 序列级的"形态保护"**：无论 reward 怎么 shape，actor 在每个 token 位置上的分布都不能偏离 ref policy 太多。这是 RAG calibration 能否成立的物理前提——`field_min` 信号反映的就是 attribute / `violation` 这些"反映 reasoning 不确定性的 token"的 max softmax prob，**只有当 token-level 分布形态被保住时，这个信号才是可信的**。

#### 2. FIPO 训练日志反推 future-KL 的实际作用（直接证据）

从 `logs/fipo_v2_hard.log`（214 个 step，simple data 1-120 + hard-mined 121-200）抽取关键 actor 指标：

| 指标 | simple data 阶段 (1-120) | hard data 阶段 (121-200) | 含义 |
|---|---|---|---|
| `actor/fipo/influence_weights_mean` | 1.0011 | 1.0024 | 平均无偏（symmetric clipping） |
| `actor/fipo/influence_weights_min` | 0.8508 | 0.8610 | **每个 step 都有 token 被 future-KL 下调到 ~0.85** |
| `actor/fipo/influence_weights_max` | 1.1546 | 1.1476 | **每个 step 都有 token 被 future-KL 上调到 ~1.15** |
| `actor/ppo_kl` | 0.000003 (近 0) | −0.0001 (近 0) | **actor 与 old policy 的 KL 全程 ≈ 0** |
| `actor/entropy` | 0.1673 | 0.1270 (**−24%**) | policy 锐化但**不**坍缩 |
| `response_length/mean` | 113.0 | 109.8 | 输出长度稳定，**无 mode collapse** |
| `actor/grad_norm` | 2.557 | 0.658 | hard 阶段梯度稀疏（与 hard-mining 引入零梯度步一致） |

**三条直接证据**：

1. **`influence_weights_min/max ≈ [0.85, 1.15]` 全程持续**：这是 future-KL 在每个 step 都在工作的硬证据。如果 future-KL 不起作用，influence_weights 应该恒等于 1.0（步 1 的初值）。**214 个 step 没有一步 weights 范围塌缩到 [1.0, 1.0]**，说明每一步都有 token 被 future-KL 拉回。
2. **`ppo_kl ≈ 0` 全程**：FIPO 的双重约束（future-KL + PPO clip）把 actor 与 ref 的 KL 压在 1e-4 量级。GRPO 默认只有单边 PPO clip，**典型 ppo_kl 在 5e-3 ~ 5e-2**（DeepSeekMath / DAPO 论文中报告的常见值），高出本项目 2 个数量级。
3. **`entropy` 单调下降但 `response_length` 不变**：entropy 从 0.167 下降到 0.127（−24%），是 policy 锐化（confidence 抬升）的硬指标，**与端到端评测中 fipo+RAG 触发率 32.2% vs sft_aux+RAG 触发率 46.2% 的 14 个点差距完美吻合**——RL 让模型整体上对 attribute/reason token 的 max prob 抬升约 0.10-0.15，这正好把 sft_aux 在 0.30-0.40 区间的样本推过 0.40 阈值。**同时 response_length 几乎不变**，说明这种 confidence 抬升不是"输出变短/变重复"的退化，而是 well-distributed 的真锐化。

#### 3. GRPO 反事实推演：会发生什么

基于上述算法差异 + 已知 RL 文献观察，纯 GRPO 在同样的 SFT-aux 起点 + reward function v2 + hard-mined data 上，**最可能出现的 3 种失败模式**：

| 失败模式 | 训练侧表现 | 对 RAG 端到端的影响 |
|---|---|---|
| **A. Reward shaping 漂移**（无 future-KL 约束） | `ppo_kl` 上行到 1e-2 量级；`influence_weights` 不存在；entropy 更激进下降但伴随 `response_length` 收缩或重复 token | 模型对错的 case 也很自信 → `field_min` 也很高 → **不该触发 RAG 的反而高 confidence 不触发，该触发的也在 0.40 以上** → hallucination 不降甚至上升 |
| **B. Mode collapse**（GRPO group 内方差饱和后） | `actor/entropy` 跌到 0.05 以下；`response_length` 趋同（变成模板回答） | RAG 检索回的多样化案例与 collapsed 模板冲突 → 模型 ignore 检索内容 → 等价于 RAG 完全失效 |
| **C. Reward hacking 长度**（reward function v2 有 length-related component 时） | `response_length` 显著上升或下降，stylistic shortcut 命中高 reward | 输出充满 reward 偏好的固定短语，**attribute 字段语义内容下降** → hallucination_rate 反升（reason 不再引用具体 attribute） |

→ **三种失败模式都直接破坏 P6.1 的 RAG 端到端正向收益**。这不是"GRPO 不行"——GRPO 在很多任务上是 SOTA RL 算法——而是**当 RL 的下游消费方是"基于 token-level confidence 做 calibration"的 RAG 系统时，token-level KL 约束变成必要条件**。

#### 4. 间接证据：v1 (rule reward, FIPO) vs v2 (hard-mined, FIPO)

我们没有 GRPO 对照，但有同算法（FIPO）下两套 reward function 的对照：

- **v1 rule-reward**（已废弃，ckpt 已删）：reward 早早饱和到 ~4.85，val_reward 不再上升
- **v2 hard-mined**（当前）：reward 卡在 ~4.71，val_reward 不动

**两个版本都触达 reward saturation**，但 v2 的训练 dynamics（entropy 收窄、`influence_weights` 持续工作、`response_length` 稳定）说明**FIPO 即使在 reward 饱和的情况下，仍然在保护 token-level calibration**。如果是 GRPO，同样饱和情况下会更快进入失败模式 A/B/C，**端到端 RAG 不会拿到 hallucination −7 个点的红利**。

#### 5. Limitations 与下一步实验

- **没有 head-to-head GRPO 实验**：本节是机制推断 + 间接证据，不是因果证明。要做严格 ablation，需要：(i) 同样 ckpt + reward + data，(ii) 把 `verl.algo.future_kl_loss` 关掉只保留 PPO clip，跑 200 step，(iii) 合并 ckpt 跑同样 RAG 端到端评测。预算 ~10 GPU·hr，价值高但非紧急
- **没扫 future-KL 强度 λ**：当前 λ 是 verl 默认值。一个有意思的扫法是 `λ ∈ {0, 0.5λ, λ, 2λ}`，看 RAG hallucination 收益曲线 — 如果 λ=0 时 hallucination 不再下降，就**直接证明** future-KL 是端到端收益的因果来源
- **可信度 vs 准确度的脱钩**：当前实验里 fipo+RAG 的 confidence 整体抬升、触发率下降，但我们**没直接测 calibration error (ECE)**。下一步可以在 val 上画 reliability diagram，对比 sft_aux vs fipo 的 calibration curve，定量衡量"RL 是否让 confidence 与 accuracy 更匹配"
- **替代解释 1：RL 学到的纯粹是 "reason 字段更引用 attribute"**：可能不是 confidence 抬升带来的 RAG 选择性提升，而是 RL 直接训出了"reason 必须 reference attribute"的偏好，所以 hallucination 自然下降。这个解释可以通过 baseline 对比排除——`fipo_v2_step160_merged` (no RAG) 的 hallucination 数字尚未跑（**下一步实验**）。如果 fipo no-RAG 的 hallucination 已经显著低于 sft_aux no-RAG（例如 < 25%），那 RAG 的增量贡献就被夸大；如果 fipo no-RAG 仍在 30% 附近，本节论证完全成立
- **替代解释 2：RAG 触发率下降是 confidence 错误抬升的副作用**：fipo 触发率 32% < sft_aux 触发率 46%，理论上"少触发"应该让 RAG 收益变小，但实际 hallucination 反而更低。一个可能解释是 fipo 触发的 32% 都是真正的 hard sample，retrieval signal-to-noise 比 sft_aux 的 46% 高得多。可以通过对比 (sft_aux 触发集 ∩ fipo 触发集) vs (sft_aux 触发集 \ fipo 触发集) 的 hallucination 数字验证

### P3.5 · 其它（低优先）

- LoRA 视觉 attn 启用：当前仅 LM + merger 加 LoRA。若评估发现细粒度视觉属性（颜色/材质）准确率掉，可白名单匹配 `visual.*qkv` / `visual.*proj`（注意视觉 encoder 默认冻结要先解冻或单独允许）
- DDP/FSDP 入口：当前两阶段都只支持单进程 + `device_map="auto"` 的 model parallelism，吞吐受限于通信。若后续要扩 batch 或上更大模型，需改写训练入口

---

## 相关文档

- 训练策略细节：[STRATEGY.md](STRATEGY.md)
- 迁移到其他机器：[MIGRATION.md](MIGRATION.md)
- 数据工程历史深入：[docs/DATA_ENGINEERING.md](docs/DATA_ENGINEERING.md)（部分信息可能与 README 不同步，以 README 为准）
