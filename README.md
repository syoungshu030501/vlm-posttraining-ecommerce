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

**FIPO 环境调研结论**（todo #2 进行中）：
- vendored `vendor/FIPO-main/` 的 verl 是 0.5.0.dev，**不支持 Qwen3-VL**（写于 2025/初）
- 官方 verl 最新版（`vendor/verl-latest/`，0.8.0.dev）已经原生支持 Qwen3-VL（`verl/models/transformers/qwen3_vl.py` 存在）
- 计划：把 FIPO 的 future-KL patch（`core_algos.py` 里 `compute_policy_loss_future_kl`、`dp_actor.py` / `megatron_actor.py` 的 actor loss 接入、约 300 LoC）forward-port 到 verl-latest，避开 Qwen3-VL 兼容这个大坑
- 依赖冲突：verl-latest 要 `vllm>=0.8.5,<=0.12.0`，当前 VLM env 是 0.19.1。**不动现有 VLM env**，给 FIPO 单独建 conda env

**用户操作步骤（机器空时）**：
1. 等 todo #2-#7 推进完成（custom reward_manager / FIPO patch 移植 / 数据 parquet / 启动脚本）
2. 我会给一行命令形如 `bash src/stage3_fipo/run_fipo_v1.sh`，内部已包含 `HF_ENDPOINT=https://hf-mirror.com` 和正确的 conda env 激活

### P3.5 · 其它（低优先）

- LoRA 视觉 attn 启用：当前仅 LM + merger 加 LoRA。若评估发现细粒度视觉属性（颜色/材质）准确率掉，可白名单匹配 `visual.*qkv` / `visual.*proj`（注意视觉 encoder 默认冻结要先解冻或单独允许）
- DDP/FSDP 入口：当前两阶段都只支持单进程 + `device_map="auto"` 的 model parallelism，吞吐受限于通信。若后续要扩 batch 或上更大模型，需改写训练入口

---

## 相关文档

- 训练策略细节：[STRATEGY.md](STRATEGY.md)
- 迁移到其他机器：[MIGRATION.md](MIGRATION.md)
- 数据工程历史深入：[docs/DATA_ENGINEERING.md](docs/DATA_ENGINEERING.md)（部分信息可能与 README 不同步，以 README 为准）
