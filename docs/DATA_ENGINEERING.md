# 数据工程文档

本文档详细记录 e-commerce-audit 项目每一批数据的**来源、处理步骤、最终产物**。
用于交接、复现、溯源。

审计日期：2026-04-20。

---

## 目录

- [0. 图片池（基础原料）](#0-图片池基础原料)
- [1. SFT 数据（Stage 1 监督微调）](#1-sft-数据stage-1-监督微调)
- [2. 幻觉三元组（Stage 1 辅助损失）](#2-幻觉三元组stage-1-辅助损失)
- [3. 偏好数据（Stage 2 奖励模型）](#3-偏好数据stage-2-奖励模型)
- [4. Stage 3 FIPO 强化学习数据](#4-stage-3-fipo-强化学习数据)
- [5. RAG 语料（Stage 4 检索增强）](#5-rag-语料stage-4-检索增强)
- [6. 目录总览](#6-目录总览)
- [7. 复现 Checklist](#7-复现-checklist)
- [8. 已知问题与限制](#8-已知问题与限制)

---

## 0. 图片池（基础原料）

### 来源
- **2550 张真实服饰商品图**，从 `DeepFashion-MultiModal` / `Fashion-Gen` 衍生数据集下载
- 脚本：[src/stage0_distill/download_dataset.py](../src/stage0_distill/download_dataset.py)
- 存放：`data/raw/images/product_XXXXX.jpg`
- 尺寸统一为 JPEG，原始分辨率保留（训练时 pipeline 再 resize）

### 筛选
- 只保留 **完整商品图**（单 SKU，无水印，无模特遮挡过重）
- 自动过滤 <50KB 的低质量图
- 最终池：2550 张，覆盖服装/鞋/包/手表四大品类

---

## 1. SFT 数据（Stage 1 监督微调）

### 产物
| 文件 | 行数 | 用途 |
|------|------|------|
| `data/sft/sft.jsonl` | 5000 | SFT 训练的主注解源 |
| `data/sft/all.parquet` | 5000 | 全量 parquet（含图片字节） |
| `data/sft/train.parquet` | 4001 | Stage 1 训练集 |
| `data/sft/val.parquet` | 497 | Stage 1 验证集 |
| `data/sft/test.parquet` | 502 | 保留测试集，训练不可见 |

### 生成过程（三批次拼接）

#### 批次 A: v1_original（499 条）
- **时间**：项目早期
- **脚本**：原始 `scripts/distill.py`（已废弃）
- **方法**：基于预设 annotations.jsonl（随机图文对）让 qwen-vl-max 写 JSON
- **特征**：`violation` 字段分布失衡（89% 违规），reason 多样性高（99%）
- **保留原因**：违规样本的推理质量好，作为高质量 violation 样本保留

#### 批次 B: balanced_v2 violation 部分（1555 条，全部违规）
- **时间**：第一轮补充
- **脚本**：[scripts/distill_balanced.py](../scripts/distill_balanced.py) v2 版
- **方法**：6 种违规类型模板（极限词/材质虚标/功效夸大/品牌侵权/价格欺诈/图文不符），让 qwen-vl-max 先真实观察图片再注入违规
- **保留原因**：违规覆盖均衡（每类 ~250 条），推理质量合格

#### 批次 C: balanced_v3 compliant 部分（2946 条，全部合规）
- **时间**：2026-04-20（本轮重跑）
- **脚本**：[scripts/distill_balanced.py](../scripts/distill_balanced.py) v3 版（已更新 `COMPLIANT_SYSTEM` prompt）
- **触发原因**：v2 的合规样本 `reason` 字段被硬编码进 prompt → 2946 条合规样本 reason 字段**完全相同**（去重率 25%）
- **修复**：prompt 改为"生成 30-80 字引用本条商品具体属性的合规说明，禁止套模板"，并给 3 个风格示例
- **成本**：43.1 min，¥97.2（qwen-vl-max-latest，并发 6，rate_limit 12/s）
- **效果**：2946 条，reason 去重率 **99.6%**，平均长度 90 字（vs 旧模板 50 字）

### SFT JSONL 格式
```json
{
  "image_file": "product_00237.jpg",
  "image": "data/raw/images/product_00237.jpg",
  "prompt": "白色皮革交叉绑带坡跟凉拖鞋",
  "response": "{\"category\": \"鞋类\", \"attributes\": {...}, \"violation\": false, \"reason\": \"...\"}",
  "violation": false,
  "category": "鞋类",
  "violation_type": null,              // balanced_v2 + v3 有，v1 没
  "gen_strategy": "balanced_v3"        // v1_original | balanced_v2 | balanced_v3
}
```

### Parquet 切分（关键：按图片分组）
- **脚本**：[src/utils/data_prep.py](../src/utils/data_prep.py)
- **分组逻辑**：按 `image_file` 唯一值切 80/10/10，同一张图的所有样本绑定在同一 split
- **修复点（2026-04-20）**：修复前是按行随机切，导致 train∩val = 352 张图泄漏；改为按图分组后**0 重叠**
- **随机种子**：42（numpy RandomState）
- **合规/违规比例稳定**：40% 违规在 train/val/test 各 split 基本一致

### 质量指标（训练集）
- reason 去重率：92.0%（train），92.2%（val），95.0%（test）
- JSON 解析成功率：99.95%（train 有 2 条 v1 样本带 markdown fence，忽略）
- 品类 top 5：服装/鞋类/手表/女装/T恤

### 成本累计
- v1 原始：~¥15
- balanced_v2：~¥148.5
- balanced_v3（重跑）：¥97.2
- **合计约 ¥261**

---

## 2. 幻觉三元组（Stage 1 辅助损失）

### 产物
- `data/sft/triplets.parquet`：**15000 条**三元组，1.9 GB（含图片字节）

### 用途
Stage 1 训练除了标准 SFT loss 外，加一个**对比损失**：
- 正样本：`(image, real_attribute)` — 图文一致
- 负样本：`(image, fake_attribute)` — 幻觉属性（颜色/材质/款式的错误值）
- 模型学会给图片编码的 embedding 与正样本对齐，与负样本拉开距离 → 减少推理时的幻觉

### 生成过程
- **脚本**：[src/utils/build_triplets.py](../src/utils/build_triplets.py)
- **输入**：`data/sft/sft.jsonl`
- **流程**：
  1. 对每条 SFT 样本，取 response JSON 的 `attributes` 字段（例：`{"颜色":"黑色","材质":"棉","款式":"修身"}`）
  2. 对每个 attr_key，从白名单池中挑一个与真值不同的 fake 值
  3. 构造 `(image, "{key}: {real}", "{key}: {fake}")`

### 修复点（2026-04-20）
- **问题 1**：生成脚本对任意 attr_key 都造三元组 → 171 种 key，长尾爆炸；而且 `款式` 没有对应的替换池 → fallback 到颜色+材质 → 出现 "款式: 藏蓝" 这种跨属性荒谬样本
- **修复**：
  - 增加 `STYLE_POOL`（修身/宽松/长袖/手提包/托特包/…）
  - 引入白名单 `TRIPLET_KEY_WHITELIST`，只对 16 个有稳定替换池的 key 造三元组
- **效果**：171 种 key → **13 种**，0 跨属性串扰，rows 16019 → 15000（-6%，剔除低信号）

### 三元组分布
- 颜色：4953 (33%)
- 款式：4924 (33%)
- 材质：4909 (33%)
- 领型/表带材质/袖长/表盘颜色等长尾：214 (1.4%)

### 格式
```
image_path, positive_attr, negative_attr, attr_key, attr_value_real, attr_value_fake, image (bytes)
```

### 注意：这不是偏好数据

- 该 parquet **不用于 Stage 2 奖励模型训练**，也不用于 Stage 3 FIPO
- 它只用于 Stage 1 的辅助对比损失（contrastive loss）
- Stage 2 的 (chosen, rejected) 偏好对另有来源，见下节

---

## 3. 偏好数据（Stage 2 奖励模型）

### 产物
- `data/preference/preference.jsonl`：**4001 对** (chosen, rejected) 偏好对
- `data/preference/preference.parquet`：**4001 行**，与 JSONL 已同步
- `data/preference/preference.jsonl.<timestamp>.bak`：自动巡检/重建时的历史备份

### 用途
**Stage 2 训练一个独立的奖励模型（RM）**，用 Bradley-Terry 目标拟合偏好对：
给定 (image, prompt)，RM 应对 chosen 打高分、rejected 打低分。
训练好的 RM 在 **Stage 3 FIPO** 中作为 reward 信号的一部分（与规则奖励相加），
而不是用 DPO 直接把策略模型往 chosen 方向拉。

### 修复点（2026-04-20）
- **问题 1**：旧版 `preference.jsonl` 只有 **426 对**，且 `same_violation_label_pairs = 408/426 (95.77%)`，绝大多数 pair 只是在措辞上有轻微差别，RM 信号太弱。
- **问题 2**：旧版 `preference.parquet` 只有 **200 行**，与 JSONL 已经不同步。
- **问题 3**：第一次本地重构虽然把区分度提升到了 `same_violation_label_ratio = 0.2999`，但由于 train split 复现逻辑错误，把 **385 条 / 393 条**样本混进了 val/test。
- **修复**：
  - 新增 `scripts/one_shot_data_guard.sh`，先并行体检 `raw / sft / preference.jsonl / preference.parquet`
  - 使用与 `data_prep.py` 一致的 `numpy RandomState(seed=42)` 复现 train split
  - 仅基于 **SFT train split** 本地重构 preference pairs
  - 重建后立即同步刷新 `preference.parquet`

### 质量指标
- chosen 有效 JSON：4001/4001
- rejected 有效 JSON：4001/4001
- `same_violation_label_pairs`：1200/4001（29.99%）
- `label_flip_pairs`：2801/4001（70.01%）
- 图片与 SFT val/test 交集：0（已完全隔离）
- `preference.jsonl` 与 `preference.parquet`：4001/4001，已同步

### 格式
```json
{
  "image_file": "product_XXXXX.jpg",
  "image": "data/raw/images/product_XXXXX.jpg",
  "prompt": "...",
  "chosen": "{...严谨 JSON...}",
  "rejected": "{...更弱或相反的审核 JSON...}"
}
```

### 当前生成策略
- `opposite_label`：从相反 `violation` 标签的 train 样本中抽 donor，作为更强的负样本
- `same_label_diff_category`：保留同标签，但替换为不同品类的审核结果，增加结构性干扰
- `self_corruption`：对当前 chosen 做 reason 弱化或直接翻转 `violation`，保留少量“自扰动”负样本

### 结论
- 当前更适合先用 **SFT train split 本地重构** 的 preference 数据训练 RM，而不是立刻重新做 API 蒸馏。
- 如果后续 RM 训练仍然收敛差、排序准确率低，再考虑用 Qwen API 重蒸馏一版更自然的 rejected。

---

## 4. Stage 3 FIPO 强化学习数据

### 背景：FIPO = GRPO 改进版
- **FIPO**（Future-KL-Penalised Policy Optimisation）在 veRL/GRPO 基础上把显式 KL 外罚改成"未来 KL"内嵌损失，同时保留 GRPO 的 group-relative advantage estimator
- **不是 DPO**：DPO 需要成对 (chosen, rejected) 做对比学习；FIPO 是 on-policy rollout，对每个 prompt 采样 n 条 response，用 reward 差拉 policy
- 启动脚本：[src/stage3_fipo/run_fipo.sh](../src/stage3_fipo/run_fipo.sh)，`LOSS_MODE=future_kl`（`vanilla` 即退化为标准 GRPO，消融 baseline）

### 输入数据
FIPO 不需要新的数据集，**直接复用 SFT parquet 作为 prompt 池**：

| 角色 | 来源 | 说明 |
|------|------|------|
| Prompt | `data/sft/train.parquet`（或 all.parquet） | 只读 `image` + `prompt` 字段，response 字段忽略 |
| Rollout | Policy model 采样（n=8） | 每个 prompt 生成 8 条 JSON 响应 |
| Reward 1 | 规则函数 `compute_reward` | JSON 可解析 +1 / 必需字段 +1 / violation 命中 gt +2 / reason 不引用属性 -0.5 / JSON 解析失败 -3 |
| Reward 2 | Stage 2 RM 打分 | 加权求和到 reward 1 之上 |
| GT 来源 | `response` 字段（SFT 标签） | 用于规则 reward 中的 violation match |

### 关键设计
- **无新数据集**：FIPO 的 prompt 池就是 SFT train split，这样 prompt 分布与 SFT 对齐
- **评估集隔离**：FIPO 在 train 上做 rollout，不会碰 val/test（Stage 2 preference 已剔除 val/test 重叠，整条链路干净）
- **Reward 混合权重**：[src/stage3_fipo/reward_fn.py](../src/stage3_fipo/reward_fn.py) 固定规则权重；RM 权重在 FIPO 训练脚本中调整
- **Group size**：每 prompt 8 个 rollout，GRPO advantage 在组内 normalise
- **Response 长度上限**：512 token（JSON 输出短，节省显存 vs 默认 20480）

### 消融脚本
- [scripts/run_ablation.sh](../scripts/run_ablation.sh) 支持 4 个变体：
  1. SFT only
  2. SFT + GRPO (n=8)
  3. SFT + FIPO (n=8)      — 核心对照
  4. SFT + Contrastive + FIPO — 完整版

### 无需额外数据
本阶段只依赖：
- SFT parquet (train.parquet / all.parquet) — prompt 和 gt 标签
- Stage 2 RM checkpoint (`models/rm_ckpt`) — reward 信号之一

---

## 5. RAG 语料（Stage 4 检索增强）

### 产物
- `data/raw/violation_cases.jsonl`：**142 条**合规违规案例
- `data/raw/violation_cases.jsonl.bak`：修复前 632 条（含模板重复）
- `data/raw/violation_rules.md`：20 条违规规则摘要（供 system prompt 注入）
- `data/raw/gd_cases.jsonl`：广东市监局原始爬取数据（未过滤）
- `data/raw/samr_cases.jsonl`：SAMR 爬取的原始语料（未过滤）

### 两个来源

#### 来源 A：模板案例（10 条）
- 人工编写的 10 种典型违规案例
- 品类覆盖：3C 认证、保质期造假、成分虚标、虚假宣传、品牌侵权、价格欺诈等
- 字段：`category, violation_type, description, evidence, penalty, case_id, text`

#### 来源 B：真实爬取案例（132 条）
- **脚本**：[scripts/crawl_gd_cases.py](../scripts/crawl_gd_cases.py)
- **来源**：广东省市场监督管理局官网 `amr.gd.gov.cn`
- **爬取栏目**：新闻发布台、图片专栏、媒体关注、办理结果
- **清洗**：[scripts/merge_rag_cases.py](../scripts/merge_rag_cases.py)
  - 白名单标题（含"典型案例/曝光/铁拳/查处/处罚"等）
  - 正文按"一、二、三"或"案例 1/2/3"切分
  - 提取罚款金额（正则 `罚款\s*([\d\.]+)\s*(万元|元)`）
  - 提取法律依据（`《...》`）
  - 提取被罚主体（`[\u4e00-\u9fa5]{2,}(?:公司|店|…)` 等）
  - 品类细粒度判定（基于全文关键词而非单词匹配）
  - 质量过滤：必须至少命中"有罚款金额 / 有法律依据"两条之一
- **字段扩展**：`penalty_amount, laws, subjects, source_url, source_title`

### 修复点（2026-04-20）
- **问题**：`violation_cases.jsonl` 里有 500 条"模板案例"，但经 prefix 去重发现**只有 10 种唯一模板各复制 50 份** → RAG 召回浪费 context
- **修复**：按 `text[:100]` 去重，模板 500 → 10（保留 10 种独立案例）
- **结果**：632 → 142（10 模板 + 132 真实），全部唯一

### 质量指标
- 模板 10 条（各品类代表案例）
- 真实案例 132 条（全部带法律依据，69 条带具体罚款金额）
- 品类分布：电子产品 46 / 广告宣传 25 / 食品 22 / 通用 13 / 认证 11 / 医药 10 / 价格 7 / 服装 5 / 化妆品 2 / 家居 1
- **分布偏斜**：服装/化妆品案例少（5+2 = 仅 5%），与图片池品类（主要是服装/鞋）不对齐 —— 未来需定向爬取

### RAG 索引构建（尚未完成）
- 脚本：[scripts/build_rag_kb.py](../scripts/build_rag_kb.py)
- 计划：
  - 用 `bge-small-zh` 做 embedding
  - FAISS IVF 索引，topK=3 检索
  - 规则 + 案例拼接进 system prompt
- **状态**：原料已就绪，待索引构建

---

## 6. 目录总览

```
data/
├── raw/
│   ├── images/                          # 2550 张商品图
│   ├── violation_cases.jsonl            # 142 条 RAG 案例 (10 模板 + 132 真实)
│   ├── violation_cases.jsonl.bak        # 632 条原始（dedup 前备份）
│   ├── violation_rules.md               # 20 条违规规则摘要
│   ├── samr_cases.jsonl                 # SAMR 原始爬取
│   └── gd_cases.jsonl                   # 广东市监局原始爬取
│
├── sft/
│   ├── sft.jsonl                        # 5000 条 SFT 注解（主源）
│   ├── sft.jsonl.bak-before-v3          # v2 版本备份
│   ├── all.parquet                      # 全量 parquet
│   ├── train.parquet                    # 4001 条训练
│   ├── val.parquet                      # 497 条验证
│   ├── test.parquet                     # 502 条测试（不可见）
│   └── triplets.parquet                 # 15000 条幻觉三元组
│
└── preference/
    ├── preference.jsonl                 # 426 对偏好数据
    └── preference.jsonl.bak             # 200 对原始备份
```

---

## 7. 复现 Checklist

从 0 复现整套数据：

```bash
# 0. 准备图片池
python src/stage0_distill/download_dataset.py --n 2550

# 1. SFT 蒸馏（v3 版，含新 COMPLIANT_SYSTEM）
export DASHSCOPE_API_KEY=sk-xxx
python scripts/distill_balanced.py \
    --image_dir data/raw/images \
    --out_file data/sft/sft.jsonl \
    --target_total 5000 --compliant_ratio 0.6 \
    --model qwen-vl-max-latest --rate_limit 12 --workers 6
# 耗时 ~90 min，¥245

# 2. SFT parquet 切分（按图分组，防泄漏）
python -m src.utils.data_prep \
    --annotation_file data/sft/sft.jsonl \
    --image_dir data/raw/images \
    --out_dir data/sft --mode sft --split

# 3. 幻觉三元组
python -m src.utils.build_triplets \
    --annotation_file data/sft/sft.jsonl \
    --image_dir data/raw/images \
    --out_file data/sft/triplets.parquet --embed

# 4. 偏好数据
python scripts/build_preference.py \
    --sft_file data/sft/sft.jsonl \
    --n_pairs 200 \
    --out_file data/preference/preference.jsonl
# 生成后必须手动过滤掉 SFT val/test 中的图（见本文第 3 节修复点）

# 5. RAG 案例库
python scripts/crawl_gd_cases.py --max_pages 30 --workers 12   # 爬广东市监局
python scripts/merge_rag_cases.py                              # 过滤+合并
python scripts/build_rag_kb.py                                 # (待实现) FAISS 索引
```

---

## 8. 已知问题与限制

1. **图片池品类偏斜**：服装占 ~60%，化妆品/食品几乎为 0 —— 模型在这些品类上泛化能力有限
2. **RAG 案例分布不匹配**：电子产品案例最多（46 条），但图片池几乎没有电子产品 —— RAG 召回可能给不出有用案例
3. **偏好数据区分度弱**：只有 9% 的 rejected 有明确缺陷，Stage 2 RM 的 BT 拟合信号有限，Stage 3 FIPO 的 RM reward 稳定性需实测
4. **v1_original 批次 2 条 markdown fence JSON 解析失败**：占比 0.04%，不影响训练
5. **字节级哈希有"假重叠"**：不同图片 JPEG 压缩后前 10KB 偶尔相同（val/test 间发现几张），但 image_file 层面完全分离
6. **RAG 索引尚未构建**：`build_rag_kb.py` 待完成

---

## 附录：修复历史

| 日期 | 问题 | 修复 |
|------|------|------|
| 2026-04-20 | v2 合规样本 reason 完全相同（2946 条同一句） | 重写 `COMPLIANT_SYSTEM` prompt，重跑 balanced_v3（¥97.2） |
| 2026-04-20 | SFT parquet 按行随机切 → train∩val=352 张图泄漏 | 改为按 `image_file` 分组切分（`data_prep.py`） |
| 2026-04-20 | triplets 长尾 attr_key（171 种）+ 跨属性串扰（"款式: 藏蓝"） | 加 `STYLE_POOL` + `TRIPLET_KEY_WHITELIST`（`build_triplets.py`） |
| 2026-04-20 | preference 信号过弱 + parquet 与 JSONL 不同步 | `one_shot_data_guard.sh` 本地重构 → **4001 对**，并同步刷新 parquet |
| 2026-04-20 | RAG 500 模板实为 10 唯一×50 复制 | prefix 去重，632 → 142 |


## Data Guard 自动巡检（2026-04-21 13:27:16）

- 脚本：`scripts/one_shot_data_guard.sh`
- 处理对象：`data/preference/preference.jsonl` 与 `data/preference/preference.parquet`
- 巡检前：JSONL 1255 条，Parquet 1255 条，同标签比例 0.6502
- 巡检后：JSONL 1255 条，Parquet 1255 条，同标签比例 0.6502
- 判定：无需修复
- 说明：本次若触发重建，使用 SFT train split 本地重建 stronger preference pairs，并已同步刷新 parquet。

