# 电商合规审核系统 — 数据与工程策略沉淀

> 本文档沉淀本卡上为 **Qwen3-VL-8B 电商合规审核系统** 准备数据、代码、知识库的**所有策略**，不论采用与否，便于迁移与复盘。

---

## 一、总体目标拆解

| Stage | 产物 | 本卡完成度 |
|---|---|---|
| 0 — 蒸馏 | SFT 样本、偏好对 | 进行中（后台，详见下文） |
| 1 — SFT | CE + SupCon + 幻觉 Triplet | ⏸ 目标卡执行 |
| 2 — RM | 奖励模型 | ⏸ 目标卡执行 |
| 3 — FIPO | token-level KL 加权 RL | ⏸ 目标卡执行 |
| 4 — Agentic RAG | FAISS (视觉) + BM25 (规则+案例) | ✅ 索引元数据就绪 |

**本卡定位：只准备数据 + 代码，不跑训练。** 训练要去高算力卡。

---

## 二、模型下载策略

### 采用方案
[scripts/download_models.sh](scripts/download_models.sh)

1. **ModelScope 优先**（国内直连，无需 HF token）
2. 回退 `hf-mirror.com`（国内 HF 镜像）
3. 最后 fallback 官方 HF
4. **并行下载**：后台 `&` 启动，`wait_all` 统一等待，失败汇总

### 清单
| 模型 | 用途 | 体积 |
|---|---|---|
| `Qwen/Qwen3-VL-8B-Instruct` | SFT+FIPO 训练底座 | ~16GB |
| `openai/clip-vit-base-patch32` | RAG 视觉检索 FAISS | ~600MB |
| `BAAI/bge-m3` | RAG 文本检索 dense 路（补强 BM25） | ~2GB |

### 决策：本卡不下模型
本卡只做数据+代码准备，模型在目标卡上 `bash scripts/download_models.sh` 一键拉下即可。

---

## 三、图片数据策略

### 失败路径（记录）
| 尝试 | 结果 | 原因 |
|---|---|---|
| `ali_pai/pai_ecommerce_product_images` | 404 | 该 ModelScope ID 不存在 |
| `tany0699/product-images-with-text` | 404 | 同上 |
| `TaoBao-MM` | 56GB 下载中止 | 内容是 feature hash 而非图片 |
| `MsDataset.load(LouisXun/fashion_iten_dataset)` | SDK 挂起 | SDK 会先扫完 43213 文件再返回 |
| 合成 `PIL.Image` 噪声图 2472 张 | 被用户判为"垃圾数据" | 非真实商品，模型学不到视觉属性 |

### 采用方案
**直接 HTTP API 下载** `LouisXun/fashion_iten_dataset`（ModelScope 开放接口）+ `ThreadPoolExecutor(16)`。

- **产物**：2550 张真实 fashion 图片（1800×2400 → 压缩到 max 1024px / JPEG quality 88）
- **落盘**：`data/raw/images/`（374MB）
- **对应标注**：2550 条 `{image_file, description}`，描述从 13 个模板里循环取（含"最新款 超低价 清仓大甩卖"等极限词触发违规）

---

## 四、标注生成（Stage 0 蒸馏）

### 教师模型选择
| 候选 | 选择 | 原因 |
|---|---|---|
| `qwen-vl-max-latest` | ✅ | 最强，专业电商语境理解好 |
| `qwen-vl-plus` | ⏸ | 便宜但能力弱一档 |
| `qwen2.5-vl-72b-instruct` | ⏸ | 慢且贵 |
| GPT-4V | ❌ | 国内访问受限 |

### 技术决策
- **DashScope OpenAI 兼容模式**：`base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"`，复用 `openai>=1.50` SDK，代码最简
- **`enable_thinking=False`**：Qwen3.5 默认开 thinking mode，蒸馏纯 JSON 不需要；同时用 regex 剥 `<think>...</think>` 兜底
- **偏好对构造**：同一张图 + 同一 prompt，`temperature=0.1`（chosen）vs `temperature=1.2`（rejected）。若两者语义相同再提到 1.5 再试一次
- **SFT 违规正负比**：`force_violation=(i % 10) < 3`，保证 30% 的样本强制生成违规标注（否则 Qwen-max 倾向保守）
- **`--resume` 断点续传**：已处理的 `image_file` 跳过，出错或中途可接着跑

### 成本估算
[scripts/check_api_cost.py](scripts/check_api_cost.py)

- `qwen-vl-max-latest`：输入 ¥0.02/1k token，输出 ¥0.06/1k token
- 每次调用 ≈ ¥0.033（1200 in + 150 out）
- 500 SFT + 200 pref (×2) = 900 calls ≈ **¥30**
- DashScope 免费额度 `qwen-vl-max` 有 100 万 token（约能撑 ~500 次调用），超出走付费

### 查额度
- 控制台：https://dashscope.console.aliyun.com/overview
- 本地估算：`python scripts/check_api_cost.py`

---

## 五、RAG 知识库策略 🎯

### 构成（分层）
| 层 | 文件 | 数量 | 来源 | 作用 |
|---|---|---|---|---|
| L1 规则 | [data/raw/rules.jsonl](data/raw/rules.jsonl) | 20 | 手工提炼（广告法/食品安全法/平台规则） | BM25 主召回 |
| L2 模板案例 | [data/raw/violation_cases.jsonl](data/raw/violation_cases.jsonl) 前 500 | 500 | 10 种子模板 × 50 扩展 | 密集 BM25 召回（解决小样本问题）|
| L3 真实案例 | [data/raw/violation_cases.jsonl](data/raw/violation_cases.jsonl) 后 132 | 132 | **北京市监局典型案例爬取** | 带法律依据+罚款金额，高权重 |
| 原始爬取 | [data/raw/samr_cases.jsonl](data/raw/samr_cases.jsonl) | 209 | 未过滤，作为审计证据 | 追溯 |

### 真实案例爬取策略

#### 采用：[scripts/crawl_samr_cases.py](scripts/crawl_samr_cases.py)
- **源站**：https://scjgj.beijing.gov.cn/zwxx/scjgdt/ (北京市市场监督管理局 · 市场监管动态)
- **爬取范围**：67 页列表 × 15 条/页 = ~1000 篇文章
- **过滤**：标题白名单 `(典型案例|曝光台|铁拳|违法|案件|查处|处罚|专项|整治|执法)` 命中，得 106 篇文章
- **拆分**：每篇按「一、二、三、...」切成独立案例，得 209 条候选
- **清洗**：命中「罚款金额」/「法律依据」/「被罚主体」至少两项，保留 132 条高质量真实案例
- **用时**：多线程 16 并发，约 1 分钟
- **反爬**：该站点为静态 HTML，未遇到 JS challenge/captcha

#### 尝试但放弃的源
| 源 | 结果 | 原因 |
|---|---|---|
| `cfws.samr.gov.cn` 市监总局裁判文书库 | 需登录 | 细节数据受登录墙保护 |
| `www.creditchina.gov.cn` 信用中国 | HTTP 412 + JS challenge | 部署了商业级反爬（`$_ts` 动态 token），需 Playwright 绕 |
| `www.samr.gov.cn` 市监总局 | 连接超时 | 本卡网络出口限制 |
| ModelScope 数据集检索 API | 500 系统错误 | API 本身不稳定 |

#### 未尝试但值得扩展的源
> 若需要继续扩充到 1000+ 条真实案例：

- **上海市监局**：`https://scjgj.sh.gov.cn/` （返回 200，结构待分析）
- **广东市监局**：`http://amr.gd.gov.cn/` （返回 200）
- **各省消协**：「京津冀消协」「浙江消协」发布的行业劝谕
- **国家标准全文公开**：`https://openstd.samr.gov.cn/` (GB/T 标准文本，权威性高)
- **curl_cffi 或 Playwright 绕 CreditChina**：反爬代价陡增，**ROI 低**

### 品类分布分析
真实案例品类偏政府重点关注领域（食品/电子/广告），对 fashion 电商的"服装/化妆品"覆盖偏弱：

```
食品 20 + 广告宣传 25 + 电子产品 44 + 医药 10 + 服装 3 + 认证 11 + 价格 7 + 通用 12
```

**缺口应对**：
- 服装类主要靠 L2 模板案例（~100 条）
- 极限词违规走 L1 规则 G002 + 真实广告案例 25 条，够用
- 后续可用 Qwen-max 扩写模板（~¥10 即可生成 500 条高质量服装/化妆品变体）

### RAG 索引策略
| 索引 | 编码器 | 内容 | 用途 |
|---|---|---|---|
| FAISS（视觉） | CLIP ViT-B/32 | 2550 张商品图 | 相似商品 k-NN，找历史合规参考 |
| BM25（文本） | 中文分词 | rules + violation_cases 合计 652 条 | 规则/案例召回 |
| 可选 dense 文本 | BGE-M3 | 同 BM25 语料 | 补强 BM25（下游做） |

**执行位置**：建议在目标卡上做（CLIP 需要 GPU 加速编码 2550 张图）。BM25 可以本卡 CPU 秒级出。

---

## 六、数据处理与训练数据构造

### JSONL → Parquet 转换
[src/utils/data_prep.py](src/utils/data_prep.py)

- **`--no_embed`**：存图片路径而非 bytes，parquet 小 10×，DataLoader 按需加载
- **train/val/test = 8:1:1**：`numpy.random.RandomState(42).permutation` 固定随机种子，可复现

### 幻觉三元组（Stage 1 辅助 loss）
[src/utils/build_triplets.py](src/utils/build_triplets.py)

#### 目的
Stage 1 SFT 除了主 CE loss，还叠加一个 **triplet margin loss** 专门打压视觉幻觉。模型对图片编码后的表征应该与**真实属性文本**更近、与**被篡改属性文本**更远：
```
L_triplet = max(0, margin + d(anchor_img, neg_text) - d(anchor_img, pos_text))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           推远幻觉                 拉近事实              margin=0.2 默认
```

#### 三元组构造流程

**输入**：`sft.jsonl` 每条的 `response` 字段里的 `attributes` dict，形如：
```json
{"颜色": "深棕色", "材质": "绒面革", "款式": "套脚便鞋"}
```

**对每个 (key, value) 对构造一条三元组**：
1. **anchor**：商品图路径（等训练时用视觉编码器过，本文件只存 path）
2. **positive_attr**：真实属性文本 `"颜色: 深棕色"`（保持 key-value 语法，让模型学到属性级对比）
3. **negative_attr**：改写值 `"颜色: 银色"`（同 key，换成池里一个不同的值）

#### 属性池（按 key 选池）
[src/utils/build_triplets.py](src/utils/build_triplets.py#L30-L52)

| key 关键词 | 值池（中英混合） |
|---|---|
| `颜色` / `color` | 红/蓝/绿/黄/黑/白/紫/橙/棕/灰/粉/金/银/米白/藏蓝/酒红 + red/blue/... (22 个) |
| `材质` / `面料` / `material` | 棉/涤纶/丝绸/羊毛/亚麻/皮革/牛仔/雪纺/尼龙/氨纶 + cotton/... (17 个) |
| `形状` / `shape` | 圆形/方形/长方形/椭圆/三角形/不规则 + round/... (11 个) |
| 其他 key | 兜底 fallback: COLOR_POOL + MATERIAL_POOL 混合池 |

**负样本选取规则**：
```python
rng = random.Random(seed)
candidates = [v for v in pool if v.lower() != real.lower()]  # 剔除真实值
neg = rng.choice(candidates) if candidates else real + "_fake"
```
- `seed = sample_idx × 100 + attr_idx`，保证每条三元组的负样本**可复现**
- 同一池内随机挑一个不等于真值的值，避免"红色 → 红色"的无效对比

#### 产出规模（基于当前 5000 SFT）
```
5000 SFT 样本 × 平均 3.2 个属性/样本 = 16019 条三元组
```

按 attr_key 分布：
| attr_key | 数量 |
|---|---|
| 颜色 | 4953 |
| 款式 | 4924 |
| 材质 | 4909 |
| 图案 | 177 |
| 领型 | 89 |
| 品牌/鞋底/设计细节/... | 长尾 967 |

**三大核心属性（颜色/款式/材质）覆盖 93%**，符合合规审核场景的重点：属性虚标几乎都集中在这三类。

#### 为什么这么做
1. **属性级而非样本级对比**：商品图整体相似度对审核没用，要让模型关注单个属性维度。
2. **文本形式保持一致**：`"颜色: 深棕色"` vs `"颜色: 银色"`，key 相同，只有 value 变 → 模型必须通过视觉特征区分。
3. **不需要额外标注**：三元组从已有 SFT response 里提取，零 API 成本。
4. **negative hard mining 的可选升级**：当前是**随机负样本**。后续可升级为 semi-hard negative（在同一 batch 内挑距离 anchor 次近但还是错的 attr），效果更强，但需要 forward pass 做 mining，训练慢一点。

#### 训练期怎么用
```python
# 伪代码 — Stage 1 loss 组合
ce_loss = cross_entropy(model(img, prompt), target_json)
img_feat = model.vision_encoder(img)            # (B, D)
pos_feat = model.text_encoder(pos_attr_text)   # (B, D)
neg_feat = model.text_encoder(neg_attr_text)   # (B, D)
triplet_loss = F.triplet_margin_loss(img_feat, pos_feat, neg_feat, margin=0.2)

loss = ce_loss + 0.3 * triplet_loss + 0.1 * supcon_loss
```
权重在 [src/stage1_sft/train.py](src/stage1_sft/train.py) 里 `--aux_loss_weight` 可调。

---

## 七、环境与包管理策略

### 采用：**uv 优先，pip 兜底**
```bash
# 优先
uv pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple/ <pkg>
# 兜底（PEP 668 环境）
pip install --break-system-packages --index-url https://pypi.tuna.tsinghua.edu.cn/simple/ <pkg>
```

### 本卡实际安装（精简）
只装 **数据准备** 需要的：
- `openai`, `dashscope` — API 调用
- `modelscope`, `datasets` — 数据拉取
- `pandas`, `pyarrow`, `pillow`, `tqdm` — 数据处理
- `requests` — 爬虫

**未安装**（等目标卡装）：`torch`, `transformers`, `accelerate`, `peft`, `deepspeed`, `vllm`, `flash-attn`, `bitsandbytes`, `faiss-gpu`, `rank-bm25`, `sentence-transformers`

---

## 八、迁移策略

### 打包
[scripts/pack_for_migration.sh](scripts/pack_for_migration.sh)

- `dist/ecommerce-audit-code.tar.gz` (<5MB)：`src/ scripts/ requirements.txt`
- `dist/ecommerce-audit-data.tar.gz` (~400MB)：images + 所有 jsonl + parquet + RAG KB
- `dist/MANIFEST.txt`：清单 + SHA256 校验和

### 目标卡跑法
见 [MIGRATION.md](MIGRATION.md)：解包 → `uv venv` 装依赖 → `bash scripts/download_models.sh` → 数据重转 parquet → FAISS 索引 → 分阶段训练。

---

## 九、数据当前状态（本文档生成时）

| 指标 | 数值 |
|---|---|
| 图片 | 2550 张（fashion 真实商品） |
| 标注模板 | 2550 条 |
| **SFT 样本** | **5000 条（3000 合规 + 2000 违规）** ✅ |
| SFT train/val/test | 4000 / 500 / 500 |
| 幻觉三元组 | 16019 条（~3.2×/样本）|
| 偏好对 | 200 条 |
| RAG 规则 | 20 条 |
| RAG 案例（模板+真实） | 632 条 |
| 真实处罚案例（带法律依据） | 132 条 |
| 总磁盘占用 | ~380MB |

### SFT 最终分布
**品类 TOP10**（按样本数）：手表 503 / 男装-衬衫 369 / 男装 336 / 服装 312 / 男鞋 287 / 女装 241 / T恤 216 / 运动鞋 178 / 衬衫 166 / 男装衬衫 161

**6 种违规类型均衡分布**：
- 品牌侵权 276
- 图文不符 270
- 极限词 269
- 价格欺诈 249
- 材质虚标 247
- 功效夸大 244

### 生成策略分层
- `v1_original` 499 条：随机模板配图 + Qwen 判审（导致 89% 违规偏斜）
- `balanced_v2` 4501 条：**看图写真实描述** + 可控合规/违规，违规样本按 6 种类型均衡抽样注入

---

## 十、Stage 0 平衡蒸馏方案（`scripts/distill_balanced.py`）

### 动机
v1 策略下，描述从 13 个模板随机挑，与图片错配，Qwen 几乎总判"图文不符 → violation=True"，导致 **89% 违规**，训练无法学到"合规是什么样"。

### 核心改动
1. **Qwen 自己看图写描述**：不再依赖 `annotations.jsonl` 的预设描述。system prompt 要求 Qwen 先观察真实图，再生成 15-35 字的卖家式描述。
2. **违规按类型注入**：6 种违规模板（极限词/材质虚标/功效夸大/品牌侵权/价格欺诈/图文不符），每个违规样本随机抽一种，在真实描述上做对应改造。这让 reason 字段能具体引用违规词。
3. **合规/违规比例可控**：`--compliant_ratio 0.6` 一键指定。生成时做前置校验：若 Qwen 返回的 `violation` 与期望不符（模型没听话），直接丢弃重试。
4. **图片均衡复用**：`--max_per_image 3`，同张图最多复用 3 次，保证 5000 样本分布在 2550 图上。

### 工程优化
- **滑动窗口并发**（workers × 3 pending）：避免 `executor.submit(4501 tasks)` 阻塞主线程 10 分钟。`FIRST_COMPLETED` 事件驱动消费，提交和写文件流水线化。
- **rate limiting 8 QPS**：DashScope 对 qwen-vl-max 单用户 QPS 限制，留一点裕度。
- **失败兜底**：429 等速率错误指数退避到 30s；JSON 解析失败直接丢该条，不阻塞整批。

### 实际指标
- 79 分钟完成 4501 次调用，**零失败**
- 实际成本 ¥148.5（与事前估算一致）
- 成功率稳定 ~57/min（= 4 workers × 0.25 call/s × 60）

### 与 v1 共存
v1 的 499 条没删，作为对照组留下来（`gen_strategy` 字段区分）。训练时可：
- 全量训（混合两种策略）
- 只用 `balanced_v2`（推荐）
- 做 ablation：v1 only vs v2 only vs mixed

### 标签偏斜的根因消除
训练期不再需要 class weighting / focal loss —— 分布已经在数据层修正到 60/40。如需更贴近真实业务分布（~95/5），也可以重抽样。

## 十一、关键教训

1. **反爬成本 > 算力成本**。政府公开数据站点分两类：静态 HTML（北京市监局）vs 商业反爬（信用中国）。优先选前者。
2. **API 蒸馏温度是最便宜的 chosen/rejected 构造方法**。同张图、同 prompt、temp=0.1 vs 1.2，省一次模型标注成本。
3. **`enable_thinking=False` 一定要关**。Qwen3.5 默认开，生成 `<think>` 污染 JSON，蒸馏用 `extra_body={"enable_thinking": False}` + regex 兜底。
4. **图片存 parquet 别 embed**。存路径 + `--no_embed`，DataLoader workers 读图，parquet 小 10×。
5. **迁移包分两份（code/data）**。代码包几 MB 便于频繁迭代；数据包几百 MB 一次性传输。
6. **DashScope 配额要看控制台**。API 返回不带 usage 头，本地只能按次数估算。
