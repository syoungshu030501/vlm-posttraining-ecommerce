# 迁移到高算力卡指南

本文档说明如何把本卡上已准备好的 **数据 + 代码** 迁移到高算力卡上跑训练。

---

## 一、本卡已完成的产物

| 类别 | 路径 | 说明 |
|---|---|---|
| 真实商品图 | `data/raw/images/` | 2550 张 fashion 图片（来自 ModelScope `LouisXun/fashion_iten_dataset`），已压缩到 max 1024px / quality 88 |
| 原始标注 | `data/raw/annotations.jsonl` | 2550 条 `image_file` + `description`（含正常描述 + 极限词/材质虚标等违规触发文本） |
| SFT 蒸馏样本 | `data/sft/sft.jsonl` | DashScope `qwen-vl-max-latest` 蒸馏产物，字段：`image_file / prompt / response(JSON) / violation / category` |
| 偏好对 | `data/preference/preference.jsonl` | temperature 0.1 vs 1.2 的对比对（用于 RM） |
| RAG 规则库 | `data/raw/rules.jsonl` | 20 条合规规则，覆盖服装/食品/化妆品/电子/家居/图片规范 |
| RAG 违规案例 | `data/raw/violation_cases.jsonl` | 500 条违规场景（规则匹配验证） |
| 代码 | `src/`, `scripts/` | 全套 Stage 0~4 实现 |

---

## 二、打包

```bash
bash scripts/pack_for_migration.sh
# 产物:
#   dist/ecommerce-audit-code.tar.gz   (<5MB)
#   dist/ecommerce-audit-data.tar.gz   (~400MB)
#   dist/MANIFEST.txt
```

或只打包数据/代码：
```bash
bash scripts/pack_for_migration.sh --code-only
bash scripts/pack_for_migration.sh --data-only
```

---

## 三、迁移后，在目标卡上执行

### 1. 解包
```bash
mkdir -p ~/ecommerce-audit && cd ~/ecommerce-audit
tar -xzf /path/to/ecommerce-audit-code.tar.gz
tar -xzf /path/to/ecommerce-audit-data.tar.gz
```

### 2. 安装依赖（uv 优先）
```bash
# 创建 venv
uv venv --python 3.11 .venv
source .venv/bin/activate

# 核心训练依赖
uv pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple/ \
    torch torchvision transformers>=4.45 accelerate>=0.34 \
    peft>=0.12 bitsandbytes trl datasets pandas pyarrow pillow \
    deepspeed flash-attn --no-build-isolation

# RAG 依赖
uv pip install faiss-gpu rank-bm25 openai modelscope tqdm

# 可选: vLLM 用于 FIPO rollout
uv pip install vllm
```

> **如果 uv 装不上 flash-attn/bitsandbytes/vllm**，退回 `pip install --break-system-packages ...`。

### 3. 下载模型（目标卡上做，~18GB）
```bash
bash scripts/download_models.sh
# 产物:
#   models/pretrained/Qwen3-VL-8B-Instruct      (~16GB)
#   models/pretrained/clip-vit-base-patch32     (~600MB)
#   models/pretrained/bge-m3                    (~2GB)
```

### 4. 数据后处理（JSONL → Parquet）
本卡已产出 `sft.jsonl` / `preference.jsonl`，目标卡只需转 parquet 即可：

```bash
# SFT: jsonl → parquet + train/val/test 划分
python -m src.utils.data_prep \
    --annotation_file data/sft/sft.jsonl \
    --image_dir data/raw/images \
    --out_dir data/sft --mode sft --split --no_embed

# 偏好对: jsonl → parquet
python -m src.utils.data_prep \
    --annotation_file data/preference/preference.jsonl \
    --image_dir data/raw/images \
    --out_dir data/preference --mode preference --no_embed

# 幻觉三元组（SupCon + Triplet loss 用）
python -m src.utils.build_triplets \
    --annotation_file data/sft/sft.jsonl \
    --image_dir data/raw/images \
    --out_file data/sft/triplets.parquet
```

> `--no_embed` 让 parquet 只存图片路径而非 bytes，更轻、读取更快（DataLoader 里按需加载）。

### 5. 构建 RAG 索引
```bash
python -m src.stage4_rag.indexer \
    --image_dir data/raw/images \
    --rule_file data/raw/rules.jsonl \
    --out_dir data/rag_index
# 产物:
#   data/rag_index/visual.faiss        (CLIP 特征 FAISS)
#   data/rag_index/image_paths.pkl
#   data/rag_index/bm25.pkl
```

### 6. 训练全流程

```bash
# 全链路 (Stage 0 已由本卡完成，跳过 distill)
SKIP_DISTILL=1 bash scripts/run_pipeline.sh

# 分阶段跑
STAGE_START=1 STAGE_END=1 bash scripts/run_pipeline.sh   # SFT
STAGE_START=2 STAGE_END=2 bash scripts/run_pipeline.sh   # RM
STAGE_START=3 STAGE_END=3 bash scripts/run_pipeline.sh   # FIPO
STAGE_START=4 STAGE_END=4 bash scripts/run_pipeline.sh   # RAG index
```

默认配置：
- Base: `models/pretrained/Qwen3-VL-8B-Instruct`
- LoRA: attention + visual projection
- SFT: epochs=3, batch=1, grad_accum=16
- FIPO: Future-KL token-level weighting，vLLM rollout

### 7. 评估
```bash
python scripts/evaluate.py \
    --model_path models/rl_ckpt/latest \
    --test_parquet data/sft/test.parquet \
    --out results/final.json
```

---

## 四、需要继续蒸馏更多数据？

本卡只蒸馏了 500 SFT + 200 pref（控制 API 成本）。想在目标卡再跑一批：

```bash
export DASHSCOPE_API_KEY="sk-1a444cab439a452cb5cb78d8a208521d"

# 再产 2000 SFT（继续写入同一文件，--resume 自动跳过已处理）
python -m src.stage0_distill.distill \
    --image_dir data/raw/images \
    --annotation_file data/raw/annotations.jsonl \
    --out_dir data/sft --mode sft --model qwen-vl-max-latest \
    --max_samples 2500 --rate_limit 8 --resume
```

---

## 五、显存估算（Qwen3-VL-8B）

| 阶段 | 单卡最低显存 | 推荐 |
|---|---|---|
| Stage 1 SFT (LoRA + flash-attn + bf16) | 24GB | 40GB+ |
| Stage 2 RM (LoRA) | 24GB | 40GB+ |
| Stage 3 FIPO (LoRA + vLLM rollout) | 40GB | 80GB × 2 |
| Stage 4 推理 + RAG | 16GB | 24GB |

---

## 六、校验（迁移到目标卡后）

```bash
# 快速检查数据
python - <<'PY'
import json, os
def count(p): return sum(1 for _ in open(p)) if os.path.exists(p) else 0
print(f"images      : {len(os.listdir('data/raw/images'))}")
print(f"annotations : {count('data/raw/annotations.jsonl')}")
print(f"sft         : {count('data/sft/sft.jsonl')}")
print(f"preference  : {count('data/preference/preference.jsonl')}")
print(f"rules       : {count('data/raw/rules.jsonl')}")
print(f"cases       : {count('data/raw/violation_cases.jsonl')}")
PY
```

期望输出（本卡打包时的基线）：
```
images      : 2550
annotations : 2550
sft         : 500     # 实际会随 distill 进度变化
preference  : 200
rules       : 20
cases       : 500
```
