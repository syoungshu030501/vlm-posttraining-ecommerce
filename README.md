# VLM-posttraining

面向**视觉-语言过程奖励模型 (Vision-Language Process Reward Model, PRM)** 的科研代码库。

本仓库当前为活跃开发分支，目标是基于
[VisualPRM400K](https://huggingface.co/datasets/OpenGVLab/VisualPRM400K) 构建
训练流程、并在
[VisualProcessBench](https://huggingface.co/datasets/OpenGVLab/VisualProcessBench)
上进行 step-level 评测。

> **历史版本（电商商品合规审核五阶段流水线）已冻结归档于
> [`legacy/`](legacy/) 目录**，参见 [legacy/README.md](legacy/README.md)。

---

## 目录

- [一、项目定位](#一项目定位)
- [二、目录结构](#二目录结构)
- [三、代码文件说明](#三代码文件说明)
- [四、数据与模型清单](#四数据与模型清单)
- [五、硬件与软件环境](#五硬件与软件环境)
- [六、当前进展](#六当前进展)
- [七、教程：安装与基础用法](#七教程安装与基础用法)
- [八、与旧版（legacy/）的关系](#八与旧版legacy的关系)

---

## 一、项目定位

### 1.1 任务

给定一条 (图像, 多轮推理链) 数据，由模型在**每一步推理**上输出一个标量奖励，
反映"该步推理过程是否正确"。这与传统 Outcome RM 仅在终点位置打分不同：
PRM 把奖励信号下沉到每个推理 step，使其可作为思维链生成的 **step-level 评判者**。

### 1.2 训练目标

采用 **Math-Shepherd 风格的 masked-MSE 回归**：
- 训练集来自 `VisualPRM400K-v1.1-Raw`，每个 step 自带 Monte-Carlo 期望准确率
  `mc_i ∈ [0, 1]`；
- 模型对 step `i` 内的所有 token 做 mean-pool，得到一个标量预测
  `pred_i`；
- 损失 = `MSE(pred_i, mc_i)`，仅对有效 step 求平均。

相对 Bradley-Terry 对比损失（仍保留在 `src/prm/model.py:prm_bt_loss`），
masked-MSE 给出 **每个 step 一个标量监督**的更密集信号，且数据侧无需
构造 chosen/rejected 配对。

### 1.3 评测

`VisualProcessBench`：2866 条样本，26950 个 step-level 二元正确性标签；
对模型预测做阈值扫描后报告 step-level F1 / Precision / Recall，
并按 math / science / chart / general 子源分桶。

---

## 二、目录结构

```
VLM-posttraining/
├── README.md                       # 本文档
├── requirements.txt                # 沿用 legacy 依赖（待新代码稳定后精简）
├── .gitignore
│
├── src/                            # 新版活跃代码
│   ├── __init__.py
│   ├── prm/                        # ProcessRewardModel 与损失
│   │   ├── __init__.py             # 公开 ProcessRewardModel / prm_bt_loss
│   │   ├── model.py                # PRM 网络主体（≈130 行）
│   │   └── losses.py               # masked-MSE 损失（≈80 行，含 self-test）
│   └── utils/                      # 与 backbone 无关的通用工具
│       ├── __init__.py
│       ├── model_loader.py         # Qwen2.5-VL / Qwen3-VL 自动适配 + LoRA
│       ├── merge_lora.py           # 把 PEFT adapter 合并回 base 权重
│       ├── tracking.py             # 可选 SwanLab 训练日志
│       └── json_utils.py           # VLM 输出的容错 JSON 抽取
│
├── data/                           # NFS 链接，git 忽略
├── models/                         # 权重目录，git 忽略
├── logs/, outputs/, swanlog/       # 训练副产物，git 忽略
│
└── legacy/                         # 旧版完整流水线（冻结，不再开发）
    ├── README.md                   # 旧版入口文档
    ├── README_orig.md              # 旧版项目最终 README（保留供溯源）
    ├── STAGE2_V3_RUNBOOK.md        # 旧版 Stage 2 v3（field-wise PRM）方案
    ├── src/                        # 五阶段源码（stage0~4 + utils + schema）
    ├── scripts/                    # 数据 / 训练 / 评估脚本
    ├── configs/                    # Hydra 配置
    ├── docs/                       # 数据工程档案 + 历史参考文档
    ├── reference/                  # 设计文档与综述笔记
    ├── results/                    # 全部旧版实验报告 JSON
    ├── requirements.txt
    ├── sitecustomize.py            # 仅用于复现 Stage 3 FIPO
    └── vendor/                     # FIPO-main（verl-latest 需自行 clone）
```

---

## 三、代码文件说明

仅列出 `src/` 下的活跃代码；`legacy/` 内每个文件的职责详见
[legacy/README.md](legacy/README.md)。

### 3.1 `src/prm/model.py` — Process Reward Model

定义 `ProcessRewardModel`：在一个**冻结**的 VLM backbone 上加一个轻量
标量头，对响应每个 token 位置都输出一个奖励分数。

```
LayerNorm(hidden) → Linear(hidden, hidden//4) → GELU → Dropout(0.1)
    → Linear(hidden//4, 1)
```

构造时即冻结 backbone；若需要训练 LoRA，应在外层（训练脚本中）显式
解冻 `lora_` 参数。

同文件还包含 `prm_bt_loss`：把响应区间内 token 奖励 mean-pool 后做
Bradley-Terry log-sigmoid，留作未来对比实验的备用损失。

### 3.2 `src/prm/losses.py` — 掩码 MSE 损失（核心训练损失）

```python
prm_mse_loss(token_rewards, step_spans, step_targets, step_valid)
```

- 输入：`token_rewards (B, T)`、每 step 的 `[start, end)` 跨度
  `step_spans (B, S, 2)`、目标 `mc_i (B, S)`、有效掩码 `(B, S)`。
- 行为：按 step 跨度对 token 奖励做 mean-pool → 与 `mc_i` 做 MSE →
  按有效 step 计数取平均。
- 文件末尾自带 `__main__` 自检（随机张量验证梯度可达 token_rewards），
  可直接 `python -m src.prm.losses` 运行。

### 3.3 `src/utils/model_loader.py` — 模型加载与 LoRA 装配

`load_model_and_processor(base_path, apply_lora=True, ...)`：
- 自动识别 Qwen2.5-VL / Qwen3-VL 家族，选择正确的
  `Auto*` 类与 attention 实现（`flash_attention_2 → sdpa → eager` 回退）；
- 若是 Qwen3.5 系列自动关闭 thinking 模式；
- 默认 `DEFAULT_LORA_CONFIG`：`r=32, alpha=64, dropout=0.05`，
  target_modules 覆盖 LM 注意力 + LM MLP + 视觉→文本 merger。

### 3.4 `src/utils/merge_lora.py`

将 PEFT adapter 合并回 base safetensors，便于 RL / 评估时直接加载
完整权重（避免 PEFT runtime 依赖）。

### 3.5 `src/utils/tracking.py`

封装 SwanLab：当且仅当环境变量 `SWANLAB_API_KEY` 存在时启用，
否则降级为 no-op。

### 3.6 `src/utils/json_utils.py`

宽容地从 VLM 输出中抽取 JSON 对象，处理：
- `markdown ```json ``` ` 围栏；
- 末尾多余字符；
- 单引号 / 尾逗号修正。

---

## 四、数据与模型清单

### 4.1 训练 / 评测数据

| 数据集 | HF id | 用途 | 缓存位置 |
|---|---|---|---|
| VisualPRM400K-v1.1-Raw | `OpenGVLab/VisualPRM400K-v1.1-Raw` | PRM 训练（含 `mc_i ∈ [0,1]` 软标签） | `/mnt/nfs/young/VLM-posttraining/data/visualprm400k_raw_cache/`（NFS） |
| VisualProcessBench | `OpenGVLab/VisualProcessBench` | 评测（step-level 二元正确性） | `/mnt/nfs/young/VLM-posttraining/data/vpb_cache/`（NFS） |

> 数据集首次拉取需 ≥ 200 GB 磁盘空间（解压后图像 + 标注）。

### 4.2 基础模型

| 模型 | 路径 | 大小 | 用途 |
|---|---|---|---|
| Qwen3-VL-8B-Instruct | `/mnt/nfs/young/VLM4reasoning/models/pretrained/Qwen3-VL-8B-Instruct` | ~17 GB | PRM backbone |

### 4.3 训练产物（计划）

| 产物 | 计划路径 | 说明 |
|---|---|---|
| LoRA adapter | `models/prm_v1/adapter/` | backbone 上的 LoRA 增量 |
| PRM head | `models/prm_v1/reward_head.pt` | 标量奖励头 state_dict |
| optimizer / scheduler | `models/prm_v1/optim_step{N}.pt` | 断点续训 |

---

## 五、硬件与软件环境

### 5.1 硬件

| 资源 | 配置 |
|---|---|
| GPU | 8 × NVIDIA L20（每卡 46 GB） |
| **GPU 黑名单** | **GPU0 持续 ECC 错误，禁止使用**；可用集合 = `{1,2,3,4,5,6,7}` |
| 实际可用并行度 | 7 GPU（训练 / 评测 launcher 必须 `export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7`） |

GPU0 的硬件限制对所有新代码硬性生效：训练入口必须显式设置
`CUDA_VISIBLE_DEVICES`，`accelerate` 的 `num_processes`、`torchrun` 的
`--nproc_per_node` 一律按 7 配置，单卡冒烟可任选 1-7 中任意一张。

### 5.2 软件

沿用 legacy 的依赖快照 [`requirements.txt`](requirements.txt)：

| 组件 | 版本 |
|---|---|
| Python | 3.12 |
| torch | 2.10.x + CUDA 12.4 wheel |
| transformers | 5.5.x |
| peft | 0.18+ |
| accelerate | 待加入（新版训练循环依赖） |
| datasets | 待加入 |
| flash_attn | 2.8.3+cu12torch2.10（可选） |

当前 `requirements.txt` 较新代码所需更宽，等新版训练 / 评测落地后会
精简成独立列表。

---

## 六、当前进展

### 6.1 已完成

| 模块 | 文件 | 状态 |
|---|---|---|
| Process Reward Model 主体 | [src/prm/model.py](src/prm/model.py) | ✅ |
| masked-MSE 损失 + 自检 | [src/prm/losses.py](src/prm/losses.py) | ✅（self-test 通过） |
| backbone 加载与 LoRA 装配 | [src/utils/model_loader.py](src/utils/model_loader.py) | ✅（沿用 legacy） |
| 旧版完整流水线归档 | [legacy/](legacy/) | ✅ |

### 6.2 进行中 / 待办

| 模块 | 计划文件 | 状态 |
|---|---|---|
| VisualPRM400K-v1.1-Raw 下载 + schema 探针 | `scripts/prepare_visualprm400k.py` | 已起草，待运行 |
| VisualPRM400K 数据加载器（multi-turn → step spans） | `src/data/visualprm400k.py` | 待写 |
| VisualProcessBench 数据加载器 | `src/data/visualprocessbench.py` | 待写 |
| accelerate 训练循环（LoRA + PRM head） | `src/training/train_prm.py` | 待写 |
| step-level F1 评测脚本 | `src/evaluation/visualprocessbench.py` | 待写 |
| 启动脚本（含 `CUDA_VISIBLE_DEVICES=1..7`） | `scripts/{train_prm,smoke_train,eval_prm}.sh` | 待写 |

详细路线图见
[legacy/STAGE2_V3_RUNBOOK.md](legacy/STAGE2_V3_RUNBOOK.md)（field-wise PRM 设计思路）以及
[legacy/reference/data-redesign-2026.md](legacy/reference/data-redesign-2026.md)。

---

## 七、教程：安装与基础用法

### 7.1 环境安装

```bash
# 推荐 conda
conda create -n VLM python=3.12 -y
conda activate VLM
pip install -r requirements.txt
```

### 7.2 快速校验：损失函数自检

```bash
# 跑 src/prm/losses.py 的 __main__ self-test，确认 masked-MSE 反向梯度正常
python -m src.prm.losses
# 预期：prm_mse_loss self-test OK: loss=1.1651, |grad|=0.5151
```

### 7.3 快速校验：PRM 模型实例化

```python
from src.prm import ProcessRewardModel, prm_bt_loss
from src.utils.model_loader import load_model_and_processor

base, processor = load_model_and_processor(
    "/mnt/nfs/young/VLM4reasoning/models/pretrained/Qwen3-VL-8B-Instruct",
    apply_lora=True,
)
prm = ProcessRewardModel(base)

# 训练前需重新解冻 LoRA 参数（ProcessRewardModel 构造时冻结了整个 backbone）
for n, p in prm.backbone.named_parameters():
    if "lora_" in n:
        p.requires_grad = True
```

### 7.4 GPU 选卡（强制）

任何启动训练 / 评估的命令都需要先排除 GPU0：

```bash
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7   # 多卡训练
# 或单卡冒烟：
export CUDA_VISIBLE_DEVICES=1
```

也可在脚本顶部加一行兜底：

```python
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2,3,4,5,6,7")
```

### 7.5 后续完整训练流程（待新代码落地后启用）

```bash
# 1. 下载并探针 VisualPRM400K schema
python scripts/prepare_visualprm400k.py --n-shards 3 --rows-per-shard 200

# 2. 单卡冒烟训练（64 样本，50 step）
bash scripts/smoke_train.sh

# 3. 7 卡完整训练
bash scripts/train_prm.sh

# 4. VisualProcessBench 评测
bash scripts/eval_prm.sh --ckpt models/prm_v1/final
```

---

## 八、与旧版（legacy/）的关系

`legacy/` 完整保留了项目最初的 **电商商品合规审核五阶段流水线**
（Stage 0 数据蒸馏 → Stage 1 SFT + LoRA → Stage 2 Bradley-Terry RM →
Stage 3 FIPO/GRPO RL → Stage 4 CLIP+FAISS RAG）。该方向因数据来源受限
（自建数据集质量瓶颈）于 2026-04 暂停开发，现作为**已冻结的归档版本**保留。

| 用途 | 路径 |
|---|---|
| 旧版入口与复现指南 | [legacy/README.md](legacy/README.md) |
| 旧版项目最终 README（完整方法 / 数字 / 踩坑） | [legacy/README_orig.md](legacy/README_orig.md) |
| 旧版 Stage 2 v3（field-wise PRM）方案 | [legacy/STAGE2_V3_RUNBOOK.md](legacy/STAGE2_V3_RUNBOOK.md) |
| 数据工程历史档案 | [legacy/docs/DATA_ENGINEERING.md](legacy/docs/DATA_ENGINEERING.md) |
| 数据重设计与综述参考 | [legacy/reference/](legacy/reference/) |
| 所有旧实验报告 | [legacy/results/](legacy/results/) |
| 旧版逐次运行调试史 | [legacy/results/runs.md](legacy/results/runs.md) |

> 新版代码不依赖 `legacy/` 中的任何模块；只有 `src/utils/` 下与 backbone
> 无关的若干工具是从 `legacy/src/utils/` 直接复用而来。复现旧实验请
> 严格在 `cd legacy/` 后再运行命令，详见 [legacy/README.md](legacy/README.md)。
