# TFI - 图像伪造检测项目 (Text & Forgery Investigation)

## 一、项目概述

本项目实现一个完整的图像伪造检测系统，完成三个子任务：

1. **伪造判别 (Task 1)**: 判断图片是否经过伪造，输出 label=0(真实) / label=1(伪造)
2. **伪造定位 (Task 2)**: 像素级定位伪造区域，输出 COCO RLE 格式的二值 mask
3. **可解释分析 (Task 3)**: 生成详细的中文鉴定分析文本，说明判断依据

最终输出为 `submit.csv`，格式：`image_name, label, location, explanation`

---

## 二、硬件与软件环境

- **GPU**: 8x AMD Instinct MI325X (每卡 256GB HBM3, 共 2TB)
- **CPU RAM**: 3TB
- **GPU 架构**: ROCm 7.1
- **推理限制**: 单卡 48GB 显存

| 包 | 版本 |
|---|------|
| PyTorch | 2.11.0.dev20260216+rocm7.1 |
| transformers | 5.2.0 |
| peft | 0.18.0 |
| deepspeed | 0.18.6 |
| accelerate | 1.9.0 |

---

## 三、数据分析

### 3.1 原始数据结构

```
train/
├── Black/          # 伪造图片 (800 张)
│   ├── Image/      # .jpg/.png 图片 (尺寸不一, 512x512 ~ 4961x7016)
│   ├── Mask/       # .png 二值 mask (0/255, 与图片同尺寸)
│   └── Caption/    # .md 中文分析文本 (260~1320 字)
└── White/          # 真实图片 (200 张)
    ├── Image/      # .jpg 图片
    └── Caption/    # .md 中文分析文本 (290~919 字)
test/
└── Image/          # 500 张测试图片 (无标签)
```

### 3.2 关键数据特征

- **图像尺寸差异大**: 从 512x512 到 4961x7016，训练时统一 resize
- **Mask 格式**: 灰度图, 0=真实区域, 255=伪造区域; 与图片完全同尺寸
- **Caption 内容**: Black 类包含坐标 `[x1, y1, x2, y2]` 和详细视觉/逻辑分析; White 类描述真实性论证
- **RLE 格式**: COCO 标准 RLE, `{"size": [H, W], "counts": "..."}`
- **类别不平衡**: Black:White = 4:1

---

## 四、系统架构

### 4.1 总体流水线

```
训练阶段 (8x MI325X 全部可用):
  ┌──────────────────────────────┐
  │ Step 1: 分割集成训练           │ ← 3架构 x 5折 = 15 个模型
  │ Step 2: 分类器训练             │ ← EfficientNet-V2-L x 5折
  │ Step 3: 397B 教师模型 LoRA 微调 │ ← LoRA r=128, 8 卡 DeepSpeed ZeRO-3
  │ Step 4: 教师生成增强数据        │ ← 多温度采样 + thinking 推理链
  │ Step 5: 8B 学生模型微调         │ ← 全量微调, 原始+增强数据 (CoT 蒸馏)
  └──────────────────────────────┘

推理阶段 (单卡 ≤ 48GB):
  测试图片
    → 阶段1: 分割集成 + TTA → mask → label + RLE        (~3GB 峰值)
    → 阶段1.5: 分类器投票 → 修正 label                    (~0.5GB 峰值)
    → 阶段2: Qwen3-VL-8B-Thinking → explanation          (~25GB 峰值)
    → 输出: submit.csv
```

### 4.2 核心设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 分割输入 | RGB + ELA + SRM (7通道) | 利用图像取证领域经典特征, 暴露压缩/噪声不一致 |
| 分割集成 | 3架构 x 5折 = 15模型 | 架构多样性 + 数据多样性, 最大化集成增益 |
| 教师模型 | Qwen3.5-397B-A17B | 最新开源 VLM, 原生视觉早期融合, DeltaNet+MoE 混合架构 |
| 教师训练 | LoRA r=128 (非全量) | 全量需 800GB/卡放不下; LoRA r=128 差距 <1% |
| 知识蒸馏 | CoT 蒸馏 (保留 thinking) | 学生不仅学结论, 还学推理过程 |
| 学生模型 | Qwen3-VL-8B-Thinking 全量微调 | BF16 ~16GB, 远低于 48GB 限制; 全量微调最大吸收 |
| 损失函数 | Focal + Dice + Boundary | 三重损失: 不平衡 + 区域 + 边缘, 各司其职 |

---

## 五、模型清单

| 模型 | 来源 | 用途 | 大小 |
|------|------|------|------|
| SegFormer-B5 | nvidia/segformer-b5-finetuned-ade-640-640 | 分割 backbone | ~340 MB |
| ConvNeXt-V2-Large | timm (fcmae_ft_in22k_in1k_384) | 分割 backbone | 749 MB |
| MaxViT-Large | timm (in21k_ft_in1k) | 分割 backbone | 806 MB |
| EfficientNet-V2-L | timm (in21k_ft_in1k) | 分类 backbone | 450 MB |
| Qwen3-VL-8B-Thinking | Qwen/Qwen3-VL-8B-Thinking | 学生 VLM | ~16 GB |
| Qwen3.5-397B-A17B | Qwen/Qwen3.5-397B-A17B | 教师 VLM | ~800 GB |

### 5.1 Qwen3.5-397B-A17B 详细架构

| 属性 | 值 |
|------|---|
| 架构类型 | `qwen3_5_moe` (Gated DeltaNet + Gated Attention + MoE) |
| 总参数 | 397B (403B 含 embeddings) |
| 活跃参数/token | 17B |
| 层数 | 60 (15 blocks x 4 layers) |
| 层布局 | 每 block: 3x DeltaNet→MoE + 1x Attention→MoE |
| 专家数 | 512 个路由专家 + 1 个共享专家 |
| 每 token 激活专家 | 10 个路由 + 1 个共享 |
| 隐藏维度 | 4096 |
| 注意力头数 | 32 (Q), 2 (KV) — GQA |
| DeltaNet 头数 | 16 (QK), 64 (V) |
| 上下文长度 | 262,144 (原生) |
| 视觉编码器 | ViT-27层, patch_size=16 |
| 视觉融合 | 早期融合 (early fusion on multimodal tokens) |

---

## 六、代码文件详解

### 6.1 `utils.py` — 工具函数库

| 函数 | 功能 |
|------|------|
| `compute_ela(image)` | Error Level Analysis, JPEG 重压缩差值, 暴露伪造区域压缩不一致 |
| `compute_srm(image)` | Spatial Rich Model 噪声残差, 8 个高通滤波核提取噪声特征 |
| `mask_to_rle(mask)` | 二值 mask → COCO RLE 编码 |
| `rle_to_mask(rle)` | RLE → numpy 二值 mask |
| `compute_iou/dice/f1()` | 评估指标 |
| `postprocess_mask()` | 形态学开闭运算 + 连通域面积过滤 |
| `describe_mask_region(mask)` | 生成区域自然语言描述 |

### 6.2 `dataset.py` — 数据集模块

| 类 | 用途 | 输入 |
|----|------|------|
| `ForgerySegDataset` | 分割训练 | 7 通道 (RGB+ELA+SRM), 输出 mask+label |
| `ForgeryClsDataset` | 分类训练 | 6 通道 (RGB+ELA), 输出 label |
| `TestImageDataset` | 推理 | 7 通道, 无标签 |
| `VLMSFTDataset` | VLM 微调 | (image_path, caption) 对话格式 |
| `create_kfold_splits()` | K-Fold 分割 | 分层分折, 保持类别比例 |

### 6.3 `train_seg_ensemble.py` — 分割集成训练

三种架构 x 5 折 = 15 个模型。

| 架构 | 参数量 | 解码器 | 特点 |
|------|--------|--------|------|
| SegFormer-B5 | ~85M | MLP Head | 层级 Transformer, 收敛快 |
| ConvNeXt-V2-L | ~235M | DeepLabV3+ (ASPP) | 强局部纹理, 多尺度空洞卷积 |
| MaxViT-L | ~212M | FPN | Multi-axis attention, 全局+局部 |

**损失函数**: 0.4*Focal + 0.4*Dice + 0.2*Boundary

**训练超参**: 768x768 输入, batch=4, AdamW lr=6e-5, OneCycleLR, 100 epochs, early stopping patience=15, BF16

### 6.4 `train_classifier.py` — 分类器训练

EfficientNet-V2-L (6 通道 RGB+ELA), 512x512, 5-fold, 类别权重 [1.0, 0.25]

### 6.5 `train_teacher.py` — 教师模型 LoRA 微调

对 Qwen3.5-397B-A17B 进行 LoRA 微调, 使用 DeepSpeed ZeRO-3 在 8 卡上分布式训练。

**LoRA 配置**: r=128, alpha=256, dropout=0.05, target_modules: q/k/v/o_proj + gate/up/down_proj

### 6.6 `merge_lora.py` — LoRA 权重合并

将 LoRA 适配器合并到基座模型, 输出独立完整模型用于推理。

### 6.7 `generate_teacher_data.py` — 教师增强数据生成

用微调后的教师模型对 1000 张训练图片生成分析文本。每张图 3 个温度 (0.7, 0.9, 1.1) 各生成一个版本, 保留 thinking 推理链, 支持断点续传和数据分片。

### 6.8 `train_student_8b.py` — 学生模型全量微调

Qwen3-VL-8B-Thinking 全量微调 (非 LoRA), 训练数据 = 原始 Caption + 教师增强 Caption (含 thinking)。

### 6.9 `inference.py` — 完整推理流水线

阶段 1: 15 个分割模型集成 + 4x TTA → mask → label + RLE
阶段 1.5: 5 个分类器投票 → 修正 label
阶段 2: Qwen3-VL-8B-Thinking → 中文鉴定分析文本

### 6.10 `rocm_compat.py` — ROCm 兼容性修复

修复 `torch._grouped_mm` 在 AMD MI325X 上的崩溃问题, 替换为逐专家顺序矩阵乘法。

### 6.11 `test_inference.py` — 推理验证脚本

验证 Qwen3.5-397B-A17B 模型加载和推理, 支持 base 模型和 LoRA 微调后模型的对比测试。

### 6.12 `split_train_val.py` — 数据集划分

固定种子 42, 8:2 分割, 符号链接, 自动验证无重叠无遗漏。

### 6.13 `ds_config_z3.json` — DeepSpeed ZeRO-3 配置

BF16, ZeRO Stage 3, micro_batch=1, grad_accum=4, overlap_comm。

---

## 七、训练结果详细记录

### 7.1 分割集成训练 (15 模型)

**训练环境**: 3 种架构分别在 GPU 0/1/2 上并行训练, 每种架构训练 5 折

| 指标 | SegFormer-B5 | ConvNeXt-V2-L | MaxViT-L |
|------|-------------|---------------|----------|
| GPU 占用 | 1 卡, ~12 GB | 1 卡, ~18 GB | 1 卡, ~16 GB |
| 每 epoch 耗时 | ~41 秒 | ~73 秒 | ~41 秒 |
| 5 折总训练时间 | ~5.7 小时 | ~10 小时 | ~5.7 小时 |
| fold0 最佳 IoU | 0.5648 | 0.4220 | 0.4798 |
| fold0 最佳 Dice | 0.6312 | 0.5007 | 0.5547 |
| fold0 分类 Acc | 0.8125 | 0.7688 | 0.7750 |
| 单模型权重大小 | 324 MB | 887 MB | 770 MB |

**产出**: 15 个 `best_model.pt`, 总计 ~9.3 GB, 保存在 `checkpoints/seg/`

### 7.2 分类器训练 (5 模型)

**训练环境**: GPU 3 上训练, 与分割训练并行

| 指标 | 值 |
|------|---|
| GPU 占用 | 1 卡, ~8 GB |
| 每 epoch 耗时 | ~10 秒 |
| 5 折总训练时间 | ~40 分钟 |
| 最佳 F1 (fold2) | 0.9160 |
| Accuracy (fold2) | 0.8625 |
| 单模型权重大小 | 450 MB |

**产出**: 5 个 `best_model.pt`, 总计 ~2.2 GB, 保存在 `checkpoints/cls/`

### 7.3 教师模型 LoRA 微调 (Qwen3.5-397B-A17B)

**训练环境**: 8x MI325X, DeepSpeed ZeRO-3, 训练两轮

| 指标 | 第一轮 | 第二轮 (最终) |
|------|--------|-------------|
| 训练数据 | train_split (800 样本) | 全量 train (1000 样本) |
| 总 steps | 75 | 96 |
| 训练时长 | ~5 小时 | ~6 小时 |
| 每 step 耗时 | ~352 秒 | ~220 秒 |
| 初始 loss | 13.83 | - |
| 初始 grad_norm | 137.7 | - |

**显存占用 (每卡)**:

| 项目 | 大小 |
|------|------|
| 冻结模型 BF16 分片 (ZeRO-3, 397B/8卡) | ~92 GB |
| LoRA 参数 (198M) | ~0.4 GB |
| 优化器状态 (AdamW, FP32 for LoRA) | ~0.8 GB |
| 激活值 (梯度检查点) | ~15-20 GB |
| **每卡总计** | **~110 GB / 256 GB (43%)** |

**CPU RAM**: ~188 GB / 3 TB (6%)
**GPU 利用率**: 100% (8 卡全满)

**LoRA 参数统计**:
```
trainable params:   198,574,080
all params:     397,000,934,896
trainable%:              0.0500
```

**产出**: `checkpoints/teacher/adapter_model.safetensors` (**379 MB**), 含 `adapter_config.json`

### 7.4 LoRA 合并

使用 `merge_lora.py` 将 379 MB LoRA 权重合并到 800 GB 基座模型:

| 指标 | 值 |
|------|---|
| 加载基座模型 | ~5 分钟 (device_map=auto, 8 卡) |
| LoRA merge_and_unload | ~1 分钟 |
| 保存合并模型 | ~10 分钟 (121 shards) |
| 总耗时 | ~16 分钟 |
| 输出大小 | 740 GB (121 个 safetensors 分片) |
| 每卡显存 | ~100 GB (加载时) |

**产出**: `models/Qwen3.5-397B-A17B-teacher/`

### 7.5 增强数据生成

用合并后的教师模型对 1000 张训练图片生成增强分析文本:

| 指标 | 值 |
|------|---|
| 模型 | Qwen3.5-397B-A17B-teacher (合并后) |
| 加载方式 | device_map=auto, 8 卡 pipeline 并行 |
| 每卡显存 | ~134 GB / 256 GB (51%) |
| CPU RAM | ~85 GB / 3 TB (3%) |
| GPU 利用率 | ~15% (pipeline 并行固有限制) |
| 输入图片 | 1000 张 (800 Black + 200 White) |
| 每张图生成版本 | 3 个 (temperature 0.7, 0.9, 1.1) |
| 每图生成耗时 | ~214 秒 (含 3 个版本) |
| max_new_tokens | 2048 |
| 总生成条数 | 3000 条 |
| 实际总耗时 | **59.5 小时** |
| 输出格式 | JSONL, 逐条写入, 支持断点续传 |

**显存分布**:

| GPU | 显存占用 |
|-----|---------|
| GPU 0 | 134 GB |
| GPU 1 | 134 GB |
| GPU 2 | 134 GB |
| GPU 3 | 133 GB |
| GPU 4 | 142 GB |
| GPU 5 | 95 GB |
| GPU 6 | 134 GB |
| GPU 7 | 134 GB |
| **合计** | **~1040 GB / 2097 GB** |

**产出**: `augmented_data/train/augmented_captions.jsonl` (3000 条, 7.9 MB)

**数据特点**: 保留完整 thinking 推理链 (`<think>...</think>` + 最终分析), 用于 CoT 蒸馏

### 7.6 学生模型全量微调 (Qwen3-VL-8B-Thinking)

**训练环境**: 8x MI325X, 数据并行 (每卡一份 8B 模型副本), accelerate 启动

**训练命令**:
```bash
accelerate launch --num_processes 8 train_student_8b.py \
    --model_name models/Qwen3-VL-8B-Thinking \
    --data_dir train \
    --augmented_dir augmented_data/train \
    --epochs 5 --lr 2e-5
```

**训练数据**: 原始 1000 条 Caption + 教师增强 3000 条 (含 thinking) = **~4000 条**

| 指标 | 值 |
|------|---|
| 训练方式 | 全量微调 (非 LoRA), 8 卡数据并行 |
| 总 steps | 315 (4000 / 8 GPU / 1 batch / 8 grad_accum x 5 epochs) |
| 每 step 耗时 | ~27-35 秒 |
| 预计总训练时长 | ~2.5-3 小时 |
| 初始 loss | 16.76 |
| step 14 loss | 13.54 (已开始下降) |

**显存占用 (每卡)**:

| 项目 | 大小 |
|------|------|
| 模型权重 BF16 | ~16 GB |
| 梯度 BF16 | ~16 GB |
| 优化器状态 (AdamW FP32) | ~32 GB |
| 激活值 (梯度检查点) | ~30-70 GB |
| **每卡总计** | **~95-140 GB / 256 GB** |

**GPU 利用率**: 100% (8 卡全满), 部分卡反向传播时峰值功耗达 950W

**Loss 曲线**:
```
step   5 | epoch 0.08 | loss 16.76  ← 起始
step  30 | epoch 0.48 | loss  4.32  ← 快速下降
step  60 | epoch 0.96 | loss  0.57  ← 第一轮结束
step  65 | epoch 1.03 | loss  0.45  ← 第二轮开始, 已收敛
step 125 | epoch 1.99 | loss  0.38  ← 趋平, 在此停止避免过拟合
```

**停止策略**: 在 2 epoch (step 126) 手动停止。loss 从 step 65 开始在 0.37-0.41 间波动不再下降, 继续训练有过拟合风险 (仅 4000 条数据对 8B 模型)。

**产出**: `checkpoints/student_8b/checkpoint-126/model.safetensors` (17 GB)

### 7.7 最终推理 (submit.csv 生成)

使用 `inference.py` 一键运行完整推理流水线, 单卡 GPU 0:

```bash
python inference.py
```

**阶段 1: 分割集成 + 多尺度 TTA**
- 15 个分割模型 (5 ConvNeXt-V2-L + 5 MaxViT-L + 5 SegFormer-B5) 逐个加载推理后释放
- **多尺度推理**: 576 + 768 + 960 三个分辨率, 概率图 resize 到统一大小后平均
- 每个模型在每个尺度做 4x TTA (原图 + 水平翻转 + 垂直翻转 + 旋转180°)
- 总推理次数: 15 模型 x 3 尺度 x 4 TTA = **180 次**
- 概率图平均 → 二值化 (阈值 0.3) → 形态学后处理 → 连通域过滤 → RLE 编码

**阶段 1.5: 智能分类器融合**
- 5 个 EfficientNet-V2-L 分类器, 计算 P(伪造) 均值
- **智能融合策略** (只在高置信度时干预, 避免偏置):
  - P(forged) < 0.2 且分割判伪造 → 翻转为真实
  - P(forged) > 0.9 且分割判真实 → 翻转为伪造
  - 中间区间不干预

**阶段 2: VLM 解释生成**
- 加载 Qwen3-VL-8B-Thinking 学生模型 (checkpoint-126, 2 epoch, loss 0.38)
- **增强版 prompt** (参考训练集 Caption 风格):
  - 要求包含篡改区域精确坐标 [x1,y1,x2,y2]
  - 要求视觉异常特征: 字体差异、边缘不自然、纹理断裂、JPEG 压缩伪影
  - 要求逻辑矛盾: 数学计算错误、日期不合理、品牌信息不存在
- 生成参数: temperature=0.3, top_p=0.9, **max_new_tokens=2048** (从 1024 提升)
- 去除 `</think>` 标签后写入 CSV

**输出**: `submit-20260221-v3.csv`

**提交版本对比**:

| 版本 | 分割阈值 | 多尺度 | 分类器 | Prompt | max_tokens | 伪造/真实 | 比赛得分 |
|------|---------|--------|--------|--------|-----------|----------|---------|
| v1 | 0.5 | 768 单尺度 | 加权投票 | 简单 | 1024 | 310/190 | **0.7502** |
| v2 | 0.3 | 768 单尺度 | 关闭 | 简单 | 1024 | 354/146 | **0.7841** |
| **v3** | **0.3** | **576+768+960** | **智能融合** | **增强** | **2048** | **372/128** | **0.7971** |
| v4 | 0.3 | 576+768+960 | 智能融合 | 增强(去坐标) | 2048 | 372/128 | **0.7947** |

**关键发现**:
- v1→v2: 降低分割阈值 (0.5→0.3) + 关闭分类器 → +0.0339，说明 v1 分类器过度干预损害了 label 准确率
- v2→v3: 多尺度推理 + 智能融合 + 增强 prompt → +0.0130，多维度同步优化的效果
- v3→v4: 去掉坐标后得分下降 0.0024，说明精确坐标对 explanation 评分有帮助
- v3 的 label 预测：372 伪造 / 128 真实 (74.4%)，训练集比例 80% 伪造 / 20% 真实
- 分类器 mean score=0.795, median=0.849, 分布偏高

**MaxViT 窗口约束修复 (v3)**:
- 576 尺度下 MaxViT 深层 feature map (18x18) 无法被 window_size=12 整除，导致 AssertionError
- 修复方案：MaxViT 只在 768 尺度推理，ConvNeXt/SegFormer 额外在 576+960 尺度推理
- 总推理组合：ConvNeXt(5) × 3 尺度 + MaxViT(5) × 1 尺度 + SegFormer(5) × 3 尺度 = 35 组


### 7.8 逐模型评估 (`evaluate.py`)

在训练集 (1000 张) 上评估每个模型的性能, 发现并修复问题:

**分割模型** (200 张采样, seg_thresh=0.3):

| 架构 | 单模型 Acc | 单模型 IoU | 说明 |
|------|-----------|-----------|------|
| ConvNeXt fold0 | 92.0% | 0.735 | |
| ConvNeXt fold1 | 93.5% | 0.776 | |
| ConvNeXt fold2 | 93.0% | 0.787 | 最佳 IoU |
| ConvNeXt fold3 | 96.0% | 0.770 | |
| ConvNeXt fold4 | 100% | 0.085 | **异常! 已重训 → IoU 0.70** |
| MaxViT fold0-4 | 92-93.5% | 0.65-0.71 | 整体弱, **已全部重训** |
| SegFormer fold0-4 | 88.5-94% | 0.73-0.75 | 稳定 |

**分类器** (200 张采样):

| 模型 | Acc | 说明 |
|------|-----|------|
| efficientnet_fold0 | 99.5% | 最佳 |
| efficientnet_fold1 | 91.0% | 最弱 |
| efficientnet_fold2 | 98.0% | |
| efficientnet_fold3 | 96.0% | |
| efficientnet_fold4 | 94.5% | |

**阈值扫描** (训练集, seg only, 无分类器):

```
   Acc  SegTh   LblTh   TP   FP   TN   FN  Pred1
0.9510   0.30  0.0010  756    5  195   44    761  ← 最优
0.9410   0.40  0.0010  746    5  195   54    751
0.9330   0.50  0.0010  738    5  195   62    743  ← v1 用的
```

### 7.9 模型重训 (2026-02-22)

基于逐模型评估发现的问题, 重训了异常和弱模型:

**convnext_fold4 重训** (原 IoU=0.085, 训练异常):
- 命令: `python train_seg_ensemble.py --arch convnext --fold 4 --gpu 0`
- 数据: `train/` (全量 1000 张, 因 train_split 目录已不存在)
- 最终: IoU ~0.70, Acc ~92%, 恢复正常
- 耗时: ~70 分钟 (85 epochs, early stopping)

**MaxViT 全部 5 折重训** (IoU 0.65-0.71, 最弱架构):
- 命令: 5 折并行在 GPU 1-5
- 数据: `train/` (全量)
- 耗时: ~70 分钟/折 (并行)

### 7.8 推理验证 (Qwen3.5-397B-A17B 教师模型)

使用 `test_inference.py` 验证模型加载和推理:

| 指标 | 值 |
|------|---|
| 模型加载时间 | 314.1 秒 (~5 分钟) |
| 分布方式 | device_map=auto, 8 卡 |
| 每卡显存 | ~100-110 GB |
| 纯文本速度 | 10.0 tok/s (512 tokens / 51s) |
| 图像理解速度 | 2.9 tok/s (512 tokens / 175s) |
| 伪造检测速度 | 9.7 tok/s (512 tokens / 53s) |

**测试结果**: 模型在伪造检测场景表现优秀, 能自动发现收据中的数学不一致 (单价 x 数量 != 总价), 进行专业级的视觉异常分析。

---

## 八、推理显存预算

顺序加载策略 (不同阶段不同时驻留):

| 阶段 | 峰值显存 |
|------|---------|
| 分割集成 (逐模型加载) | ~1-2 GB |
| 分类器 (逐模型加载) | ~0.5 GB |
| VLM 8B BF16 + KV Cache + 生成 | ~25 GB |
| **总峰值** | **~25 GB << 48 GB** |

---

## 九、性能提升技术汇总

| 技术 | 预期提升 | 适用任务 | 实现文件 |
|------|---------|---------|---------|
| 多流输入 RGB+ELA+SRM (7ch) | +3~5% IoU | 分割 | dataset.py |
| 三架构集成 (SegFormer+ConvNeXt+MaxViT) | +2~4% IoU | 分割 | train_seg_ensemble.py |
| 5-Fold 交叉验证 (15模型) | +1~2% IoU | 分割+分类 | train_seg_ensemble.py |
| TTA 4x (原图+翻转+旋转) | +1~2% IoU | 分割 | inference.py |
| **多尺度推理 (576+768+960)** | **+2~3% IoU** | 分割 | inference.py |
| 三重损失 Focal+Dice+Boundary | +1~2% IoU | 分割 | train_seg_ensemble.py |
| 后处理 (形态学+连通域) | +0.5~1% IoU | 分割 | utils.py |
| **逐模型评估 + 坏模型重训** | **+2~5% IoU** | 分割 | evaluate.py |
| 397B 教师 CoT 蒸馏 → 8B 学生 | +8~15% 生成质量 | 解释 | train_teacher.py |
| 学生全量微调 (非 LoRA) | +2~5% | 解释 | train_student_8b.py |
| Thinking 推理链保留 | +3~5% 分析深度 | 解释 | generate_teacher_data.py |
| **增强版 prompt (坐标+证据+逻辑)** | **+2~3% 文本质量** | 解释 | inference.py |
| **max_new_tokens 2048** | **+1~2% 文本质量** | 解释 | inference.py |
| 信息融合 (seg 结果→VLM prompt) | +2~3% 一致性 | 解释 | inference.py |
| **智能分类器融合 (高置信度干预)** | **+1~2% Acc** | 分类 | inference.py |
| 阈值扫描优化 (0.3 vs 0.5) | +2% Acc | 分类 | evaluate.py |

---

## 十、项目文件总览

```
/wekafs/datongxu/tfi/
├── README.md                    # 本文档
├── PROJECT.md                   # 项目内部详细记录 (含调试日志)
├── requirements.txt             # 依赖清单
├── ds_config_z3.json            # DeepSpeed ZeRO-3 配置
├── rocm_compat.py               # ROCm 兼容性修复 (grouped_mm)
│
├── split_train_val.py           # 数据集划分 (8:2, 符号链接)
├── utils.py                     # 工具函数 (ELA/SRM/RLE/指标/后处理)
├── dataset.py                   # 数据集类 (分割/分类/VLM SFT/K-Fold)
├── train_seg_ensemble.py        # 分割集成训练 (3架构 x 5折 = 15模型)
├── train_classifier.py          # 分类器训练 (EfficientNet-V2-L x 5折)
├── train_teacher.py             # 教师模型 LoRA 微调 (Qwen3.5-397B, DeepSpeed)
├── merge_lora.py                # LoRA 权重合并到基座模型
├── generate_teacher_data.py     # 教师增强数据生成 (含 thinking, 断点续传)
├── train_student_8b.py          # 学生模型全量微调 (Qwen3-VL-8B)
├── inference.py                 # 完整推理流水线 (分割→分类→VLM)
├── test_inference.py            # Qwen3.5 推理验证 (支持 LoRA 对比)
│
├── checkpoints/
│   ├── seg/                     # 15 个分割模型 (9.3 GB)
│   │   ├── segformer_fold{0-4}/best_model.pt
│   │   ├── convnext_fold{0-4}/best_model.pt
│   │   └── maxvit_fold{0-4}/best_model.pt
│   ├── cls/                     # 5 个分类器 (2.2 GB)
│   │   └── efficientnet_fold{0-4}/best_model.pt
│   ├── teacher/                 # 教师 LoRA 权重 (379 MB)
│   │   ├── adapter_model.safetensors
│   │   └── adapter_config.json
│   └── student_8b/              # 学生完整模型 (已完成)
│       └── checkpoint-126/
│           ├── model.safetensors   # 17 GB (推理用)
│           ├── config.json
│           └── trainer_state.json  # loss 曲线记录
│
├── augmented_data/
│   └── train/
│       └── augmented_captions.jsonl  # 3000 条增强数据 (7.9 MB, 已完成)
│
├── cache/                       # 推理中间结果缓存 (断点续传用)
│   ├── seg_results.json         # 分割集成结果
│   ├── cls_scores.json          # 分类器投票结果
│   ├── explanations.json        # VLM 生成文本 (逐条保存)
│   └── prob_maps/               # 500 个分割概率图 .npy (供 CRF 后处理)
│
├── evaluate.py                  # 训练集阈值扫描, 参数优化
│
├── submit_example.csv           # 提交格式示例
├── submit-20260221.csv          # v1 提交 (得分 0.7502)
├── submit-20260221-v2.csv       # v2 提交 (得分 0.7841)
├── submit-20260221-v3.csv       # v3 提交 (得分 0.7971, 当前最佳)
├── submit-20260222-v4.csv       # v4 提交 (得分 0.7947, v3去坐标)
│
├── gen_v4.py                    # v4 后处理: 从 v3 去掉坐标信息
├── gen_v5.py                    # v5 后处理: 纯CRF + 6组参数扫描
├── gen_v5_hybrid.py             # v5 混合策略: 保留v3标签, CRF只改mask
│
├── submit-20260222-v5a_crf_t03.csv          # v5a: 纯CRF, 阈值0.3 (伪造254)
├── submit-20260222-v5b_crf_t025.csv         # v5b: 纯CRF, 阈值0.25 (伪造255)
├── submit-20260222-v5c_crf_t035.csv         # v5c: 纯CRF, 阈值0.35 (伪造254)
├── submit-20260222-v5d_crf_nocls.csv        # v5d: 纯CRF, 无分类器 (伪造195)
├── submit-20260222-v5e_nocrf_t03.csv        # v5e: 无CRF对照 (伪造371, ≈v3)
├── submit-20260222-v5f_crf_aggr_cls.csv     # v5f: CRF+激进分类器 (伪造313)
├── submit-20260222-v5g_hybrid_default.csv   # v5g: 混合+默认CRF (伪造372=v3)
├── submit-20260222-v5h_hybrid_gentle.csv    # v5h: 混合+温和CRF (伪造372=v3, 推荐)
└── submit-20260222-v5i_hybrid_lower_thresh.csv  # v5i: 混合+低阈值CRF (伪造372=v3)
```

---

## 十一、v5 改进计划与结果

### 11.1 当前瓶颈分析

从 v1(0.7502) → v3(0.7971)，提升幅度逐渐变小。比赛评分是 label + mask + explanation 的综合分。

| 维度 | 当前状态 | 瓶颈 | 改进空间 |
|------|---------|------|---------|
| Label | 74.4% 预测为伪造 (训练集 80%) | 可能漏判 ~5% 伪造图 | 中等 |
| Mask | 15 模型集成 + 多尺度 TTA | 边界不够精细, 无 CRF 细化 | **较大** |
| Explanation | 608 字均长, 含坐标 | 风格与 GT 有差异 | 中等 |

### 11.2 改进方向

#### 方向 1: DenseCRF Mask 细化

用原始 RGB 图像的颜色信息细化分割 mask 边界。利用 `pydensecrf2` 实现。

**原理**: CRF 利用像素颜色相似性约束 mask 边界——颜色相似的像素应属于同一类别，使 mask 边缘更紧贴物体轮廓。

**关键参数**:
```python
crf_params = {
    "n_iters": 5,           # CRF 迭代次数
    "sxy_gauss": 3,         # 高斯核空间范围
    "sxy_bilateral": 50,    # 双边滤波空间范围
    "srgb_bilateral": 13,   # 双边滤波颜色范围
}
```

#### 方向 2: 参数扫描 (多版本择优)

`gen_v5.py` 生成 6 个版本的 CSV，分别测试不同组合。

#### 方向 3: VLM Prompt 风格对齐

对比 GT Caption 和 v3 输出，发现 GT 有更严格的结构。已在 `inference.py` 中更新 `system_prompt` 和 `user_prompt`。

### 11.3 v5 实验结果

#### Step 1: 概率图缓存生成

重跑 `inference.py` Stage 1，将 15 模型集成的概率图保存为 `cache/prob_maps/*.npy` (500 个文件)。
后续的 CRF 细化和参数扫描全部在这些概率图上进行，无需重新跑分割模型。

#### Step 2: 纯 CRF + 参数扫描 (`gen_v5.py`)

对 500 张测试图运行 6 种参数配置，每种约 10 分钟，总计约 1 小时：

| 版本 | CRF | 分割阈值 | 分类器覆盖 | 伪造 | 真实 | 来源脚本 | 说明 |
|------|-----|---------|-----------|------|------|---------|------|
| **v5a** | ✓ | 0.30 | lo=0.2, hi=0.9 | 254 | 246 | `gen_v5.py` | CRF + 当前最优参数 |
| **v5b** | ✓ | 0.25 | lo=0.2, hi=0.9 | 255 | 245 | `gen_v5.py` | 更宽松阈值 |
| **v5c** | ✓ | 0.35 | lo=0.2, hi=0.9 | 254 | 246 | `gen_v5.py` | 更严格阈值 |
| **v5d** | ✓ | 0.30 | 关闭 | 195 | 305 | `gen_v5.py` | CRF 但不用分类器 |
| **v5e** | ✗ | 0.30 | lo=0.2, hi=0.9 | 371 | 129 | `gen_v5.py` | 无 CRF 对照组 (≈v3) |
| **v5f** | ✓ | 0.30 | lo=0.15, hi=0.85 | 313 | 187 | `gen_v5.py` | 更激进的分类器回拉 |

**关键发现**: CRF 过于激进
- 纯 CRF (v5a-d) 将大量 "伪造" 标签翻转为 "真实" (254~195 vs v3 的 372)
- CRF 把许多低置信度伪造区域清除，导致 mask 面积过小，label_threshold 判定为"真实"
- 即使降低阈值 (v5b: 0.25) 也几乎没有帮助 (255 vs 254)
- 激进分类器 (v5f) 恢复了约一半标签 (313)，但仍远低于 v3 (372)
- v5e (无 CRF) 与 v3 基本一致 (371 vs 372)，验证了对照组

#### Step 3: 混合策略 (`gen_v5_hybrid.py`)

**核心思路**: 保留 v3 原始标签不变，只对被标记为"伪造"的图像应用 CRF 改善 mask 边界。
- 如果 CRF 后 mask 变为空 → 回退使用 v3 原始 mask (不改变标签)
- 如果 CRF 后 mask 非空 → 使用 CRF 细化后的 mask (不改变标签)

这样标签准确率完全不受影响，只有 mask 质量可能提升。

| 版本 | CRF 强度 | CRF 处理 | Mask 更新 | 伪造 | 真实 | 来源脚本 | 说明 |
|------|---------|----------|----------|------|------|---------|------|
| **v5g** | 标准 | 372 张 | 217 张 (58%) | 372 | 128 | `gen_v5_hybrid.py` | 标签=v3，CRF 默认参数 |
| **v5h** | 温和 | 372 张 | 263 张 (71%) | 372 | 128 | `gen_v5_hybrid.py` | 标签=v3，温和 CRF |
| **v5i** | 标准+低阈值 | 372 张 | 232 张 (62%) | 372 | 128 | `gen_v5_hybrid.py` | 标签=v3，阈值 0.2 |

**混合策略三种 CRF 强度对比**:
- **v5g (默认 CRF)**: `n_iters=5, sxy_bilateral=50, srgb_bilateral=13, compat_bilateral=10`，217/372 (58%) mask 被 CRF 成功更新
- **v5h (温和 CRF)**: `n_iters=3, sxy_bilateral=80, srgb_bilateral=5, compat_bilateral=5`，263/372 (71%) mask 被更新，CRF 强度更低保留了更多原始信息
- **v5i (低阈值)**: 阈值 0.2 (vs 默认 0.3)，`min_area=80`，232/372 (62%) mask 被更新

**所有版本标签一致**: v5g/h/i 全部保持 372 伪造 / 128 真实，与 v3 完全相同。

### 11.4 所有提交版本汇总

| 版本 | 伪造 | 真实 | 比赛得分 | 说明 |
|------|------|------|---------|------|
| v1 | 310 | 190 | **0.7502** | 初始版本，seg_thresh=0.5 |
| v2 | 354 | 146 | **0.7841** | seg_thresh=0.3，关闭分类器 |
| **v3** | **372** | **128** | **0.7971** | 多尺度+智能融合+增强prompt (当前最佳) |
| v4 | 372 | 128 | **0.7947** | v3 去掉坐标信息 |
| v5a | 254 | 246 | 待提交 | 纯 CRF (CRF 翻转大量标签) |
| v5b | 255 | 245 | 待提交 | 纯 CRF + 低阈值 0.25 |
| v5c | 254 | 246 | 待提交 | 纯 CRF + 高阈值 0.35 |
| v5d | 195 | 305 | 待提交 | 纯 CRF + 无分类器 (最激进) |
| v5e | 371 | 129 | 待提交 | 无 CRF 对照组 (≈v3) |
| v5f | 313 | 187 | 待提交 | CRF + 激进分类器回拉 |
| **v5g** | **372** | **128** | **待提交** | **混合: v3标签 + 默认CRF改mask** |
| **v5h** | **372** | **128** | **待提交** | **混合: v3标签 + 温和CRF改mask (推荐)** |
| **v5i** | **372** | **128** | **待提交** | **混合: v3标签 + 低阈值CRF改mask** |

### 11.5 推荐提交顺序

1. **v5h** (混合温和 CRF) — 最推荐。标签=v3，mask 更新率最高 (71%)，CRF 温和不破坏原始信息
2. **v5g** (混合默认 CRF) — 第二选择。标准 CRF，mask 更新 58%
3. **v5i** (混合低阈值 CRF) — 第三选择。低阈值+CRF，mask 更新 62%
4. (可选) **v5f** (CRF + 激进分类器) — 如果混合策略无效，尝试 CRF 全量版本

### 11.6 后续可选改进

1. **重跑 VLM explanation**: 删除 `cache/explanations.json` 后运行 `python inference.py`，用已更新的 prompt 模板重新生成解释文本
2. **训练更强分割模型**: 增加训练数据增强 (旋转/缩放/颜色抖动)、增加训练分辨率 (768→1024)
3. **集成更多架构**: 引入 Swin Transformer V2 或 InternImage 作为第四种分割 backbone


## 十二、Bug 修复历史

| 日期 | Bug | 原因 | 修复 |
|------|-----|------|------|
| 02-17 | CPU OOM (Qwen3-235B) | 缺少 `HfDeepSpeedConfig` 预初始化 | 加入 ZeRO-3 pre-init |
| 02-18 | `torch._grouped_mm` 崩溃 | ROCm grouped GEMM 不支持 | `rocm_compat.py` 顺序回退 |
| 02-18 | DeepSpeed `"auto"` TypeError | DeepSpeed 0.18.6 早期验证 | 显式整数替换 |
| 02-18 | PEFT Conv3d ValueError | ZeRO-3 展平参数 | 硬编码 `target_modules` |
| 02-19 | PyTorch ROCm 被 CUDA 覆盖 | vLLM 安装 CUDA 版 torch | 重装 ROCm nightly |
| 02-19 | `causal_conv1d` ABI 错误 | PyTorch 版本升级不兼容 | 源码重编译 |
| 02-20 | `_grouped_mm` bias 参数 | PyTorch 2.11.0 新增参数 | 更新 fallback 签名 |
| 02-21 | OOM 加载合并模型 (4卡) | 740GB 模型超 4 卡容量 | 改用 8 卡加载 |
| 02-21 | GitHub LFS 2GB 限制 | student model.safetensors 17GB | 用 HuggingFace Hub |
| 02-22 | MaxViT 576 AssertionError | height(18) % window(12) ≠ 0 | 按架构分尺度推理 |

---

## 十三、交接摘要 (供新对话窗口参考)

### 当前最佳得分: v3 = 0.7971

### 比赛得分记录

| 版本 | 得分 | 伪造/真实 | 关键变化 |
|------|------|----------|---------|
| v1 | 0.7502 | 310/190 | 初始版本 |
| v2 | 0.7841 | 354/146 | 降阈值+关分类器 (+0.0339) |
| **v3** | **0.7971** | **372/128** | **多尺度+智能融合+增强prompt (+0.0130)** |
| v4 | 0.7947 | 372/128 | v3去坐标 (-0.0024) |
| v5g-i | 待提交 | 372/128 | 混合CRF改善mask (标签=v3) |

### 已完成的工作

1. **模型训练**: 15 个分割模型 + 5 个分类器 + 397B 教师 LoRA + 8B 学生全量微调，全部完成
2. **推理流水线**: `inference.py` 实现三阶段流水线 (分割→分类→VLM)，支持缓存和断点续传
3. **4 个版本提交**: v1(0.7502) → v2(0.7841) → v3(0.7971) → v4(0.7947)
4. **v5 参数扫描完成**: 9 个 v5 变体已生成 (6 纯CRF + 3 混合策略)
5. **代码全部在 GitHub**: `https://github.com/Datxu-ai/True-or-Fake-Image`

### v5 结果分析

- **纯 CRF (v5a-d)**: CRF 过于激进，将大量伪造标签翻转为真实 (254~195 vs v3的372)，不推荐
- **无 CRF 对照 (v5e)**: ≈v3，验证基线正确
- **CRF+激进分类器 (v5f)**: 恢复部分标签 (313)，但仍低于 v3
- **混合策略 (v5g/h/i)**: 保留 v3 标签，只用 CRF 改善 mask 边界，推荐提交 v5h (温和CRF，71% mask 更新)

### 下一步操作

```bash
cd /wekafs/datongxu/tfi

# 1. 提交 v5h (混合温和CRF) 看是否超过 v3
# 文件: submit-20260222-v5h_hybrid_gentle.csv

# 2. 如果 v5h 效果不明显，尝试 v5g 或 v5f

# 3. (可选) 重跑 VLM 用改进后的 prompt
rm cache/explanations.json
python inference.py

# 4. (可选) 提交不同 v5 版本对比分数
```

### 关键文件路径

| 文件 | 说明 |
|------|------|
| `inference.py` | 主推理流水线 (已更新: 保存概率图 + 新 prompt) |
| `gen_v5.py` | 纯 CRF + 6 组参数扫描 (已完成) |
| `gen_v5_hybrid.py` | 混合策略: 保留 v3 标签 + CRF 改 mask (已完成) |
| `cache/prob_maps/` | 500 个分割概率图 .npy (已生成) |
| `cache/cls_scores.json` | 分类器分数缓存 |
| `cache/explanations.json` | VLM 解释缓存 (旧 prompt 生成) |
| `submit-20260222-v5h_hybrid_gentle.csv` | 推荐提交版本 |
