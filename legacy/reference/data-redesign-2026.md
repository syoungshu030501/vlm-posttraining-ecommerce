# 数据集重构与多模态 PRM 适配 — 文献调研与落地决策

> 触发问题：当前数据集"过于简单"，导致 SFT F1 在 98.8% 早早饱和、GRPO 训练 89% step `grad_norm = 0`、reward saturation 严重。本文档围绕**已重新训完的 token-level scalar PRM 升级为 field-wise PRM + 多类违规分类 head**、**数据全栈重做**、**项目从电商合规扩到通用 VLM 推理**三个决策展开。
>
> 形态参考 [SoK-agentic-RAG-summary.md](SoK-agentic-RAG-summary.md)：前 1-2 节梳理外部工作，后半给本项目落地决策。

---

## §0 章节地图

- §1 现状诊断：数据"过于简单"的 5 个错位
- §2-§7 五篇核心对口工作 + 辅助工作
- §8 项目重构方向：field-wise PRM + 多类违规 + 选定 VisualPRM 路线
- §9 三份数据集详细方案（B 优先）
- §10 工程量、优先级、评估
- §11 take-aways

## §1 现状诊断：数据"过于简单"的 5 个错位

| # | 错位 | 当前症状（README 已记录） | 根本原因 |
|---|---|---|---|
| 1 | **任务深度天花板低** | F1 在 SFT 阶段就饱和到 98.83%（[../README.md](../README.md) §7.2），RL 不动 F1 | violation 是 binary 二分类，JSON 4 字段，每条样本推理深度 1-2 步 |
| 2 | **难度静态 vs 训练动态错位** | mine_hard_samples 是 RL 启动前一次性挖矿；19/19 step `score ∈ [4.5, 5.0]`，89% step `grad=0` | 当前 hard 池跑过一次就再不更新，actor 学会几条 hard 后剩下变 easy |
| 3 | **教师标注高度同质** | 即使修了 prompt，reason 唯一度也只 99.6%（[../STRATEGY.md](../STRATEGY.md) §3） | qwen-vl-max 一个老师写所有 reason，stylistic distribution 收敛 |
| 4 | **奖励信号粒度 vs 数据真值粒度错位** | PRM 已升级为 token-level credit assignment + 你下一版要做 field-wise；但 preference 数据仍是 ORM 形态（整条 chosen / rejected） | 多 head 拿不到独立监督信号——等价于共享一个 reward 然后强行分摊 → SupCon 无法收敛 |
| 5 | **领域分布过窄** | 服装+鞋占 70%，3093 张图 95% 是商品图 | 已被你确认要 broaden 到 math/chart/doc/reasoning |

**核心错位是 #4**：PRM 升级了但训练它的数据没升级，跟"模型训不好"是两回事。

---

## §2 StructVRM — 跟你 field-wise PRM 完全同源（核心对口 #1）

> **ByteDance Seed**, arxiv [2508.05383](https://arxiv.org/abs/2508.05383), 2025-08。最对口论文。

### 2.1 为什么对口
StructVRM 解决的问题与你升级 PRM 的动机**完全一致**：

> 传统 RLVR 用 binary 单分回报，"3/4 正确"和"全错"奖励一样；这种 all-or-nothing 信号是稀疏且无信息量的，导致策略学习收敛慢甚至 collapse。

→ 你的 violation binary 在 SFT 已 98.8% 正确，RL 没法继续学；attribute / reason 子任务被埋没。

### 2.2 核心方案
- **model-based verifier 输出 score vector**：每个 sub-question 一个 binary score；最终 reward = mean over sub-answers
- 不再依赖字符串 exact-match，而是用 verifier 模型**判断语义/数学等价**
- 训练流程：SFT (50K+ multimodal CoT) → RL (PPO) with structured reward

### 2.3 在你项目里的映射
| StructVRM 概念 | 你项目的对应 |
|---|---|
| sub-question score vector | **field-wise PRM heads**：`category / attributes / violation_prob / violation_type / reason_align` |
| verifier 模型 | 你刚训完的 PRM 升级版 |
| partial credit 机制 | violation 取消硬分类（你已决定）→ continuous prob + multi-class type |
| semantic equivalence | reason ↔ attributes BGE 余弦（你 reward_v2 已用）|

### 2.4 抄什么、不抄什么
- **抄**：score vector 的输出形态、partial credit 的 reward 聚合（mean / weighted-sum）、verifier 训练流程（SFT + 验证集）
- **不抄**：他们 50K SFT 全是 STEM 学科题，跟电商合规域差距大；只取数据格式不取数据本身

---

## §3 VisualPRM400K — 你选定的数据路线（核心对口 #2）

> **OpenGVLab/InternVL**, arxiv [2503.10291](https://arxiv.org/abs/2503.10291), ICLR 2026 poster。
> HF: [OpenGVLab/VisualPRM400K-v1.1](https://huggingface.co/datasets/OpenGVLab/VisualPRM400K-v1.1) (formatted) / [VisualPRM400K-v1.1-Raw](https://huggingface.co/datasets/OpenGVLab/VisualPRM400K-v1.1-Raw) (raw with mc scores)。

### 3.1 数据集形态
每条样本：`(image I, question Q, step-by-step solution {s_0,...,s_n}, expected accuracy {mc_0,...,mc_n})`

- 平均每条响应 5.6 steps × 126.9 词
- `mc_i ∈ R≥0`：从 step `s_i` 开始 MCTS 续推 N 次，N_correct / N
- formatted 版本：`mc_i > 0` 转成 `+`，否则 `-`，多轮对话格式
- 总规模：400K samples × 2M steps

### 3.2 自动 step 标签生成 pipeline（关键 trick）
```
对每个 sample (I, Q, full_solution):
  for each step s_i in solution:
    fix prefix s_0...s_i
    用 policy 续生 K 次（典型 K=8-16）
    解析每次结果是否正确
    mc_i = #correct_continuations / K
  step_i 标记为 correct iff mc_i > 0
```
**核心 = 把 OmegaPRM/Math-Shepherd 的 MCTS 思路移植到多模态**，零人工标注。

### 3.3 配套基准
- **VisualProcessBench**：2,866 samples × 26,950 step-wise 人工标签
- 用于评估 PRM 是否能正确识别 incorrect step
- 实测：现有开源 MLLM 在 step-correctness judge 上普遍弱

### 3.4 在你项目里的应用
你**选定走这条路线**，落地时这样用：

| 用途 | 直接拿 VisualPRM400K-v1.1 用 | 自己造 |
|---|---|---|
| **数据集 A（SFT 多领域扩充）** | ✅ 取 100K 子集做 SFT，混入电商集 | 电商合规集保持原样 |
| **数据集 B（PRM 训练）** | ⚠️ 只能拿"step-wise correctness"格式参考 | 电商任务 step 划分要重新定义；attribute-level grounded 标签也得自造 |
| **数据集 C（RL 池）** | ✅ 取 mc ∈ [0.2, 0.8] 的样本（中等难度） | 电商样本要自己跑 MCTS |

**关键 adaptation**：你要把"step"概念替换为"field"。原 VisualPRM 是 step `s_i` → `mc_i`；你要做的是 field `f_j ∈ {category, attributes, violation, reason}` → `mc_j`。这就是 **field-wise PRM 在数据上的具体形态**。

---

## §4 VL-Rethinker — 直接解你的 reward saturation（核心对口 #3）

> arxiv [2504.08837](https://arxiv.org/abs/2504.08837), NeurIPS 2025。
> 主页 [tiger-ai-lab.github.io/VL-Rethinker](https://tiger-ai-lab.github.io/VL-Rethinker/)，配 ViRL39K（38,870 queries）。

### 4.1 为什么对口
你 README §6.3.5 描述的 reward saturation 跟 VL-Rethinker 的 **vanishing advantages problem** 是同一个东西，且他们给出了比你"一次性 hard mining"更系统的解法。

> "vanishing advantages：当一个 query group 内所有 rollout reward 相同（全对或全错），advantage = 0，gradient = 0；reward uniformity 随训练加剧，模型停止探索更深的 reasoning"

### 4.2 SSR (Selective Sample Replay)
- 维护一个 replay buffer `B_replay`，存历史 high-|advantage| 样本
- 每个 RL step 在当前 batch 之外，按 `P(j) ∝ |A_j|` 从 buffer 抽 k 条样本拼进来
- 让"决策边界附近的样本"得到反复训练 → 把 vanishing advantages 消掉
- **本质 = online curriculum**，比 offline 的 hard mining 灵活

### 4.3 Forced Rethinking（次要）
- 在 rollout 末尾追加一个 `<rethink>` trigger token，强制模型再生第二段反思后再给最终答案
- Qwen2.5-VL 不会自发 self-reflection，需要这种 explicit nudge
- 你的电商任务输出短（100-150 token），可能用不上；但 broaden 到 math 后值得加

### 4.4 在你项目里的映射
| VL-Rethinker | 你的项目 |
|---|---|
| ViRL39K 数据集（38,870 queries 跨 STEM + 社科）| 你 broaden 阶段直接借鉴主题分布；规模放到 30-50K 起 |
| SSR replay buffer | **C 数据集设计的核心机制**（详见 §9.2）|
| Forced Rethinking trigger | broaden 阶段对 math/chart 任务可选启用 |

### 4.5 SSR vs 你当前 hard mining 的对比
| 维度 | 你当前做法 | SSR |
|---|---|---|
| 难度更新频率 | 训练前一次性 | 每 step 动态 |
| 来源 | offline 全集 rollout | online buffer + current batch |
| 难度信号 | reward < 4.5（fixed threshold）| `|advantage|` 排序（adaptive）|
| 实现复杂度 | 中（需要单卡跑全集 rollout）| 低（~50 LoC，verl 里加 buffer 类）|

**结论**：你的 hard mining 是 SSR 的 v0 版；升级时 SSR 优先。

---

## §5 Vision-G1 — 多轮 RL curriculum（核心对口 #4）

> arxiv [2508.12680](https://arxiv.org/abs/2508.12680), 2025-08。

### 5.1 关键贡献
- **数据收集**：46 公开数据源 × 8 维度（infographic / math / spatial / cross-image / GUI / medical / commonsense / general science），统一格式 → 40K
- **Influence function-based filtering**：用影响函数估计每条样本对 holdout loss 的贡献，去掉负贡献样本
- **Multi-round RL data curriculum**：
  ```
  Round 1: 训 100-200 step → 用当前 ckpt 全集 rollout 算 mc_i
            → 留 mc_i ∈ [0.2, 0.8] 中等难度样本
  Round 2: 用 round-1 ckpt 重复以上
  Round 3: 收敛
  ```

### 5.2 在你项目里的映射
- broaden 后用同样思路做"难度自适应"
- "mc_i ∈ [0.2, 0.8]"是关键阈值——比你当前"reward < 4.5"更直接反映难度
- **每轮重新计算 mc_i** 是关键，避免你当前一次性 hard mining 的失效问题

### 5.3 抄什么、不抄什么
- **抄**：multi-round 思路 + difficulty filter `mc ∈ [0.2, 0.8]` + 跨域混合
- **不抄**：influence function（实现复杂，对 8B 单节点 ROI 偏低；先用 mc 过滤即可）

---

## §6 VLAA-Thinking — 警告：SFT 会污染下游 RL（核心对口 #5）

> arxiv [2504.11468](https://arxiv.org/abs/2504.11468), ICCV 2025。UCSC。

### 6.1 关键发现（直接挑战你当前 SFT-first 流程）
> "SFT can undermine subsequent RL by inducing **pseudo reasoning paths** imitated from teacher models. These paths involve prolonged, hesitant, less informative steps and incorrect reasoning, despite resembling native RL paths."

→ 你的 SFT 用 qwen-vl-max 蒸馏，本质就是 imitation learning；如果 SFT 学了过多假 reasoning，FIPO 阶段反而被锚定在低质 reasoning 模板上。

### 6.2 数据集分两批
- **VLAA-Thinking-SFT**: 126K 高质量 step-by-step 视觉推理 trace（教格式）
- **VLAA-Thinking-RL (GRPO)**: 25K 更难的 RL 集（不再有 CoT 答案，只有最终答案 + 验证函数）
- 来源：CLEVR_Math, GeoQA170K, Math PUMA, ArxivQA, ChartQA, ...
- 6 步 pipeline：caption → distill reasoning → answer rewrite → **verify**（关键）

### 6.3 给你项目的启示
1. **SFT 集和 RL 集要分开**，不要全是同一份 qwen-vl-max 蒸馏数据
2. **SFT 集 reasoning 长度要控**：你电商任务 reason 平均 ≤80 字是好事，符合 VLAA 建议；但 broaden 到 math 后要 cap 在 256 tokens 内，避免学太多 hesitant 模板
3. **RL 集只保留 verifiable 答案**：放弃 reason 整段对齐，只保留 violation_type / numeric answer 这种容易 verify 的字段做 reward

---

## §7 辅助工作（不主用，但要在论文/答辩时知道）

| 工作 | arxiv | 一行总结 | 跟你项目的相关点 |
|---|---|---|---|
| **HARMO / Beyond Monolithic Rewards** | [2510.05283](https://arxiv.org/abs/2510.05283) | model-based + rule-based 混合 reward + 多 aspect（accuracy / instruction-adherence / length-penalty） | 你 reward_v2 的 5 组件就是 rule-based aspect；可补 instruction-adherence head |
| **Curr-ReFT** | [2503.07065](https://arxiv.org/abs/2503.07065) | 小 VLM RL 的"砖墙现象" + 难度感知 reward design | 直接对应你 8B 在 6 卡 L20 的规模 |
| **R1-OneVision** | [2503.10615](https://arxiv.org/abs/2503.10615) | cross-modal formalization：图先转结构化文本再 reason | 你直接喂图+JSON 输出是简化版；broaden 到 math 后值得加 image→text 中间步骤 |
| **MMR1 / R1-V / Visual-RFT / Perception-R1** | 多篇 | R1-style RLVR 在视觉任务的早期工作 | 都是 binary reward，被 StructVRM/HARMO 替代 |
| **Infinity-MM** | 2410.18558 | 40M 多模态 instruction，自动标签合成 | 你 broaden 阶段做大规模合成时的参考 |
| **VisionFoundry** | 2604.09531 | 任务关键词 → LLM 出问 → T2I 出图 → VLM 验证 | 合规审核对 hard case 缺图时可用此合成 |
| **MindGYM** | 2503.09499 | self-challenge synthetic Q & A | 400 sample +16% 提升，small data 下值得试 |
| **Hierarchical Multi-Step Reward** | 2503.13551 | 同时评估 individual + consecutive step | 可叠加在 field-wise PRM 上做二级聚合 |

---

## §8 项目重构方向

### 8.1 PRM 头部设计（你的下一版）

```
当前 PRM:  scalar reward → token-level credit assignment → binary violation
                                       ↓
下一版 PRM (本文档采用):
   shared backbone (sft_aux_merged frozen)
   ├── category_head        : multi-class 分类（10 粗粒度品类，CE loss）
   ├── attributes_head      : sequence-level，每个 attr (key,val) 输出一个 grounded ∈ [0,1]
   │                          (BCE loss; key 数变长用 Transformer-on-Sequence)
   ├── violation_prob_head  : continuous regression ∈ [0,1]（BCE 或 MSE，取消 binary）
   ├── violation_type_head  : multi-class N+1 分类（"无违规" + N 类违规，CE loss，N 可扩展）
   └── reason_align_head    : continuous regression ∈ [0,1]（MSE，目标 = BGE 余弦）
```

**为什么 SupCon 不收敛**：当前 violation 是 binary，正负样本各占 50%，二者特征差异本来就不大；SupCon 要求 anchor 与同类样本 embedding 拉近、与异类推远，当类内/类间方差差不多时 contrastive loss 失效。换成 `violation_prob` continuous + `violation_type` multi-class 后，supervision 信号更细，无需 SupCon。

### 8.2 数据形态升级：从 ORM 到 field-wise

每条样本要从当前 `(prompt, chosen, rejected)` 升级为：

```jsonc
{
  "image": "path",
  "prompt": "...",
  "response": "<full JSON output from policy>",
  // ↓ 每个 head 一份独立监督信号
  "labels": {
    "category": {"gt": "服装", "pred_correct": true},
    "attributes": [
      {"key": "颜色", "val": "黑色", "grounded": true,  "mc": 1.0},
      {"key": "材质", "val": "皮革", "grounded": false, "mc": 0.2}   // 幻觉属性
    ],
    "violation_prob": 0.85,        // soft target
    "violation_type": "极限词",     // multi-class
    "reason_align": 0.78           // BGE 余弦
  },
  // ↓ 跟 VisualPRM 对齐的元数据
  "meta": {
    "mc_violation": 0.6,           // 在 violation 字段处分叉续推 8 次的 correct 率
    "mc_reason_align": 0.4,        // reason 续生 8 次的 BGE 余弦均值 > 0.6 的比例
    "policy_ckpt": "fipo_v2_step160_merged"
  }
}
```

### 8.3 路线选定：VisualPRM 路线

你的决策是 **走 VisualPRM 路线**（混合电商专用造数）：

| 阶段 | 直接拿 OpenGVLab/VisualPRM400K-v1.1 | 自己跑 |
|---|---|---|
| SFT 集 A | ✅ 取 ~80K 子集（math + chart + general），混入电商集 6685 → 总 ~90K | 电商集保持原样 |
| RM/PRM 集 B | ❌ 格式参考用，不直接拿 | ✅ 必须重造（field-wise label 是论文里没有的）|
| RL 池 C | ⚠️ 取 mc ∈ [0.2, 0.8] 的子集 ~30K 当难度热身 | 电商难例多轮 mine（按 §5 Vision-G1 多轮） |

**优先级（你已确认）**：B 优先 → C 次 → A 最后

---

## §9 三份数据集详细方案（B 优先）

### 9.1 数据集 B：RM/PRM 训练集（**主写**）

#### 9.1.1 目标规模
- 总量 **8K-12K 条 field-wise 样本**（每条样本本身比当前 preference 信息量大 5×，所以不需要 2000→20000 的暴力扩张）
- 数据源构成：
  - 电商合规 5K（你已有 2000 + mine 出 3000）
  - VisualPRM 借鉴：math/chart 各 1.5K（取 STEM 子集，扩 PRM 的领域泛化）
  - 通用 OOD 1K（OpenAI red team / 平台内审 user log，测 OOD generalization）

#### 9.1.2 核心：每个 field 的真值怎么自动生成

| Field | 真值生成方法 | 借鉴来源 | 自动化程度 |
|---|---|---|---|
| `category.gt` | qwen-vl-max 单样本投票 | 现有 | ✅ 100% |
| `attributes[].grounded` | **MC rollout**：fix 前缀到 attribute 字段，policy 续生 N=8 → 看 attribute (key,val) 出现率 mc_i | VisualPRM/OmegaPRM 移植 | ✅ 100% |
| `attributes[].mc` | 同上 | VisualPRM | ✅ 100% |
| `violation_prob` | **双教师投票**：qwen-vl-max + Qwen3-VL-72B + InternVL3-78B 三者各跑 5 个 temp，21 个判断的 violation 比例 = soft prob | StructVRM 半自动 | ✅ 100% |
| `violation_type` | 现有 6 类 + 扩到 **N=10**：极限词 / 材质虚标 / 功效夸大 / 品牌侵权 / 价格欺诈 / 图文不符 / **涉黄涉政 / 医疗夸大 / 虚假代言 / 违禁品**（最后 4 类要新爬 50-100 案例样本）| HARMO 多 aspect | ✅ 80%（爬数据要人工筛） |
| `reason_align` | BGE-zh 余弦（你已用）| 现有 reward_v2 | ✅ 100% |

#### 9.1.3 MC rollout 的具体实现（关键 trick）

```python
# 仿 VisualPRM 思路，move到 field 粒度
def mc_field_label(image, prompt, gold_response, field_name, K=8):
    """
    给定 (image, prompt, gold_response)，在 field_name 处分叉，
    用 policy 续生 K 次，估计该字段的 mc。
    """
    # 1. 找到 gold_response 中 field_name 的位置
    prefix = gold_response[: position_of(field_name)]
    # 2. 用 policy (sft_aux_merged) 在 prefix 上续生 K 次（temp=0.7-1.0）
    rollouts = [policy.generate(image, prompt, prefix=prefix) for _ in range(K)]
    # 3. 用 verifier 判断每个 rollout 的 field 部分是否正确
    correct = sum(1 for r in rollouts if verifier(r, gold_response, field_name))
    return correct / K  # mc_i ∈ [0, 1]
```

**成本估计**：8K samples × 5 fields × 8 rollouts = 320K rollouts；按 sft_aux_merged 单条 ~0.5s（短 prefix），单卡 L20 跑 ~45h；6 卡并行 ~8h，可接受。

#### 9.1.4 训练 PRM 的 loss 形态

```python
loss = (
    1.0 * CE(category_logits, category_gt)
  + 0.5 * BCE_pointwise(attr_grounded_logits, attr_grounded_labels)  # 序列上每个 attr 一个 BCE
  + 1.0 * BCE(violation_prob_logit, violation_prob_target)            # soft target
  + 1.0 * CE(violation_type_logits, violation_type_gt)                # multi-class
  + 0.3 * MSE(reason_align_pred, reason_align_target)                 # continuous
  # 不再有 SupCon / Triplet（它们在 binary 时本来就不收敛）
)
```

#### 9.1.5 评估
- **field-wise pair-acc**：每个 head 单独 holdout 200 对算 pair-acc
- **field-wise mean_margin**：跟当前 mean_margin=11.21 对比，看每个 head 是否 margin > 5
- **multi-class violation_type 准确率**：N+1 分类的 macro-F1（每个违规类型平均 50 条 holdout）
- **OOD generalization**：拿 1K OOD 样本算 violation_prob 与 GT 的 ECE（expected calibration error）

#### 9.1.6 工程量
- 数据生产：6 卡 ~8h MC rollout + 1-2d 双教师 API 调用 (~¥150-200)
- PRM 训练：单卡 SFT-style ~12-18h
- **总计：~3-4 天**，是 B 优先后第一周内就能交差的事

---

### 9.2 数据集 C：RL 训练池（多轮 curriculum）

#### 9.2.1 目标规模
- 静态种子池：5K（混电商 3K + VisualPRM math/chart 取 mc∈[0.2,0.8] 子集 2K）
- 动态 SSR replay buffer：训练中维护，cap 5K

#### 9.2.2 多轮 curriculum（仿 Vision-G1）

```
Round 0 (热身): 静态种子池 5K + reward_fn v3 (field-wise)
                训 200 step

Round 1: 用 round-0 ckpt 全集 rollout (~5K × n=8 = 40K rollouts，6 卡 ~3h)
         按 mc ∈ [0.2, 0.8] 留中等难度 → 期望剩 2-3K
         混入 SSR buffer 高 |advantage| 1K → 总 3-4K
         继续训 200 step

Round 2: 同上，期望平均 mc 慢慢从 0.5 → 0.3
         即模型已经学得很好，这时该停了
```

#### 9.2.3 SSR 实现
- verl `compute_advantages` 后增加 buffer.push((sample, |advantage|))
- 每个 step 从 buffer 按 P ∝ |A| 采样 batch_size × 0.3 比例的 replay 样本
- buffer 用 reservoir sampling 保持 cap=5K

#### 9.2.4 工程量
- 一轮 RL：~6-8h（6 卡 L20）+ rollout 重打分 ~3h
- 三轮 = 30h，~1.5d 训练 + 0.5d 数据 → 总 2-2.5d
- **依赖 B 完成**（reward_fn v3 需要 field-wise PRM）

---

### 9.3 数据集 A：SFT 主集（broaden）

#### 9.3.1 直接复用现成数据
- 80K 取自 [VisualPRM400K-v1.1](https://huggingface.co/datasets/OpenGVLab/VisualPRM400K-v1.1) formatted 版（multi-turn 对话格式，已含 step correctness token）
- 6.7K 电商合规集保持
- 总 ~87K，按 image 分组 80/10/10 切分

#### 9.3.2 schema 统一
- 增加可选字段 `reasoning_steps`（math/chart 任务有，电商任务空）
- output 仍用 `{category, attributes, violation, reason}` schema（电商）+ `{answer, steps}` schema（math）
- 用 task-specific system prompt 区分

#### 9.3.3 关键约束（VLAA-Thinking 警告）
- SFT 集 reasoning 长度 cap **256 tokens**（防止学太多 hesitant 模板）
- 老师多样性：尽量选 VisualPRM400K 里多个老师产的样本（HF 上 raw 版有 source 字段）

#### 9.3.4 工程量
- 数据下载（HF）：1-2h
- 格式统一脚本：1d
- LoRA SFT 单卡：1-2 个 epoch ~24-30h
- **总计 3-4d**，但**不阻塞 B 和 C**

---

## §10 工程量、优先级、评估

### 10.1 优先级与时间线（你已确认 B 优先）

| 周 | 主任务 | 子任务 | 验收 |
|---|---|---|---|
| W1 | **数据集 B 生产** | (1) 扩 violation_type 到 N=10，爬 50-100 案例补 4 类 (2) MC rollout 跑 8K 样本 × 5 fields × 8 rollouts (3) 双教师 violation_prob 投票 | `data/preference_v3/{train,holdout}.parquet` 总 8-12K 行 |
| W1-W2 | **PRM v3 训练** | field-wise heads + 5 个 loss 联合优化 | 每 head pair-acc / mean_margin 跑出，跟 v2-aux 横比 |
| W2 | **reward_fn v3** | 用 PRM v3 替代 reward_fn v2 的 5 组件，但 BGE 余弦保留作 sanity | sanity test 8 case 仍 reward 正向 |
| W2-W3 | **数据集 C 启动** | 种子池 5K 混合 + SSR 实现 | RL Round 0 跑通 |
| W3-W4 | **三轮 RL curriculum** | Round 0/1/2 + 每轮重打分 | 平均 mc 从 0.5 → 0.3，hallucination 比当前 23.04% 再降 |
| W4-W5 | **数据集 A 扩** | 拉 VisualPRM400K-v1.1 80K + 格式统一 + 重训 SFT | broaden 后 SFT 在 MathVista/ChartQA 也有合理基线 |

总计 **4-5 周**，单人节奏。

### 10.2 评估指标体系（参考 SoK trajectory-level 思路）

| 类别 | 指标 | 目标值 |
|---|---|---|
| **PRM 质量** | 每 head pair-acc | 各 head ≥ 0.85 |
| | violation_type macro-F1 | ≥ 0.75（10 类，每类 ~50 条 holdout） |
| | OOD ECE | ≤ 0.10 |
| **RL 健康** | grad_norm 非零 step 比例 | ≥ 80%（当前 11%） |
| | reward std 跨 group | ≥ 0.4（当前接近 0） |
| | response_length | 稳定不缩水（无 mode collapse） |
| **下游任务** | 电商 hallucination_rate | ≤ 0.20（当前 0.2304） |
| | broaden 后 MathVista | 看 SFT 后能否 ≥ 60%（baseline 参考点） |
| | broaden 后 ChartQA | ≥ 70% |
| **trajectory** | mc_field_drift | RL 前后 attribute mc 变化的均值（看 RL 是否真在改 grounding） |

### 10.3 三个最大风险

| 风险 | 触发条件 | 缓解 |
|---|---|---|
| **MC rollout 噪声大** | sft_aux_merged 在 attribute 字段续生不稳定，mc 波动大 | K 从 8 提到 16；用 majority voting 而非 raw frequency |
| **field-wise PRM 训不动** | 5 个 loss 互相打架，某 head 收敛某 head 发散 | 阶段性训练：先训前 3 个 head，冻结后训剩下 2 个；或用 GradNorm 自适应权重 |
| **broaden 后电商任务 forgetting** | 87K 多领域 SFT 后，电商集 F1 反掉 | replay rate 30%（每 batch 至少 30% 是电商），仿 VL-Rethinker SSR 同思路 |

---

## §11 关键 Take-aways（一行复述）

1. **数据简单不是规模问题，是结构问题**——你 PRM 升级到 field-wise，但 preference 数据还是 ORM 形态，多 head 拿不到独立监督，所以 SupCon 不收敛。
2. **直接对口的 paper 是 StructVRM**（field-wise / score vector）+ **VisualPRM400K**（MCTS 自动 step 标签）+ **VL-Rethinker SSR**（解 reward saturation）。
3. **violation 取消硬分类是对的**——continuous prob + multi-class type 比 binary 信号丰富 5×，且能扩展类别（涉黄/医疗夸大等）不需要重训 head 结构。
4. **PRM 数据自动生成的核心 trick**：把 OmegaPRM 的 MCTS 思路从 step 粒度移植到 field 粒度——每个字段在响应里分叉，policy 续生 K 次，看输出一致性 mc_i 当 soft label。
5. **RL 数据要动态而非静态**——VL-Rethinker 的 SSR 比你当前一次性 hard mining 更省、更稳；Vision-G1 的多轮 curriculum 让 mc ∈ [0.2, 0.8] 自适应。
6. **broaden 路线选 VisualPRM400K-v1.1 直接拿**——80K 样本 step 标注完备，省 99% 数据生产成本；但 RM/PRM 的 field-wise 真值必须自造（论文没有现成的）。
7. **SFT 集和 RL 集要分开**——VLAA-Thinking 警告 SFT 蒸馏会引入 pseudo reasoning paths，broaden 阶段两批数据分开做。
8. **优先级 B → C → A**：先验证 field-wise PRM 是否真有意义（4 周内可验证），再花更多算力扩 RL/SFT。

---

> **更新记录**
> - 2026-05-15：基于"已训完 token-level scalar PRM、计划升级为 field-wise + 多类违规分类"决策初次撰写。配套 §10 时间线进入主 README 的 §13 路线图（待补）。
