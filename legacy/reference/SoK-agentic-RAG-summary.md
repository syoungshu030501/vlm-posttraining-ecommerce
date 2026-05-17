# SoK: Agentic RAG — 文献总结与本项目落地决策

> 原文：Mishra et al., *SoK: Agentic Retrieval-Augmented Generation: Taxonomy, Architectures, Evaluation, and Research Directions*. arXiv:2603.07379, 2026-03。
>
> 本总结围绕**电商商品合规审核 VLM 后训练**项目（[../README.md](../README.md)）的 Stage 4 RAG 升级需求展开，前五节梳理综述要点，第六节给出本项目的方案选型与不选项的理由。

---

## 1. 论文定位

把"agentic RAG"作为**独立范式**与传统 RAG 划清边界。核心论点：

- 传统 RAG = 静态 pipeline：retrieve once → generate once。无 mid-loop 修正、无 tool 选择、无 memory。
- Agentic RAG = **由 LLM 作为 policy** 在 finite-horizon POMDP 上反复决策：何时检索、用哪个工具、是否反思、什么时候终止。
- 全文贡献 4 条：(a) POMDP 形式化；(b) 多维 taxonomy；(c) 模块化架构分解 + 设计模式；(d) 系统性风险图谱与 doctoral-scale 研究方向。

是 SoK（Systematization of Knowledge），不是新方法论文——因此适合用作选型参考而非"按这个 repo 复现"。

---

## 2. 形式化定义（Section III-D / III-E）

把 agentic RAG 写成 finite-horizon POMDP：

| 符号 | 含义 |
|---|---|
| $S_{env}$ | 知识库 $C$ 中"任务真正需要的隐状态" |
| $\mathcal{A} = \mathcal{A}_{ret} \cup \mathcal{A}_{reason} \cup \mathcal{A}_{tool} \cup \{STOP\}$ | 离散动作空间 |
| $\Omega$ | 观测空间（retrieval 返回的文本块、tool 输出） |
| $O(o_t \mid s_t, a_t)$ | 观测函数 |
| $\pi_\theta(a_t \mid M_t)$ | 由 LLM 实现的随机策略（in-context 或 fine-tuned） |
| $M_t$ | 工作记忆（observable history，belief state 的近似） |

**判定 agentic vs iterative-RAG 的两个充分条件**：

1. **Policy autonomy**：动作选择（含 STOP）由 $\pi_\theta$ 而非启发式规则决定
2. **State-aware control**：下一步动作依赖到 $t-1$ 步的历史 $M_t$，而不仅是初始 query $q$

只满足循环但用固定 trigger（如"每 n 个 token 检索一次"）→ 不算 agentic。

---

## 3. 架构模块（Section V — 五大组件）

| 模块 | 输入 | 输出 | 控制信号 | 反馈环 |
|---|---|---|---|---|
| **Planner** | user query, global state | sub-task graph | depth/max-step | plan failure 时 self-correct |
| **Reasoning Engine (Controller)** | sub-task, local memory | action / tool call | confidence threshold | observation-triggered replan |
| **Retrieval Subsystem** | query | passages / structured rows | hierarchical interface（lexical / dense / span-extract） | staged ranking |
| **Memory System** | session events | persistent / episodic store | dynamic pruning | retrieval-induced interference |
| **Tool Orchestration Layer** | reasoning engine 的 tool call | API / sub-agent 输出 | sequential / parallel / loop router | structured error recovery |
| **Verification Module** | draft answer | constraint check + structured feedback | factuality / format guard | PPAR 闭环 + HITL escalation |

**两个工程要点**：

- **Agent-Computer Interface (ACI)**：与 tool 通信必须是结构化、可校验、错误反馈精炼，不能直接吐 stack trace 给 LLM（否则 context 爆掉）。
- **Continuum Memory Architecture (CMA)**：episodic memory 不是 KV store，是带"persist / decay / interfere"动力学的子系统——简单 RAG 上叠加 memory 反而引入 memory poisoning 风险（见 §5）。

---

## 4. 六种设计模式（Section VI / Table VI）

| Pattern | 核心控制问题 | 终止条件 | 代表工作 |
|---|---|---|---|
| **Plan-Then-Retrieve** | "synthesize 前要回答哪些子问题？" | 全部 sub-question 答完 | Self-Ask, Plan-and-Solve |
| **Retrieve-Reflect-Refine** | "这些 passage 够吗？要不要改 query？" | 反思判定足够 / 预算耗尽 | Self-RAG, Iter-RetGen |
| **Decomposition-Based (interleaved)** | "当前推理状态下还缺什么？" | 答案达到充分证据 | IRCoT, ReAct |
| **Tool-Augmented Loop** | "现在该调哪个工具？" | tool 结果稳定 / verifier halt | Toolformer, CRITIC |
| **Multi-Agent Collaboration** | "哪个角色处理这个任务？" | 跨 agent 共识 | AutoGen, MetaGPT |
| **Retrieval-Grounded Self-Verification (CoVe)** | "哪些 claim 要证据校验？" | verification 通过 / abstain | Chain-of-Verification, GopherCite, Search-R2 |

**模式间关系**：不互斥；生产级系统经常 stack（CoVe 套在 Multi-Agent 之上做后置校验）。

---

## 5. 风险图谱（Section IX）

| 风险 | 触发条件 | 严重度 |
|---|---|---|
| **Cascading hallucination** | 一步幻觉被后续 retrieval/reasoning 当 fact 接受 | 极高 — 综述实测 legal-RAG 工具 hallucination 33% |
| **Prompt injection in retrieval** | 检索语料含对抗指令 | 极高 — 5 条恶意文档 = 90% ASR |
| **Tool misuse / cascading errors** | 错选工具、参数错构、API 失败被当合法输出 | 高 |
| **Memory poisoning** | 长期 memory 写入污染信息 | 高 — 跨 session 持久 |
| **Systemic risk amplification** | 上述四项在 iterative loop 里复合 | 极高 |

综述明确指出：**单独看任一风险都可控，但 iterative agentic loop 把它们放大**。所以"加一层 reflection 永远是好的"是错的——多 loop 步意味着多次暴露在 prompt injection 与 cascading hallucination 下。

---

## 6. 本项目落地决策

### 6.1 现状对照（Stage 4 RAG v1）

按综述判据，[src/stage4_rag/inference.py](../src/stage4_rag/inference.py) **不算 agentic**：

| 综述要求 | v1 实现 | 差距 |
|---|---|---|
| Policy autonomy | confidence gate `field_min < 0.40` 是固定启发式 | ✗ 未通过 |
| State-aware control | 检索一次后直接拼 prompt，无 reflection | ✗ 未通过 |
| Verification | 无 — retrieved context 注入即采纳 | ✗ 未通过 |
| Tool orchestration | 无 — visual + text 同时硬触发，无路由 | ✗ 未通过 |
| Cascading hallucination 防护 | 无 | ✗ 高风险 |

**README §7.2 实测的 "sft_aux + RAG → hallucination 升至 32.4%"** 就是 cascading hallucination 的教科书症状——retrieved context 强化了模型本来就错的判断。

### 6.2 选型（CoVe + Plan-Then-Retrieve + 轻量 OCR）

任务约束筛选：单图、单条、单跳、结构化 JSON 输出 → 六个 pattern 中：

#### ★ 主线：Retrieval-Grounded Self-Verification (CoVe)

**为什么选**：
- 综述 §VI-F 把 CoVe 列为 **medical / legal / compliance domain 的首选**，与本项目高度同构（"audit decision + 可审计 reason"）。
- 直击核心 pain：本项目 hallucination 定义就是"reason 提到 attrs 没有的属性"，正是 claim-level verification 适用场景。
- 与 FIPO 的 token-level confidence 互补：FIPO 给 token 概率视图，CoVe 给 claim 语义视图。

**控制流**（落地版）：
```
1. policy 第一次 generate → audit JSON v0
2. claim extractor 拆 reason → ["图中可见 Nike logo", "含棉量低", ...]
3. for each claim:
     视觉检索 top-3 + BM25 top-3 → 拼 evidence pack
     BGE 算 claim ↔ evidence 余弦
     if 余弦 < 0.45: mark as unsupported
4. reason rewriter: 删除 unsupported claims（**prune-only，不 add**）
5. 输出 JSON v1，可选附 evidence 引用
```

**关键安全约束**（综述 §IX-F）：CoVe 的 retrieval 结果**只用于剪枝**，不用于补 reason。否则 cascading hallucination 直接拉满——综述明确警告"retrieve-then-add" mode 在本场景必失败。

**预期收益**：hallucination 从 23.04% 进一步降到 17-19% 区间（外推自 reward_fn v2 的 BGE 阈值经验）。

#### 副线：Plan-Then-Retrieve（按品类路由的检索）

**为什么选**：
- 现 BM25 是全语料（170 文档）召回，对单条查询毫无针对性。
- 与本项目"硬违规扩充"（烟草/管制）天然契合：一旦 category 命中硬违规清单，强制走 banned-list 检索路径，跳过 confidence gate。
- 综述 §VI-A 警告"plan 错全错"——我们用 category 作为 plan 的离散 anchor（远比开放 sub-question 稳定），失败模式可控。

**控制流**：
```
轻量 category classifier（可用 v0 输出的 category 字段）
  → 命中硬违规白名单（烟草/酒/管制药品/枪支仿制品）→ 强制 RAG 走 banned-list
  → 服装 / 食品 / 化妆品 / 电子 → 路由到品类专属规则集 + 案例集
  → 其它 → fallback 到当前全语料 BM25
```

实现复杂度：低（~30 LoC，主要修 [src/stage4_rag/indexer.py](../src/stage4_rag/indexer.py) 加 category 字段索引）。

#### 选做：Tool-Augmented（OCR + 局部 crop）

**为什么选（弱推荐）**：
- 综述 §VI-D 明确说 tool routing reliability 是 first-class failure point；我们只引入 1-2 个高确定性 tool 控制风险。
- 现实需求：violation_cases 至少 30% 涉及"看包装文字"（保质期、SC、禁用词），Qwen3-VL 在小字 OCR 上不可靠 → PaddleOCR 是性价比最高的 tool。

**控制流**（限定循环深度 ≤ 1）：
```
当 confidence 低 + category ∈ {食品, 化妆品, 电子产品}:
  调用 PaddleOCR(image) → ocr_text
  ocr_text 注入 user prompt
  二次 generate
```

**安全约束**：
- 不允许 LLM 自由决定何时调 OCR（避免 tool hallucination — §IX-C 警告）
- 调用层做 deterministic routing，不进 reasoning loop

### 6.3 不选的 pattern 与理由

| 不选 | 理由 |
|---|---|
| **IRCoT / Decomposition-Based** | 本任务非 multi-hop；综述 §VI-C 说 prompt prefix 累积"extremely expensive"，我们已经 max_prompt=8192 没冗余 |
| **Multi-Agent Collaboration** | 综述 §VI-E 标注"highest token amplification profile"；8B 单模型 role-split 收益不抵 token 翻 2-3 倍 |
| **Human-in-the-Loop** | 跟生产自动化目标矛盾；综述 §VI-G 自己也说 HITL "fundamentally breaks continuous system autonomy" |
| **Long-term Memory / CMA** | 单条审核任务无 session continuity；引入即承担 memory poisoning 风险（§IX-E）；ROI 极低 |

### 6.4 评估方式（响应综述 §VII 的呼吁）

综述强烈批评仅用 final-answer accuracy 评估 agentic 系统会"obscure failure modes"。本项目落地时新增 trajectory-level 指标：

| 指标 | 含义 | 监控点 |
|---|---|---|
| `claim_extracted_count` | CoVe 抽出的 claim 数（per sample） | 太低 = extractor 失效；太高 = reason 过长 |
| `claim_pruned_rate` | 被剪枝 claim 占比 | 过高 → 模型生成原本就糟；过低 → CoVe 没起作用 |
| `evidence_recall@k` | 视觉检索 top-k 内出现真正相关 case 的比例 | retrieval misalignment 早期信号 |
| `cascading_hallu_delta` | RAG-on vs RAG-off 时 hallucination 差值 | **若 > 0** 立即报警（v1 已经踩过这坑） |
| `tool_call_success_rate` | OCR 工具调用解析成功率 | tool misuse 监控（§IX-C） |

### 6.5 落地节奏

参考综述 §X 的研究方向（stable adaptive retrieval / cost-aware orchestration / formal evaluation / oversight）：

1. **第一批**（与 hallucination 指标 H0/H1/H2 同步）：CoVe 落地 + 新评估指标，纯验证收益；
2. **第二批**（与硬违规数据扩充同步）：Plan-Then-Retrieve 路由生效；
3. **第三批**（生产化）：OCR 工具接入 + cascading_hallu_delta 监控。

---

## 7. 关键 take-aways（一行复述）

1. v1 RAG 不是 agentic，是 confidence-gated 单步 retrieval — 这正是 README §7.2 "sft_aux+RAG 反而升幻觉" 的根因。
2. 本场景**最适合 CoVe + Plan-Then-Retrieve**，CoVe 必须 prune-only。
3. 不要为了 agentic 而 agentic — 多步循环每加一层都要付 cascading hallucination 的代价。
4. Trajectory-level metric 比 terminal accuracy 重要 — 至少要监控 `cascading_hallu_delta` 和 `claim_pruned_rate`。
5. Tool 越少越好（OCR 是性价比最高的一个，停在那就够）。

---

> **更新记录**
> - 2026-05-03：基于 v2-opd 分支需求初次撰写，与 §6 决策同步至 README §10.5 OPD 升级方案。
