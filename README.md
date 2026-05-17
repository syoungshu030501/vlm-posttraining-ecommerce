# VLM-posttraining

A research repository for **vision-language Process Reward Models (PRMs)**.

## What this repo is

This is the active codebase for a VLM PRM benchmark seeded by
[VisualPRM400K](https://huggingface.co/datasets/OpenGVLab/VisualPRM400K) and
evaluated against
[VisualProcessBench](https://huggingface.co/datasets/OpenGVLab/VisualProcessBench).
The PRM scores reasoning chains at the *token level* and is trained with a
Bradley-Terry pairwise objective in which per-token rewards are mean-pooled
over the response mask before the BT log-sigmoid is taken — so gradients
reach every response token rather than only the final one.

This is the structural property that distinguishes a Process Reward Model
from a classic Outcome RM and is what makes a PRM useful as a step-level
critic for chain-of-thought generations.

## What this repo *was*

Until April 2026 the repository was an end-to-end e-commerce compliance
audit pipeline (five stages: API distillation → SFT → outcome-RM →
FIPO/GRPO RL → CLIP+FAISS RAG). The data was self-built and quality was
limited, so the project was pivoted to a public-dataset benchmark.

The frozen pipeline is preserved verbatim under [`legacy/`](legacy/) — see
[legacy/README.md](legacy/README.md) for how to reproduce it and
[legacy/README_orig.md](legacy/README_orig.md) for the original write-up.

## What's in `src/` today

The active `src/` tree is intentionally minimal — only the canonical PRM
plus the four backbone-agnostic utilities that survive the pivot:

```
src/
├── prm/
│   ├── __init__.py             # re-exports ProcessRewardModel, prm_bt_loss
│   └── model.py                # mean-pool token-level PRM (≈130 LOC)
└── utils/
    ├── model_loader.py         # Qwen2.5-VL / Qwen3-VL family detection + LoRA
    ├── merge_lora.py           # fold PEFT adapter into base weights
    ├── tracking.py             # optional SwanLab logging
    └── json_utils.py           # tolerant JSON extraction for VLM outputs
```

### PRM head

```
LayerNorm(hidden) → Linear(hidden, hidden//4) → GELU → Dropout(0.1)
    → Linear(hidden//4, 1)
```

The backbone is frozen at construction; if you want LoRA, apply it to the
base model *before* wrapping with `ProcessRewardModel`.

## Status

**Stub only.** The PRM module is in place, but the benchmark loaders, the
training loop, and the eval harness against VisualProcessBench are the next
deliverables. The dataset (`data/`) and checkpoint (`models/`) directories
are NFS-backed and gitignored; nothing in this repo will pull weights or
data automatically.

Concretely, what's *missing* before training can start:

- `src/data/visualprm400k.py` — HF dataset loader yielding
  `(image, prompt, chosen_steps, rejected_steps)` pairs with response masks.
- `src/training/train_prm.py` — PRM training loop with `prm_bt_loss`.
- `src/evaluation/visualprocessbench.py` — eval harness reporting
  step-level accuracy / F1.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The current `requirements.txt` is inherited from the legacy pipeline and is
larger than the new code path strictly needs; it will be slimmed once the
benchmark code lands and we know the real dependency set.

## Quick check

```python
from src.prm import ProcessRewardModel, prm_bt_loss
from src.utils.model_loader import load_model_and_processor

base, processor = load_model_and_processor(
    "Qwen/Qwen3-VL-8B-Instruct", apply_lora=True
)
prm = ProcessRewardModel(base)
```

## Citation / License

To be added once a release tag is cut.
