# legacy/ — frozen e-commerce VLM compliance audit pipeline

This directory is an **archive** of the project's original work: a
five-stage VLM post-training pipeline for electronic-commerce listing
compliance auditing (image + title/description → violation type + binary
verdict, both pointwise and with retrieval augmentation).

It is kept intact so the original numbers can be reproduced, but
**no new development happens here**. New work lives at the repository
root — see [../README.md](../README.md).

## Original write-up

[README_orig.md](README_orig.md) is the verbatim copy of the project's
top-level README at the time of the pivot. It contains the full
methodology, evaluation numbers, and run logs. Start there.

[STAGE2_V3_RUNBOOK.md](STAGE2_V3_RUNBOOK.md) captures the in-flight Stage 2
v3 (field-wise PRM) work that was paused at the time of the freeze.

## Layout

```
legacy/
├── README_orig.md              # original top-level README
├── STAGE2_V3_RUNBOOK.md        # Stage 2 v3 (field-wise PRM) plan
├── requirements.txt            # snapshot of root requirements at freeze
├── sitecustomize.py            # repo-root path injection (legacy-only)
├── configs/                    # Hydra train/model configs
├── docs/                       # DATA_ENGINEERING, INTERVIEW (gitignored), ...
├── reference/                  # data-redesign-2026 + SoK-agentic-RAG summary
├── scripts/                    # data prep + training launchers
│   ├── data/                   # S0Data, S1Data, S2Data(V3), S4Data(_v3), SAData
│   └── *.sh                    # download / launch / pipeline drivers
├── src/                        # the five stages
│   ├── schema.py               # VIOLATION_TYPES, COARSE_CATEGORIES, SYSTEM_PROMPT
│   ├── stage0_distill/         # API distillation
│   ├── stage1_sft/             # supervised fine-tune (LoRA)
│   ├── stage2_rm/              # reward model (outcome + process + v3 field-wise)
│   ├── stage3_fipo/            # FIPO RL (future-KL) + GRPO
│   ├── stage4_rag/             # CLIP+FAISS+BM25 retrieval + inference
│   └── utils/                  # data_prep, build_triplets, model_loader, ...
└── vendor/
    ├── FIPO-main/              # FIPO/veRL training framework (Stage 3 only)
    └── verl-latest/            # gitignored; clone separately to reproduce
```

## Reproducing the original work

All imports inside `legacy/` resolve relative to `legacy/` itself (e.g.
`from src.schema import VIOLATION_TYPES` → `legacy/src/schema.py`). Run
every legacy command from inside this directory:

```bash
cd legacy
pip install -r requirements.txt

# Stage 0 — distill labels from a capable API model
python -m src.stage0_distill.distill --config configs/train.yaml

# Stage 1 — SFT with LoRA
python -m src.stage1_sft.train     --config configs/train.yaml

# Stage 2 — outcome / process reward model
python -m src.stage2_rm.train      --config configs/train.yaml

# Stage 3 — FIPO RL (requires vendor/verl-latest; see ../README.md note)
bash src/stage3_fipo/run_fipo.sh

# Stage 4 — retrieval + RAG inference
python -m src.stage4_rag.inference --config configs/train.yaml
```

The `vendor/verl-latest/` clone (~250MB) is intentionally gitignored;
re-clone it if you need to rerun Stage 3:

```bash
git clone https://github.com/volcengine/verl vendor/verl-latest
```

Large artefacts (`data/`, `models/`, `logs/`, `swanlog/`, `outputs/`,
`results/`) are all gitignored — they live on NFS in the original
environment and are not redistributable here.
