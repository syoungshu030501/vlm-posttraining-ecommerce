#!/usr/bin/env bash
# ============================================================
# Ablation study runner (Plan §10.2)
#
# 4 experiment groups:
#   1. Baseline       — SFT only
#   2. +GRPO          — SFT + GRPO (n=8)
#   3. +FIPO          — SFT + FIPO (n=8)     ← core comparison
#   4. +Contrast+FIPO — SFT(+contrastive) + FIPO
# ============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Paths
MODEL_PATH="${MODEL_PATH:-/mnt/nfs/young/VLM4reasoning/models/pretrained/Qwen3-VL-8B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-data/sft/train.parquet}"
TEST_DATA="${TEST_DATA:-data/sft/test.parquet}"
PREF_DATA="${PREF_DATA:-data/preference/preference.parquet}"
TRIPLET_DATA="${TRIPLET_DATA:-data/sft/triplets.parquet}"
RESULTS_DIR="${PROJECT_ROOT}/results/ablation"
mkdir -p "$RESULTS_DIR"

PY="${CONDA_PREFIX:-$(conda env list | grep ecom-audit | awk '{print $NF}')}/bin/python"

# ============================================================
# Experiment 1: Baseline (SFT only, no contrastive)
# ============================================================
echo "============================================================"
echo "  Experiment 1/4: Baseline (SFT only)"
echo "============================================================"
SFT_BASELINE_DIR="models/ablation/baseline_sft"
$PY -m src.stage1_sft.train \
    --model_path "$MODEL_PATH" \
    --train_parquet "$TRAIN_DATA" \
    --out_dir "$SFT_BASELINE_DIR" \
    --epochs 3 --batch_size 1 --grad_accum 16 \
    --flash_attn

# Merge LoRA
$PY -m src.utils.merge_lora \
    --base_model "$MODEL_PATH" \
    --lora_path "${SFT_BASELINE_DIR}/best" \
    --out_dir "models/ablation/baseline_merged"

# Evaluate baseline
$PY scripts/evaluate.py \
    --model_path "models/ablation/baseline_merged" \
    --test_parquet "$TEST_DATA" \
    --out "${RESULTS_DIR}/01_baseline.json"

# ============================================================
# Experiment 2: +GRPO (SFT + GRPO RL, no FIPO)
# ============================================================
echo "============================================================"
echo "  Experiment 2/4: +GRPO"
echo "============================================================"
# Use FIPO script but with loss_mode=vanilla (standard PPO loss)
LOSS_MODE=vanilla \
    N_GPUS="${N_GPUS:-1}" \
    bash src/stage3_fipo/run_fipo.sh \
        "models/ablation/baseline_merged" \
        "models/rm_ckpt" \
        "models/ablation/grpo_rl" \
        "$TRAIN_DATA"

$PY scripts/evaluate.py \
    --model_path "models/ablation/grpo_rl/latest" \
    --test_parquet "$TEST_DATA" \
    --out "${RESULTS_DIR}/02_grpo.json"

# ============================================================
# Experiment 3: +FIPO (SFT + FIPO RL) — core comparison
# ============================================================
echo "============================================================"
echo "  Experiment 3/4: +FIPO"
echo "============================================================"
N_GPUS="${N_GPUS:-1}" \
    bash src/stage3_fipo/run_fipo.sh \
        "models/ablation/baseline_merged" \
        "models/rm_ckpt" \
        "models/ablation/fipo_rl" \
        "$TRAIN_DATA"

$PY scripts/evaluate.py \
    --model_path "models/ablation/fipo_rl/latest" \
    --test_parquet "$TEST_DATA" \
    --out "${RESULTS_DIR}/03_fipo.json"

# ============================================================
# Experiment 4: +Contrastive + FIPO
# ============================================================
echo "============================================================"
echo "  Experiment 4/4: +Contrastive + FIPO"
echo "============================================================"
SFT_CONTRASTIVE_DIR="models/ablation/contrastive_sft"
$PY -m src.stage1_sft.train \
    --model_path "$MODEL_PATH" \
    --train_parquet "$TRAIN_DATA" \
    --triplet_parquet "$TRIPLET_DATA" \
    --out_dir "$SFT_CONTRASTIVE_DIR" \
    --epochs 3 --batch_size 1 --grad_accum 16 \
    --flash_attn

$PY -m src.utils.merge_lora \
    --base_model "$MODEL_PATH" \
    --lora_path "${SFT_CONTRASTIVE_DIR}/best" \
    --out_dir "models/ablation/contrastive_merged"

N_GPUS="${N_GPUS:-1}" \
    bash src/stage3_fipo/run_fipo.sh \
        "models/ablation/contrastive_merged" \
        "models/rm_ckpt" \
        "models/ablation/contrastive_fipo_rl" \
        "$TRAIN_DATA"

$PY scripts/evaluate.py \
    --model_path "models/ablation/contrastive_fipo_rl/latest" \
    --test_parquet "$TEST_DATA" \
    --out "${RESULTS_DIR}/04_contrastive_fipo.json"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "  Ablation Results"
echo "============================================================"
for f in "${RESULTS_DIR}"/*.json; do
    echo "--- $(basename "$f") ---"
    cat "$f"
    echo ""
done
