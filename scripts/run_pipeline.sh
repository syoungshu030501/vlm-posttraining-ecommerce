#!/usr/bin/env bash
# ============================================================
# End-to-end pipeline: Stage 0 → Stage 4
#
# Run the full training pipeline from data preparation to
# final evaluation. Each stage can be run independently
# by setting STAGE_START and STAGE_END.
#
# Usage:
#   bash scripts/run_pipeline.sh                     # Full pipeline
#   STAGE_START=1 STAGE_END=1 bash scripts/run_pipeline.sh  # SFT only
#   STAGE_START=3 bash scripts/run_pipeline.sh       # RL onwards
# ============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

STAGE_START="${STAGE_START:-0}"
STAGE_END="${STAGE_END:-4}"

# ---- Python env ----
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PY="${PYTHON_BIN}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PY="${VIRTUAL_ENV}/bin/python"
else
    PY="$(command -v python3 || command -v python)"
fi
if [[ -z "$PY" || ! -x "$PY" ]]; then
    echo "[ERROR] Could not locate a usable Python interpreter."
    exit 1
fi
echo "Python: $($PY --version)"

# ---- Paths ----
MODEL_ROOT="${MODEL_ROOT:-/mnt/nfs/young/VLM4reasoning/models/pretrained}"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/models}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/Qwen3-VL-8B-Instruct}"
RAW_ANN="${RAW_ANN:-data/raw/annotations.jsonl}"
RAW_IMAGES="${RAW_IMAGES:-data/raw/images}"
SFT_DIR="data/sft"
PREF_DIR="data/preference"
SFT_CKPT="${SFT_CKPT:-${CKPT_ROOT}/sft_ckpt}"
SFT_MERGED="${SFT_MERGED:-${CKPT_ROOT}/sft_merged}"
RM_CKPT="${RM_CKPT:-${CKPT_ROOT}/rm_ckpt}"
RL_CKPT="${RL_CKPT:-${CKPT_ROOT}/rl_ckpt}"
RAG_INDEX="data/rag_index"
N_GPUS="${N_GPUS:-1}"
mkdir -p "$CKPT_ROOT" "$RESULTS_DIR"

should_run() { [[ "$1" -ge "$STAGE_START" && "$1" -le "$STAGE_END" ]]; }

# ============================================================
# Stage 0: Data preparation + distillation
# ============================================================
if should_run 0; then
echo ""
echo "============================================================"
echo "  Stage 0: Data Preparation"
echo "============================================================"

# 0a. Distill SFT data (requires API key)
if [[ -n "${OPENAI_API_KEY:-}" || -n "${DASHSCOPE_API_KEY:-}" ]]; then
    echo ">>> Distilling SFT samples..."
    $PY -m src.stage0_distill.distill \
        --image_dir "$RAW_IMAGES" \
        --annotation_file "$RAW_ANN" \
        --out_dir "$SFT_DIR" \
        --mode sft \
        --model "${TEACHER_MODEL:-qwen-vl-plus}"

    echo ">>> Distilling preference pairs..."
    $PY -m src.stage0_distill.distill \
        --image_dir "$RAW_IMAGES" \
        --annotation_file "$RAW_ANN" \
        --out_dir "$PREF_DIR" \
        --mode preference \
        --model "${TEACHER_MODEL:-qwen-vl-plus}"
else
    echo "[SKIP] No OPENAI_API_KEY set. Using existing data in ${SFT_DIR}/"
fi

# 0b. Convert JSONL → Parquet + split
if [[ -f "${SFT_DIR}/sft.jsonl" ]]; then
    echo ">>> Converting JSONL to parquet..."
    $PY -m src.utils.data_prep \
        --annotation_file "${SFT_DIR}/sft.jsonl" \
        --image_dir "$RAW_IMAGES" \
        --out_dir "$SFT_DIR" \
        --mode sft --split
fi

# 0c. Build triplets for hallucination loss
if [[ -f "$RAW_ANN" ]]; then
    echo ">>> Building hallucination triplets..."
    $PY -m src.utils.build_triplets \
        --annotation_file "$RAW_ANN" \
        --image_dir "$RAW_IMAGES" \
        --out_file "${SFT_DIR}/triplets.parquet"
fi

echo "[Stage 0 done]"
fi

# ============================================================
# Stage 1: LoRA SFT
# ============================================================
if should_run 1; then
echo ""
echo "============================================================"
echo "  Stage 1: LoRA SFT (CE + SupCon + Triplet)"
echo "============================================================"
$PY -m src.stage1_sft.train \
    --model_path "$MODEL_PATH" \
    --train_parquet "${SFT_DIR}/train.parquet" \
    --triplet_parquet "${SFT_DIR}/triplets.parquet" \
    --out_dir "$SFT_CKPT" \
    --epochs 3 \
    --batch_size 1 \
    --grad_accum 16 \
    --project_name "${SWANLAB_PROJECT:-vlm-posttraining}" \
    --experiment_name "${SWANLAB_STAGE1_NAME:-stage1-sft}" \
    --flash_attn

echo ">>> Merging LoRA into base model..."
$PY -m src.utils.merge_lora \
    --base_model "$MODEL_PATH" \
    --lora_path "${SFT_CKPT}/best" \
    --out_dir "$SFT_MERGED"

echo ">>> Evaluating SFT model..."
$PY scripts/evaluate.py \
    --model_path "$SFT_MERGED" \
    --test_parquet "${SFT_DIR}/test.parquet" \
    --project_name "${SWANLAB_PROJECT:-vlm-posttraining}" \
    --experiment_name "${SWANLAB_STAGE1_EVAL_NAME:-stage1-sft-eval}" \
    --out "${RESULTS_DIR}/stage1_sft.json"

echo "[Stage 1 done]"
fi

# ============================================================
# Stage 2: Reward Model
# ============================================================
if should_run 2; then
echo ""
echo "============================================================"
echo "  Stage 2: Reward Model (Bradley-Terry)"
echo "============================================================"
$PY -m src.stage2_rm.train \
    --model_path "$MODEL_PATH" \
    --train_parquet "${PREF_DIR}/preference.parquet" \
    --out_dir "$RM_CKPT" \
    --epochs 3 \
    --project_name "${SWANLAB_PROJECT:-vlm-posttraining}" \
    --experiment_name "${SWANLAB_STAGE2_NAME:-stage2-rm}" \
    --flash_attn

echo "[Stage 2 done]"
fi

# ============================================================
# Stage 3: FIPO RL
# ============================================================
if should_run 3; then
echo ""
echo "============================================================"
echo "  Stage 3: FIPO RL Optimization"
echo "============================================================"
PYTHON_BIN="$PY" N_GPUS="$N_GPUS" bash src/stage3_fipo/run_fipo.sh \
    "$SFT_MERGED" \
    "$RM_CKPT" \
    "$RL_CKPT" \
    "${SFT_DIR}/train.parquet"

echo ">>> Evaluating FIPO model..."
$PY scripts/evaluate.py \
    --model_path "${RL_CKPT}/latest" \
    --test_parquet "${SFT_DIR}/test.parquet" \
    --project_name "${SWANLAB_PROJECT:-vlm-posttraining}" \
    --experiment_name "${SWANLAB_STAGE3_EVAL_NAME:-stage3-fipo-eval}" \
    --out "${RESULTS_DIR}/stage3_fipo.json"

echo "[Stage 3 done]"
fi

# ============================================================
# Stage 4: RAG Index (optional)
# ============================================================
if should_run 4; then
echo ""
echo "============================================================"
echo "  Stage 4: Building RAG indices"
echo "============================================================"
$PY -m src.stage4_rag.indexer \
    --image_dir "$RAW_IMAGES" \
    --rule_file "data/raw/rules.jsonl" \
    --out_dir "$RAG_INDEX"

echo "[Stage 4 done]"
fi

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "  Results in: ${RESULTS_DIR}"
echo "============================================================"
ls -lh "${RESULTS_DIR}" 2>/dev/null || echo "  (no results yet)"
