#!/usr/bin/env bash
# ============================================================
# RM head architecture ablation: v0 (baseline, already running)
# vs v1 (bias=True + LayerNorm).
#
# This script polls the v0 log until it sees "RM training complete.",
# then immediately launches v1 on the same GPU. Both runs share the
# same backbone (frozen Qwen3-VL base) and same data, so only the
# head architecture differs.
#
# Usage:
#   bash scripts/run_rm_head_ablation.sh \
#       <gpu_id> \
#       <v0_log_file> \
#       [v1_experiment_name]
#
# Example:
#   bash scripts/run_rm_head_ablation.sh 3 \
#       logs/runs/rm_vlm-posttraining-ecommerce-RM_20260421_195755.log
# ============================================================
set -euo pipefail

GPU_ID="${1:?need gpu id}"
V0_LOG="${2:?need v0 log path}"
V1_NAME="${3:-vlm-posttraining-ecommerce-RM-headv1}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

POLL_INTERVAL="${POLL_INTERVAL:-60}"
COMPLETE_MARKER="RM training complete."

echo "[watcher] waiting for v0 to finish: $V0_LOG"
echo "[watcher] poll every ${POLL_INTERVAL}s, marker='${COMPLETE_MARKER}'"

while true; do
    if [[ -f "$V0_LOG" ]] && grep -q "$COMPLETE_MARKER" "$V0_LOG"; then
        echo "[watcher] v0 finished at $(date '+%F %T')"
        break
    fi
    sleep "$POLL_INTERVAL"
done

# Quick safety: make sure GPU is actually free before launching v1
sleep 30
echo "[watcher] launching v1 on GPU ${GPU_ID}: ${V1_NAME}"

PY="/home/young/miniconda3/envs/VLM/bin/python"
MODEL_PATH="/mnt/nfs/young/VLM4reasoning/models/pretrained/Qwen3-VL-8B-Instruct"
PREF_PARQUET="data/preference/preference.parquet"
RM_CKPT_V1="${PROJECT_ROOT}/models/rm_ckpt_headv1"

bash scripts/launch_training.sh rm "$GPU_ID" "$V1_NAME" \
    "$PY" -m src.stage2_rm.train \
        --model_path "$MODEL_PATH" \
        --train_parquet "$PREF_PARQUET" \
        --out_dir "$RM_CKPT_V1" \
        --epochs 3 \
        --batch_size 2 \
        --lr 1e-4 \
        --project_name "vlm-posttraining-ecommerce" \
        --experiment_name "$V1_NAME" \
        --head_bias \
        --head_layernorm
