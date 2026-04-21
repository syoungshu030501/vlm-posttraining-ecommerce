#!/usr/bin/env bash
# Launches a training stage with timestamped log and metadata header.
# Usage:
#   bash scripts/launch_training.sh <stage> <gpu_ids> <experiment_name> [extra args...]
# Example:
#   bash scripts/launch_training.sh sft 1 vlm-posttraining-ecommerce-SFT
set -euo pipefail

STAGE="$1"
GPU_IDS="$2"
EXP_NAME="$3"
shift 3

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PY="/home/young/miniconda3/envs/VLM/bin/python"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/logs/runs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${STAGE}_${EXP_NAME}_${TS}.log"

# SwanLab credentials read from ~/.swanlab/.netrc (saved by `swanlab login`)
if [[ -z "${SWANLAB_API_KEY:-}" && -f "$HOME/.swanlab/.netrc" ]]; then
    export SWANLAB_API_KEY="$(awk '/password/ {print $2; exit}' "$HOME/.swanlab/.netrc")"
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="/mnt/nfs/young/VLM4reasoning/models/pretrained/Qwen3-VL-8B-Instruct"
PROJECT_NAME="vlm-posttraining-ecommerce"

{
    echo "=========================================================="
    echo "  Run metadata"
    echo "=========================================================="
    echo "stage              : $STAGE"
    echo "experiment_name    : $EXP_NAME"
    echo "swanlab_project    : $PROJECT_NAME"
    echo "started_at         : $(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "host               : $(hostname)"
    echo "user               : $(whoami)"
    echo "cwd                : $(pwd)"
    echo "git_commit         : $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
    echo "git_status         :"; git status --short 2>/dev/null | sed 's/^/    /' || true
    echo "python             : $PY"
    echo "python_version     : $($PY --version 2>&1)"
    echo "torch              : $($PY -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)')"
    echo "transformers       : $($PY -c 'import transformers; print(transformers.__version__)')"
    echo "peft               : $($PY -c 'import peft; print(peft.__version__)')"
    echo "swanlab            : $($PY -c 'import swanlab; print(swanlab.__version__)')"
    echo "model_path         : $MODEL_PATH"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "GPU snapshot:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv | sed 's/^/    /'
    echo "----------------------------------------------------------"
    echo "command            : $@"
    echo "log_file           : $LOG_FILE"
    echo "=========================================================="
    echo
} | tee "$LOG_FILE"

exec "$@" 2>&1 | tee -a "$LOG_FILE"
