#!/usr/bin/env bash
# ============================================================
# Stage 3: FIPO RL launch script
# Adapted from FIPO/recipe/fipo/run_fipo_qwen2.5_32b.sh
# for single-node 8-GPU (or 4-GPU) setup with Qwen2.5-VL-7B/9B
# ============================================================

set -euo pipefail

# ---- Path configuration ----
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIPO_ROOT="${FIPO_ROOT:-${PROJECT_ROOT}/vendor/FIPO}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SFT_CKPT="${1:-${PROJECT_ROOT}/models/sft_ckpt/epoch-3}"     # SFT checkpoint (LoRA merged)
RM_CKPT="${2:-${PROJECT_ROOT}/models/rm_ckpt}"
OUTPUT_DIR="${3:-${PROJECT_ROOT}/models/rl_ckpt}"
TRAIN_DATA="${4:-${PROJECT_ROOT}/data/sft/train.parquet}"

# ---- Cluster config ----
N_GPUS=${N_GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29500}

# ---- FIPO core parameters ----
# Key: adv_estimator stays as grpo; FIPO only changes the policy loss function
# Set LOSS_MODE=vanilla from env to run standard GRPO (ablation baseline)
LOSS_MODE="${LOSS_MODE:-future_kl}"   # future_kl=FIPO, vanilla=standard GRPO
DECAY_RATE=12.0               # Short responses (≤512 tok) → smaller half-life
CHUNK_SIZE=64
FUTURE_KL_START="include_current"
FUTURE_KL_WINDOW=-1           # Full sequence
FUTURE_KL_AVERAGE=False
FUTURE_KL_CLIP_RATIO=0.2
FUTURE_KL_CLIP_HIGH_ONLY=True
SAFETY_THRESH=5.0             # Tighter than math tasks due to multimodal reward noise

# ---- GRPO advantage estimator (unchanged by FIPO) ----
ADV_ESTIMATOR="grpo"

# ---- KL control (FIPO internalises KL, no external KL penalty) ----
USE_KL_IN_REWARD=False
KL_COEF=0.0
USE_KL_LOSS=False
KL_LOSS_COEF=0.0

# ---- Clipping ----
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
CLIP_RATIO_C=10.0

# ---- Sequence lengths ----
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=512       # JSON output is short; saves compute vs default 20480
N_RESP_PER_PROMPT=8           # 16 in math tasks; 8 to fit 4-GPU setup

# ---- Training ----
TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=4
EPOCHS=8
LR=1e-6

echo "============================================================"
echo "  FIPO RL training"
echo "  SFT ckpt : ${SFT_CKPT}"
echo "  N GPUs   : ${N_GPUS}"
echo "  Output   : ${OUTPUT_DIR}"
echo "============================================================"

cd "${FIPO_ROOT}" || { echo "FIPO not found: ${FIPO_ROOT}"; exit 1; }

"${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${N_GPUS}" \
    --master_port="${MASTER_PORT}" \
    verl/trainer/main_ppo.py \
    \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    \
    loss_mode="${LOSS_MODE}" \
    decay_rate="${DECAY_RATE}" \
    chunk_size="${CHUNK_SIZE}" \
    future_kl_start="${FUTURE_KL_START}" \
    future_kl_window="${FUTURE_KL_WINDOW}" \
    future_kl_average="${FUTURE_KL_AVERAGE}" \
    future_kl_clip_ratio="${FUTURE_KL_CLIP_RATIO}" \
    future_kl_clip_high_only="${FUTURE_KL_CLIP_HIGH_ONLY}" \
    safety_thresh="${SAFETY_THRESH}" \
    \
    use_kl_in_reward="${USE_KL_IN_REWARD}" \
    kl_coef="${KL_COEF}" \
    use_kl_loss="${USE_KL_LOSS}" \
    kl_loss_coef="${KL_LOSS_COEF}" \
    \
    clip_ratio_low="${CLIP_RATIO_LOW}" \
    clip_ratio_high="${CLIP_RATIO_HIGH}" \
    clip_ratio_c="${CLIP_RATIO_C}" \
    \
    data.train_files="${TRAIN_DATA}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.image_key="image" \
    \
    actor_rollout_ref.model.path="${SFT_CKPT}" \
    actor_rollout_ref.model.freeze_vision_encoder=true \
    actor_rollout_ref.rollout.n="${N_RESP_PER_PROMPT}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    \
    reward_model.reward_manager=dapo \
    \
    trainer.total_epochs="${EPOCHS}" \
    trainer.project_name="ecom-audit-fipo" \
    trainer.experiment_name="qwen-vl-fipo-run1" \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="${OUTPUT_DIR}"
