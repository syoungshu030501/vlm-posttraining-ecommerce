#!/usr/bin/env bash
# FIPO v1 single-node launch script for VLM e-commerce audit.
#
# Prereqs (one-time):
#     1. bash scripts/setup_fipo_env.sh                       (~30 min)
#     2. python -m src.stage3_fipo.prepare_fipo_data \
#            --in_train data/sft/train.parquet \
#            --in_val   data/sft/val.parquet \
#            --out_dir  data/fipo --max_train 2000 --max_val 200
#
# Then:
#     bash src/stage3_fipo/run_fipo_v1.sh
#
# Override knobs via env vars, e.g.:
#     N_GPUS=2 BATCH_SIZE=8 bash src/stage3_fipo/run_fipo_v1.sh

set -euo pipefail

# --------------------------------------------------------------------- env
ENV_NAME="${ENV_NAME:-VLM_FIPO}"
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}:${PWD}/vendor/verl-latest:${PYTHONPATH:-}"

# --------------------------------------------------------------------- paths
PROJECT_NAME="${PROJECT_NAME:-vlm-posttraining-ecommerce}"
EXP_NAME="${EXP_NAME:-FIPO-v1-rule-reward}"
MODEL_PATH="${MODEL_PATH:-${PWD}/models/sft_aux_merged}"
TRAIN_FILE="${TRAIN_FILE:-${PWD}/data/fipo/train.parquet}"
VAL_FILE="${VAL_FILE:-${PWD}/data/fipo/val.parquet}"
CKPTS_DIR="${CKPTS_DIR:-${PWD}/models/rl_ckpt/${EXP_NAME}}"

# --------------------------------------------------------------------- shape
N_GPUS="${N_GPUS:-1}"               # 1×L20 minimum (8B + LoRA-merged + vllm rollout)
BATCH_SIZE="${BATCH_SIZE:-4}"        # train_prompt_bsz
N_RESP="${N_RESP:-8}"                # rollouts per prompt
MINI_BSZ="${MINI_BSZ:-2}"            # PPO mini-batch (prompts)
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
MAX_RESP_LEN="${MAX_RESP_LEN:-1024}"
GEN_TP="${GEN_TP:-1}"

# --------------------------------------------------------------------- FIPO knobs
LOSS_MODE="${LOSS_MODE:-future_kl}"  # set to "vanilla" to ablate FIPO -> standard GRPO
DECAY_RATE="${DECAY_RATE:-12.0}"     # smaller than 32B's 32.0 (shorter responses)
CHUNK_SIZE="${CHUNK_SIZE:-128}"
FKL_CLIP="${FKL_CLIP:-0.2}"
FKL_CLIP_HIGH_ONLY="${FKL_CLIP_HIGH_ONLY:-false}"
SAFETY_THRESH="${SAFETY_THRESH:-4.0}"

# --------------------------------------------------------------------- launch
mkdir -p "${CKPTS_DIR}"

python -m src.stage3_fipo.main_fipo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.prompt_key=prompt \
    data.image_key=images \
    data.truncation=left \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESP_LEN} \
    data.train_batch_size=${BATCH_SIZE} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BSZ} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE} \
    +actor_rollout_ref.actor.policy_loss.decay_rate=${DECAY_RATE} \
    +actor_rollout_ref.actor.policy_loss.chunk_size=${CHUNK_SIZE} \
    +actor_rollout_ref.actor.policy_loss.future_kl_clip_ratio=${FKL_CLIP} \
    +actor_rollout_ref.actor.policy_loss.future_kl_clip_high_only=${FKL_CLIP_HIGH_ONLY} \
    +actor_rollout_ref.actor.policy_loss.safety_thresh=${SAFETY_THRESH} \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${N_RESP} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LEN + MAX_RESP_LEN)) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    reward_model.reward_manager=vlm_audit_v2 \
    reward_model.enable=False \
    trainer.logger='["console"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=20 \
    trainer.total_epochs=2 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    "$@"
