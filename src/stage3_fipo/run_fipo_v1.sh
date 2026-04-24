#!/usr/bin/env bash
# FIPO v1 single-node launch script for VLM e-commerce audit.
#
# Prereqs (one-time):
#     1. (already done) verl-latest installed --no-deps into the VLM conda env;
#        tensordict / codetiming / torchdata / pybind11 / pylatexenc added.
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

# Always clean up our own ray/vLLM workers on exit, otherwise GPU residue
# (e.g. a leaked VLLM::Worker holding 30GB) breaks the next launch with OOM.
cleanup() {
    pkill -9 -f VLLM 2>/dev/null || true
    pkill -9 -f "ray::"  2>/dev/null || true
    pkill -9 -P $$ 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --------------------------------------------------------------------- env
ENV_NAME="${ENV_NAME:-VLM}"
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# bge-small-zh-v1.5 (used by reward_fn v2 semantic alignment) is already
# cached at ~/.cache/huggingface/hub. hf-mirror occasionally hangs on the
# HEAD etag check, blocking every RewardLoopWorker for 5+ min of retries.
# Force offline mode so we trust the local cache (one-time download done).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# When CUDA_VISIBLE_DEVICES is explicitly set by the user (which we do for
# multi-tenant GPU sharing), Ray will NOT auto-isolate per-actor GPUs.
# Without this flag, all 6 actor workers see {0..5} and torch defaults each
# rank to device 0 → NCCL "Duplicate GPU detected" at FSDP broadcast.
# With this flag, verl's worker.py:_setup_env_cuda_visible_devices uses
# ray.get_accelerator_ids() to pin each worker to its own physical GPU.
# (Ref: vendor/verl-latest/verl/workers/rollout/sglang_rollout/sglang_rollout.py:213)
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Activate sitecustomize's torch.cuda.set_device remap so that non-contiguous
# CUDA_VISIBLE_DEVICES (e.g. "0,3,4,5,6,7" when GPU 1/2 are taken) works with
# verl's worker.set_device(physical_id) call. Inherited by every Ray worker.
export FIPO_PATCH_VERL="${FIPO_PATCH_VERL:-1}"

# Push Ray's CPU-RAM OOM kill threshold from the default 0.95 to 0.97.
# We have 944GB of host RAM; FSDP param_offload alone for 8B-actor + 8B-ref
# (~96GB combined) plus rollout buffers + bge encoder spikes near 0.95 when
# rollout sequences are long. Bumping the threshold gives ~20GB of headroom
# without disabling Ray's safety net entirely.
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.97}"

# --------------------------------------------------------------------- paths
PROJECT_NAME="${PROJECT_NAME:-vlm-posttraining-ecommerce}"
EXP_NAME="${EXP_NAME:-FIPO-v1-rule-reward}"
# verl supports console / wandb / swanlab / mlflow / tensorboard.
# Override with LOGGERS=console,swanlab when running for-real experiments.
LOGGERS="${LOGGERS:-console}"
MODEL_PATH="${MODEL_PATH:-${PWD}/models/sft_aux_merged}"
TRAIN_FILE="${TRAIN_FILE:-${PWD}/data/fipo/train.parquet}"
VAL_FILE="${VAL_FILE:-${PWD}/data/fipo/val.parquet}"
# RL ckpts are large (~99GB each, FSDP shard + optim state + HF dump).
# Default to NFS to avoid filling local SSD; override with CKPTS_DIR env var.
CKPTS_DIR="${CKPTS_DIR:-/mnt/nfs/young/VLM4reasoning/rl_ckpt/${EXP_NAME}}"

# --------------------------------------------------------------------- shape
N_GPUS="${N_GPUS:-1}"               # 1×L20 minimum (8B + LoRA-merged + vllm rollout)
BATCH_SIZE="${BATCH_SIZE:-6}"        # train_prompt_bsz; must satisfy (BATCH_SIZE * ROLLOUT_N) % (N_GPUS * MICRO_BSZ_PER_GPU) == 0
N_RESP="${N_RESP:-8}"                # rollouts per prompt
MINI_BSZ="${MINI_BSZ:-3}"            # PPO mini-batch (prompts); (MINI_BSZ * ROLLOUT_N) must also be divisible by minimal_bsz
MICRO_BSZ_PER_GPU="${MICRO_BSZ_PER_GPU:-1}"   # PPO micro-batch (per-GPU, conservative)
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-8192}"   # Qwen3-VL: 1 image up to ~3k image tokens + text;
                                           # image tokens cannot be truncated, so headroom matters
MAX_RESP_LEN="${MAX_RESP_LEN:-1024}"
GEN_TP="${GEN_TP:-1}"

# --------------------------------------------------------------------- distributed strategy
# verl-latest supports fsdp / fsdp2 / megatron. We use fsdp2 by default
# (per-param sharding, better CPU offload, finer wrap for Qwen3-VL vision tower).
# Drop to "fsdp" if FSDP2 hits DTensor/vLLM-swap edge cases.
ACTOR_STRATEGY="${ACTOR_STRATEGY:-fsdp2}"
REF_STRATEGY="${REF_STRATEGY:-fsdp2}"

# --------------------------------------------------------------------- FIPO knobs
# verl-latest's PolicyLossConfig is a strict dataclass and rejects unknown
# Hydra fields, so future_kl knobs are passed via env vars (read inside
# src/stage3_fipo/verl_patches/future_kl_loss.py).
LOSS_MODE="${LOSS_MODE:-future_kl}"  # set to "vanilla" to ablate FIPO -> standard GRPO
export FIPO_DECAY_RATE="${FIPO_DECAY_RATE:-12.0}"
export FIPO_CHUNK_SIZE="${FIPO_CHUNK_SIZE:-128}"
export FIPO_FKL_CLIP_RATIO="${FIPO_FKL_CLIP_RATIO:-0.2}"
export FIPO_FKL_CLIP_HIGH_ONLY="${FIPO_FKL_CLIP_HIGH_ONLY:-false}"
export FIPO_SAFETY_THRESH="${FIPO_SAFETY_THRESH:-4.0}"

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
    actor_rollout_ref.actor.strategy=${ACTOR_STRATEGY} \
    actor_rollout_ref.ref.strategy=${REF_STRATEGY} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BSZ} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE} \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${N_RESP} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LEN + MAX_RESP_LEN)) \
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LEN + MAX_RESP_LEN)) \
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LEN} \
    actor_rollout_ref.rollout.response_length=${MAX_RESP_LEN} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    reward.reward_manager.source=importlib \
    reward.reward_manager.name=VLMAuditRewardManager \
    reward.reward_manager.module.path="${PWD}/src/stage3_fipo/verl_patches/reward_manager.py" \
    reward.reward_model.enable=False \
    trainer.logger="$(python -c "import sys,json; print(json.dumps([s.strip() for s in sys.argv[1].split(',') if s.strip()]))" "${LOGGERS}")" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=40 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.total_epochs=2 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    "$@"
