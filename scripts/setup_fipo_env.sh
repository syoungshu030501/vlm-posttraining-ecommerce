#!/usr/bin/env bash
# One-time setup for the FIPO conda environment.
# Isolated from the existing VLM env (whose vllm 0.19 conflicts with verl 0.8).
#
# Usage:
#     bash scripts/setup_fipo_env.sh
#
# Total ~25-40 min depending on network. Re-runnable.

set -euo pipefail

ENV_NAME="${ENV_NAME:-VLM_FIPO}"
PY_VER="${PY_VER:-3.12}"
TORCH_VER="${TORCH_VER:-2.6.0+cu124}"
VLLM_VER="${VLLM_VER:-0.10.2}"   # within verl-latest's >=0.8.5,<=0.12.0 range
TENSORDICT_VER="${TENSORDICT_VER:-0.8.3}"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[setup] env ${ENV_NAME} already exists; activating to install/update deps"
else
    echo "[setup] creating conda env ${ENV_NAME} (python ${PY_VER}) ..."
    conda create -n "${ENV_NAME}" "python=${PY_VER}" -y
fi

conda activate "${ENV_NAME}"

PIP_OPTS=("--index-url" "https://mirrors.aliyun.com/pypi/simple/")

echo "[setup] installing torch ${TORCH_VER} (cu124, aliyun mirror) ..."
pip install \
    "https://mirrors.aliyun.com/pytorch-wheels/cu124/torch-${TORCH_VER}-cp312-cp312-linux_x86_64.whl" \
    "https://mirrors.aliyun.com/pytorch-wheels/cu124/torchvision-0.21.0+cu124-cp312-cp312-linux_x86_64.whl"

echo "[setup] installing verl runtime deps ..."
pip install "${PIP_OPTS[@]}" \
    "tensordict==${TENSORDICT_VER}" \
    "vllm==${VLLM_VER}" \
    "transformers>=4.51,<4.55" \
    "accelerate" "datasets" "peft" "hydra-core" "omegaconf" \
    "ray[default]" "codetiming" "dill" "pyarrow>=19.0.0" \
    "sentence-transformers" "pandas" "pillow" "swanlab" "wandb"

echo "[setup] installing verl-latest from local source (no-deps) ..."
pip install --no-deps -e ./vendor/verl-latest

echo "[setup] sanity import checks ..."
python - <<'PY'
import importlib
for mod in [
    "torch", "vllm", "tensordict", "verl",
    "verl.trainer.ppo.core_algos",
    "verl.workers.reward_manager",
    "verl.models.transformers.qwen3_vl",
    "sentence_transformers",
]:
    importlib.import_module(mod)
    print(f"  ok  {mod}")
print("[setup] all imports OK")
PY

echo "[setup] DONE. Activate with:  conda activate ${ENV_NAME}"
