#!/usr/bin/env bash
# ============================================================
# 一键环境搭建脚本
# 在目标机器上运行：bash scripts/setup_env.sh
# 支持国内镜像自动探测
# ============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-ecom-audit}"
PYTHON_VER="${PYTHON_VER:-3.10}"
CUDA_VER="${CUDA_VER:-cu124}"        # cu118 / cu121 / cu124
MODEL_DIR="${PROJECT_ROOT}/models/pretrained"

echo "============================================================"
echo "  E-commerce Audit — Environment Setup"
echo "  Project root : ${PROJECT_ROOT}"
echo "  Env name     : ${ENV_NAME}"
echo "  CUDA target  : ${CUDA_VER}"
echo "============================================================"

# ------------------------------------------------------------------
# 0. Probe mirrors
# ------------------------------------------------------------------
probe_url() { curl -sk --connect-timeout 5 -o /dev/null -w "%{http_code}" "$1" 2>/dev/null; }

PIP_MIRROR=""
for m in "https://mirrors.aliyun.com/pypi/simple" \
         "https://pypi.tuna.tsinghua.edu.cn/simple" \
         "https://mirrors.ustc.edu.cn/pypi/simple"; do
    code=$(probe_url "$m")
    if [[ "$code" =~ ^(200|301|302)$ ]]; then
        PIP_MIRROR="$m"
        echo "[probe] pip mirror: $PIP_MIRROR"
        break
    fi
done

GH_MIRROR=""
for m in "https://bgithub.xyz" "https://gitclone.com" "https://mirror.ghproxy.com"; do
    code=$(probe_url "$m")
    if [[ "$code" =~ ^(200|301|302)$ ]]; then
        GH_MIRROR="$m"
        echo "[probe] GitHub mirror: $GH_MIRROR"
        break
    fi
done

HF_MIRROR=""
code=$(probe_url "https://hf-mirror.com")
if [[ "$code" =~ ^(200|301|302)$ ]]; then
    HF_MIRROR="https://hf-mirror.com"
    echo "[probe] HuggingFace mirror: $HF_MIRROR"
fi

PIP_INDEX_ARG=""
[[ -n "$PIP_MIRROR" ]] && PIP_INDEX_ARG="-i $PIP_MIRROR"

# ------------------------------------------------------------------
# 1. Conda environment
# ------------------------------------------------------------------
echo ""
echo ">>> Step 1: Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VER})"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Environment already exists, skipping creation."
else
    conda create -n "$ENV_NAME" python="$PYTHON_VER" -y
fi

CONDA_PREFIX=$(conda env list | grep "^${ENV_NAME}[[:space:]]" | awk '{print $NF}')
PY="${CONDA_PREFIX}/bin/python"
PIP="${CONDA_PREFIX}/bin/pip"
echo "  Python: $($PY --version)"
echo "  pip:    $($PIP --version)"

# ------------------------------------------------------------------
# 2. PyTorch
# ------------------------------------------------------------------
echo ""
echo ">>> Step 2: Installing PyTorch (${CUDA_VER})"
if $PY -c "import torch; print(torch.__version__)" 2>/dev/null; then
    echo "  PyTorch already installed, skipping."
else
    $PIP install torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/${CUDA_VER}" \
        --no-cache-dir
fi
$PY -c "import torch; print(f'  torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}')"

# ------------------------------------------------------------------
# 3. pip requirements
# ------------------------------------------------------------------
echo ""
echo ">>> Step 3: Installing pip requirements"
$PIP install -r "${PROJECT_ROOT}/requirements.txt" $PIP_INDEX_ARG --no-cache-dir

# ------------------------------------------------------------------
# 4. flash-attn
# ------------------------------------------------------------------
echo ""
echo ">>> Step 4: Installing flash-attn (compilation)"
if $PY -c "import flash_attn; print(f'  flash_attn {flash_attn.__version__}')" 2>/dev/null; then
    echo "  flash-attn already installed, skipping."
else
    GPU_ARCH=$($PY -c "import torch; cc=torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')" 2>/dev/null || echo "8.9")
    LARGE_TMP="${PROJECT_ROOT}/.tmp_flash"
    mkdir -p "$LARGE_TMP"
    echo "  GPU arch: ${GPU_ARCH} | TMPDIR: ${LARGE_TMP}"
    TMPDIR="$LARGE_TMP" \
        TORCH_CUDA_ARCH_LIST="$GPU_ARCH" \
        FLASH_ATTENTION_FORCE_BUILD=TRUE \
        MAX_JOBS=4 \
        $PIP install flash-attn --no-build-isolation --no-cache-dir $PIP_INDEX_ARG
    rm -rf "$LARGE_TMP"
fi

# ------------------------------------------------------------------
# 5. FIPO (veRL fork)
# ------------------------------------------------------------------
echo ""
echo ">>> Step 5: Cloning & installing FIPO framework"
FIPO_DIR="${PROJECT_ROOT}/vendor/FIPO"
if [[ -d "${FIPO_DIR}/.git" ]]; then
    echo "  FIPO already cloned, skipping."
else
    mkdir -p "${PROJECT_ROOT}/vendor"
    # Clear any old insteadOf rules
    git config --global --unset "url.https://mirror.ghproxy.com/https://github.com/.insteadOf" 2>/dev/null || true
    git config --global --unset "url.https://gitclone.com/github.com/.insteadOf" 2>/dev/null || true
    git config --global --unset "url.https://bgithub.xyz/.insteadOf" 2>/dev/null || true

    REPO_URL="https://github.com/qwenpilot/FIPO"
    if [[ -n "$GH_MIRROR" ]]; then
        case "$GH_MIRROR" in
            *gitclone.com*)
                REPO_URL="https://gitclone.com/github.com/qwenpilot/FIPO"
                ;;
            *)
                REPO_URL="${GH_MIRROR}/qwenpilot/FIPO"
                ;;
        esac
    fi
    git clone --depth=1 "$REPO_URL" "$FIPO_DIR"
fi
cd "$FIPO_DIR" && $PIP install -e ".[vllm]" $PIP_INDEX_ARG --no-cache-dir
cd "$PROJECT_ROOT"

# ------------------------------------------------------------------
# 6. Download models
# ------------------------------------------------------------------
echo ""
echo ">>> Step 6: Downloading models"
[[ -n "$HF_MIRROR" ]] && export HF_ENDPOINT="$HF_MIRROR"
bash "${PROJECT_ROOT}/scripts/download_models.sh"

# ------------------------------------------------------------------
# 7. Verify
# ------------------------------------------------------------------
echo ""
echo ">>> Step 7: Verification"
$PY -c "
import torch, transformers, peft, accelerate, datasets
print(f'torch       : {torch.__version__} (CUDA {torch.cuda.is_available()})')
print(f'transformers: {transformers.__version__}')
print(f'peft        : {peft.__version__}')
print(f'accelerate  : {accelerate.__version__}')
print(f'datasets    : {datasets.__version__}')
try:
    import vllm; print(f'vllm        : {vllm.__version__}')
except ImportError:
    print('vllm        : NOT installed (optional for RL)')
try:
    import flash_attn; print(f'flash_attn  : {flash_attn.__version__}')
except ImportError:
    print('flash_attn  : NOT installed')
"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "  Activate: conda activate ${ENV_NAME}"
echo "============================================================"
