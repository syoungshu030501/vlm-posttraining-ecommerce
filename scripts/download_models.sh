#!/usr/bin/env bash
# ============================================================
# 模型下载脚本 — ModelScope 优先 / HF 镜像备用 / 并行下载
# ============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 主模型（大体积，放 NFS 共享存储，避免占本地盘）
PRIMARY_MODEL_DIR="${PRIMARY_MODEL_DIR:-/mnt/nfs/young/VLM4reasoning/models/pretrained}"
# RAG 小模型（几百 MB ~ 2GB，放项目目录，随代码迁移）
RAG_MODEL_DIR="${RAG_MODEL_DIR:-${PROJECT_ROOT}/models/pretrained}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/downloads}"
mkdir -p "$PRIMARY_MODEL_DIR" "$RAG_MODEL_DIR" "$LOG_DIR"

# 主训练模型: Qwen3-VL-8B-Instruct (~16GB)
PRIMARY_MODEL_HF_ID="${PRIMARY_MODEL_HF_ID:-Qwen/Qwen3-VL-8B-Instruct}"
PRIMARY_MODEL_MS_ID="${PRIMARY_MODEL_MS_ID:-Qwen/Qwen3-VL-8B-Instruct}"
PRIMARY_MODEL_LOCAL_NAME="${PRIMARY_MODEL_LOCAL_NAME:-Qwen3-VL-8B-Instruct}"

# ── 选择下载后端 ──────────────────────────────────────────────
# 优先级: ModelScope(国内直连) > hf-mirror.com > 官方 HF
USE_MODELSCOPE="${USE_MODELSCOPE:-1}"
if [[ "$USE_MODELSCOPE" == "1" ]] && ! python3 -c "import modelscope" &>/dev/null; then
    echo "[warn] modelscope 未安装，回退到 huggingface-cli"
    USE_MODELSCOPE=0
fi

if [[ "$USE_MODELSCOPE" != "1" ]] && [[ -z "${HF_ENDPOINT:-}" ]]; then
    code=$(curl -sk --connect-timeout 5 -o /dev/null -w "%{http_code}" "https://hf-mirror.com" 2>/dev/null || echo "000")
    if [[ "$code" =~ ^(200|301|302)$ ]]; then
        export HF_ENDPOINT="https://hf-mirror.com"
    fi
    echo "[hf] HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}"
fi

# ── 下载函数 ──────────────────────────────────────────────────
# 用法: download_model <hf_id> <ms_id> <local_name>
#   hf_id       HuggingFace repo id  (e.g. Qwen/Qwen3-VL-8B-Instruct)
#   ms_id       ModelScope repo id   (通常与 hf_id 相同)
#   local_name  本地目录名           (models/pretrained/<local_name>)
download_model() {
    local hf_id="$1"
    local ms_id="$2"
    local local_name="$3"
    local dest_dir="${4:-${RAG_MODEL_DIR}}"
    local target="${dest_dir}/${local_name}"
    local log="${LOG_DIR}/${local_name}.log"

    if [[ -f "${target}/config.json" ]]; then
        echo "[skip] ${local_name} — 已存在"
        return 0
    fi

    mkdir -p "$target"

    if [[ "$USE_MODELSCOPE" == "1" ]]; then
        echo ">>> [ModelScope] ${ms_id} → ${target}"
        modelscope download \
            --model "${ms_id}" \
            --local_dir "${target}" \
            2>&1 | tee "${log}" && echo "[done] ${local_name}" && return 0
        echo "[warn] ModelScope 失败，回退到 HF"
    fi

    if [[ -z "${HF_ENDPOINT:-}" ]]; then
        code=$(curl -sk --connect-timeout 5 -o /dev/null -w "%{http_code}" "https://hf-mirror.com" 2>/dev/null || echo "000")
        [[ "$code" =~ ^(200|301|302)$ ]] && export HF_ENDPOINT="https://hf-mirror.com"
    fi
    echo ">>> [HF${HF_ENDPOINT:+ mirror}] ${hf_id} → ${target}"
    huggingface-cli download "${hf_id}" \
        --local-dir "${target}" \
        --local-dir-use-symlinks False \
        --resume-download \
        2>&1 | tee "${log}" && echo "[done] ${local_name}"
}

# ── 并行下载：后台启动，统一等待 ─────────────────────────────
declare -A pids   # local_name -> PID

launch() {
    local hf_id="$1" ms_id="$2" local_name="$3" dest_dir="${4:-${RAG_MODEL_DIR}}"
    download_model "$hf_id" "$ms_id" "$local_name" "$dest_dir" &
    pids["$local_name"]=$!
    echo "[launch] ${local_name} → ${dest_dir} (pid=${pids[$local_name]})"
}

wait_all() {
    local failed=0
    for name in "${!pids[@]}"; do
        if wait "${pids[$name]}"; then
            echo "[ok] ${name}"
        else
            echo "[FAIL] ${name} — 检查日志: ${LOG_DIR}/${name}.log"
            failed=$((failed + 1))
        fi
    done
    [[ $failed -eq 0 ]] || { echo ""; echo "[ERROR] ${failed} 个模型下载失败"; exit 1; }
}

# ==============================================================
# 需要下载的模型
# ==============================================================
echo ""
echo "============================================================"
echo "  开始并行下载模型 (USE_MODELSCOPE=${USE_MODELSCOPE})"
echo "============================================================"
echo ""

# ── 主模型：Qwen3-VL-8B-Instruct → NFS 共享存储 ─────────────
launch "${PRIMARY_MODEL_HF_ID}" \
       "${PRIMARY_MODEL_MS_ID}" \
       "${PRIMARY_MODEL_LOCAL_NAME}" \
       "${PRIMARY_MODEL_DIR}"

# ── RAG 视觉检索：CLIP（FAISS 索引用） → 项目目录 ───────────
launch "openai/clip-vit-base-patch32" \
       "openai-mirror/clip-vit-base-patch32" \
       "clip-vit-base-patch32" \
       "${RAG_MODEL_DIR}"

# ── RAG 文本检索：BGE-M3（补强 BM25） → 项目目录 ────────────
launch "BAAI/bge-m3" \
       "BAAI/bge-m3" \
       "bge-m3" \
       "${RAG_MODEL_DIR}"

# ── 等待所有后台任务 ─────────────────────────────────────────
echo ""
echo ">>> 等待所有下载完成..."
wait_all

# ==============================================================
echo ""
echo "============================================================"
echo "  全部模型下载完毕"
echo "    主模型  → ${PRIMARY_MODEL_DIR}"
echo "    RAG 小模型 → ${RAG_MODEL_DIR}"
echo "============================================================"
echo "[primary]"; ls -lh "${PRIMARY_MODEL_DIR}/" 2>/dev/null || true
echo "[rag]"; ls -lh "${RAG_MODEL_DIR}/" 2>/dev/null || true
