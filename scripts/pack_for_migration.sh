#!/usr/bin/env bash
# ============================================================
# 打包数据+代码，用于迁移到高算力卡
#
# 产物:
#   dist/ecommerce-audit-code.tar.gz   — 代码 + 脚本 + 配置（轻量 <5MB）
#   dist/ecommerce-audit-data.tar.gz   — 图片 + 标注 + RAG KB (~400MB)
#   dist/MANIFEST.txt                  — 文件清单 + 校验和
#   dist/MIGRATION.md                  — 迁移指南
#
# 用法:
#   bash scripts/pack_for_migration.sh
#   bash scripts/pack_for_migration.sh --data-only   # 只打包数据
#   bash scripts/pack_for_migration.sh --code-only   # 只打包代码
# ============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DIST_DIR="${PROJECT_ROOT}/dist"
mkdir -p "$DIST_DIR"

PACK_CODE=1
PACK_DATA=1
for arg in "$@"; do
    case "$arg" in
        --code-only) PACK_DATA=0 ;;
        --data-only) PACK_CODE=0 ;;
    esac
done

# ── 代码包：src/ + scripts/ + 配置（不含 __pycache__、logs、models） ──
if [[ "$PACK_CODE" == "1" ]]; then
    echo ">>> 打包代码..."
    CODE_TAR="${DIST_DIR}/ecommerce-audit-code.tar.gz"
    tar -czf "$CODE_TAR" \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.git' \
        --exclude='logs/*' \
        --exclude='.pytest_cache' \
        src/ scripts/ \
        requirements.txt pyproject.toml setup.py 2>/dev/null || true
    # 兜底：至少把 src/scripts 放进去
    [[ -f "$CODE_TAR" ]] || tar -czf "$CODE_TAR" --exclude='__pycache__' --exclude='*.pyc' src/ scripts/
    echo "    $CODE_TAR ($(du -h "$CODE_TAR" | cut -f1))"
fi

# ── 数据包：data/raw + data/sft + data/preference（不含 parquet 里的 image bytes 重复） ──
if [[ "$PACK_DATA" == "1" ]]; then
    echo ">>> 打包数据..."
    DATA_TAR="${DIST_DIR}/ecommerce-audit-data.tar.gz"
    # raw/images 必带；sft/preference jsonl+parquet；rules/violation_cases
    tar -czf "$DATA_TAR" \
        data/raw/images \
        data/raw/annotations.jsonl \
        data/raw/rules.jsonl \
        data/raw/violation_cases.jsonl \
        $(ls data/sft/*.jsonl data/sft/*.parquet 2>/dev/null) \
        $(ls data/preference/*.jsonl data/preference/*.parquet 2>/dev/null) \
        2>/dev/null || tar -czf "$DATA_TAR" data/raw data/sft data/preference
    echo "    $DATA_TAR ($(du -h "$DATA_TAR" | cut -f1))"
fi

# ── 清单 ──
MANIFEST="${DIST_DIR}/MANIFEST.txt"
{
    echo "# E-commerce Audit — Migration Manifest"
    echo "# Generated: $(date -Iseconds)"
    echo "# Host: $(hostname)"
    echo ""
    echo "## 数据统计"
    echo "- 图片:       $(ls data/raw/images 2>/dev/null | wc -l) 张"
    echo "- 标注:       $(wc -l < data/raw/annotations.jsonl 2>/dev/null || echo 0) 条"
    echo "- SFT:        $(wc -l < data/sft/sft.jsonl 2>/dev/null || echo 0) 条"
    echo "- 偏好对:     $(wc -l < data/preference/preference.jsonl 2>/dev/null || echo 0) 条"
    echo "- 规则:       $(wc -l < data/raw/rules.jsonl 2>/dev/null || echo 0) 条"
    echo "- 违规案例:   $(wc -l < data/raw/violation_cases.jsonl 2>/dev/null || echo 0) 条"
    echo ""
    echo "## 包体积"
    du -sh data/raw/images data/raw/*.jsonl data/sft data/preference 2>/dev/null || true
    echo ""
    echo "## 校验和"
    for f in "$DIST_DIR"/*.tar.gz; do
        [[ -f "$f" ]] && echo "$(sha256sum "$f")"
    done
} > "$MANIFEST"
cat "$MANIFEST"
echo ""
echo ">>> 打包完成: $DIST_DIR"
ls -lh "$DIST_DIR"
