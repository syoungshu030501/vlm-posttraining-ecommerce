#!/usr/bin/env python
"""一次性把 .hf_staging/{dataset,model} 推到 Hugging Face。

用法：
    export HF_TOKEN='hf_...'
    python scripts/upload_to_hf.py --target dataset                       # 仅传数据集
    python scripts/upload_to_hf.py --target model --num-workers 2         # 仅传模型，并发降到 2
    python scripts/upload_to_hf.py --target both                          # 都传

设计要点：
- 使用 `upload_large_folder`（自动多线程 + 断点续传 + LFS 处理大权重）。
- staging 全是软链接，脚本自动 resolve（HF API 对 symlink 透明）。
- 跑完不会泄漏 token 到日志，token 仅从 env 读取。
- 自动重试 EADDRNOTAVAIL / ConnectError 等本地网络瞬时错误。
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from huggingface_hub import HfApi

DATASET_REPO = "HappierYang/VLM-posttraining-dataset"
MODEL_REPO = "HappierYang/VLM-posttraining-FIPO"

ROOT = Path(__file__).resolve().parent.parent
STAGE_DATASET = ROOT / ".hf_staging" / "dataset"
STAGE_MODEL = ROOT / ".hf_staging" / "model"

IGNORE_PATTERNS = [
    "*.bak",
    "*.pre_*.bak",
    "*.20*.bak",
    "*.leaked_bak",
    ".DS_Store",
    "__pycache__",
    "*.pyc",
]


def push(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    folder: Path,
    num_workers: int,
    max_retries: int = 8,
) -> None:
    print(f"[upload] target = {repo_type}://{repo_id}")
    print(f"[upload] folder = {folder}")
    print(f"[upload] num_workers = {num_workers}")
    if not folder.is_dir():
        raise FileNotFoundError(f"staging folder missing: {folder}")

    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=False)

    backoff = 5.0
    for attempt in range(1, max_retries + 1):
        try:
            api.upload_large_folder(
                folder_path=str(folder),
                repo_id=repo_id,
                repo_type=repo_type,
                ignore_patterns=IGNORE_PATTERNS,
                num_workers=num_workers,
                print_report=True,
            )
            break
        except Exception as e:  # noqa: BLE001
            msg = repr(e)
            transient = any(
                tag in msg
                for tag in (
                    "EADDRNOTAVAIL",
                    "Cannot assign requested address",
                    "ConnectError",
                    "ReadTimeout",
                    "RemoteDisconnected",
                    "ConnectionResetError",
                    "ProtocolError",
                )
            )
            if attempt == max_retries or not transient:
                raise
            print(
                f"[upload] transient error (attempt {attempt}/{max_retries}): {msg}\n"
                f"[upload] sleeping {backoff:.0f}s, then resuming…"
            )
            time.sleep(backoff)
            backoff = min(backoff * 1.7, 120.0)

    print(
        f"[upload] ✅ done: https://huggingface.co/"
        f"{'datasets/' if repo_type=='dataset' else ''}{repo_id}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["dataset", "model", "both"], default="both")
    ap.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="并发上传 worker 数（默认 2，端口紧张时更稳；网络好可上调到 4）",
    )
    ap.add_argument("--max-retries", type=int, default=8)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN env var is required (export HF_TOKEN='hf_...')."
        )
    api = HfApi(token=token)

    if args.target in ("dataset", "both"):
        push(api, DATASET_REPO, "dataset", STAGE_DATASET,
             num_workers=args.num_workers, max_retries=args.max_retries)

    if args.target in ("model", "both"):
        push(api, MODEL_REPO, "model", STAGE_MODEL,
             num_workers=args.num_workers, max_retries=args.max_retries)


if __name__ == "__main__":
    main()
