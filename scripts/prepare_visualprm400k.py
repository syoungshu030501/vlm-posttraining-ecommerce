"""Download + probe OpenGVLab/VisualPRM400K-v1.1-Raw.

The dataset ships as a single ``annotations.zip`` plus a separate
``images/`` tree on the HF hub. Its JSONL shards have heterogeneous columns
(some include an ``image`` field, some don't; some include ``analysis``),
which makes ``datasets.load_dataset`` fail out of the box. We side-step the
loader and pull files directly with ``huggingface_hub``.

What this script does
---------------------
1. Resolves a target NFS cache dir, refusing to start if free space is low.
2. Pulls (or symlinks from the HF cache) ``annotations.zip`` and unpacks it.
3. Walks the first ``--n-shards`` JSONL files and prints:
   - shard filename + row count
   - the union of keys observed in the first 200 rows
   - one fully-rendered example row (truncated)
4. Optionally pulls a small subset of images (``--with-images``) so the
   loader / smoke tests can be wired up without waiting for the full
   image tree.

After running this once, write ``src/data/visualprm400k.py`` against the
real schema — don't guess.

Usage
-----
    python scripts/prepare_visualprm400k.py \
        --cache_dir /mnt/nfs/young/VLM-posttraining/data/visualprm400k_raw_cache \
        --n-shards 3 \
        --rows-per-shard 200

    # also fetch image dir (slow):
    python scripts/prepare_visualprm400k.py --with-images
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

REPO_ID = "OpenGVLab/VisualPRM400K-v1.1-Raw"
REPO_TYPE = "dataset"
ANNOTATIONS_FILE = "annotations.zip"
MIN_FREE_GB = 200


def _check_free_space(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(path).free
    free_gb = free_bytes / 1e9
    if free_gb < MIN_FREE_GB:
        print(
            f"[FATAL] only {free_gb:.1f} GB free at {path}; need ≥{MIN_FREE_GB} GB.",
            file=sys.stderr,
        )
        sys.exit(2)
    print(f"  free space at {path}: {free_gb:.1f} GB (OK)")


def _list_repo_files() -> list[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    return sorted(files)


def _download_annotations(cache_dir: Path) -> Path:
    print(f"  downloading {ANNOTATIONS_FILE} from {REPO_ID} ...")
    local = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=ANNOTATIONS_FILE,
        cache_dir=str(cache_dir / "hf_cache"),
    )
    print(f"  -> {local}")
    return Path(local)


def _unpack(zip_path: Path, out_dir: Path) -> Path:
    target = out_dir / "annotations"
    if target.exists() and any(target.iterdir()):
        print(f"  reusing existing unpacked annotations at {target}")
        return target
    target.mkdir(parents=True, exist_ok=True)
    print(f"  unpacking {zip_path.name} -> {target} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    print("  unpack complete")
    return target


def _probe_shard(jsonl_path: Path, rows: int) -> dict:
    """Read the first ``rows`` lines, return a small summary."""
    keys_seen: Counter[str] = Counter()
    first_row: dict | None = None
    has_mc_i = 0
    n = 0
    with jsonl_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"    [WARN] {jsonl_path.name} line {n}: {exc}")
                continue
            if first_row is None:
                first_row = obj
            for k in obj:
                keys_seen[k] += 1
            # crude mc_i probe: keys containing "mc" or list/dict values with floats
            for k, v in obj.items():
                if isinstance(k, str) and "mc" in k.lower():
                    has_mc_i += 1
                    break
            n += 1
            if n >= rows:
                break
    return {
        "rows_probed": n,
        "keys": dict(keys_seen),
        "rows_with_mc_key": has_mc_i,
        "first_row": first_row,
    }


def _format_row(row: dict, max_chars: int = 1200) -> str:
    s = json.dumps(row, ensure_ascii=False, indent=2)
    if len(s) > max_chars:
        s = s[:max_chars] + f"\n... (truncated, full length {len(s)} chars)"
    return s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        default="/mnt/nfs/young/VLM-posttraining/data/visualprm400k_raw_cache",
        help="NFS-backed cache for annotations + (optional) images.",
    )
    parser.add_argument(
        "--n-shards",
        type=int,
        default=3,
        help="How many JSONL files to probe.",
    )
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=200,
        help="How many rows per shard to read when probing keys.",
    )
    parser.add_argument(
        "--with-images",
        action="store_true",
        help="Also snapshot_download the images dir (slow; ~50-100 GB).",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List repo files and exit (no downloads).",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    _check_free_space(cache_dir)

    print(f"\n== listing files in {REPO_ID} ==")
    files = _list_repo_files()
    print(f"  total files in repo: {len(files)}")
    for f in files[:20]:
        print(f"    {f}")
    if len(files) > 20:
        print(f"    ... and {len(files) - 20} more")
    if args.list_only:
        return

    print("\n== fetching annotations.zip ==")
    zip_path = _download_annotations(cache_dir)
    ann_dir = _unpack(zip_path, cache_dir)

    shard_paths = sorted(ann_dir.rglob("*.jsonl"))
    print(f"\n== {len(shard_paths)} JSONL shards in {ann_dir} ==")
    for sp in shard_paths[:25]:
        size_mb = sp.stat().st_size / 1e6
        print(f"    {sp.relative_to(ann_dir)}  ({size_mb:.1f} MB)")
    if len(shard_paths) > 25:
        print(f"    ... and {len(shard_paths) - 25} more")

    print(f"\n== probing first {args.n_shards} shards ==")
    for sp in shard_paths[: args.n_shards]:
        print(f"\n--- shard: {sp.relative_to(ann_dir)} ---")
        summary = _probe_shard(sp, args.rows_per_shard)
        print(f"  rows probed: {summary['rows_probed']}")
        print(f"  rows whose top-level keys mention 'mc': {summary['rows_with_mc_key']}")
        print(f"  key frequencies:")
        for k, c in sorted(summary["keys"].items(), key=lambda kv: -kv[1]):
            print(f"    {k:30s}  {c}")
        if summary["first_row"] is not None:
            print(f"  first row:\n{_format_row(summary['first_row'])}")

    if args.with_images:
        print("\n== fetching images/ tree (slow) ==")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            allow_patterns=["images/*", "images/**"],
            cache_dir=str(cache_dir / "hf_cache"),
        )
        print("  images download complete")

    print("\nDONE. Use the printed schema to write src/data/visualprm400k.py.")


if __name__ == "__main__":
    main()
