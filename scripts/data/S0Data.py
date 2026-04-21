"""
Visual dedup: identify equivalence classes in data/raw/images via JPEG re-encode + MD5 hash.
For each class with >1 members, keep the canonical (smallest product_XXXXX id) and delete others.
Outputs:
  - data/raw/dedup_map.json  : {deleted_filename: canonical_filename}
  - data/raw/dedup_classes.json : [[class_members...], ...] (only multi-member)
Does NOT touch jsonl; downstream scripts use dedup_map to filter.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMG_DIR = Path("data/raw/images")
OUT_MAP = Path("data/raw/dedup_map.json")
OUT_CLASSES = Path("data/raw/dedup_classes.json")


def visual_hash(p: Path) -> str:
    try:
        img = Image.open(p).convert("RGB")
    except Exception as e:
        print(f"[WARN] failed to open {p}: {e}", file=sys.stderr)
        return ""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return hashlib.md5(buf.getvalue()).hexdigest()


def main(dry_run: bool = False) -> None:
    files = sorted(IMG_DIR.glob("*.jpg"))
    print(f"Scanning {len(files)} images...")
    buckets: dict[str, list[str]] = defaultdict(list)
    for p in tqdm(files):
        h = visual_hash(p)
        if not h:
            continue
        buckets[h].append(p.name)

    dup_classes = [sorted(v) for v in buckets.values() if len(v) > 1]
    dup_classes.sort(key=lambda g: g[0])
    print(f"Total hashes: {len(buckets)}, duplicate classes: {len(dup_classes)}")

    dedup_map: dict[str, str] = {}
    kept = 0
    removed = 0
    for grp in dup_classes:
        canonical = grp[0]  # sorted lexicographically, smallest product_XXXXX
        kept += 1
        for victim in grp[1:]:
            dedup_map[victim] = canonical
            removed += 1
    print(f"Canonical kept: {kept}, victims to delete: {removed}")

    OUT_MAP.write_text(json.dumps(dedup_map, indent=2, ensure_ascii=False))
    OUT_CLASSES.write_text(json.dumps(dup_classes, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT_MAP} and {OUT_CLASSES}")

    if dry_run:
        print("Dry run — not deleting files.")
        return

    for victim in dedup_map:
        (IMG_DIR / victim).unlink()
    remaining = len(list(IMG_DIR.glob("*.jpg")))
    print(f"Deleted {len(dedup_map)} files. Remaining images: {remaining}")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
