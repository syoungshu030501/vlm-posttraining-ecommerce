"""
Stage-A (原料采集) 增量图片抓取：用 Pexels API 给缺类目补图。

用法:
    export PEXELS_API_KEY=xxx
    python scripts/data/SAData.py \\
        --category food,cosmetics,electronics \\
        --per_category 200 \\
        --out_dir data/raw/images

设计要点：
  - 每个粗类目映射多组查询词（避免单关键词导致分布过窄）
  - 命名规则 `product_{next_id:05d}.jpg`，next_id 基于现有 data/raw/images/ 的最大编号+1
  - JPEG 再编码 (quality=90) + MD5 dedup：与现有 2493 张图池共用视觉等价类判据
    * 与现存图 hash 相同 → 跳过
    * 本轮内部重复 → 跳过
  - resume-safe：中断后再跑只补缺失部分
  - Pexels 免费额度 200 req/hour、20000 req/month；加 rate limiting 0.5s/req 很保守
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = ROOT / "data/raw/images"
MANIFEST = ROOT / "data/raw/images_manifest.json"  # 记录本脚本写入的每张图的来源
PEXELS_URL = "https://api.pexels.com/v1/search"


# 每个粗类目 → 查询词（多组，保证分布）
QUERIES: dict[str, list[str]] = {
    "food": [
        "packaged food product", "snack package", "canned food",
        "bottled beverage", "cereal box", "chocolate bar packaging",
        "instant noodles package", "biscuit packaging",
    ],
    "cosmetics": [
        "cosmetics bottle", "lipstick product", "skincare serum bottle",
        "face cream jar", "perfume bottle", "makeup palette",
        "shampoo bottle", "cosmetic tube product",
    ],
    "electronics": [
        "smartphone product", "wireless earbuds", "laptop computer",
        "bluetooth speaker", "smart watch device", "camera product",
        "power bank charger", "tablet device",
    ],
}


def visual_md5(img_bytes: bytes) -> str | None:
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return hashlib.md5(buf.getvalue()).hexdigest()


def scan_existing_hashes(img_dir: Path) -> tuple[dict[str, str], int]:
    """Return ({md5: filename}, max product id)."""
    hashes: dict[str, str] = {}
    max_id = 0
    pat = re.compile(r"product_(\d+)\.jpg")
    for p in img_dir.glob("*.jpg"):
        m = pat.match(p.name)
        if m:
            max_id = max(max_id, int(m.group(1)))
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        hashes[hashlib.md5(buf.getvalue()).hexdigest()] = p.name
    return hashes, max_id


def fetch_pexels(api_key: str, query: str, page: int, per_page: int = 80) -> list[dict]:
    r = requests.get(
        PEXELS_URL,
        headers={"Authorization": api_key},
        params={"query": query, "per_page": per_page, "page": page,
                "orientation": "square"},
        timeout=20,
    )
    r.raise_for_status()
    return r.json().get("photos", [])


def download(url: str, retries: int = 3) -> bytes | None:
    for i in range(retries):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                return r.content
        except requests.RequestException:
            pass
        time.sleep(1 + i)
    return None


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"entries": []}


def save_manifest(m: dict) -> None:
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True,
                    help="comma-separated categories from: "
                         + ",".join(QUERIES.keys()))
    ap.add_argument("--per_category", type=int, default=200)
    ap.add_argument("--out_dir", default=str(IMG_DIR))
    ap.add_argument("--rate_sleep", type=float, default=0.6,
                    help="seconds between requests (Pexels allows 200/h)")
    args = ap.parse_args()

    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        print("[ERROR] PEXELS_API_KEY not set", file=sys.stderr)
        return 1

    requested = [c.strip() for c in args.category.split(",") if c.strip()]
    for c in requested:
        if c not in QUERIES:
            print(f"[ERROR] unknown category: {c}. known: {list(QUERIES)}", file=sys.stderr)
            return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f">>> Scanning existing images in {out_dir} ...")
    existing_hashes, max_id = scan_existing_hashes(out_dir)
    print(f"    existing: {len(existing_hashes)} unique images, max product id = {max_id}")

    manifest = load_manifest()
    next_id = max_id + 1

    for cat in requested:
        print(f"\n=== Category: {cat}  target: {args.per_category} ===")
        collected = 0
        # 统计本类目已在 manifest 中的数量，实现 resume-safe
        already = sum(1 for e in manifest["entries"] if e["category"] == cat)
        if already >= args.per_category:
            print(f"    [skip] manifest already has {already} entries for {cat}")
            continue
        remaining = args.per_category - already
        print(f"    need to fetch: {remaining} more (already {already})")

        queries = QUERIES[cat]
        q_idx = 0
        page = 1
        seen_in_run: set[str] = set()

        while collected < remaining:
            query = queries[q_idx % len(queries)]
            try:
                photos = fetch_pexels(api_key, query, page=page)
            except Exception as e:
                print(f"    [warn] fetch error q={query} page={page}: {e}")
                q_idx += 1
                page = 1
                time.sleep(2)
                continue

            if not photos:
                # 当前 query 已抓尽，切下一个
                q_idx += 1
                page = 1
                if q_idx >= len(queries) * 4:
                    print("    [stop] exhausted all queries")
                    break
                continue

            for ph in photos:
                if collected >= remaining:
                    break
                url = ph.get("src", {}).get("large") or ph.get("src", {}).get("original")
                if not url:
                    continue
                data = download(url)
                if not data:
                    continue
                h = visual_md5(data)
                if not h or h in existing_hashes or h in seen_in_run:
                    continue
                seen_in_run.add(h)

                fname = f"product_{next_id:05d}.jpg"
                # Re-encode to normalized JPEG q=90 so downstream hash is stable
                img = Image.open(io.BytesIO(data)).convert("RGB")
                img.save(out_dir / fname, format="JPEG", quality=90)
                existing_hashes[h] = fname

                manifest["entries"].append({
                    "file": fname,
                    "category": cat,
                    "query": query,
                    "source": "pexels",
                    "source_url": ph.get("url"),
                    "photographer": ph.get("photographer"),
                    "md5": h,
                })
                next_id += 1
                collected += 1
                if collected % 25 == 0:
                    print(f"    [{cat}] {collected}/{remaining}")
                    save_manifest(manifest)

                time.sleep(args.rate_sleep)

            # 下一页或下一个 query
            if len(photos) < 80:
                q_idx += 1
                page = 1
            else:
                page += 1
            time.sleep(args.rate_sleep)

        save_manifest(manifest)
        print(f"    [done] {cat}: fetched {collected} (total manifest entries for {cat}: {already + collected})")

    print(f"\n>>> Total images now in {out_dir}: {len(list(out_dir.glob('*.jpg')))}")
    print(f">>> Manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
