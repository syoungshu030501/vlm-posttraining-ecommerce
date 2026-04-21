"""
Data guard (visual-class aware).

Checks:
  1. File row counts: sft.jsonl, preference.jsonl, parquets
  2. Split reproducibility: parquet train/val/test rowcount matches deterministic split
  3. Visual-class leakage: after JPEG re-encode + MD5, each image hashes to a unique class;
     preference ∩ val = 0, preference ∩ test = 0, train ∩ val = 0, train ∩ test = 0 at class level
  4. Preference contract: same coarse category; violation flip consistent with pair_strategy
  5. JSON parse rate for SFT response / preference chosen/rejected
  6. Image path existence (no deleted victims referenced)
  7. Coarse-category distribution report (non-failing, informational)
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.schema import coarse_category, same_coarse  # noqa: E402

SFT_JSONL = ROOT / "data/sft/sft.jsonl"
PREF_JSONL = ROOT / "data/preference/preference.jsonl"
IMG_DIR = ROOT / "data/raw/images"
SFT_TRAIN_PQ = ROOT / "data/sft/train.parquet"
SFT_VAL_PQ = ROOT / "data/sft/val.parquet"
SFT_TEST_PQ = ROOT / "data/sft/test.parquet"
PREF_PQ = ROOT / "data/preference/preference.parquet"
TRIPLET_PQ = ROOT / "data/sft/triplets.parquet"
SEED = 42
RATIOS = (0.8, 0.1, 0.1)


def load_jsonl(p: Path) -> list[dict]:
    rows = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def reproduce_split(sft_rows: list[dict]) -> tuple[set[str], set[str], set[str]]:
    seen, ss = [], set()
    for r in sft_rows:
        g = r["image_file"]
        if g not in ss:
            seen.append(g)
            ss.add(g)
    idx = np.random.RandomState(SEED).permutation(len(seen))
    n_train = int(len(seen) * RATIOS[0])
    n_val = int(len(seen) * RATIOS[1])
    return (
        {seen[i] for i in idx[:n_train]},
        {seen[i] for i in idx[n_train : n_train + n_val]},
        {seen[i] for i in idx[n_train + n_val :]},
    )


def visual_hash(p: Path) -> str | None:
    try:
        img = Image.open(p).convert("RGB")
    except Exception:
        return None
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return hashlib.md5(buf.getvalue()).hexdigest()


def safe_load(text: str) -> dict | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def main() -> int:
    fail = 0

    print("=" * 70)
    print("DATA GUARD v2 — visual-class aware")
    print("=" * 70)

    # --- 1. File rowcount ---
    sft_rows = load_jsonl(SFT_JSONL)
    pref_rows = load_jsonl(PREF_JSONL)
    print(f"\n[1] Rowcounts")
    print(f"  sft.jsonl: {len(sft_rows)}")
    print(f"  preference.jsonl: {len(pref_rows)}")

    train_df = pd.read_parquet(SFT_TRAIN_PQ)
    val_df = pd.read_parquet(SFT_VAL_PQ)
    test_df = pd.read_parquet(SFT_TEST_PQ)
    pref_df = pd.read_parquet(PREF_PQ)
    trip_df = pd.read_parquet(TRIPLET_PQ)
    print(f"  train.parquet: {len(train_df)}")
    print(f"  val.parquet:   {len(val_df)}")
    print(f"  test.parquet:  {len(test_df)}")
    print(f"  preference.parquet: {len(pref_df)}")
    print(f"  triplets.parquet:   {len(trip_df)}")

    if len(train_df) + len(val_df) + len(test_df) != len(sft_rows):
        print(f"  [FAIL] train+val+test != sft.jsonl rows")
        fail += 1
    if len(pref_df) != len(pref_rows):
        print(f"  [FAIL] preference parquet/jsonl mismatch")
        fail += 1

    # --- 2. Split reproducibility ---
    train_cls, val_cls, test_cls = reproduce_split(sft_rows)
    print(f"\n[2] Split reproducibility (by image_file group)")
    print(f"  train classes: {len(train_cls)}, val: {len(val_cls)}, test: {len(test_cls)}")
    # Check parquet train rows' image paths are a subset of train_cls images
    # Prefer the `image_file` column if present (newer parquets preserve it);
    # fall back to parsing the `image` column when it holds a path string.
    def _img_names(df: pd.DataFrame) -> set[str]:
        if "image_file" in df.columns:
            return {Path(str(p)).name for p in df["image_file"].tolist()}
        return {Path(str(p)).name for p in df["image"].astype(str).tolist()}

    train_img_files = _img_names(train_df)
    val_img_files = _img_names(val_df)
    test_img_files = _img_names(test_df)
    if not train_img_files.issubset(train_cls):
        print(f"  [FAIL] train.parquet has images outside reproduced train set")
        fail += 1
    if not val_img_files.issubset(val_cls):
        print(f"  [FAIL] val.parquet has images outside reproduced val set")
        fail += 1
    if not test_img_files.issubset(test_cls):
        print(f"  [FAIL] test.parquet has images outside reproduced test set")
        fail += 1
    print(f"  train.parquet image_files ⊂ train_cls: {train_img_files.issubset(train_cls)}")
    print(f"  val.parquet image_files ⊂ val_cls:     {val_img_files.issubset(val_cls)}")
    print(f"  test.parquet image_files ⊂ test_cls:   {test_img_files.issubset(test_cls)}")

    # --- 3. Visual-class leakage ---
    print(f"\n[3] Visual-class leakage (JPEG re-encode + MD5)")
    # Compute hash for every image referenced
    all_imgs = train_img_files | val_img_files | test_img_files
    pref_img_files = _img_names(pref_df) if len(pref_df) else set()
    all_imgs |= pref_img_files

    img_hash: dict[str, str] = {}
    missing = 0
    for img in sorted(all_imgs):
        p = IMG_DIR / img
        if not p.exists():
            missing += 1
            continue
        h = visual_hash(p)
        if h:
            img_hash[img] = h
    print(f"  hashed images: {len(img_hash)}, missing files: {missing}")
    if missing > 0:
        print(f"  [FAIL] {missing} images referenced but missing on disk")
        fail += 1

    # Build class -> {split} membership
    train_hashes = {img_hash[f] for f in train_img_files if f in img_hash}
    val_hashes = {img_hash[f] for f in val_img_files if f in img_hash}
    test_hashes = {img_hash[f] for f in test_img_files if f in img_hash}
    pref_hashes = {img_hash[f] for f in pref_img_files if f in img_hash}

    tv = train_hashes & val_hashes
    tt = train_hashes & test_hashes
    vt = val_hashes & test_hashes
    pv = pref_hashes & val_hashes
    ptest = pref_hashes & test_hashes
    pt = pref_hashes & train_hashes

    print(f"  train_cls ∩ val_cls:   {len(tv)}")
    print(f"  train_cls ∩ test_cls:  {len(tt)}")
    print(f"  val_cls   ∩ test_cls:  {len(vt)}")
    print(f"  pref_cls  ∩ val_cls:   {len(pv)}")
    print(f"  pref_cls  ∩ test_cls:  {len(ptest)}")
    print(f"  pref_cls  ⊂ train_cls: {pref_hashes.issubset(train_hashes)} (overlap {len(pt)}/{len(pref_hashes)})")

    for name, s in [("train∩val", tv), ("train∩test", tt), ("val∩test", vt),
                    ("pref∩val", pv), ("pref∩test", ptest)]:
        if s:
            print(f"  [FAIL] {name} leakage = {len(s)}")
            fail += 1
    if not pref_hashes.issubset(train_hashes):
        print(f"  [FAIL] preference has classes not in train")
        fail += 1

    # --- 4. Preference contract ---
    print(f"\n[4] Preference contract (coarse-category contract via src.schema)")
    bad_cat, bad_flip = 0, 0
    strat_same_coarse = defaultdict(lambda: [0, 0])  # strategy -> [match, mismatch]
    for r in pref_rows:
        c = safe_load(r["chosen"])
        rj = safe_load(r["rejected"])
        if not c or not rj:
            continue
        strat = r.get("pair_strategy", "?")
        if same_coarse(c.get("category"), rj.get("category")):
            strat_same_coarse[strat][0] += 1
        else:
            bad_cat += 1
            strat_same_coarse[strat][1] += 1
        flip_required = any(s in strat for s in ("missed_cue", "over_strict"))
        flipped = bool(c.get("violation")) != bool(rj.get("violation"))
        if flip_required != flipped:
            bad_flip += 1
    print(f"  coarse-category mismatch: {bad_cat}")
    print(f"  flip contract violation: {bad_flip}")
    for strat, (ok, bad) in sorted(strat_same_coarse.items()):
        print(f"    {strat}: same-coarse {ok}/{ok+bad}")
    if bad_cat > 0 or bad_flip > 0:
        fail += 1

    # --- 5. JSON parse rate ---
    print(f"\n[5] JSON parse rate")
    sft_bad = sum(1 for r in sft_rows if safe_load(r["response"]) is None)
    pref_bad = sum(1 for r in pref_rows
                   if safe_load(r["chosen"]) is None or safe_load(r["rejected"]) is None)
    print(f"  sft unparseable: {sft_bad}/{len(sft_rows)}")
    print(f"  pref unparseable: {pref_bad}/{len(pref_rows)}")
    if sft_bad or pref_bad:
        fail += 1

    # --- 6. Distributions ---
    print(f"\n[6] Distributions")
    vios = Counter(bool(safe_load(r["response"]).get("violation"))
                   for r in sft_rows if safe_load(r["response"]))
    print(f"  SFT violation: {dict(vios)}")
    strat = Counter(r.get("pair_strategy", "?") for r in pref_rows)
    print(f"  Preference pair_strategy: {dict(strat)}")
    cats = Counter(safe_load(r["response"]).get("category", "?")
                   for r in sft_rows if safe_load(r["response"]))
    print(f"  SFT categories: {len(cats)} unique")
    coarse_sft = Counter(coarse_category(safe_load(r["response"]).get("category"))
                         for r in sft_rows if safe_load(r["response"]))
    total_sft = sum(coarse_sft.values()) or 1
    coarse_pct = {k: f"{v} ({v*100/total_sft:.1f}%)" for k, v in coarse_sft.most_common()}
    print(f"  SFT coarse categories: {coarse_pct}")
    # Preference coarse distribution (by chosen side)
    coarse_pref = Counter(coarse_category(safe_load(r["chosen"]).get("category"))
                          for r in pref_rows if safe_load(r["chosen"]))
    print(f"  Pref coarse categories: {dict(coarse_pref.most_common())}")

    # --- 7. Triplet sanity ---
    print(f"\n[7] Triplets")
    trip_imgs = {Path(p).name for p in trip_df["image_path"].astype(str).tolist()}
    trip_in_train = trip_imgs & train_cls
    trip_in_eval = (trip_imgs - train_cls) & (val_cls | test_cls)
    print(f"  triplet images: {len(trip_imgs)}, in_train: {len(trip_in_train)}, in_eval (leak): {len(trip_in_eval)}")
    if trip_in_eval:
        print(f"  [FAIL] triplets reference eval images")
        fail += 1

    print("\n" + "=" * 70)
    if fail == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"FAILED: {fail} check(s) failed")
    print("=" * 70)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
