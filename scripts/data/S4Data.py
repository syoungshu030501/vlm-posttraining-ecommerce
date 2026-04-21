"""
Stage-4 data: RAG violation-case corpus.

Subcommands:
  gd     crawl 广东省市场监督管理局 (multi-section)
  samr   crawl 北京市市场监督管理局「市场监管动态」
  merge  filter raw crawls + merge with template cases into the unified RAG KB

Examples:
  python scripts/data/S4Data.py gd    --max_pages 30 --workers 12
  python scripts/data/S4Data.py samr  --max_pages 67 --workers 12
  python scripts/data/S4Data.py merge
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Shared HTTP / parsing utilities
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

TITLE_WHITELIST = re.compile(
    r"(典型案例|曝光台|曝光|铁拳|违法|案件|查处|处罚|专项|整治|执法|抽查|抽检|不合格|召回)"
)
CASE_SPLIT = re.compile(r"(?:^|\n|\s)([一二三四五六七八九十]{1,3}[、.]|案例\s*\d+[、.:：]?|\d+[、.]\s)")
RE_PENALTY = re.compile(r"罚款\s*([\d\.]+)\s*(万元|元)")
RE_LAW = re.compile(r"《([^》]{3,30})》")
RE_COMPANY = re.compile(r"([一-龥]{2,}(?:公司|店|中心|工作室|诊所|医院|商行|企业|社|厂))")


def fetch(url: str, retries: int = 3, timeout: int = 15) -> str | None:
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout, verify=False)
            r.encoding = r.apparent_encoding or "utf-8"
            if r.status_code == 200:
                return r.text
        except requests.RequestException:
            pass
        time.sleep(1 + i)
    return None


def clean_html(html: str) -> str:
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&(nbsp|emsp|ensp);", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_cases(text: str) -> list[str]:
    parts = CASE_SPLIT.split(text)
    cases, buf = [], ""
    for i, chunk in enumerate(parts):
        if i % 2 == 1:
            if buf.strip():
                cases.append(buf.strip())
            buf = chunk
        else:
            buf += chunk
    if buf.strip():
        cases.append(buf.strip())
    return [c for c in cases if len(c) > 60]


def extract_fields(case_text: str, extra_kw: list[tuple[str, str]] | None = None) -> dict:
    penalty = RE_PENALTY.search(case_text)
    laws = list(set(RE_LAW.findall(case_text)))
    companies = list(dict.fromkeys(RE_COMPANY.findall(case_text)))[:3]
    kw_map = [
        ("食品", "食品"), ("药品", "医药"), ("医疗", "医药"), ("化妆品", "化妆品"),
        ("服装", "服装"), ("纺织", "服装"), ("电子", "电子产品"), ("电器", "电子产品"),
        ("广告", "广告宣传"), ("特种设备", "特种设备"), ("玩具", "玩具"),
        ("口罩", "医药"), ("保健", "保健品"), ("标签", "食品"), ("充电", "电子产品"),
    ] + (extra_kw or [])
    category = "通用"
    for kw, cat in kw_map:
        if kw in case_text:
            category = cat
            break
    return {
        "text": case_text[:800],
        "category": category,
        "penalty_amount": penalty.group(0) if penalty else None,
        "laws": laws[:5],
        "subjects": companies,
    }


def _dedup_by_url(items: list[dict]) -> list[dict]:
    seen, uniq = set(), []
    for it in items:
        if it["url"] not in seen:
            seen.add(it["url"])
            uniq.append(it)
    return uniq


def _disable_ssl_warnings() -> None:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ---------------------------------------------------------------------------
# Source: 广东省 (multi-section)
# ---------------------------------------------------------------------------

GD_SECTIONS = [
    ("xwfbt", "https://amr.gd.gov.cn/zwdt/xwfbt"),
    ("tpzl", "https://amr.gd.gov.cn/zwdt/tpzl"),
    ("mtgz", "https://amr.gd.gov.cn/zwdt/mtgz"),
    ("bljg", "https://amr.gd.gov.cn/zwgk/bljg"),
]

GD_LIST_RE = re.compile(
    r'<a\s+href="(https://amr\.gd\.gov\.cn/[^"]*/content/post_\d+\.html)"[^>]*>([^<]{5,120})</a>',
    re.S,
)


def gd_parse_list_page(html: str) -> list[dict]:
    items = [
        {"url": m.group(1), "title": m.group(2).strip()}
        for m in GD_LIST_RE.finditer(html)
        if TITLE_WHITELIST.search(m.group(2))
    ]
    return _dedup_by_url(items)


def gd_parse_detail(html: str) -> str:
    text = clean_html(html)
    m = re.search(r"(首页.*?)(相关链接|分享|打印\s+关闭|附件|上一篇|\s+$)", text, re.S)
    if m:
        text = m.group(1)
    return text


def gd_crawl_section(name: str, base: str, max_pages: int) -> list[dict]:
    all_items = []
    for p in range(max_pages + 1):
        url = f"{base}/index.html" if p == 0 else f"{base}/index_{p}.html"
        html = fetch(url)
        if not html:
            print(f"[skip {name} page {p}]", flush=True)
            continue
        items = gd_parse_list_page(html)
        for it in items:
            it["section"] = name
        print(f"[{name} page {p}] {len(items)} candidates", flush=True)
        all_items.extend(items)
        if not items and p > 2:
            break
    return all_items


def gd_crawl_detail(item: dict) -> list[dict]:
    html = fetch(item["url"])
    if not html:
        return []
    cases = split_cases(gd_parse_detail(html))
    out = []
    for i, ct in enumerate(cases):
        rec = extract_fields(ct)
        rec.update({
            "source_title": item["title"],
            "source_url": item["url"],
            "source_section": f"gd/{item['section']}",
            "case_seq": i + 1,
        })
        out.append(rec)
    return out


def cmd_gd(args: argparse.Namespace) -> None:
    _disable_ssl_warnings()
    all_items = []
    for name, url in GD_SECTIONS:
        print(f"\n=== 扫描栏目 {name} ===", flush=True)
        all_items.extend(gd_crawl_section(name, url, args.max_pages))
    uniq = _dedup_by_url(all_items)
    print(f"\n>>> 共 {len(uniq)} 篇候选文章，开始爬取详情", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out_path.open("w", encoding="utf-8") as fout, \
         ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(gd_crawl_detail, it) for it in uniq]
        for n, fut in enumerate(as_completed(futures)):
            try:
                cases = fut.result()
            except Exception as e:
                cases = []
                print(f"  [err] {e}", flush=True)
            for c in cases:
                fout.write(json.dumps(c, ensure_ascii=False) + "\n")
                total += 1
            if (n + 1) % 20 == 0:
                print(f"  [{n+1}/{len(uniq)}] cases={total}", flush=True)

    print(f"\n>>> 完成 {total} 条案例 → {out_path}")
    print(f">>> 文件大小 {out_path.stat().st_size / 1024:.1f} KB")


# ---------------------------------------------------------------------------
# Source: 北京市 SAMR
# ---------------------------------------------------------------------------

SAMR_BASE = "https://scjgj.beijing.gov.cn/zwxx/scjgdt"
SAMR_LIST_RE = re.compile(
    r'href="(\./\d{6}/t\d{8}_\d+\.html?)"\s+title="([^"]+)"',
    re.S,
)


def samr_parse_list_page(html: str) -> list[dict]:
    items = []
    for m in SAMR_LIST_RE.finditer(html):
        url = m.group(1).lstrip("./")
        title = m.group(2).strip()
        if TITLE_WHITELIST.search(title):
            items.append({"url": f"{SAMR_BASE}/{url}", "title": title})
    return _dedup_by_url(items)


def samr_parse_detail(html: str) -> str:
    text = clean_html(html)
    m = re.search(r"(首页.*?)(相关新闻|分享到|附件下载|\s+$)", text, re.S)
    if m:
        text = m.group(1)
    return text


def samr_crawl_list(max_pages: int) -> list[dict]:
    all_items = []
    for p in range(max_pages + 1):
        url = f"{SAMR_BASE}/index.html" if p == 0 else f"{SAMR_BASE}/index_{p}.html"
        html = fetch(url)
        if not html:
            print(f"[skip page {p}] fetch failed")
            continue
        items = samr_parse_list_page(html)
        print(f"[page {p}] {len(items)} case-like articles")
        all_items.extend(items)
    return _dedup_by_url(all_items)


def samr_crawl_detail(item: dict) -> list[dict]:
    html = fetch(item["url"])
    if not html:
        return []
    cases = split_cases(samr_parse_detail(html))
    out = []
    for i, ct in enumerate(cases):
        rec = extract_fields(ct)
        rec.update({
            "source_title": item["title"],
            "source_url": item["url"],
            "case_seq": i + 1,
        })
        out.append(rec)
    return out


def cmd_samr(args: argparse.Namespace) -> None:
    _disable_ssl_warnings()
    print(f">>> 扫描列表页 1..{args.max_pages}")
    items = samr_crawl_list(args.max_pages)
    print(f">>> 共找到 {len(items)} 篇候选文章，开始爬取正文")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out_path.open("w", encoding="utf-8") as fout, \
         ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(samr_crawl_detail, it): it for it in items}
        for n, fut in enumerate(as_completed(futures)):
            try:
                cases = fut.result()
            except Exception as e:
                cases = []
                print(f"  [err] {e}")
            for c in cases:
                fout.write(json.dumps(c, ensure_ascii=False) + "\n")
                total += 1
                if args.max_cases and total >= args.max_cases:
                    break
            if args.max_cases and total >= args.max_cases:
                break
            if (n + 1) % 20 == 0:
                print(f"  [{n+1}/{len(items)}] cases so far: {total}")
        fout.flush()

    print(f"\n>>> 完成: {total} 条案例 → {out_path}")
    print(f">>> 文件大小: {out_path.stat().st_size / 1024:.1f} KB")


# ---------------------------------------------------------------------------
# Merge: raw crawls → unified RAG KB
# ---------------------------------------------------------------------------

CATEGORY_RULES = [
    ("医药", ["药品", "药店", "医疗器械", "医疗美容", "诊所", "医院", "针剂"]),
    ("食品", ["食品", "餐饮", "饭店", "食堂", "年夜饭", "保质期", "食品安全", "烘焙", "肉制品", "乳制品"]),
    ("化妆品", ["化妆品", "美白", "祛斑", "防晒", "护肤"]),
    ("电子产品", ["电子", "电器", "手机", "充电", "电梯", "电池", "家电", "电线", "插座"]),
    ("服装", ["服装", "服饰", "纺织", "童装", "鞋", "羽绒服", "棉", "纤维"]),
    ("广告宣传", ["广告", "虚假宣传", "商标", "不正当竞争", "极限词", "最高级", "国家级"]),
    ("价格", ["价格", "哄抬", "欺诈", "明码标价", "虚标", "原价"]),
    ("认证", ["认证", "资质", "许可证", "检验"]),
    ("特种设备", ["特种设备", "电梯", "锅炉", "压力容器"]),
]

MERGE_CLEAN_PATTERNS = [
    r"首页\s*政务公开[^.。]*",
    r"时间：\s*\d{4}年\s*\d+月\s*\d+日",
    r"来源：[^\s]+",
    r"分享：\s*X",
    r"【字体：[^】]+】",
    r"打印\s+收藏",
]


def refine_category(text: str, default: str) -> str:
    for cat, kws in CATEGORY_RULES:
        if any(k in text for k in kws):
            return cat
    return default


def is_real_case(rec: dict) -> bool:
    evidence = sum([
        bool(rec.get("penalty_amount")),
        bool(rec.get("laws")),
        bool(rec.get("subjects")),
    ])
    return evidence >= 2


def clean_merge_text(t: str) -> str:
    for pat in MERGE_CLEAN_PATTERNS:
        t = re.sub(pat, "", t)
    return re.sub(r"\s+", " ", t).strip()


def _load_jsonl(p: Path) -> list[dict]:
    rows = []
    if not p.exists():
        return rows
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cmd_merge(args: argparse.Namespace) -> None:
    template = _load_jsonl(Path(args.template))
    print(f"[load] template cases: {len(template)}")

    # 建立已存在案例的文本指纹集合 + 最大 REAL 编号，避免重复入库与 ID 冲突
    def fingerprint(rec: dict) -> str | None:
        t = (rec.get("text") or "")[:200]
        u = rec.get("source") or rec.get("source_url") or ""
        if not t:
            return None
        return hashlib.md5(f"{u}|{t}".encode("utf-8")).hexdigest()

    existing_fp = {fp for fp in (fingerprint(r) for r in template) if fp}
    max_real_id = 0
    for r in template:
        cid = r.get("case_id", "")
        if cid.startswith("REAL") and cid[4:].isdigit():
            max_real_id = max(max_real_id, int(cid[4:]))
    print(f"[dedup] existing fingerprints: {len(existing_fp)}, max REAL id: {max_real_id}")

    real_raw: list[dict] = []
    for src in args.sources:
        rows = _load_jsonl(Path(src))
        print(f"[load] {src}: {len(rows)} raw rows")
        real_raw.extend(rows)

    real_cases = []
    next_id = max_real_id + 1
    skipped_dup = 0
    for rec in real_raw:
        if not is_real_case(rec):
            continue
        text = clean_merge_text(rec["text"])
        cat = refine_category(text, rec.get("category", "通用"))
        merged_rec = {
            "case_id": f"REAL{next_id:04d}",
            "category": cat,
            "violation_type": "真实处罚案例",
            "penalty_amount": rec.get("penalty_amount"),
            "laws": rec.get("laws", []),
            "subjects": rec.get("subjects", []),
            "source": rec.get("source_url"),
            "source_title": rec.get("source_title"),
            "text": text[:600],
        }
        fp = fingerprint(merged_rec)
        if fp and fp in existing_fp:
            skipped_dup += 1
            continue
        if fp:
            existing_fp.add(fp)
        real_cases.append(merged_rec)
        next_id += 1
    print(f"[filter] real cases after evidence ≥2 filter: {len(real_cases)}  (dedup skipped: {skipped_dup})")

    merged = list(template) + real_cases

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fout:
        for rec in merged:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    real_by_cat = collections.Counter(
        r["category"] for r in merged if r.get("case_id", "").startswith("REAL")
    )
    with_penalty = sum(1 for r in merged if r.get("penalty_amount"))
    with_laws = sum(1 for r in merged if r.get("laws"))
    print(f"\n[write] {len(merged)} total cases → {out}")
    print(f"[real by category]: {dict(real_by_cat)}")
    print(f"  with penalty_amount : {with_penalty}")
    print(f"  with law citation   : {with_laws}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_gd = sub.add_parser("gd", help="crawl 广东省 multi-section")
    p_gd.add_argument("--max_pages", type=int, default=30)
    p_gd.add_argument("--workers", type=int, default=12)
    p_gd.add_argument("--out", default="data/raw/gd_cases.jsonl")
    p_gd.set_defaults(func=cmd_gd)

    p_samr = sub.add_parser("samr", help="crawl 北京市 SAMR")
    p_samr.add_argument("--max_pages", type=int, default=67)
    p_samr.add_argument("--workers", type=int, default=12)
    p_samr.add_argument("--out", default="data/raw/samr_cases.jsonl")
    p_samr.add_argument("--max_cases", type=int, default=0, help="0 = no cap")
    p_samr.set_defaults(func=cmd_samr)

    p_mg = sub.add_parser("merge", help="filter + merge raw crawls into RAG KB")
    p_mg.add_argument("--sources", nargs="+",
                      default=["data/raw/samr_cases.jsonl", "data/raw/gd_cases.jsonl"],
                      help="raw crawl JSONL files to merge")
    p_mg.add_argument("--template", default="data/raw/violation_cases.jsonl",
                      help="pre-existing template KB (will be preserved as base)")
    p_mg.add_argument("--out", default="data/raw/violation_cases.jsonl",
                      help="output KB path (default overwrites the template file)")
    p_mg.set_defaults(func=cmd_merge)

    args = ap.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
