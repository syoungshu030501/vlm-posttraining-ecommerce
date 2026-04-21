"""
构建 RAG 知识库 — 电商合规规则文档

生成两类文件：
  data/raw/rules.jsonl           — 平台合规规则（BM25 + 密集检索用）
  data/raw/violation_cases.jsonl — 历史违规案例（相似案例召回用）

用法：
    python scripts/build_rag_kb.py
    python scripts/build_rag_kb.py --out_dir data/raw --n_cases 200

运行完毕后构建 FAISS + BM25 索引：
    python -m src.stage4_rag.indexer \
        --image_dir data/raw/images \
        --rule_file data/raw/rules.jsonl \
        --out_dir data/rag_index
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ==============================================================
# 平台合规规则库
# 参考: 天猫/淘宝/京东/拼多多平台规则，广告法，消费者权益保护法
# ==============================================================

RULES: list[dict] = [
    # ── 通用规则 ────────────────────────────────────────────────
    {
        "id": "G001",
        "category": "通用",
        "title": "禁止虚假宣传",
        "text": (
            "商品描述、主图、详情页中的文字和图片必须真实反映商品实际情况，"
            "不得夸大商品功能、效果或品质。商品属性（颜色、材质、尺寸、重量）"
            "须与实物一致，修图不得改变商品实质外观。"
        ),
        "keywords": ["虚假宣传", "描述不符", "以次充好", "修图过度"],
    },
    {
        "id": "G002",
        "category": "通用",
        "title": "禁用极限词",
        "text": (
            "商品标题、描述中禁止使用绝对化用语，包括但不限于：最、第一、国家级、"
            "最高级、最佳、极致、顶级、首选、无与伦比、史上最。"
            "违反广告法第九条相关规定。"
        ),
        "keywords": ["极限词", "绝对化用语", "最好", "第一", "国家级"],
    },
    {
        "id": "G003",
        "category": "通用",
        "title": "价格标注规范",
        "text": (
            "商品展示的原价（划线价）必须是曾经真实成交的价格，不得虚标原价制造"
            "打折假象。促销活动须明确说明活动时间和优惠范围。"
        ),
        "keywords": ["虚假打折", "原价虚标", "价格欺诈"],
    },
    {
        "id": "G004",
        "category": "通用",
        "title": "品牌与知识产权",
        "text": (
            "商品不得使用他人注册商标或与知名品牌相似的名称、logo 进行搭售或混淆。"
            "非官方授权店铺不得在商品名称中使用品牌词作为主体。"
        ),
        "keywords": ["品牌侵权", "商标仿冒", "未授权使用"],
    },
    # ── 服装/纺织品 ─────────────────────────────────────────────
    {
        "id": "C001",
        "category": "服装",
        "title": "材质成分标注",
        "text": (
            "纺织品须标注纤维成分及含量，误差不超过 5%。"
            "标注「纯棉」须含棉量不低于 95%，标注「真丝」须蚕丝含量不低于 50%。"
            "图片中可见的面料实际外观须与标注材质一致。"
        ),
        "keywords": ["材质造假", "成分不符", "虚标纯棉", "假真丝"],
    },
    {
        "id": "C002",
        "category": "服装",
        "title": "颜色与尺码一致性",
        "text": (
            "主图颜色须与实物高度一致，过度美化、调色导致色差明显（色差值超过10）的商品"
            "须在详情页注明颜色以实物为准。尺码表须与实物尺寸对应，误差不超过 2cm。"
        ),
        "keywords": ["色差", "尺码不符", "颜色失真"],
    },
    {
        "id": "C003",
        "category": "服装",
        "title": "童装安全要求",
        "text": (
            "14岁以下儿童服装须符合 GB 18401 安全技术规范，帽绳、领绳不得使用"
            "带绳结设计，甲醛含量须不超过 20mg/kg（婴幼儿）。"
            "须标注安全技术类别（A类/B类/C类）。"
        ),
        "keywords": ["童装安全", "GB18401", "绳结风险", "甲醛超标"],
    },
    # ── 食品 ────────────────────────────────────────────────────
    {
        "id": "F001",
        "category": "食品",
        "title": "强制标签信息",
        "text": (
            "预包装食品主图或详情页须清晰展示：品名、配料表、净含量、"
            "生产日期、保质期、生产许可证（SC编号）、执行标准、生产商信息。"
            "以上信息须与实物一致，不得遮挡或模糊处理。"
        ),
        "keywords": ["食品标签", "SC编号", "保质期", "配料表"],
    },
    {
        "id": "F002",
        "category": "食品",
        "title": "禁止宣称疾病预防和治疗",
        "text": (
            "普通食品不得宣称具有医疗、疾病预防、治疗功效。"
            "保健食品须标注蓝帽子标志，其功效须在批准范围内，"
            "不得超范围宣传。"
        ),
        "keywords": ["食品禁止疾病宣传", "医疗功效", "保健食品违规"],
    },
    {
        "id": "F003",
        "category": "食品",
        "title": "进口食品要求",
        "text": (
            "进口食品须提供有效期内的入境货物检验检疫证明，主图须展示中文标签。"
            "跨境食品须注明产地国、进口商信息，不得声称无添加而实际含有添加剂。"
        ),
        "keywords": ["进口食品", "中文标签", "检疫证明"],
    },
    # ── 化妆品 ──────────────────────────────────────────────────
    {
        "id": "M001",
        "category": "化妆品",
        "title": "产品备案与注册",
        "text": (
            "国产普通化妆品须完成备案，国产特殊化妆品（防晒、美白、染发等）须获得"
            "注册批件。进口化妆品须提供注册/备案号。主图须展示完整包装，"
            "产品名称须与注册名一致。"
        ),
        "keywords": ["化妆品备案", "注册批件", "特殊化妆品"],
    },
    {
        "id": "M002",
        "category": "化妆品",
        "title": "禁止宣称医疗效果",
        "text": (
            "化妆品不得宣称具有医疗功效，禁用词包括：治疗、消炎、抗菌（除特定条件外）、"
            "激素、医学级、医美级（非医疗器械）。"
            "不得使用前后对比图暗示疾病治疗效果。"
        ),
        "keywords": ["化妆品禁用词", "医疗宣称", "治疗功效", "医美违规"],
    },
    {
        "id": "M003",
        "category": "化妆品",
        "title": "成分标注规范",
        "text": (
            "全成分须按《化妆品标签管理办法》规定标注，成分名称使用 INCI 命名或"
            "中文通用名称。不得标注不在配方中的成分或虚标含量。"
        ),
        "keywords": ["成分标注", "INCI命名", "虚标成分"],
    },
    # ── 电子产品 ─────────────────────────────────────────────────
    {
        "id": "E001",
        "category": "电子产品",
        "title": "3C认证要求",
        "text": (
            "列入强制性产品认证目录的电子产品（手机、充电器、电源适配器、电线等）"
            "须通过 CCC（3C）认证，主图或详情须展示 3C 标志，"
            "商品编号须可在官网查询。"
        ),
        "keywords": ["3C认证", "CCC认证", "强制认证缺失"],
    },
    {
        "id": "E002",
        "category": "电子产品",
        "title": "手机入网许可",
        "text": (
            "手机、平板须取得工业和信息化部入网许可证（MIIT）。"
            "图片中 IMEI 不得模糊处理（隐私保护除外），"
            "网络制式须与描述一致。"
        ),
        "keywords": ["入网许可", "手机合规", "MIIT"],
    },
    {
        "id": "E003",
        "category": "电子产品",
        "title": "锂电池运输规范",
        "text": (
            "含锂电池商品须在详情页说明电池容量（mAh/Wh）和安全认证（UN38.3）。"
            "锂电池容量超过 100Wh 的商品在部分快递渠道受限，须在商品页面告知。"
        ),
        "keywords": ["锂电池", "UN38.3", "电池容量标注"],
    },
    # ── 家居/日用品 ─────────────────────────────────────────────
    {
        "id": "H001",
        "category": "家居",
        "title": "家用电器安全认证",
        "text": (
            "家用电器须通过 3C 认证或相关行业标准认证。"
            "小家电（电饭煲、热水壶、吹风机等）须标注额定电压、功率、"
            "防水等级（如涉及）。"
        ),
        "keywords": ["家电安全", "小家电认证", "额定功率"],
    },
    {
        "id": "H002",
        "category": "家居",
        "title": "家具甲醛限量",
        "text": (
            "室内用木制家具甲醛释放量须满足 GB 18580 限量要求（不超过 0.124mg/m3）。"
            "不得以「零甲醛」「无甲醛」作为绝对化用语进行宣传，除非有有效检测报告。"
        ),
        "keywords": ["甲醛超标", "家具环保", "GB18580", "零甲醛违规"],
    },
    # ── 图片质量规范 ─────────────────────────────────────────────
    {
        "id": "IMG001",
        "category": "图片规范",
        "title": "主图要求",
        "text": (
            "商品主图须以白底为主，不得包含大量牛皮癣文字（促销文字面积不超过图片面积20%）。"
            "图片分辨率不低于 800x800 像素，商品须占图片面积 60% 以上。"
            "不得使用假冒品牌官方图。"
        ),
        "keywords": ["主图规范", "牛皮癣", "图片质量", "白底"],
    },
    {
        "id": "IMG002",
        "category": "图片规范",
        "title": "水印与版权",
        "text": (
            "不得在主图中添加影响商品展示的大尺寸水印，水印高度不超过图片高度的5%。"
            "商品图片须为自有版权或已获授权，不得盗用他人商品图。"
        ),
        "keywords": ["水印违规", "图片版权", "盗图"],
    },
]

# ==============================================================
# 历史违规案例库（用于 FAISS 相似违规案例召回）
# ==============================================================

VIOLATION_CASE_TEMPLATES: list[dict] = [
    {
        "category": "服装",
        "violation_type": "材质虚假标注",
        "description": "卖家将聚酯纤维（涤纶）标注为纯棉，主图展示柔软触感但实物为化纤面料",
        "evidence": "纤维成分检测：涤纶 92%，棉 8%，不符合纯棉不低于95%标准",
        "penalty": "商品下架，扣分12分，退款处理",
    },
    {
        "category": "食品",
        "violation_type": "保质期造假",
        "description": "商品主图展示的保质期标签被贴纸覆盖重新标注，实际已过期",
        "evidence": "图片中可见标签边缘有覆盖痕迹，两层标签可辨别",
        "penalty": "关店处理，涉嫌食品安全犯罪移交市场监管部门",
    },
    {
        "category": "化妆品",
        "violation_type": "违禁词宣传",
        "description": "面霜详情页使用治疗湿疹、消炎抑菌等医疗功效宣传",
        "evidence": "详情页截图中包含违规词汇，属于化妆品违规宣传",
        "penalty": "商品下架，扣分6分",
    },
    {
        "category": "电子产品",
        "violation_type": "3C认证缺失",
        "description": "充电器商品页面无3C认证标志，详情页无认证信息",
        "evidence": "主图及详情页均未显示CCC认证标志，查询数据库无对应认证编号",
        "penalty": "商品强制下架，缴纳违约金",
    },
    {
        "category": "服装",
        "violation_type": "颜色严重失真",
        "description": "商品主图颜色经过严重修图处理，实物为灰色，图片显示为亮白色",
        "evidence": "买家实拍对比图与主图色差超过25，超过行业容差标准",
        "penalty": "扣分6分，强制允许退货退款",
    },
    {
        "category": "食品",
        "violation_type": "无SC生产许可证",
        "description": "糕点类商品详情页无SC编号或SC编号为无效格式",
        "evidence": "SC编号格式错误，国家市场监管局官网无法查询",
        "penalty": "商品下架，要求提供合规证明",
    },
    {
        "category": "家居",
        "violation_type": "虚假环保宣传",
        "description": "板式家具宣称零甲醛，但无第三方检测报告",
        "evidence": "商品无有效检测报告，宣传用语属于绝对化用语违规",
        "penalty": "要求修改描述，扣分3分",
    },
    {
        "category": "通用",
        "violation_type": "极限词违规",
        "description": "商品标题包含全网最低价、史上最强等极限词",
        "evidence": "违反广告法第九条，不得使用最等最高级表述",
        "penalty": "强制修改标题，扣分3分",
    },
    {
        "category": "化妆品",
        "violation_type": "成分虚标",
        "description": "护肤品宣称含有高浓度玻尿酸，实际成分表中透明质酸钠排名靠后",
        "evidence": "成分表位置与宣传浓度不符，属于误导消费者",
        "penalty": "下架整改，扣分6分",
    },
    {
        "category": "电子产品",
        "violation_type": "虚假参数",
        "description": "移动电源标称20000mAh，实测容量不足12000mAh",
        "evidence": "第三方检测报告显示实际容量与标称差距超过40%",
        "penalty": "下架，缴纳赔偿金，扣分12分",
    },
]


def build_rules(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rule in RULES:
            f.write(json.dumps(rule, ensure_ascii=False) + "\n")
    print(f"[rules] {len(RULES)} 条规则 -> {out_path}")


def build_violation_cases(out_path: Path, n: int, seed: int = 42) -> None:
    rng = random.Random(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(n):
            tpl = rng.choice(VIOLATION_CASE_TEMPLATES)
            case = dict(tpl)
            case["case_id"] = f"VC{i + 1:04d}"
            # BM25 检索用的拼接文本
            case["text"] = (
                f"品类:{case['category']} "
                f"违规类型:{case['violation_type']} "
                f"{case['description']} "
                f"证据:{case['evidence']}"
            )
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"[cases] {n} 条案例 -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/raw")
    ap.add_argument("--n_cases", type=int, default=200)
    args = ap.parse_args()

    out = Path(args.out_dir)
    build_rules(out / "rules.jsonl")
    build_violation_cases(out / "violation_cases.jsonl", n=args.n_cases)

    print()
    print("下一步: 构建 FAISS + BM25 索引")
    print(
        "  python -m src.stage4_rag.indexer \\\n"
        "      --image_dir data/raw/images \\\n"
        "      --rule_file data/raw/rules.jsonl \\\n"
        "      --out_dir   data/rag_index"
    )


if __name__ == "__main__":
    main()
