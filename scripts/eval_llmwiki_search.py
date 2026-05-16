"""
LLMWiki 搜索评估脚本

评估指标：
- Hit@K: 前 K 个结果中是否包含任一期望页面
- MRR: 第一个期望页面排名的倒数
- Precision@3: 前 3 个结果中有多少是期望页面
- Top-1 Accuracy: 第一个结果是否是最佳期望页面

用法：
    cd D:/Agent/xiaozhi-esp32-server-main/main/xiaozhi-server
    python scripts/eval_llmwiki_search.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 直接导入搜索模块中的核心函数
sys.path.insert(0, str(PROJECT_ROOT / "plugins_func" / "functions"))
from search_from_llmwiki import _load_markdown_documents, _rank_documents


# ============================================================
# 知识页 slug 映射（用于评估匹配）
# ============================================================

WIKI_ROOT = PROJECT_ROOT / "knowledge_base" / "llmwiki" / "clinical-nutrition"

# slug → 文件相对路径的映射，用于在结果中识别命中了哪个页面
SLUG_TO_PATH = {
    "index": "_index.md",
    "type-2-diabetes": "diseases/type-2-diabetes.md",
    "diabetes-medical-nutrition-therapy": "guidelines/diabetes-medical-nutrition-therapy.md",
    "diabetes-carbohydrate-gi-gl": "guidelines/diabetes-carbohydrate-gi-gl.md",
    "diabetes-special-situations-and-red-flags": "guidelines/diabetes-special-situations-and-red-flags.md",
    "diabetes-breakfast-decision-guide": "guidelines/diabetes-breakfast-decision-guide.md",
    "balanced-breakfast": "guidelines/balanced-breakfast.md",
    "common-breakfast-items": "foods/common-breakfast-items.md",
    "peanut-allergy": "allergies/peanut-allergy.md",
}


# ============================================================
# 评估数据集
# ============================================================

@dataclass
class WikiEvalCase:
    question: str
    expected_slugs: list[str]  # 期望命中的页面 slug，按相关性排序
    category: str  # 精确匹配 / 主题匹配 / 场景匹配 / 安全边界 / 跨页


EVAL_DATASET: list[WikiEvalCase] = [
    # ================================================================
    # 全新题目集（第二套）：与第一套完全不同的问法和角度
    # ================================================================

    # ---- 安全边界：第一套最弱的类别，重点测试 ----
    WikiEvalCase(
        question="突然头晕冒汗是不是低血糖？",
        expected_slugs=["diabetes-special-situations-and-red-flags"],
        category="安全边界",
    ),
    WikiEvalCase(
        question="糖尿病患者用胰岛素后吃饭要注意什么？",
        expected_slugs=["diabetes-special-situations-and-red-flags", "type-2-diabetes"],
        category="安全边界",
    ),
    WikiEvalCase(
        question="血糖忽高忽低是怎么回事？",
        expected_slugs=["diabetes-special-situations-and-red-flags", "type-2-diabetes"],
        category="安全边界",
    ),
    WikiEvalCase(
        question="糖尿病合并痛风饮食怎么处理？",
        expected_slugs=["diabetes-special-situations-and-red-flags", "type-2-diabetes"],
        category="安全边界",
    ),

    # ---- 精确匹配：用专业术语提问 ----
    WikiEvalCase(
        question="花生酱会引起过敏吗？",
        expected_slugs=["peanut-allergy"],
        category="精确匹配",
    ),
    WikiEvalCase(
        question="什么是血糖负荷GL？",
        expected_slugs=["diabetes-carbohydrate-gi-gl"],
        category="精确匹配",
    ),
    WikiEvalCase(
        question="MNT对糖尿病患者有什么帮助？",
        expected_slugs=["diabetes-medical-nutrition-therapy", "type-2-diabetes"],
        category="精确匹配",
    ),

    # ---- 主题匹配：口语化表达 ----
    WikiEvalCase(
        question="血糖偏高怎么调整饮食？",
        expected_slugs=["type-2-diabetes", "diabetes-carbohydrate-gi-gl"],
        category="主题匹配",
    ),
    WikiEvalCase(
        question="糖尿病人每天能吃多少肉？",
        expected_slugs=["diabetes-medical-nutrition-therapy", "type-2-diabetes"],
        category="主题匹配",
    ),
    WikiEvalCase(
        question="什么样的碳水不容易升糖？",
        expected_slugs=["diabetes-carbohydrate-gi-gl"],
        category="主题匹配",
    ),
    WikiEvalCase(
        question="糖尿病前期需要控制饮食吗？",
        expected_slugs=["type-2-diabetes", "diabetes-medical-nutrition-therapy"],
        category="主题匹配",
    ),

    # ---- 场景匹配：日常饮食场景 ----
    WikiEvalCase(
        question="早上来不及做饭吃什么方便？",
        expected_slugs=["diabetes-breakfast-decision-guide", "common-breakfast-items", "balanced-breakfast"],
        category="场景匹配",
    ),
    WikiEvalCase(
        question="全麦面包比白面包好在哪？",
        expected_slugs=["diabetes-carbohydrate-gi-gl", "common-breakfast-items"],
        category="场景匹配",
    ),
    WikiEvalCase(
        question="早餐只喝一杯燕麦粥行不行？",
        expected_slugs=["diabetes-breakfast-decision-guide", "diabetes-carbohydrate-gi-gl"],
        category="场景匹配",
    ),
    WikiEvalCase(
        question="糖尿病人能吃水果吗？什么时间吃好？",
        expected_slugs=["diabetes-carbohydrate-gi-gl", "type-2-diabetes"],
        category="场景匹配",
    ),

    # ---- 跨页：综合性问题 ----
    WikiEvalCase(
        question="超重的糖尿病患者怎么吃早餐？",
        expected_slugs=["diabetes-breakfast-decision-guide", "type-2-diabetes", "balanced-breakfast"],
        category="跨页",
    ),
    WikiEvalCase(
        question="糖尿病患者饮食和运动怎么配合？",
        expected_slugs=["type-2-diabetes", "diabetes-medical-nutrition-therapy"],
        category="跨页",
    ),
    WikiEvalCase(
        question="控糖饮食的总原则是什么？",
        expected_slugs=["type-2-diabetes", "diabetes-medical-nutrition-therapy", "diabetes-carbohydrate-gi-gl"],
        category="跨页",
    ),
    WikiEvalCase(
        question="用降糖药的人低血糖风险怎么预防？",
        expected_slugs=["diabetes-special-situations-and-red-flags", "type-2-diabetes"],
        category="跨页",
    ),
]


# ============================================================
# 评估逻辑
# ============================================================

@dataclass
class WikiEvalResult:
    question: str
    category: str
    hit_at_1: bool
    hit_at_3: bool
    hit_at_4: bool
    mrr: float
    precision_at_3: float
    top1_slug: str
    matched_slugs: list[str] = field(default_factory=list)
    retrieved_slugs: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


def evaluate_case(
    documents: list[dict],
    case: WikiEvalCase,
    top_k: int = 4,
) -> WikiEvalResult:
    start = time.perf_counter()
    ranked = _rank_documents(case.question, documents, top_k=top_k, snippet_chars=480)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # 将结果的 relative_path 映射回 slug
    path_to_slug = {v: k for k, v in SLUG_TO_PATH.items()}
    retrieved_slugs: list[str] = []
    for item in ranked:
        rel_path = item["relative_path"]
        slug = path_to_slug.get(rel_path, rel_path)
        retrieved_slugs.append(slug)

    expected_set = set(case.expected_slugs)

    # Hit@K
    hit_at_1 = bool(retrieved_slugs) and retrieved_slugs[0] in expected_set
    hit_at_3 = any(s in expected_set for s in retrieved_slugs[:3])
    hit_at_4 = any(s in expected_set for s in retrieved_slugs[:4])

    # MRR
    mrr = 0.0
    for i, slug in enumerate(retrieved_slugs):
        if slug in expected_set:
            mrr = 1.0 / (i + 1)
            break

    # Precision@3
    precision_at_3 = sum(1 for s in retrieved_slugs[:3] if s in expected_set) / 3.0

    # Matched slugs
    matched_slugs = [s for s in retrieved_slugs if s in expected_set]

    return WikiEvalResult(
        question=case.question,
        category=case.category,
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        hit_at_4=hit_at_4,
        mrr=mrr,
        precision_at_3=precision_at_3,
        top1_slug=retrieved_slugs[0] if retrieved_slugs else "",
        matched_slugs=matched_slugs,
        retrieved_slugs=retrieved_slugs,
        elapsed_ms=elapsed_ms,
    )


def run_evaluation() -> None:
    print("=" * 70)
    print("LLMWiki Search Evaluation")
    print("=" * 70)

    if not WIKI_ROOT.exists():
        print(f"ERROR: Wiki root not found: {WIKI_ROOT}")
        return

    excluded_dirs = {"raw"}
    documents = _load_markdown_documents(WIKI_ROOT, excluded_dirs)
    print(f"  Loaded {len(documents)} wiki pages")

    results: list[WikiEvalResult] = []
    for i, case in enumerate(EVAL_DATASET, 1):
        print(f"\n[{i}/{len(EVAL_DATASET)}] [{case.category}] {case.question}")
        result = evaluate_case(documents, case)
        results.append(result)

        status = "HIT" if result.hit_at_4 else "MISS"
        top1_mark = " *" if result.hit_at_1 else ""
        print(f"  {status} | MRR={result.mrr:.2f} | P@3={result.precision_at_3:.1f} | "
              f"Top1={result.top1_slug}{top1_mark} | {result.elapsed_ms:.1f}ms")
        if result.matched_slugs:
            print(f"  Matched: {', '.join(result.matched_slugs)}")
        print(f"  Retrieved: {', '.join(result.retrieved_slugs)}")

    # 汇总
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    n = len(results)
    hit1_rate = sum(1 for r in results if r.hit_at_1) / n
    hit3_rate = sum(1 for r in results if r.hit_at_3) / n
    hit4_rate = sum(1 for r in results if r.hit_at_4) / n
    avg_mrr = sum(r.mrr for r in results) / n
    avg_p3 = sum(r.precision_at_3 for r in results) / n
    avg_latency = sum(r.elapsed_ms for r in results) / n

    print(f"  Test cases:      {n}")
    print(f"  Top-1 Accuracy:  {hit1_rate:.1%}")
    print(f"  Hit@3:           {hit3_rate:.1%}")
    print(f"  Hit@4:           {hit4_rate:.1%}")
    print(f"  MRR:             {avg_mrr:.3f}")
    print(f"  Precision@3:     {avg_p3:.1%}")
    print(f"  Avg Latency:     {avg_latency:.1f}ms")

    # 按类别汇总
    print("\n  By category:")
    categories = sorted(set(r.category for r in results))
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        cat_n = len(cat_results)
        cat_hit = sum(1 for r in cat_results if r.hit_at_4) / cat_n
        cat_mrr = sum(r.mrr for r in cat_results) / cat_n
        print(f"    {cat}: Hit@4={cat_hit:.0%} MRR={cat_mrr:.2f} ({cat_n} cases)")

    # 失败案例
    misses = [r for r in results if not r.hit_at_4]
    if misses:
        print(f"\n  Missed cases ({len(misses)}):")
        for r in misses:
            print(f"    - [{r.category}] {r.question}")
            print(f"      Expected: {', '.join(EVAL_DATASET[results.index(r)].expected_slugs)}")
            print(f"      Got: {', '.join(r.retrieved_slugs)}")

    # Top-1 错误案例
    wrong_top1 = [r for r in results if not r.hit_at_1 and r.hit_at_4]
    if wrong_top1:
        print(f"\n  Wrong Top-1 but hit in results ({len(wrong_top1)}):")
        for r in wrong_top1:
            print(f"    - [{r.category}] {r.question}")
            print(f"      Top-1: {r.top1_slug}, Expected: {EVAL_DATASET[results.index(r)].expected_slugs[0]}")

    # JSON 报告
    report = {
        "summary": {
            "test_cases": n,
            "top1_accuracy": round(hit1_rate, 4),
            "hit_at_3": round(hit3_rate, 4),
            "hit_at_4": round(hit4_rate, 4),
            "mrr": round(avg_mrr, 4),
            "precision_at_3": round(avg_p3, 4),
            "avg_latency_ms": round(avg_latency, 1),
            "wiki_pages": len(documents),
        },
        "by_category": {
            cat: {
                "count": len([r for r in results if r.category == cat]),
                "hit_at_4": round(sum(1 for r in results if r.category == cat and r.hit_at_4) / max(1, len([r for r in results if r.category == cat])), 4),
                "mrr": round(sum(r.mrr for r in results if r.category == cat) / max(1, len([r for r in results if r.category == cat])), 4),
            }
            for cat in categories
        },
        "details": [
            {
                "question": r.question,
                "category": r.category,
                "hit_at_1": r.hit_at_1,
                "hit_at_3": r.hit_at_3,
                "hit_at_4": r.hit_at_4,
                "mrr": r.mrr,
                "precision_at_3": r.precision_at_3,
                "top1_slug": r.top1_slug,
                "matched_slugs": r.matched_slugs,
                "retrieved_slugs": r.retrieved_slugs,
                "elapsed_ms": round(r.elapsed_ms, 1),
            }
            for r in results
        ],
    }

    report_path = PROJECT_ROOT / "scripts" / "llmwiki_eval_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Report saved to: {report_path}")


if __name__ == "__main__":
    run_evaluation()
