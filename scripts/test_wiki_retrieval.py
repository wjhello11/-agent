"""
Test wiki retrieval for the 4 clinical nutrition PDFs.

Tests whether user questions about each PDF's content correctly route
to the expected wiki pages.

Usage:
    cd D:/Agent/xiaozhi-esp32-server-main/main/xiaozhi-server
    python scripts/test_wiki_retrieval.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "plugins_func" / "functions"))

from search_from_llmwiki import _load_markdown_documents, _rank_documents

WIKI_ROOT = PROJECT_ROOT / "knowledge_base" / "llmwiki" / "clinical-nutrition"


@dataclass
class RetrievalCase:
    case_id: str
    category: str
    question: str
    expected_slugs: list[str]
    top1_preferred: str
    expected_source_markers: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    case_id: str
    category: str
    question: str
    hit_at_1: bool
    hit_at_4: bool
    top1_slug: str
    retrieved_slugs: list[str] = field(default_factory=list)
    retrieved_sources: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


# Test cases for each PDF
TEST_CASES: list[RetrievalCase] = [
    # === 成人肥胖指南 ===
    RetrievalCase(
        case_id="OB01",
        category="肥胖-诊断",
        question="BMI多少算肥胖？",
        expected_slugs=["overview"],
        top1_preferred="overview",
    ),
    RetrievalCase(
        case_id="OB02",
        category="肥胖-能量",
        question="肥胖的人每天应该吃多少热量？",
        expected_slugs=["energy-control", "principles", "overview"],
        top1_preferred="energy-control",
    ),
    RetrievalCase(
        case_id="OB03",
        category="肥胖-食物",
        question="肥胖患者应该怎么选择食物？",
        expected_slugs=["food-selection"],
        top1_preferred="food-selection",
    ),
    RetrievalCase(
        case_id="OB04",
        category="肥胖-中医",
        question="中医怎么调理肥胖？有哪些食药物质？",
        expected_slugs=["tcm-diet-therapy"],
        top1_preferred="tcm-diet-therapy",
    ),
    RetrievalCase(
        case_id="OB05",
        category="肥胖-安全",
        question="减重速度多少是安全的？",
        expected_slugs=["safe-weight-loss"],
        top1_preferred="safe-weight-loss",
    ),
    # === 糖尿病膳食指南 ===
    RetrievalCase(
        case_id="DM01",
        category="糖尿病-碳水",
        question="糖尿病患者碳水化合物应该占多少比例？",
        expected_slugs=["principles", "diabetes-carbohydrate-gi-gl", "diabetes-medical-nutrition-therapy"],
        top1_preferred="principles",
    ),
    RetrievalCase(
        case_id="DM02",
        category="糖尿病-早餐",
        question="糖尿病早餐怎么搭配？",
        expected_slugs=["food-selection", "diabetes-breakfast-decision-guide"],
        top1_preferred="food-selection",
    ),
    RetrievalCase(
        case_id="DM03",
        category="糖尿病-纤维",
        question="膳食纤维每天要吃多少？",
        expected_slugs=["principles", "diabetes-medical-nutrition-therapy"],
        top1_preferred="principles",
    ),
    # === 高尿酸血症与痛风 ===
    RetrievalCase(
        case_id="GOUT01",
        category="痛风-饮食",
        question="尿酸高不能吃什么食物？",
        expected_slugs=["food-selection", "overview"],
        top1_preferred="food-selection",
    ),
    RetrievalCase(
        case_id="GOUT02",
        category="痛风-饮水",
        question="痛风患者每天喝多少水？",
        expected_slugs=["energy-control"],
        top1_preferred="energy-control",
    ),
    RetrievalCase(
        case_id="GOUT03",
        category="痛风-中医",
        question="痛风中医怎么食养？",
        expected_slugs=["tcm-diet-therapy"],
        top1_preferred="tcm-diet-therapy",
    ),
    RetrievalCase(
        case_id="GOUT04",
        category="痛风-运动",
        question="痛风患者适合做什么运动？",
        expected_slugs=["exercise-sleep"],
        top1_preferred="exercise-sleep",
    ),
    # === 高血压防治指南 ===
    RetrievalCase(
        case_id="HT01",
        category="高血压-盐",
        question="高血压每天盐不能超过多少？",
        expected_slugs=["dietary-salt-sodium", "overview", "dietary-principles"],
        top1_preferred="dietary-salt-sodium",
    ),
    RetrievalCase(
        case_id="HT02",
        category="高血压-合并症",
        question="高血压合并肥胖怎么吃？",
        expected_slugs=["dietary-principles", "overview"],
        top1_preferred="dietary-principles",
    ),
    RetrievalCase(
        case_id="HT03",
        category="高血压-中医",
        question="高血压中医食养有什么建议？",
        expected_slugs=["tcm-diet-therapy", "overview"],
        top1_preferred="tcm-diet-therapy",
    ),
]


def _slug_for_document(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    slug = str(metadata.get("slug") or "").strip()
    if slug:
        return slug
    relative_path = Path(str(item.get("relative_path") or ""))
    if relative_path.name == "_index.md":
        return "index"
    return relative_path.stem


def _source_markers_for_case(case: RetrievalCase) -> list[str]:
    if case.expected_source_markers:
        return case.expected_source_markers
    prefix = case.case_id.upper()
    if prefix.startswith("OB"):
        return ["成人肥胖", "adult-obesity", "肥胖"]
    if prefix.startswith("DM"):
        return ["糖尿病", "diabetes"]
    if prefix.startswith("GOUT"):
        return ["高尿酸", "痛风", "gout", "hyperuricemia"]
    if prefix.startswith("HT"):
        return ["高血压", "hypertension"]
    return []


def _document_identity_text(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    source_bits = [
        str(item.get("relative_path") or ""),
        str(item.get("title") or ""),
        str(metadata.get("source_document") or ""),
        str(metadata.get("source_name") or ""),
        str(metadata.get("source") or ""),
    ]
    source_documents = metadata.get("source_documents")
    if isinstance(source_documents, list):
        source_bits.extend(str(value) for value in source_documents)
    return " ".join(source_bits).lower()


def _source_label(item: dict[str, Any]) -> str:
    relative_path = str(item.get("relative_path") or "")
    parts = Path(relative_path).parts
    return "/".join(parts[-3:]) if len(parts) >= 3 else relative_path


def _matches_expected_document(item: dict[str, Any], case: RetrievalCase) -> bool:
    slug = _slug_for_document(item)
    if slug not in set(case.expected_slugs):
        return False
    markers = _source_markers_for_case(case)
    if not markers:
        return True
    identity = _document_identity_text(item)
    return any(marker.lower() in identity for marker in markers)


def evaluate_case(documents: list[dict[str, Any]], case: RetrievalCase) -> RetrievalResult:
    start = time.perf_counter()
    ranked = _rank_documents(case.question, documents, top_k=4, snippet_chars=480)
    elapsed_ms = (time.perf_counter() - start) * 1000

    retrieved_slugs = [_slug_for_document(item) for item in ranked]
    retrieved_sources = [_source_label(item) for item in ranked]

    hit_at_1 = bool(ranked) and _matches_expected_document(ranked[0], case)
    hit_at_4 = any(_matches_expected_document(item, case) for item in ranked[:4])

    return RetrievalResult(
        case_id=case.case_id,
        category=case.category,
        question=case.question,
        hit_at_1=hit_at_1,
        hit_at_4=hit_at_4,
        top1_slug=retrieved_slugs[0] if retrieved_slugs else "",
        retrieved_slugs=retrieved_slugs,
        retrieved_sources=retrieved_sources,
        elapsed_ms=elapsed_ms,
    )


def main():
    print("=" * 70)
    print("Wiki Retrieval Test - Clinical Nutrition PDFs")
    print("=" * 70)

    documents = _load_markdown_documents(WIKI_ROOT, {"raw", "templates"})
    print(f"Loaded {len(documents)} wiki pages:")
    for doc in documents:
        rel = doc.get("relative_path", "?")
        title = doc.get("title", "?")
        print(f"  {rel}: {title}")
    print()

    results: list[RetrievalResult] = []
    for case in TEST_CASES:
        result = evaluate_case(documents, case)
        results.append(result)
        status = "HIT" if result.hit_at_4 else "MISS"
        top1_mark = " *" if result.hit_at_1 else ""
        print(f"[{result.case_id}] {result.category}: {result.question}")
        print(f"  {status} | Top1={result.top1_slug}{top1_mark} | "
              f"Hit@1={int(result.hit_at_1)} Hit@4={int(result.hit_at_4)} | "
              f"{result.elapsed_ms:.1f}ms")
        print(f"  Retrieved: {', '.join(result.retrieved_slugs)}")
        print(f"  Sources: {', '.join(result.retrieved_sources)}")
        print()

    # Summary
    total = len(results)
    hit1 = sum(1 for r in results if r.hit_at_1)
    hit4 = sum(1 for r in results if r.hit_at_4)

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Total cases: {total}")
    print(f"  Hit@1: {hit1}/{total} ({hit1/total*100:.1f}%)")
    print(f"  Hit@4: {hit4}/{total} ({hit4/total*100:.1f}%)")

    # Per-category breakdown
    categories = sorted(set(r.category.split("-")[0] for r in results))
    for cat in categories:
        cat_results = [r for r in results if r.category.startswith(cat)]
        cat_hit4 = sum(1 for r in cat_results if r.hit_at_4)
        print(f"  {cat}: {cat_hit4}/{len(cat_results)} Hit@4")

    # Failures
    failures = [r for r in results if not r.hit_at_4]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for r in failures:
            print(f"  [{r.case_id}] {r.question}")
            expected_case = next(c for c in TEST_CASES if c.case_id == r.case_id)
            print(f"    Expected: {expected_case.expected_slugs}")
            print(f"    Expected source markers: {_source_markers_for_case(expected_case)}")
            print(f"    Got: {r.retrieved_slugs}")
            print(f"    Sources: {r.retrieved_sources}")
    else:
        print("\nAll cases passed!")

    return 0 if hit4 == total else 1


if __name__ == "__main__":
    sys.exit(main())
