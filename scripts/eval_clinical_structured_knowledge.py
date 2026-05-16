"""
Clinical structured knowledge evaluation.

This checks clinical_knowledge.db independently from RAG/Wiki so table rows,
exchange portions, recipes, therapeutic recipes, MET values, and diagnostic
thresholds do not get hidden behind narrative retrieval scores.

Usage:
    cd D:/Agent/xiaozhi-esp32-server-main/main/xiaozhi-server
    python scripts/eval_clinical_structured_knowledge.py
    python scripts/eval_clinical_structured_knowledge.py --strict
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import load_config
from core.clinical_nutrition.structured_knowledge import search_structured_knowledge


REPORT_PATH = PROJECT_ROOT / "scripts" / "clinical_structured_knowledge_eval_report.json"


@dataclass
class StructuredCase:
    case_id: str
    category: str
    question: str
    expected_types: list[str]
    expected_keywords: list[str]
    expected_source_markers: list[str] = field(default_factory=list)


@dataclass
class StructuredResult:
    case_id: str
    category: str
    question: str
    passed: bool
    hit_type: bool
    hit_keywords: bool
    hit_source: bool
    top_types: list[str]
    top_sources: list[str]
    matched_keywords: list[str]
    elapsed_ms: float


CASES: list[StructuredCase] = [
    StructuredCase(
        case_id="S01",
        category="肥胖-诊断阈值",
        question="BMI 多少算肥胖？",
        expected_types=["diagnostic_threshold", "table_row"],
        expected_keywords=["BMI", "28"],
        expected_source_markers=["肥胖", "adult-obesity"],
    ),
    StructuredCase(
        case_id="S02",
        category="高血压-诊断阈值",
        question="血压多少算高血压？",
        expected_types=["diagnostic_threshold", "table_row"],
        expected_keywords=["收缩压", "舒张压", "140", "90"],
        expected_source_markers=["高血压", "hypertension"],
    ),
    StructuredCase(
        case_id="S03",
        category="高血压-限盐",
        question="高血压一天盐最多吃多少？",
        expected_types=["nutrition_target", "table_row"],
        expected_keywords=["盐", "钠"],
        expected_source_markers=["高血压", "hypertension"],
    ),
    StructuredCase(
        case_id="S04",
        category="糖尿病-交换份",
        question="一份谷薯类交换份大概是多少主食？",
        expected_types=["food_exchange_portion", "table_row"],
        expected_keywords=["谷薯", "90", "25"],
        expected_source_markers=["糖尿病", "diabetes"],
    ),
    StructuredCase(
        case_id="S05",
        category="肥胖-MET",
        question="跳绳的 MET 大概是多少？",
        expected_types=["activity_met", "table_row"],
        expected_keywords=["跳绳", "10.2", "高"],
        expected_source_markers=["肥胖", "adult-obesity"],
    ),
    StructuredCase(
        case_id="S06",
        category="肥胖-食谱",
        question="冬季 1600kcal 减肥食谱",
        expected_types=["recipe_plan"],
        expected_keywords=["冬季", "1600", "早餐", "晚餐"],
        expected_source_markers=["肥胖", "adult-obesity"],
    ),
    StructuredCase(
        case_id="S07",
        category="肥胖-食养方",
        question="铁皮石斛玉竹煲瘦肉怎么做？",
        expected_types=["therapeutic_recipe"],
        expected_keywords=["铁皮石斛", "玉竹", "瘦肉", "制作方法"],
        expected_source_markers=["肥胖", "adult-obesity"],
    ),
    StructuredCase(
        case_id="S08",
        category="高尿酸-食养方",
        question="痛风有什么食养方？",
        expected_types=["therapeutic_recipe", "recipe_plan"],
        expected_keywords=["痛风"],
        expected_source_markers=["高尿酸", "痛风", "gout"],
    ),
]


def _resolve_db_path() -> Path:
    config = load_config()
    clinical_knowledge = config.get("clinical_knowledge") or {}
    configured = clinical_knowledge.get("db_path") or "data/clinical_knowledge.db"
    path = Path(str(configured))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _row_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in ("type", "title", "source_name", "citation", "content", "raw_text", "search_text")
    )


def evaluate_case(db_path: Path, case: StructuredCase, limit: int) -> StructuredResult:
    start = time.perf_counter()
    rows = search_structured_knowledge(db_path, case.question, limit=limit)
    elapsed_ms = (time.perf_counter() - start) * 1000

    texts = [_row_text(row) for row in rows]
    all_text = "\n".join(texts)
    top_types = [str(row.get("type") or "") for row in rows]
    top_sources = [str(row.get("source_name") or "") for row in rows]

    hit_type = any(row_type in case.expected_types for row_type in top_types)
    matched_keywords = [keyword for keyword in case.expected_keywords if keyword.lower() in all_text.lower()]
    hit_keywords = len(matched_keywords) == len(case.expected_keywords)
    if case.expected_source_markers:
        hit_source = any(marker.lower() in all_text.lower() for marker in case.expected_source_markers)
    else:
        hit_source = True

    return StructuredResult(
        case_id=case.case_id,
        category=case.category,
        question=case.question,
        passed=bool(rows) and hit_type and hit_keywords and hit_source,
        hit_type=hit_type,
        hit_keywords=hit_keywords,
        hit_source=hit_source,
        top_types=top_types,
        top_sources=top_sources,
        matched_keywords=matched_keywords,
        elapsed_ms=elapsed_ms,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any case fails.")
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()

    db_path = _resolve_db_path()
    if not db_path.exists():
        print(f"clinical_knowledge.db not found: {db_path}")
        return 1 if args.strict else 0

    results = [evaluate_case(db_path, case, args.limit) for case in CASES]
    passed = sum(1 for result in results if result.passed)

    print("=" * 72)
    print("Clinical Structured Knowledge Eval")
    print("=" * 72)
    print(f"DB: {db_path}")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{result.case_id}] {status} {result.category}: {result.question}")
        print(f"  type={int(result.hit_type)} keywords={int(result.hit_keywords)} source={int(result.hit_source)} {result.elapsed_ms:.1f}ms")
        print(f"  matched_keywords={result.matched_keywords}")
        print(f"  top_types={result.top_types[:4]}")
        print(f"  top_sources={result.top_sources[:4]}")

    report = {
        "version": "clinical_structured_knowledge_eval_v1",
        "db_path": str(db_path),
        "summary": {
            "test_cases": len(results),
            "passed": passed,
            "pass_rate": round(passed / len(results), 4) if results else 0.0,
        },
        "results": [result.__dict__ for result in results],
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"Passed: {passed}/{len(results)} ({passed / len(results) * 100:.1f}%)")
    print(f"Report: {REPORT_PATH}")
    return 0 if passed == len(results) or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
