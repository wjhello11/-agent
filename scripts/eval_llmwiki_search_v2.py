"""
LLMWiki commercial-style routing evaluation v2.

The original wiki evaluator is kept for backwards compatibility. This version
loads a versioned JSON dataset and resolves page slugs from wiki frontmatter,
so newly added wiki pages can be tested without editing a hard-coded map.

Usage:
    cd D:/Agent/xiaozhi-esp32-server-main/main/xiaozhi-server
    python scripts/eval_llmwiki_search_v2.py
    python scripts/eval_llmwiki_search_v2.py --strict
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
sys.path.insert(0, str(PROJECT_ROOT / "plugins_func" / "functions"))

from search_from_llmwiki import _load_markdown_documents, _rank_documents


WIKI_ROOT = PROJECT_ROOT / "knowledge_base" / "llmwiki" / "clinical-nutrition"
DATASET_PATH = PROJECT_ROOT / "scripts" / "eval_datasets" / "llmwiki_eval_v2.json"
REPORT_PATH = PROJECT_ROOT / "scripts" / "llmwiki_eval_v2_report.json"


@dataclass
class WikiEvalCase:
    case_id: str
    category: str
    question: str
    expected_slugs: list[str]
    top1_preferred: str
    reference_intent: str
    expected_source_markers: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WikiEvalCase":
        expected = [str(item) for item in payload.get("expected_slugs", [])]
        return cls(
            case_id=str(payload.get("id", "")),
            category=str(payload.get("category", "")),
            question=str(payload.get("question", "")),
            expected_slugs=expected,
            top1_preferred=str(payload.get("top1_preferred") or (expected[0] if expected else "")),
            reference_intent=str(payload.get("reference_intent", "")),
            expected_source_markers=[str(item) for item in payload.get("expected_source_markers", [])],
        )


@dataclass
class WikiEvalResult:
    case_id: str
    category: str
    question: str
    hit_at_1: bool
    hit_at_3: bool
    hit_at_4: bool
    preferred_top1: bool
    mrr: float
    precision_at_3: float
    top1_slug: str
    matched_slugs: list[str] = field(default_factory=list)
    retrieved_slugs: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


def _load_cases(path: Path) -> tuple[dict[str, Any], list[WikiEvalCase]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, [WikiEvalCase.from_dict(item) for item in payload.get("cases", [])]


def _slug_for_document(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    slug = str(metadata.get("slug") or "").strip()
    if slug:
        return slug
    relative_path = Path(str(item.get("relative_path") or ""))
    if relative_path.name == "_index.md":
        return "index"
    return relative_path.stem


def _document_identity_text(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    bits = [
        str(item.get("relative_path") or ""),
        str(item.get("title") or ""),
        str(metadata.get("source_document") or ""),
        str(metadata.get("source_name") or ""),
        str(metadata.get("source") or ""),
    ]
    source_documents = metadata.get("source_documents")
    if isinstance(source_documents, list):
        bits.extend(str(value) for value in source_documents)
    return " ".join(bits).lower()


def _matches_expected(item: dict[str, Any], case: WikiEvalCase) -> bool:
    slug = _slug_for_document(item)
    if slug not in set(case.expected_slugs):
        return False
    if not case.expected_source_markers:
        return True
    identity = _document_identity_text(item)
    return any(marker.lower() in identity for marker in case.expected_source_markers)


def evaluate_case(documents: list[dict[str, Any]], case: WikiEvalCase, top_k: int) -> WikiEvalResult:
    start = time.perf_counter()
    ranked = _rank_documents(case.question, documents, top_k=top_k, snippet_chars=480)
    elapsed_ms = (time.perf_counter() - start) * 1000
    retrieved_slugs = [_slug_for_document(item) for item in ranked]

    hit_at_1 = bool(ranked) and _matches_expected(ranked[0], case)
    hit_at_3 = any(_matches_expected(item, case) for item in ranked[:3])
    hit_at_4 = any(_matches_expected(item, case) for item in ranked[:4])
    preferred_top1 = bool(retrieved_slugs) and retrieved_slugs[0] == case.top1_preferred
    mrr = 0.0
    for index, item in enumerate(ranked):
        if _matches_expected(item, case):
            mrr = 1.0 / (index + 1)
            break
    precision_at_3 = sum(1 for item in ranked[:3] if _matches_expected(item, case)) / 3.0
    matched_slugs = [slug for item, slug in zip(ranked, retrieved_slugs) if _matches_expected(item, case)]

    return WikiEvalResult(
        case_id=case.case_id,
        category=case.category,
        question=case.question,
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        hit_at_4=hit_at_4,
        preferred_top1=preferred_top1,
        mrr=mrr,
        precision_at_3=precision_at_3,
        top1_slug=retrieved_slugs[0] if retrieved_slugs else "",
        matched_slugs=matched_slugs,
        retrieved_slugs=retrieved_slugs,
        elapsed_ms=elapsed_ms,
    )


def _summary(results: list[WikiEvalResult]) -> dict[str, Any]:
    count = len(results)
    if count == 0:
        return {
            "test_cases": 0,
            "top1_accuracy": 0.0,
            "preferred_top1_accuracy": 0.0,
            "hit_at_3": 0.0,
            "hit_at_4": 0.0,
            "mrr": 0.0,
            "precision_at_3": 0.0,
            "avg_latency_ms": 0.0,
        }
    return {
        "test_cases": count,
        "top1_accuracy": round(sum(1 for item in results if item.hit_at_1) / count, 4),
        "preferred_top1_accuracy": round(sum(1 for item in results if item.preferred_top1) / count, 4),
        "hit_at_3": round(sum(1 for item in results if item.hit_at_3) / count, 4),
        "hit_at_4": round(sum(1 for item in results if item.hit_at_4) / count, 4),
        "mrr": round(sum(item.mrr for item in results) / count, 4),
        "precision_at_3": round(sum(item.precision_at_3 for item in results) / count, 4),
        "avg_latency_ms": round(sum(item.elapsed_ms for item in results) / count, 1),
    }


def run(dataset_path: Path, report_path: Path, top_k: int, strict: bool) -> int:
    payload, cases = _load_cases(dataset_path)
    documents = _load_markdown_documents(WIKI_ROOT, {"raw", "templates"})

    print("=" * 76)
    print(f"LLMWiki Search Evaluation v2: {payload.get('version')}")
    print("=" * 76)
    print(f"  Loaded wiki pages: {len(documents)}")

    results: list[WikiEvalResult] = []
    for index, case in enumerate(cases, start=1):
        result = evaluate_case(documents, case, top_k)
        results.append(result)
        status = "HIT" if result.hit_at_4 else "MISS"
        top1_mark = " *" if result.preferred_top1 else ""
        print(f"\n[{index}/{len(cases)}] {case.case_id} [{case.category}] {case.question}")
        print(
            f"  {status} | Top1={result.top1_slug}{top1_mark} | "
            f"Hit@3={int(result.hit_at_3)} Hit@4={int(result.hit_at_4)} "
            f"MRR={result.mrr:.2f} P@3={result.precision_at_3:.1f} | {result.elapsed_ms:.1f}ms"
        )
        if result.matched_slugs:
            print(f"  Matched: {', '.join(result.matched_slugs)}")
        print(f"  Retrieved: {', '.join(result.retrieved_slugs)}")

    summary = _summary(results)
    print("\n" + "=" * 76)
    print("Summary")
    print("=" * 76)
    for key, value in summary.items():
        print(f"  {key}: {value}")

    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({item.category for item in results}):
        category_results = [item for item in results if item.category == category]
        by_category[category] = _summary(category_results)
    print("\n  By category:")
    for category, values in by_category.items():
        print(f"    {category}: Hit@4={values['hit_at_4']:.0%} MRR={values['mrr']:.2f} ({values['test_cases']} cases)")

    misses = [item for item in results if not item.hit_at_4]
    wrong_top1 = [item for item in results if not item.hit_at_1 and item.hit_at_4]
    if misses:
        print(f"\n  Missed cases ({len(misses)}):")
        for item in misses:
            print(f"    - {item.case_id} {item.question} -> {', '.join(item.retrieved_slugs)}")
    if wrong_top1:
        print(f"\n  Wrong Top-1 but hit in results ({len(wrong_top1)}):")
        case_map = {case.case_id: case for case in cases}
        for item in wrong_top1:
            expected = case_map[item.case_id].top1_preferred
            print(f"    - {item.case_id} {item.question}: top1={item.top1_slug}, preferred={expected}")

    report = {
        "dataset": payload,
        "summary": summary,
        "by_category": by_category,
        "details": [item.__dict__ for item in results],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Report saved to: {report_path}")

    if strict and misses:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    return run(args.dataset, args.report, max(1, args.top_k), args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
