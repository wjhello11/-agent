"""
Clinical RAG commercial-style evaluation v2.

This evaluator keeps the original page-based script intact and adds a more robust
versioned test set:
- RAG evidence retrieval is judged by expected source + keyword coverage.
- Structured table/recipe/MET queries are routed to clinical_knowledge.db.
- Safety redline queries are routed to the deterministic rule engine.
- Visual, short-term-memory, and final-answer abstention cases are recorded as
  manual generation contracts instead of being falsely marked pass/fail.

Usage:
    cd D:/Agent/xiaozhi-esp32-server-main/main/xiaozhi-server
    python scripts/eval_clinical_rag_v2.py
    python scripts/eval_clinical_rag_v2.py --strict
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import load_config
from config.logger import setup_logging
from core.clinical_nutrition.clinical_rag import ClinicalRAGService
from core.clinical_nutrition.structured_knowledge import search_structured_knowledge
from core.clinical_safety import ClinicalSafetyInterceptor


DATASET_PATH = PROJECT_ROOT / "scripts" / "eval_datasets" / "clinical_rag_eval_v2.json"
REPORT_PATH = PROJECT_ROOT / "scripts" / "clinical_rag_eval_v2_report.json"

AUTOMATED_ROUTES = {"rag", "structured_knowledge", "safety"}


@dataclass
class EvalCase:
    case_id: str
    category: str
    question: str
    expected_route: str
    expected_sources: list[str]
    expected_keywords: list[str]
    reference_answer: str
    expected_rule_ids: list[str] = field(default_factory=list)
    memory_context: str = ""
    manual_reason: str = ""
    must_not_contain: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalCase":
        return cls(
            case_id=str(payload.get("id", "")),
            category=str(payload.get("category", "")),
            question=str(payload.get("question", "")),
            expected_route=str(payload.get("expected_route", "")),
            expected_sources=[str(item) for item in payload.get("expected_sources", [])],
            expected_keywords=[str(item) for item in payload.get("expected_keywords", [])],
            reference_answer=str(payload.get("reference_answer", "")),
            expected_rule_ids=[str(item) for item in payload.get("expected_rule_ids", [])],
            memory_context=str(payload.get("memory_context", "")),
            manual_reason=str(payload.get("manual_reason", "")),
            must_not_contain=[str(item) for item in payload.get("must_not_contain", [])],
        )


@dataclass
class EvalResult:
    case_id: str
    category: str
    question: str
    expected_route: str
    status: str
    hit_at_3: bool = False
    hit_at_6: bool = False
    mrr: float = 0.0
    answer_covered: bool = False
    source_covered: bool = False
    matched_sources: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    matched_rule_ids: list[str] = field(default_factory=list)
    failure_class: str = ""
    top_titles: list[str] = field(default_factory=list)
    top_citations: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    manual_reason: str = ""


def _load_cases(path: Path) -> tuple[dict[str, Any], list[EvalCase]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, [EvalCase.from_dict(item) for item in payload.get("cases", [])]


def _resolve_structured_db(config: dict[str, Any]) -> Path:
    knowledge_config = config.get("clinical_knowledge") or {}
    plugin_config = (config.get("plugins") or {}).get("search_clinical_structured_knowledge") or {}
    configured = knowledge_config.get("db_path") or plugin_config.get("db_path") or "data/clinical_knowledge.db"
    path = Path(str(configured))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _normalize(value: Any) -> str:
    text = str(value or "").lower().replace("～", "-").replace("—", "-").replace("－", "-")
    return re.sub(r"\s+", "", text)


def _item_text(item: dict[str, Any]) -> str:
    fields = [
        item.get("title"),
        item.get("section_title"),
        item.get("citation"),
        item.get("source_document"),
        item.get("source_name"),
        item.get("content"),
        item.get("text"),
        item.get("raw_text"),
        item.get("search_text"),
        item.get("type"),
    ]
    metadata = item.get("metadata")
    if metadata:
        fields.append(json.dumps(metadata, ensure_ascii=False))
    return "\n".join(str(field or "") for field in fields)


def _item_source_text(item: dict[str, Any]) -> str:
    fields = [
        item.get("title"),
        item.get("citation"),
        item.get("source_document"),
        item.get("source_name"),
    ]
    return "\n".join(str(field or "") for field in fields)


def _source_matches(source: str, item: dict[str, Any], route: str) -> bool:
    source_norm = _normalize(source)
    if source_norm in {"clinical_knowledge.db", "结构化知识库"}:
        return route == "structured_knowledge"
    if source_norm in {"安全规则", "clinical_safety_rules.json"}:
        return route == "safety"
    text = _normalize(_item_source_text(item))
    return source_norm in text


def _matched_sources(case: EvalCase, items: list[dict[str, Any]], route: str) -> list[str]:
    matches: list[str] = []
    for source in case.expected_sources:
        if any(_source_matches(source, item, route) for item in items):
            matches.append(source)
    return matches


def _matched_keywords(case: EvalCase, text: str) -> list[str]:
    normalized_text = _normalize(text)
    matches: list[str] = []
    for keyword in case.expected_keywords:
        normalized_keyword = _normalize(keyword)
        if normalized_keyword and normalized_keyword in normalized_text:
            matches.append(keyword)
    return matches


def _coverage_ok(case: EvalCase, matches: list[str]) -> bool:
    if not case.expected_keywords:
        return True
    threshold = max(1, (len(case.expected_keywords) + 1) // 2)
    return len(matches) >= threshold


def _evaluate_items(case: EvalCase, items: list[dict[str, Any]], route: str, elapsed_ms: float) -> EvalResult:
    all_text = "\n".join(_item_text(item) for item in items)
    matched_keywords = _matched_keywords(case, all_text)
    matched_sources_all = _matched_sources(case, items, route)

    relevant_indices: list[int] = []
    for index, item in enumerate(items[:6]):
        if case.expected_sources and any(_source_matches(source, item, route) for source in case.expected_sources):
            relevant_indices.append(index)
            continue
        item_keywords = _matched_keywords(case, _item_text(item))
        if _coverage_ok(case, item_keywords):
            relevant_indices.append(index)

    hit_at_3 = any(index < 3 for index in relevant_indices)
    hit_at_6 = any(index < 6 for index in relevant_indices)
    mrr = 1.0 / (relevant_indices[0] + 1) if relevant_indices else 0.0
    answer_covered = _coverage_ok(case, matched_keywords)
    source_covered = bool(matched_sources_all) or not case.expected_sources
    passed = hit_at_6 and answer_covered and source_covered

    failure_class = ""
    if not items:
        failure_class = "missing_context"
    elif not source_covered:
        failure_class = "wrong_document"
    elif not answer_covered:
        failure_class = "missing_keywords"
    elif items and str(items[0].get("chunk_type") or items[0].get("type") or "").lower() == "table" and route == "rag":
        failure_class = "table_pollution"

    return EvalResult(
        case_id=case.case_id,
        category=case.category,
        question=case.question,
        expected_route=case.expected_route,
        status="pass" if passed else "fail",
        hit_at_3=hit_at_3,
        hit_at_6=hit_at_6,
        mrr=mrr,
        answer_covered=answer_covered,
        source_covered=source_covered,
        matched_sources=matched_sources_all,
        matched_keywords=matched_keywords,
        failure_class=failure_class,
        top_titles=[str(item.get("title") or item.get("section_title") or item.get("type") or "") for item in items[:6]],
        top_citations=[str(item.get("citation") or item.get("source_name") or "") for item in items[:6]],
        elapsed_ms=elapsed_ms,
    )


def _evaluate_rag(case: EvalCase, service: ClinicalRAGService, top_k: int) -> EvalResult:
    start = time.perf_counter()
    items = service.search(case.question, top_k=top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return _evaluate_items(case, items, "rag", elapsed_ms)


def _evaluate_structured(case: EvalCase, db_path: Path, top_k: int) -> EvalResult:
    start = time.perf_counter()
    items = search_structured_knowledge(db_path, case.question, limit=top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return _evaluate_items(case, items, "structured_knowledge", elapsed_ms)


def _evaluate_safety(case: EvalCase, interceptor: ClinicalSafetyInterceptor) -> EvalResult:
    start = time.perf_counter()
    interception = interceptor.evaluate(query=case.question, memory_context=case.memory_context)
    elapsed_ms = (time.perf_counter() - start) * 1000
    rule_ids = [item.rule_id for item in interception.findings]
    text = "\n".join(
        [
            " ".join(rule_ids),
            interception.response_text,
            interception.prompt_context,
            json.dumps(interception.extracted_context, ensure_ascii=False),
        ]
    )
    matched_keywords = _matched_keywords(case, text)
    matched_rule_ids = [rule_id for rule_id in case.expected_rule_ids if rule_id in rule_ids]
    rule_ok = bool(matched_rule_ids) if case.expected_rule_ids else bool(interception.findings)
    answer_covered = _coverage_ok(case, matched_keywords)
    forbidden_hits = [item for item in case.must_not_contain if _normalize(item) in _normalize(text)]
    passed = rule_ok and answer_covered and not forbidden_hits
    if not interception.findings:
        failure_class = "missing_context"
    elif not rule_ok:
        failure_class = "wrong_rule"
    elif not answer_covered:
        failure_class = "missing_keywords"
    elif forbidden_hits:
        failure_class = "unsafe_answer"
    else:
        failure_class = ""

    return EvalResult(
        case_id=case.case_id,
        category=case.category,
        question=case.question,
        expected_route=case.expected_route,
        status="pass" if passed else "fail",
        hit_at_3=passed,
        hit_at_6=passed,
        mrr=1.0 if passed else 0.0,
        answer_covered=answer_covered,
        source_covered=bool(interception.findings),
        matched_sources=["安全规则"] if interception.findings else [],
        matched_keywords=matched_keywords,
        matched_rule_ids=matched_rule_ids,
        failure_class=failure_class,
        top_titles=rule_ids,
        top_citations=[interception.response_text[:160]],
        elapsed_ms=elapsed_ms,
    )


def _manual_result(case: EvalCase) -> EvalResult:
    return EvalResult(
        case_id=case.case_id,
        category=case.category,
        question=case.question,
        expected_route=case.expected_route,
        status="manual",
        failure_class="manual_generation_contract",
        manual_reason=case.manual_reason or "This case requires end-to-end generation or device workflow review.",
    )


def _summary(results: list[EvalResult]) -> dict[str, Any]:
    automated = [item for item in results if item.status != "manual"]
    manual = [item for item in results if item.status == "manual"]
    passed = [item for item in automated if item.status == "pass"]
    safety = [item for item in automated if item.expected_route == "safety"]
    if automated:
        hit_at_6 = sum(1 for item in automated if item.hit_at_6) / len(automated)
        recall_at_3 = sum(1 for item in automated if item.hit_at_3) / len(automated)
        coverage = sum(1 for item in automated if item.answer_covered) / len(automated)
        mrr = sum(item.mrr for item in automated) / len(automated)
        avg_latency = sum(item.elapsed_ms for item in automated) / len(automated)
    else:
        hit_at_6 = recall_at_3 = coverage = mrr = avg_latency = 0.0
    safety_hit = sum(1 for item in safety if item.status == "pass") / len(safety) if safety else None
    return {
        "total_cases": len(results),
        "automated_cases": len(automated),
        "manual_cases": len(manual),
        "passed_cases": len(passed),
        "pass_rate": round(len(passed) / len(automated), 4) if automated else 0.0,
        "hit_at_6": round(hit_at_6, 4),
        "recall_at_3": round(recall_at_3, 4),
        "answer_coverage": round(coverage, 4),
        "mrr": round(mrr, 4),
        "automated_safety_hit": round(safety_hit, 4) if safety_hit is not None else None,
        "avg_latency_ms": round(avg_latency, 1),
    }


def run(dataset_path: Path, report_path: Path, top_k: int, strict: bool) -> int:
    payload, cases = _load_cases(dataset_path)
    config = load_config()
    logger = setup_logging()
    rag_service = ClinicalRAGService(project_root=PROJECT_ROOT, config=config, logger=logger)
    structured_db = _resolve_structured_db(config)
    safety = ClinicalSafetyInterceptor(PROJECT_ROOT / "knowledge_base" / "rules" / "clinical_safety_rules.json")

    print("=" * 76)
    print(f"Clinical RAG Evaluation v2: {payload.get('version')}")
    print("=" * 76)

    results: list[EvalResult] = []
    for index, case in enumerate(cases, start=1):
        print(f"\n[{index}/{len(cases)}] {case.case_id} [{case.category}] {case.question}")
        if case.expected_route == "rag":
            result = _evaluate_rag(case, rag_service, top_k)
        elif case.expected_route == "structured_knowledge":
            result = _evaluate_structured(case, structured_db, top_k)
        elif case.expected_route == "safety":
            result = _evaluate_safety(case, safety)
        else:
            result = _manual_result(case)
        results.append(result)

        if result.status == "manual":
            print(f"  MANUAL | route={result.expected_route} | {result.manual_reason}")
            continue
        print(
            f"  {result.status.upper()} | route={result.expected_route} | "
            f"Hit@3={int(result.hit_at_3)} Hit@6={int(result.hit_at_6)} "
            f"Coverage={int(result.answer_covered)} MRR={result.mrr:.2f} | {result.elapsed_ms:.0f}ms"
        )
        if result.matched_sources:
            print(f"  Sources: {', '.join(result.matched_sources)}")
        if result.matched_keywords:
            print(f"  Keywords: {', '.join(result.matched_keywords)}")
        if result.matched_rule_ids:
            print(f"  Rules: {', '.join(result.matched_rule_ids)}")
        if result.failure_class:
            print(f"  Failure: {result.failure_class}")

    summary = _summary(results)
    print("\n" + "=" * 76)
    print("Summary")
    print("=" * 76)
    for key, value in summary.items():
        print(f"  {key}: {value}")

    failures = [item for item in results if item.status == "fail"]
    manual = [item for item in results if item.status == "manual"]
    if failures:
        print(f"\n  Failed automated cases ({len(failures)}):")
        for item in failures:
            print(f"    - {item.case_id} {item.question} [{item.failure_class}]")
    if manual:
        print(f"\n  Manual contract cases ({len(manual)}):")
        for item in manual:
            print(f"    - {item.case_id} {item.question}")

    report = {
        "dataset": payload,
        "summary": summary,
        "details": [item.__dict__ for item in results],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Report saved to: {report_path}")

    if strict and failures:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    return run(args.dataset, args.report, max(1, args.top_k), args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
