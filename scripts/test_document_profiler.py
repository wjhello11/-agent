"""
Test script for the AI Document Structure Compiler.

Tests:
1. Pydantic schema validation
2. DocumentProfiler fallback plan (no LLM)
3. DocumentProfiler with LLM (if configured)
4. StructuredKnowledgeStore.ingest_from_plan()

Usage:
    cd D:/Agent/xiaozhi-esp32-server-main/main/xiaozhi-server
    python scripts/test_document_profiler.py
    python scripts/test_document_profiler.py --pdf "data/knowledge_uploads/clinical-nutrition/成人肥胖指南.pdf"
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_schemas():
    """Test Pydantic schema validation."""
    print("\n" + "=" * 60)
    print("Test 1: Pydantic Schema Validation")
    print("=" * 60)

    from core.clinical_nutrition.ingestion_schemas import (
        ActivityMET,
        DiagnosticThreshold,
        ExchangePortion,
        GuideTable,
        GuideTableRow,
        IngestionBlock,
        IngestionPlan,
        NeedsReviewItem,
        NutritionTarget,
        RecipePlan,
        SafetyRuleCandidate,
        TherapeuticRecipe,
        BLOCK_TYPE_SCHEMA_MAP,
        get_schema_for_block_type,
    )

    # Test IngestionPlan
    plan = IngestionPlan(
        document_type="clinical_guideline",
        knowledge_types=["诊断阈值", "营养目标"],
        blocks=[
            IngestionBlock(
                block_id="b001",
                block_type="diagnostic_threshold",
                section_path="第一章",
                page_start=1,
                page_end=5,
                confidence=0.85,
                should_store_in="structured",
            ),
            IngestionBlock(
                block_id="b002",
                block_type="narrative_guideline",
                section_path="第二章",
                page_start=6,
                page_end=10,
                confidence=0.9,
                should_store_in="wiki",
            ),
        ],
        total_pages=10,
    )
    assert len(plan.blocks) == 2
    assert plan.blocks[0].should_store_in == "structured"
    print(f"  IngestionPlan: {len(plan.blocks)} blocks, type={plan.document_type}")

    # Test DiagnosticThreshold
    dt = DiagnosticThreshold(
        indicator="BMI",
        threshold=">=28.0",
        unit="kg/m2",
        population="成人",
        source_document="肥胖指南",
        source_pages=[8],
    )
    assert dt.indicator == "BMI"
    print(f"  DiagnosticThreshold: {dt.indicator} {dt.threshold} {dt.unit}")

    # Test ExchangePortion
    ep = ExchangePortion(
        food_name="米饭",
        exchange_group="谷薯类",
        serving_amount="25g",
        energy_kcal=90,
        carbohydrate_g=20,
        protein_g=2,
        fat_g=0.5,
        source_document="糖尿病膳食指南",
        source_pages=[15],
    )
    assert ep.energy_kcal == 90
    print(f"  ExchangePortion: {ep.food_name} {ep.serving_amount} {ep.energy_kcal}kcal")

    # Test RecipePlan
    rp = RecipePlan(
        title="春季食谱1",
        season="春季",
        energy_kcal=1500,
        source_document="肥胖指南",
        source_pages=[68, 69],
    )
    assert rp.season == "春季"
    print(f"  RecipePlan: {rp.title} {rp.energy_kcal}kcal")

    # Test TherapeuticRecipe
    tr = TherapeuticRecipe(
        syndrome="痰湿证",
        title="薏米山药粥",
        ingredients=["薏米30g", "山药50g"],
        method="煮粥",
        source_document="肥胖指南",
        source_pages=[70],
    )
    assert len(tr.ingredients) == 2
    print(f"  TherapeuticRecipe: {tr.title} ({tr.syndrome})")

    # Test ActivityMET
    met = ActivityMET(
        category="有氧运动",
        activity_name="快走",
        met=3.5,
        intensity="中等",
        source_document="肥胖指南",
        source_pages=[20],
    )
    assert met.met == 3.5
    print(f"  ActivityMET: {met.activity_name} {met.met}MET")

    # Test SafetyRuleCandidate
    src = SafetyRuleCandidate(
        trigger_condition="BMI>35",
        risk_description="心血管风险增加",
        safety_recommendation="建议就医评估",
        severity="warn",
        source_document="肥胖指南",
        source_pages=[10],
    )
    assert src.severity == "warn"
    print(f"  SafetyRuleCandidate: {src.trigger_condition} -> {src.safety_recommendation}")

    # Test NutritionTarget
    nt = NutritionTarget(
        nutrient="碳水化合物",
        target_value="45%-60%",
        population="成人肥胖",
        source_document="肥胖指南",
        source_pages=[15],
    )
    assert nt.target_value == "45%-60%"
    print(f"  NutritionTarget: {nt.nutrient} {nt.target_value}")

    # Test GuideTable
    gt = GuideTable(
        title="表1 食物交换份",
        table_type="food_exchange",
        page_start=15,
        page_end=16,
        rows=[
            GuideTableRow(label="米饭", columns={"能量": "90kcal", "碳水": "20g"}),
        ],
        source_document="肥胖指南",
    )
    assert len(gt.rows) == 1
    print(f"  GuideTable: {gt.title} ({len(gt.rows)} rows)")

    # Test NeedsReviewItem
    nr = NeedsReviewItem(
        document_id="test_doc",
        block_id="b003",
        block_type="generic_table",
        page_start=20,
        page_end=20,
        raw_text="some table text",
        schema_errors="missing required field",
    )
    assert nr.review_status == "pending"
    print(f"  NeedsReviewItem: {nr.block_type} p.{nr.page_start}")

    # Test schema map
    assert len(BLOCK_TYPE_SCHEMA_MAP) == 10
    assert get_schema_for_block_type("diagnostic_threshold") == DiagnosticThreshold
    assert get_schema_for_block_type("unknown_type") is None
    print(f"  BLOCK_TYPE_SCHEMA_MAP: {len(BLOCK_TYPE_SCHEMA_MAP)} types")

    print("\n  All schema tests PASSED!")


def test_fallback_plan():
    """Test DocumentProfiler without LLM (fallback mode)."""
    print("\n" + "=" * 60)
    print("Test 2: DocumentProfiler Fallback Plan (no LLM)")
    print("=" * 60)

    from core.clinical_nutrition.document_profiler import DocumentProfiler

    # Simulate pages
    pages = [
        {"page_number": 1, "text": "中国成人肥胖指南（2024版）"},
        {"page_number": 2, "text": "目录\n第一章 概述\n第二章 诊断标准\n第三章 治疗"},
        {"page_number": 3, "text": "肥胖的定义：BMI>=28.0 kg/m2为中国成人肥胖标准。" * 5},
        {"page_number": 4, "text": ""},
        {"page_number": 5, "text": "腰围标准：男性>=90cm，女性>=85cm为中心性肥胖。" * 5},
    ]

    profiler = DocumentProfiler(source_name="test_guide.pdf")
    plan = profiler.generate_ingestion_plan(pages, source_name="test_guide.pdf")

    assert plan.total_pages == 5
    assert len(plan.blocks) == 5
    assert plan.source_document == "test_guide.pdf"

    # Check that empty page is skipped
    empty_blocks = [b for b in plan.blocks if b.should_store_in == "skip"]
    assert len(empty_blocks) >= 1, f"Expected at least 1 skip block, got {len(empty_blocks)}"

    # Check that non-empty pages are assigned to rag
    rag_blocks = [b for b in plan.blocks if b.should_store_in == "rag"]
    assert len(rag_blocks) >= 1, f"Expected at least 1 rag block, got {len(rag_blocks)}"

    print(f"  Plan: {len(plan.blocks)} blocks, {plan.total_pages} pages")
    for b in plan.blocks:
        print(f"    {b.block_id}: p.{b.page_start}-{b.page_end} -> {b.should_store_in} ({b.block_type})")

    print("\n  Fallback plan test PASSED!")


def test_page_coverage():
    """Test that all pages are covered in the ingestion plan."""
    print("\n" + "=" * 60)
    print("Test 3: Page Coverage Guarantee")
    print("=" * 60)

    from core.clinical_nutrition.document_profiler import DocumentProfiler

    pages = [
        {"page_number": i, "text": f"Page {i} content with enough text to not be skipped. " * 10}
        for i in range(1, 21)
    ]
    # Add a gap: skip page 10 and 15 from any block
    pages[9]["text"] = ""
    pages[14]["text"] = ""

    profiler = DocumentProfiler(source_name="test_coverage.pdf")
    plan = profiler.generate_ingestion_plan(pages, source_name="test_coverage.pdf")

    # Verify all pages are covered
    covered = set()
    for block in plan.blocks:
        for p in range(block.page_start, block.page_end + 1):
            covered.add(p)

    all_pages = {p["page_number"] for p in pages}
    missing = sorted(all_pages - covered)

    assert not missing, f"Missing pages: {missing}"
    print(f"  All {len(all_pages)} pages covered by {len(plan.blocks)} blocks")
    print(f"  Empty pages (skip): {len([b for b in plan.blocks if b.should_store_in == 'skip'])}")

    print("\n  Page coverage test PASSED!")


def test_extraction_without_llm():
    """Test structured extraction without LLM (should return empty)."""
    print("\n" + "=" * 60)
    print("Test 4: Structured Extraction (no LLM)")
    print("=" * 60)

    from core.clinical_nutrition.document_profiler import DocumentProfiler
    from core.clinical_nutrition.ingestion_schemas import IngestionBlock, IngestionPlan

    plan = IngestionPlan(
        blocks=[
            IngestionBlock(
                block_id="b001",
                block_type="diagnostic_threshold",
                page_start=8,
                page_end=8,
                should_store_in="structured",
            ),
        ],
    )
    pages = [{"page_number": 8, "text": "BMI>=28.0 kg/m2"}]

    profiler = DocumentProfiler(source_name="test.pdf")
    results = profiler.extract_structured_blocks(plan, pages)

    assert results["extracted"] == {}
    assert len(results["needs_review"]) == 1  # Should fail without LLM
    print(f"  Extracted: {results['extracted']}")
    print(f"  Needs review: {len(results['needs_review'])} items")

    print("\n  Extraction (no LLM) test PASSED!")


def test_with_real_pdf():
    """Test with a real PDF if available."""
    print("\n" + "=" * 60)
    print("Test 5: Real PDF Test")
    print("=" * 60)

    # Find the obesity guide PDF by pattern
    pdf_dir = PROJECT_ROOT / "data" / "knowledge_uploads" / "clinical-nutrition"
    pdf_candidates = list(pdf_dir.glob("*obesity*")) + list(pdf_dir.glob("*122807Z*"))
    if not pdf_candidates:
        # Try Chinese name
        pdf_candidates = list(pdf_dir.glob("*成人肥胖*")) + list(pdf_dir.glob("*肥胖指南*"))
    if not pdf_candidates:
        # Fallback: use first PDF
        pdf_candidates = list(pdf_dir.glob("*.pdf"))
    pdf_path = pdf_candidates[0] if pdf_candidates else None
    if not pdf_path or not pdf_path.exists():
        print(f"  SKIPPED: No PDF found in {pdf_dir}")
        return

    from config.config_loader import load_config
    from config.logger import setup_logging
    from core.clinical_nutrition.knowledge_ingestion import (
        KnowledgeIngestionService,
        extract_document_pages_for_ingestion,
    )

    config = load_config()
    logger = setup_logging()

    print(f"  Loading PDF: {pdf_path.name}")
    pages = extract_document_pages_for_ingestion(pdf_path)
    print(f"  Pages: {len(pages)}")

    service = KnowledgeIngestionService(
        project_root=PROJECT_ROOT,
        config=config,
        logger=logger,
    )

    print("  Running document profiler...")
    start = time.perf_counter()
    result = service._run_document_profiler(
        pages=pages,
        source_path=pdf_path,
    )
    elapsed = time.perf_counter() - start

    plan = result.get("ingestion_plan") or {}
    extraction = result.get("extraction_results") or {}
    stats = extraction.get("stats") or {}

    print(f"  Profiler used: {result.get('profiler_used')}")
    print(f"  Error: {result.get('profiler_error') or 'none'}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Document type: {plan.get('document_type', 'N/A')}")
    print(f"  Knowledge types: {plan.get('knowledge_types', [])}")
    print(f"  Blocks: {len(plan.get('blocks') or [])}")
    print(f"  Extraction stats: {stats}")
    print(f"  Needs review: {len(extraction.get('needs_review') or [])}")

    if plan.get("blocks"):
        print("\n  Block summary:")
        for block in plan["blocks"][:15]:
            print(f"    {block.get('block_id')}: p.{block.get('page_start')}-{block.get('page_end')} "
                  f"-> {block.get('should_store_in')} ({block.get('block_type')})")
        if len(plan["blocks"]) > 15:
            print(f"    ... and {len(plan['blocks']) - 15} more blocks")

    print("\n  Real PDF test DONE!")


def main():
    print("=" * 60)
    print("AI Document Structure Compiler - Test Suite")
    print("=" * 60)

    test_schemas()
    test_fallback_plan()
    test_page_coverage()
    test_extraction_without_llm()

    # Check if --pdf argument is provided
    if "--pdf" in sys.argv:
        idx = sys.argv.index("--pdf")
        if idx + 1 < len(sys.argv):
            pdf_path = Path(sys.argv[idx + 1])
            if pdf_path.exists():
                test_with_real_pdf()
            else:
                print(f"\n  PDF not found: {pdf_path}")
        else:
            print("\n  --pdf requires a path argument")
    else:
        print("\n  Skipping real PDF test. Pass --pdf <path> to run document profiler on a PDF.")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
