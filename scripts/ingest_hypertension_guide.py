"""
高血压指南完整入库流程脚本

流程：
1. 注册文档到 RAG
2. 索引文档（分块 + 向量化 + FTS5）
3. 测试 RAG 搜索
4. 创建 LLMWiki draft（需要 LLM API）
5. 测试 Wiki 搜索

用法：
    cd D:/Agent/xiaozhi-esp32-server-main/main/xiaozhi-server
    python scripts/ingest_hypertension_guide.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import load_config
from config.logger import setup_logging
from core.clinical_nutrition.clinical_rag import ClinicalRAGService


PDF_NAME = "20260509T201000Z_中国高血压防治指南(2024年修订版).pdf"
PDF_PATH = PROJECT_ROOT / "data" / "knowledge_uploads" / "clinical-nutrition" / PDF_NAME


def step_register(service: ClinicalRAGService) -> str:
    """Step 1: 注册文档"""
    print("\n" + "=" * 60)
    print("Step 1: Register document in RAG")
    print("=" * 60)

    doc = service.register_document(
        PDF_PATH,
        original_name="中国高血压防治指南(2024年修订版).pdf",
        content_type="guideline",
    )
    document_id = doc["document_id"]
    print(f"  Document ID: {document_id}")
    print(f"  Title: {doc.get('title')}")
    print(f"  Status: {doc.get('status')}")
    return document_id


def step_index(service: ClinicalRAGService, document_id: str) -> dict:
    """Step 2: 索引文档（分块 + 向量化 + FTS5）"""
    print("\n" + "=" * 60)
    print("Step 2: Index document (chunk + embed + FTS5)")
    print("=" * 60)

    start = time.perf_counter()
    result = service.index_document(document_id)
    elapsed = time.perf_counter() - start

    print(f"  Status: {result.get('status')}")
    print(f"  Pages: {result.get('page_count')}")
    print(f"  Chunks: {result.get('chunk_count')}")
    print(f"  Embedded: {result.get('embedded_count')}")
    print(f"  Error: {result.get('error_message') or 'none'}")
    print(f"  Time: {elapsed:.1f}s")
    return result


def step_search_test(service: ClinicalRAGService) -> None:
    """Step 3: 测试 RAG 搜索"""
    print("\n" + "=" * 60)
    print("Step 3: RAG search verification")
    print("=" * 60)

    test_questions = [
        "高血压的诊断标准是什么？",
        "高血压患者应该限制多少盐？",
        "高血压合并糖尿病怎么降压？",
        "老年高血压有什么特点？",
        "高血压的药物治疗原则是什么？",
    ]

    for q in test_questions:
        print(f"\n  Q: {q}")
        results = service.search(q, top_k=3)
        if not results:
            print("  A: (no results)")
            continue
        for i, r in enumerate(results[:3], 1):
            source = r.get("source_name", "?")
            pages = f"pp.{r.get('page_start', '?')}-{r.get('page_end', '?')}"
            score = r.get("score", 0)
            text_preview = str(r.get("text", ""))[:80].replace("\n", " ")
            print(f"  [{i}] score={score:.3f} {source} {pages}")
            print(f"      {text_preview}...")


def step_wiki_draft(service_cls, config, logger) -> None:
    """Step 4: 创建 LLMWiki draft"""
    print("\n" + "=" * 60)
    print("Step 4: Create LLMWiki draft (requires LLM API)")
    print("=" * 60)

    try:
        from core.clinical_nutrition.knowledge_ingestion import KnowledgeIngestionService

        ingestion = KnowledgeIngestionService(
            project_root=PROJECT_ROOT,
            config=config,
            logger=logger,
        )

        start = time.perf_counter()
        draft = ingestion.create_draft(
            source_path=PDF_PATH,
            title="中国高血压防治指南（2024年修订版）",
            topic="高血压防治",
        )
        elapsed = time.perf_counter() - start

        draft_id = draft.get("draft_id", "?")
        wiki_pages = draft.get("wiki_pages", [])
        rules = draft.get("rules_draft", {}).get("rules", [])

        print(f"  Draft ID: {draft_id}")
        print(f"  Wiki pages: {len(wiki_pages)}")
        print(f"  Safety rules: {len(rules)}")
        print(f"  Time: {elapsed:.1f}s")

        if wiki_pages:
            print("\n  Generated wiki pages:")
            for wp in wiki_pages:
                title = wp.get("title", "?")
                slug = wp.get("slug", "?")
                chars = len(wp.get("content", ""))
                print(f"    - {title} ({slug}, {chars} chars)")

        # 保存 draft 到 wiki 目录
        wiki_dir = PROJECT_ROOT / "knowledge_base" / "llmwiki" / "clinical-nutrition" / "guidelines"
        wiki_dir.mkdir(parents=True, exist_ok=True)
        for wp in wiki_pages:
            slug = wp.get("slug", "unknown")
            content = wp.get("content", "")
            if content:
                out_path = wiki_dir / f"{slug}.md"
                out_path.write_text(content, encoding="utf-8")
                print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")

    except Exception as exc:
        print(f"  LLMWiki draft failed: {exc}")
        print("  (This is expected if LLM API is not available)")


def step_wiki_search_test() -> None:
    """Step 5: 测试 Wiki 搜索"""
    print("\n" + "=" * 60)
    print("Step 5: Wiki search verification")
    print("=" * 60)

    try:
        sys.path.insert(0, str(PROJECT_ROOT / "plugins_func" / "functions"))
        from search_from_llmwiki import _load_markdown_documents, _rank_documents

        wiki_root = PROJECT_ROOT / "knowledge_base" / "llmwiki" / "clinical-nutrition"
        docs = _load_markdown_documents(wiki_root, {"raw"})
        print(f"  Wiki pages loaded: {len(docs)}")

        test_questions = [
            "高血压的诊断标准是多少？",
            "高血压患者每天吃多少盐？",
            "降压药有哪些类型？",
        ]

        for q in test_questions:
            print(f"\n  Q: {q}")
            ranked = _rank_documents(q, docs, top_k=3, snippet_chars=300)
            if not ranked:
                print("  A: (no results)")
                continue
            for i, r in enumerate(ranked[:3], 1):
                print(f"  [{i}] score={r['score']:.2f} {r['title']}")
                print(f"      {r['snippet'][:100]}...")

    except Exception as exc:
        print(f"  Wiki search test failed: {exc}")


def main() -> None:
    print("=" * 60)
    print("Hypertension Guide - Full Ingestion Pipeline")
    print("=" * 60)

    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found: {PDF_PATH}")
        return

    config = load_config()
    logger = setup_logging()
    service = ClinicalRAGService(
        project_root=PROJECT_ROOT,
        config=config,
        logger=logger,
    )

    # Step 1: Register
    document_id = step_register(service)

    # Step 2: Index
    result = step_index(service, document_id)

    # Step 3: Search test (only if indexed successfully)
    if result.get("status") in ("indexed", "indexed_partial_vector"):
        step_search_test(service)
    else:
        print(f"\n  Skipping search test - document status: {result.get('status')}")

    # Step 4: Wiki draft
    step_wiki_draft(type(service), config, logger)

    # Step 5: Wiki search test
    step_wiki_search_test()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
