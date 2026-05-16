"""
Batch approve wiki drafts for the 4 clinical nutrition PDFs.

Approves existing wiki compiler v2 drafts for:
1. 成人肥胖指南
2. 糖尿病膳食指南
3. 高尿酸血症与痛风
Then generates and approves wiki content for:
4. 高血压防治指南

Usage:
    cd D:/Agent/xiaozhi-esp32-server-main/main/xiaozhi-server
    python scripts/approve_drafts.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Best draft for each PDF (latest draft with wiki content)
EXISTING_DRAFTS = {
    "成人肥胖指南": "20260507033626-610446e9b8-be0d81",
    "糖尿病膳食指南": "20260507041635-57e2e5478c-bd19f7",
    "高尿酸血症与痛风": "20260508114420-7179e6fd24-39b0f7",
}

HYPERTENSION_PDF = "data/knowledge_uploads/clinical-nutrition/20260509T201000Z_中国高血压防治指南(2024年修订版).pdf"


def main():
    from config.config_loader import load_config
    from config.logger import setup_logging
    from core.clinical_nutrition.knowledge_ingestion import KnowledgeIngestionService

    config = load_config()
    logger = setup_logging()
    service = KnowledgeIngestionService(
        project_root=PROJECT_ROOT,
        config=config,
        logger=logger,
    )

    wiki_root = service._target_wiki_root()
    print(f"Wiki root: {wiki_root}")
    print()

    results = []

    # Step 1: Approve 3 existing drafts
    for name, draft_id in EXISTING_DRAFTS.items():
        print(f"{'=' * 60}")
        print(f"Approving: {name}")
        print(f"  Draft ID: {draft_id}")

        draft = service.get_draft(draft_id)
        if not draft:
            print(f"  ERROR: Draft not found!")
            results.append((name, "FAIL", "draft not found"))
            continue

        wiki_pages = draft.get("wiki_pages") or []
        print(f"  Wiki pages: {len(wiki_pages)}")
        for p in wiki_pages:
            print(f"    - {p.get('slug')}: {p.get('title')}")

        try:
            start = time.perf_counter()
            approved = service.approve_draft(draft_id)
            elapsed = time.perf_counter() - start
            print(f"  Approved in {elapsed:.1f}s")
            print(f"  Wiki target: {approved.get('wiki_target_path', '?')}")
            results.append((name, "OK", f"{len(wiki_pages)} pages"))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, "FAIL", str(e)[:80]))

        print()

    # Step 2: Generate wiki for hypertension guide
    print(f"{'=' * 60}")
    print(f"Generating wiki for: 高血压防治指南")
    pdf_path = PROJECT_ROOT / HYPERTENSION_PDF
    if not pdf_path.exists():
        print(f"  ERROR: PDF not found at {pdf_path}")
        results.append(("高血压防治指南", "FAIL", "PDF not found"))
    else:
        print(f"  PDF: {pdf_path.name}")
        print(f"  Running create_draft() with LLM wiki compiler...")
        try:
            start = time.perf_counter()
            draft = service.create_draft(
                source_path=pdf_path,
                title="中国高血压防治指南（2024年修订版）",
                topic="高血压",
            )
            elapsed = time.perf_counter() - start
            draft_id = draft.get("draft_id", "?")
            wiki_pages = draft.get("wiki_pages") or []
            print(f"  Draft created in {elapsed:.1f}s: {draft_id}")
            print(f"  Wiki pages: {len(wiki_pages)}")
            for p in wiki_pages:
                print(f"    - {p.get('slug')}: {p.get('title')}")

            if wiki_pages:
                print(f"  Approving draft...")
                approved = service.approve_draft(draft_id)
                print(f"  Approved! Wiki target: {approved.get('wiki_target_path', '?')}")
                results.append(("高血压防治指南", "OK", f"{len(wiki_pages)} pages"))
            else:
                print(f"  WARNING: No wiki pages generated, skipping approval")
                results.append(("高血压防治指南", "WARN", "no wiki pages"))
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append(("高血压防治指南", "FAIL", str(e)[:80]))

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for name, status, detail in results:
        icon = "OK" if status == "OK" else ("WARN" if status == "WARN" else "FAIL")
        print(f"  [{icon}] {name}: {detail}")

    # Verify wiki directory
    print(f"\nWiki pages in {wiki_root}:")
    for md_file in sorted(wiki_root.rglob("*.md")):
        rel = md_file.relative_to(wiki_root)
        size = md_file.stat().st_size
        print(f"  {rel} ({size} bytes)")


if __name__ == "__main__":
    main()
