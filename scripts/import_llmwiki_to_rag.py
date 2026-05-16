from __future__ import annotations

import argparse
from pathlib import Path

from config.config_loader import get_project_dir, load_config
from config.logger import setup_logging
from core.clinical_nutrition.clinical_rag import ClinicalRAGService


def main() -> int:
    parser = argparse.ArgumentParser(description="Import existing LLMWiki markdown pages into Clinical RAG.")
    parser.add_argument(
        "--wiki-root",
        default="knowledge_base/llmwiki/clinical-nutrition",
        help="Markdown directory to seed into the RAG store.",
    )
    args = parser.parse_args()

    project_root = Path(get_project_dir()).resolve()
    wiki_root = Path(args.wiki_root)
    if not wiki_root.is_absolute():
        wiki_root = project_root / wiki_root

    service = ClinicalRAGService(
        project_root=project_root,
        config=load_config(),
        logger=setup_logging(),
    )
    result = service.seed_markdown_directory(wiki_root)
    print(result)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
