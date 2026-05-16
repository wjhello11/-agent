from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MAX_SOURCE_CHARS = 30000
DEFAULT_LLM_TIMEOUT_SECONDS = 240
DEFAULT_CHUNK_TARGET_CHARS = 6000
DEFAULT_CHUNK_OVERLAP_CHARS = 300
DEFAULT_MAX_CHUNK_SUMMARY_CHARS = 1800
DEFAULT_WIKI_COMPILER_CHUNK_CHARS = 4200
DEFAULT_WIKI_COMPILER_CHUNK_OVERLAP_CHARS = 350
WIKI_COMPILER_VERSION = "wiki_compiler_v2"
INGESTION_STATUS_UPLOADED = "uploaded"
INGESTION_STATUS_PROFILED = "profiled"
INGESTION_STATUS_EXTRACTED = "extracted"
INGESTION_STATUS_REVIEWED = "reviewed"
INGESTION_STATUS_APPROVED = "approved"
INGESTION_STATUS_PUBLISHED = "published"
INGESTION_STATUS_FAILED = "failed"
INGESTION_STATUS_NEEDS_OCR = "needs_ocr"


class KnowledgeIngestionService:
    """Builds human-reviewable LLMWiki pages and safety-rule drafts from uploads."""

    def __init__(self, *, project_root: Path, config: dict[str, Any], logger):
        self.project_root = Path(project_root)
        self.config = config
        self.logger = logger
        self.draft_root = self.project_root / "data" / "knowledge_ingestion" / "drafts"

    def list_drafts(self) -> list[dict[str, Any]]:
        if not self.draft_root.exists():
            return []
        drafts = []
        for path in self.draft_root.iterdir():
            if not path.is_dir():
                continue
            manifest = _read_json(path / "manifest.json", {})
            if manifest:
                drafts.append(manifest)
        return sorted(drafts, key=lambda item: item.get("updated_at", ""), reverse=True)

    def get_draft(self, draft_id: str) -> dict[str, Any] | None:
        draft_dir = self._draft_dir(draft_id)
        manifest = _read_json(draft_dir / "manifest.json", {})
        if not manifest:
            return None
        wiki_pages = _read_wiki_pages(draft_dir)
        return {
            **manifest,
            "wiki_markdown": _read_text(draft_dir / "wiki_index.md")
            or _read_text(draft_dir / "wiki_page.md"),
            "wiki_pages": wiki_pages,
            "rules_draft": _read_json(draft_dir / "rules_draft.json", {"rules": []}),
            "source_excerpt": _read_text(draft_dir / "source_excerpt.txt"),
            "chunk_summaries": _read_json(draft_dir / "chunk_summaries.json", []),
            "coverage_report": _read_json(draft_dir / "coverage_report.json", {}),
            "llm_review": _read_json(draft_dir / "llm_review.json", {}),
            "document_quality": _read_json(draft_dir / "document_quality.json", {}),
            "document_profile": _read_json(draft_dir / "document_profile.json", {}),
            "ingestion_plan": _read_json(draft_dir / "ingestion_plan.json", {}),
            "structured_extraction": _read_json(draft_dir / "structured_extraction.json", {}),
        }

    def create_draft(
        self,
        *,
        source_path: Path,
        title: str = "",
        topic: str = "",
    ) -> dict[str, Any]:
        source_path = source_path.resolve()
        try:
            source_path.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError("source file must be inside project root") from exc
        if not source_path.exists() or not source_path.is_file():
            raise ValueError("source file not found")

        pages = extract_document_pages_for_ingestion(source_path)
        document_quality = _build_document_quality_report(pages, source_path.name)
        source_text = "\n\n".join(str(page.get("text") or "") for page in pages).strip()
        source_title = _clean_source_stem_title(source_path.stem)
        display_title = _clean_ingestion_display_text(title) or _clean_ingestion_display_text(topic) or source_title
        display_topic = _clean_ingestion_display_text(topic)

        ingestion = self.config.get("knowledge_ingestion") or {}
        source_excerpt = _build_source_excerpt(
            source_text,
            _safe_int(
                ingestion.get("source_max_chars"),
                DEFAULT_MAX_SOURCE_CHARS,
                minimum=8000,
                maximum=60000,
            ),
        )
        draft_id = self._new_draft_id(source_path, source_excerpt)
        draft_dir = self._draft_dir(draft_id)
        draft_dir.mkdir(parents=True, exist_ok=True)

        if not source_text.strip() or document_quality.get("needs_ocr"):
            now = _utc_now()
            manifest = {
                "draft_id": draft_id,
                "status": INGESTION_STATUS_NEEDS_OCR,
                "source_path": _relative_to(self.project_root, source_path),
                "source_name": source_path.name,
                "title": display_title,
                "topic": display_topic,
                "wiki_target_path": "",
                "rules_draft_path": "knowledge_base/rules/clinical_safety_rule_drafts.json",
                "created_at": now,
                "updated_at": now,
                "llm_used": False,
                "llm_error": "source document has no readable text; OCR is required before automatic ingestion",
                "source_char_count": len(source_text),
                "source_excerpt_char_count": len(source_excerpt),
                "ingestion_mode": WIKI_COMPILER_VERSION,
                "page_count": len(pages),
                "review_status": "needs_ocr",
                "needs_review_count": 0,
            }
            _write_text(draft_dir / "source_excerpt.txt", source_excerpt)
            _write_json(draft_dir / "document_quality.json", document_quality)
            _write_json(draft_dir / "document_profile.json", {
                "document_type": "other",
                "knowledge_types": [],
                "source_document": source_path.name,
                "page_count": len(pages),
                "quality_status": document_quality.get("quality_status", "needs_ocr"),
                "suggested_status": INGESTION_STATUS_NEEDS_OCR,
                "confidence": 1.0,
                "summary": "文档没有可读文本，需先 OCR 后再入库。",
            })
            _write_json(draft_dir / "manifest.json", manifest)
            return self.get_draft(draft_id) or manifest

        # Run document profiler (LLM-based block classification + structured extraction)
        profiler_result = self._run_document_profiler(
            pages=pages,
            source_path=source_path,
            document_id=draft_id,
            quality_report=document_quality,
        )

        use_compiler_v2 = bool(ingestion.get("wiki_compiler_v2", True))
        if use_compiler_v2:
            llm_payload = self._generate_wiki_compiler_v2(
                pages=pages,
                source_path=source_path,
                source_text=source_text,
                source_excerpt=source_excerpt,
                title=display_title,
                topic=display_topic,
                ingestion_plan=profiler_result.get("ingestion_plan"),
            )
            wiki_markdown = llm_payload.get("wiki_index_markdown") or ""
            rules_draft = _normalize_rule_drafts(
                llm_payload.get("rules"),
                source_name=source_path.name,
            )
        else:
            llm_payload = self._generate_with_llm(
                source_text=source_text,
                source_name=source_path.name,
                title=display_title,
                topic=display_topic,
            )
            wiki_markdown = _normalize_wiki_markdown(
                payload=llm_payload.get("wiki") or {},
                fallback_title=display_title,
                source_name=source_path.name,
                source_text=source_excerpt,
            )
            rules_draft = _normalize_rule_drafts(
                llm_payload.get("rules"),
                source_name=source_path.name,
            )

        now = _utc_now()
        review_status_for_state = str(llm_payload.get("review_status") or "").lower()
        status = (
            INGESTION_STATUS_FAILED
            if review_status_for_state in {"failed", "review_failed"}
            else INGESTION_STATUS_REVIEWED
            if llm_payload.get("llm_review")
            else INGESTION_STATUS_EXTRACTED
        )
        manifest = {
            "draft_id": draft_id,
            "status": status,
            "source_path": _relative_to(self.project_root, source_path),
            "source_name": source_path.name,
            "title": _clean_ingestion_display_text(_frontmatter_title(wiki_markdown)) or display_title,
            "topic": display_topic,
            "wiki_target_path": "",
            "rules_draft_path": "knowledge_base/rules/clinical_safety_rule_drafts.json",
            "created_at": now,
            "updated_at": now,
            "llm_used": llm_payload.get("llm_used", False),
            "llm_error": llm_payload.get("llm_error", ""),
            "source_char_count": len(source_text),
            "source_excerpt_char_count": len(source_excerpt),
            "ingestion_mode": llm_payload.get("ingestion_mode", "single_pass"),
            "chunk_count": llm_payload.get("chunk_count", 0),
            "chunk_success_count": llm_payload.get("chunk_success_count", 0),
            "page_count": llm_payload.get("page_count", len(pages)),
            "covered_page_count": llm_payload.get("covered_page_count", 0),
            "uncovered_pages": llm_payload.get("uncovered_pages", []),
            "wiki_page_count": llm_payload.get("wiki_page_count", 0),
            "review_status": llm_payload.get("review_status", "draft"),
            "document_quality_status": document_quality.get("quality_status", "ok"),
            "document_profile_type": (profiler_result.get("document_profile") or {}).get("document_type", ""),
            "related_rag_document_id": llm_payload.get("related_rag_document_id", ""),
            "related_structured_document_id": llm_payload.get("related_structured_document_id", ""),
        }
        _write_text(draft_dir / "source_excerpt.txt", source_excerpt)
        _write_json(draft_dir / "document_quality.json", document_quality)
        if llm_payload.get("ingestion_mode") == WIKI_COMPILER_VERSION:
            pages_dir = draft_dir / "pages"
            pages_dir.mkdir(parents=True, exist_ok=True)
            for page in llm_payload.get("wiki_pages") or []:
                filename = f"{_slugify(page.get('slug') or page.get('title') or 'page')}.md"
                _write_text(pages_dir / filename, str(page.get("markdown") or ""))
            _write_text(draft_dir / "wiki_index.md", wiki_markdown)
            _write_json(draft_dir / "coverage_report.json", llm_payload.get("coverage_report") or {})
            _write_json(draft_dir / "llm_review.json", llm_payload.get("llm_review") or {})
        else:
            _write_text(draft_dir / "wiki_page.md", wiki_markdown)
        _write_json(draft_dir / "rules_draft.json", rules_draft)
        _write_json(draft_dir / "chunk_summaries.json", llm_payload.get("chunk_summaries") or [])
        # Save document profiler results
        _write_json(draft_dir / "document_profile.json", profiler_result.get("document_profile") or {})
        _write_json(draft_dir / "ingestion_plan.json", profiler_result.get("ingestion_plan") or {})
        _write_json(draft_dir / "structured_extraction.json", profiler_result.get("extraction_results") or {})
        manifest["profiler_used"] = profiler_result.get("profiler_used", False)
        manifest["profiler_error"] = profiler_result.get("profiler_error", "")
        manifest["structured_extraction_stats"] = (
            profiler_result.get("extraction_results", {}).get("stats") or {}
        )
        manifest["needs_review_count"] = len(
            profiler_result.get("extraction_results", {}).get("needs_review") or []
        )
        _write_json(draft_dir / "manifest.json", manifest)
        return self.get_draft(draft_id) or manifest

    def approve_draft(self, draft_id: str) -> dict[str, Any]:
        draft = self.get_draft(draft_id)
        if not draft:
            raise ValueError("draft not found")
        self._validate_publish_gate(draft)

        wiki_root = self._target_wiki_root()
        title = _clean_wiki_publish_title(draft.get("title") or draft.get("source_name") or "uploaded clinical note")
        if draft.get("ingestion_mode") == WIKI_COMPILER_VERSION and draft.get("wiki_pages"):
            version_id = datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")
            target_path = _unique_dir(wiki_root / "guidelines" / f"{_slugify(title)}-{version_id}")
            target_path.mkdir(parents=True, exist_ok=True)
            for page in draft.get("wiki_pages") or []:
                page_slug = _slugify(page.get("slug") or page.get("title") or "page")
                _write_text(target_path / f"{page_slug}.md", page.get("markdown") or "")
            _write_text(target_path / "index.md", draft.get("wiki_markdown") or "")
            self._update_global_wiki_index(
                title=str(title),
                target_index=target_path / "index.md",
                source_name=str(draft.get("source_name") or ""),
            )
        else:
            target_dir = wiki_root / "uploads"
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = _unique_path(target_dir / f"{_slugify(title)}.md")
            _write_text(target_path, draft.get("wiki_markdown") or "")

        rules_target = self.project_root / "knowledge_base" / "rules" / "clinical_safety_rule_drafts.json"
        existing = _read_json(rules_target, {"version": "draft", "rules": []})
        rules = existing.get("rules") if isinstance(existing.get("rules"), list) else []
        for rule in (draft.get("rules_draft") or {}).get("rules", []):
            if not isinstance(rule, dict):
                continue
            rule["source_draft_id"] = draft_id
            rule["review_status"] = "draft_needs_review"
            rules.append(rule)
        existing["rules"] = rules
        existing["updated_at"] = _utc_now()
        rules_target.parent.mkdir(parents=True, exist_ok=True)
        _write_json(rules_target, existing)

        # Ingest structured extraction results into StructuredKnowledgeStore
        structured_extraction = draft.get("structured_extraction") or {}
        if structured_extraction.get("extracted") or structured_extraction.get("needs_review"):
            try:
                from core.clinical_nutrition.structured_knowledge import StructuredKnowledgeStore

                store = StructuredKnowledgeStore(
                    project_root=self.project_root,
                    config=self.config,
                )
                source_path_str = draft.get("source_path") or ""
                source_path_resolved = self.project_root / source_path_str if source_path_str else None
                pages = []
                if source_path_resolved and source_path_resolved.exists():
                    pages = extract_document_pages_for_ingestion(source_path_resolved)

                document_meta = {
                    "document_id": draft_id,
                    "original_name": draft.get("source_name") or "",
                    "stored_path": source_path_str,
                    "title": title,
                    "source_hash": "",
                }
                structured_result = store.ingest_from_plan(
                    document=document_meta,
                    pages=pages,
                    extraction_results=structured_extraction,
                )
                manifest_path_tmp = self._draft_dir(draft_id) / "manifest.json"
                manifest_tmp = _read_json(manifest_path_tmp, {})
                manifest_tmp["structured_ingestion"] = structured_result
                manifest_tmp.pop("structured_ingestion_error", None)
                _write_json(manifest_path_tmp, manifest_tmp)
            except Exception as exc:
                self.logger.bind(tag="knowledge_ingestion").error(
                f"Structured ingestion from plan failed: {exc}"
                )
                manifest_path_tmp = self._draft_dir(draft_id) / "manifest.json"
                manifest_tmp = _read_json(manifest_path_tmp, {})
                manifest_tmp["structured_ingestion_error"] = str(exc)
                _write_json(manifest_path_tmp, manifest_tmp)
                raise ValueError(f"structured ingestion failed: {exc}") from exc

        manifest_path = self._draft_dir(draft_id) / "manifest.json"
        manifest = _read_json(manifest_path, {})
        manifest.update(
            {
                "status": INGESTION_STATUS_PUBLISHED,
                "wiki_target_path": _relative_to(self.project_root, target_path),
                "rules_draft_path": _relative_to(self.project_root, rules_target),
                "review_status": "published",
                "published_at": _utc_now(),
                "published_by": "console_admin",
                "updated_at": _utc_now(),
            }
        )
        _write_json(manifest_path, manifest)
        return self.get_draft(draft_id) or manifest

    def _validate_publish_gate(self, draft: dict[str, Any]) -> None:
        if draft.get("status") == INGESTION_STATUS_NEEDS_OCR:
            raise ValueError("文档需要 OCR，不能发布。")
        coverage = draft.get("coverage_report") or {}
        uncovered = coverage.get("uncovered_pages") or draft.get("uncovered_pages") or []
        if uncovered:
            raise ValueError(f"存在未覆盖页，不能发布：{_format_page_ranges(uncovered)}")
        review = draft.get("llm_review") or {}
        review_status = str(review.get("overall_status") or draft.get("review_status") or "").lower()
        if review_status in {"failed", "review_failed"}:
            raise ValueError("LLM 复核失败，不能发布。")
        if not coverage.get("total_pages") and draft.get("ingestion_mode") == WIKI_COMPILER_VERSION:
            raise ValueError("缺少覆盖率报告，不能发布。")

    def _update_global_wiki_index(self, *, title: str, target_index: Path, source_name: str) -> None:
        wiki_root = self._target_wiki_root()
        index_path = wiki_root / "_index.md"
        relative = _relative_to(wiki_root, target_index)
        existing = _read_text(index_path)
        existing = _remove_prior_wiki_index_entries(
            existing,
            title=title,
            source_name=source_name,
        )
        line = f"- [{title}]({relative})：来源 {source_name}"
        if line in existing:
            return
        if not existing.strip():
            existing = "# Clinical Nutrition Wiki\n"
        text = existing.rstrip() + "\n" + line + "\n"
        _write_text(index_path, text)

    def _target_wiki_root(self) -> Path:
        ingestion = self.config.get("knowledge_ingestion") or {}
        configured = ingestion.get("target_wiki_root") or "knowledge_base/llmwiki/clinical-nutrition"
        path = Path(configured)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def _new_draft_id(self, source_path: Path, text: str) -> str:
        digest = hashlib.sha1(f"{source_path}:{text[:2000]}".encode("utf-8")).hexdigest()[:10]
        return f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{digest}-{uuid.uuid4().hex[:6]}"

    def _draft_dir(self, draft_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(draft_id or "")).strip(".-")
        if not safe:
            raise ValueError("invalid draft_id")
        return self.draft_root / safe

    def _run_document_profiler(
        self,
        *,
        pages: list[dict[str, Any]],
        source_path: Path,
        document_id: str = "",
        quality_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run DocumentProfiler to generate ingestion plan and extract structured data.

        Returns:
            {
                "ingestion_plan": IngestionPlan dict,
                "document_profile": DocumentProfile dict,
                "extraction_results": {extracted, needs_review, stats},
                "profiler_used": bool,
                "profiler_error": str,
            }
        """
        ingestion = self.config.get("knowledge_ingestion") or {}
        profile_llm_options = self._knowledge_ingestion_llm_options(ingestion, stage="profile")

        if not profile_llm_options:
            return {
                "document_profile": {},
                "ingestion_plan": {},
                "extraction_results": {},
                "profiler_used": False,
                "profiler_error": "no LLM configured for profile stage",
            }

        from core.clinical_nutrition.document_profiler import DocumentProfiler

        def llm_caller(prompt: str, max_tokens: int) -> dict[str, Any]:
            return self._call_llm_chat(
                base_url=profile_llm_options["base_url"],
                api_key=profile_llm_options["api_key"],
                models=profile_llm_options.get("models") or [profile_llm_options["model"]],
                prompt=prompt,
                timeout_seconds=profile_llm_options["timeout_seconds"],
                max_tokens=max_tokens,
                response_format_json=True,
            )

        source_name = source_path.name
        profiler = DocumentProfiler(
            llm_caller=llm_caller,
            source_name=source_name,
            document_id=document_id,
        )

        try:
            profile = profiler.profile_document(
                pages,
                source_name=source_name,
                quality_report=quality_report or {},
            )
            plan = profiler.generate_ingestion_plan(pages, source_name=source_name)
            if profile.document_type != "other":
                plan.document_type = profile.document_type
            if profile.knowledge_types:
                plan.knowledge_types = sorted(set(plan.knowledge_types) | set(profile.knowledge_types))
            extraction_results = profiler.extract_structured_blocks(plan, pages)
            return {
                "document_profile": profile.model_dump(),
                "ingestion_plan": plan.model_dump(),
                "extraction_results": {
                    "extracted": extraction_results["extracted"],
                    "needs_review": [
                        item.model_dump() if hasattr(item, "model_dump") else item
                        for item in extraction_results["needs_review"]
                    ],
                    "stats": extraction_results["stats"],
                },
                "profiler_used": True,
                "profiler_error": "",
            }
        except Exception as exc:
            self.logger.bind(tag="knowledge_ingestion").error(
                f"Document profiler failed: {exc}"
            )
            return {
                "document_profile": {},
                "ingestion_plan": {},
                "extraction_results": {},
                "profiler_used": False,
                "profiler_error": str(exc),
            }

    def _generate_wiki_compiler_v2(
        self,
        *,
        pages: list[dict[str, Any]],
        source_path: Path,
        source_text: str,
        source_excerpt: str,
        title: str,
        topic: str,
        ingestion_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source_name = source_path.name
        fallback_title = _clean_ingestion_display_text(title) or _clean_ingestion_display_text(topic) or source_path.stem
        topic = _clean_ingestion_display_text(topic)
        ingestion = self.config.get("knowledge_ingestion") or {}
        chunk_llm_options = self._knowledge_ingestion_llm_options(ingestion, stage="chunk")
        review_llm_options = self._knowledge_ingestion_llm_options(ingestion, stage="review")
        target_chars = _safe_int(
            ingestion.get("wiki_compiler_chunk_chars"),
            DEFAULT_WIKI_COMPILER_CHUNK_CHARS,
            minimum=1800,
            maximum=9000,
        )
        overlap_chars = _safe_int(
            ingestion.get("wiki_compiler_chunk_overlap_chars"),
            DEFAULT_WIKI_COMPILER_CHUNK_OVERLAP_CHARS,
            minimum=0,
            maximum=1000,
        )
        max_llm_chunks = _safe_int(
            ingestion.get("wiki_compiler_max_llm_chunks"),
            8,
            minimum=0,
            maximum=200,
        )
        max_llm_failures = _safe_int(
            ingestion.get("wiki_compiler_max_llm_failures"),
            1,
            minimum=0,
            maximum=20,
        )
        chunks = _build_wiki_compiler_chunks(
            pages,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
        )
        for chunk in chunks:
            chunk["chunk_count"] = len(chunks)
        related = self._find_related_document_ids(source_path)
        chunk_summaries: list[dict[str, Any]] = []
        chunk_errors: list[str] = []
        llm_used = False
        llm_failures = 0
        skip_remaining_llm = False

        for chunk in chunks:
            parsed: dict[str, Any] = {}
            structured_type = _structured_candidate_type(chunk, str(chunk.get("text") or ""))
            skip_llm_for_structured = structured_type in {"recipe_plan", "therapeutic_recipe"}
            if (
                chunk_llm_options
                and not skip_llm_for_structured
                and not skip_remaining_llm
                and int(chunk.get("chunk_index") or 0) <= max_llm_chunks
            ):
                try:
                    parsed = self._call_llm_json(
                        base_url=chunk_llm_options["base_url"],
                        api_key=chunk_llm_options["api_key"],
                        model=chunk_llm_options["model"],
                        models=chunk_llm_options.get("models"),
                        prompt=_wiki_compiler_chunk_prompt(
                            chunk=chunk,
                            chunk_count=len(chunks),
                            source_name=source_name,
                            title=fallback_title,
                            topic=topic,
                        ),
                        timeout_seconds=chunk_llm_options["timeout_seconds"],
                        max_tokens=chunk_llm_options["max_tokens"],
                    )
                    llm_used = llm_used or bool(parsed)
                except Exception as exc:
                    chunk_errors.append(f"chunk {chunk['chunk_index']}: {exc}")
                    llm_failures += 1
                    if max_llm_failures and llm_failures >= max_llm_failures:
                        skip_remaining_llm = True
                        chunk_errors.append("remaining chunks: skipped LLM after fail-fast threshold")
            elif skip_llm_for_structured:
                parsed = {}
            elif chunk_llm_options and max_llm_chunks < len(chunks):
                chunk_errors.append(
                    f"chunk {chunk['chunk_index']}: skipped LLM in synchronous compiler"
                )
            if not parsed:
                parsed = _fallback_chunk_summary(chunk, source_name)
            parsed = _normalize_wiki_chunk_summary(parsed, chunk)
            chunk_summaries.append(parsed)

        wiki_pages = _build_wiki_pages_from_summaries(
            chunk_summaries,
            title=fallback_title,
            source_name=source_name,
            related_rag_document_id=related.get("rag_document_id", ""),
            related_structured_document_id=related.get("structured_document_id", ""),
            review_status="draft",
        )
        if not wiki_pages:
            fallback = _fallback_generation(
                source_excerpt or source_text,
                source_name,
                fallback_title,
                topic,
                "wiki compiler produced no pages",
            )
            wiki_pages = [
                {
                    "slug": "overview",
                    "title": fallback_title,
                    "source_pages": _all_source_page_numbers(pages),
                    "markdown": _normalize_wiki_markdown(
                        payload=fallback.get("wiki") or {},
                        fallback_title=fallback_title,
                        source_name=source_name,
                        source_text=source_excerpt,
                    ),
                }
            ]

        coverage_report = _build_coverage_report(
            project_root=self.project_root,
            config=self.config,
            source_path=source_path,
            source_pages=pages,
            chunks=chunks,
            wiki_pages=wiki_pages,
            related=related,
            ingestion_plan=ingestion_plan,
        )
        wiki_index = _build_wiki_index_markdown(
            title=fallback_title,
            source_name=source_name,
            wiki_pages=wiki_pages,
            coverage_report=coverage_report,
            related=related,
        )
        llm_review = self._review_wiki_compiler_output(
            llm_options=review_llm_options if (llm_used or not skip_remaining_llm) else {},
            source_name=source_name,
            title=fallback_title,
            wiki_pages=wiki_pages,
            coverage_report=coverage_report,
            chunk_summaries=chunk_summaries,
        )
        if chunk_llm_options and not llm_used and chunk_errors:
            issues = llm_review.setdefault("issues", [])
            issues.append("LLM chunk 抽取未完成，当前草案使用结构化回退摘要，必须人工复核。")
            llm_review["overall_status"] = "needs_human_review"
            llm_review["confidence"] = min(float(llm_review.get("confidence") or 0.7), 0.68)
        review_status = str(llm_review.get("overall_status") or "needs_human_review")

        rules: list[dict[str, Any]] = []
        for summary in chunk_summaries:
            for rule in summary.get("rules") or []:
                if isinstance(rule, dict):
                    rules.append(rule)

        return {
            "llm_used": llm_used,
            "llm_error": "; ".join(chunk_errors),
            "ingestion_mode": WIKI_COMPILER_VERSION,
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "chunk_success_count": len(chunk_summaries),
            "chunk_summaries": chunk_summaries,
            "wiki_pages": wiki_pages,
            "wiki_index_markdown": wiki_index,
            "coverage_report": coverage_report,
            "covered_page_count": len(coverage_report.get("covered_pages") or []),
            "uncovered_pages": coverage_report.get("uncovered_pages") or [],
            "wiki_page_count": len(wiki_pages),
            "llm_review": llm_review,
            "review_status": review_status,
            "rules": rules,
            "related_rag_document_id": related.get("rag_document_id", ""),
            "related_structured_document_id": related.get("structured_document_id", ""),
        }

    def _knowledge_ingestion_llm_options(self, ingestion: dict[str, Any], *, stage: str = "") -> dict[str, Any]:
        if not ingestion.get("enabled", True):
            return {}
        llm_config = ingestion.get("llm") or {}
        api_key = str(llm_config.get("api_key") or "").strip()
        stage_model_key = {
            "chunk": "wiki_compiler_chunk_model",
            "review": "wiki_compiler_review_model",
            "synthesis": "wiki_compiler_synthesis_model",
            "profile": "document_profiler_model",
            "extract": "document_profiler_model",
        }.get(stage, "")
        # Profile/extract stages default to chunk model (faster) if no dedicated model configured
        fallback_model_key = ""
        if stage in ("profile", "extract") and not ingestion.get(stage_model_key):
            fallback_model_key = "wiki_compiler_chunk_model"
        model = str(
            (ingestion.get(stage_model_key) if stage_model_key else "")
            or llm_config.get(stage_model_key)
            or (ingestion.get(fallback_model_key) if fallback_model_key else "")
            or (llm_config.get(fallback_model_key) if fallback_model_key else "")
            or llm_config.get("model")
            or ""
        ).strip()
        model_candidates = _model_candidates_for_stage(
            ingestion=ingestion,
            llm_config=llm_config,
            stage=stage,
            primary_model=model,
        )
        if model_candidates:
            model = model_candidates[0]
        if not api_key or not model:
            return {}
        if stage in ("profile", "extract"):
            timeout_seconds = _safe_int(
                ingestion.get("document_profiler_timeout_seconds")
                or ingestion.get("wiki_compiler_timeout_seconds"),
                180,
                minimum=60,
                maximum=300,
            )
        else:
            timeout_seconds = _safe_int(
                ingestion.get("wiki_compiler_timeout_seconds"),
                45,
                minimum=30,
                maximum=240,
            )
        max_tokens = _safe_int(
            ingestion.get("wiki_compiler_max_tokens") or llm_config.get("max_tokens") or ingestion.get("max_tokens"),
            3072,
            minimum=1024,
            maximum=8000,
        )
        base_url = str(
            llm_config.get("openai_base_url")
            or llm_config.get("base_url")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).strip()
        return {
            "api_key": api_key,
            "model": model,
            "models": model_candidates or [model],
            "base_url": base_url,
            "timeout_seconds": timeout_seconds,
            "max_tokens": max_tokens,
            "stage": stage,
        }

    def _review_wiki_compiler_output(
        self,
        *,
        llm_options: dict[str, Any],
        source_name: str,
        title: str,
        wiki_pages: list[dict[str, Any]],
        coverage_report: dict[str, Any],
        chunk_summaries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        deterministic = _deterministic_wiki_review(
            wiki_pages=wiki_pages,
            coverage_report=coverage_report,
            chunk_summaries=chunk_summaries,
        )
        if not llm_options:
            return deterministic
        try:
            parsed = self._call_llm_json(
                base_url=llm_options["base_url"],
                api_key=llm_options["api_key"],
                model=llm_options["model"],
                models=llm_options.get("models"),
                prompt=_wiki_compiler_review_prompt(
                    source_name=source_name,
                    title=title,
                    wiki_pages=wiki_pages,
                    coverage_report=coverage_report,
                    chunk_summaries=chunk_summaries,
                ),
                timeout_seconds=llm_options["timeout_seconds"],
                max_tokens=min(llm_options["max_tokens"], 2048),
            )
            if isinstance(parsed, dict) and parsed:
                parsed.setdefault("review_method", f"llm:{llm_options['model']}")
                parsed.setdefault("reviewed_at", _utc_now())
                parsed.setdefault("overall_status", deterministic["overall_status"])
                parsed.setdefault("issues", deterministic["issues"])
                return parsed
        except Exception as exc:
            deterministic["issues"].append(f"LLM 复核调用失败：{exc}")
            deterministic["overall_status"] = "needs_human_review"
            deterministic["confidence"] = min(float(deterministic.get("confidence") or 0.7), 0.68)
        return deterministic

    def _find_related_document_ids(self, source_path: Path) -> dict[str, str]:
        return {
            "rag_document_id": _find_related_rag_document_id(
                self.project_root,
                self.config,
                source_path,
            ),
            "structured_document_id": _find_related_structured_document_id(
                self.project_root,
                self.config,
                source_path,
            ),
        }

    def _generate_with_llm(
        self,
        *,
        source_text: str,
        source_name: str,
        title: str,
        topic: str,
    ) -> dict[str, Any]:
        ingestion = self.config.get("knowledge_ingestion") or {}
        if not ingestion.get("enabled", True):
            return _fallback_generation(source_text, source_name, title, topic, "knowledge ingestion disabled")
        llm_config = ingestion.get("llm") or {}
        api_key = str(llm_config.get("api_key") or "").strip()
        model = str(llm_config.get("model") or "").strip()
        timeout_seconds = _safe_int(
            llm_config.get("timeout_seconds") or ingestion.get("timeout_seconds"),
            DEFAULT_LLM_TIMEOUT_SECONDS,
            minimum=30,
            maximum=600,
        )
        max_tokens = _safe_int(
            llm_config.get("max_tokens") or ingestion.get("max_tokens"),
            4096,
            minimum=1024,
            maximum=12000,
        )
        base_url = str(
            llm_config.get("openai_base_url")
            or llm_config.get("base_url")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).strip()
        if not api_key or not model:
            return _fallback_generation(source_text, source_name, title, topic, "missing ingestion llm config")
        model_candidates = _model_candidates_for_stage(
            ingestion=ingestion,
            llm_config=llm_config,
            stage="synthesis",
            primary_model=model,
        )

        chunk_target_chars = _safe_int(
            ingestion.get("chunk_target_chars"),
            DEFAULT_CHUNK_TARGET_CHARS,
            minimum=2500,
            maximum=12000,
        )
        chunk_overlap_chars = _safe_int(
            ingestion.get("chunk_overlap_chars"),
            DEFAULT_CHUNK_OVERLAP_CHARS,
            minimum=0,
            maximum=1200,
        )
        chunk_summary_chars = _safe_int(
            ingestion.get("chunk_summary_chars"),
            DEFAULT_MAX_CHUNK_SUMMARY_CHARS,
            minimum=800,
            maximum=4000,
        )
        chunks = _split_document_for_ingestion(
            source_text,
            target_chars=chunk_target_chars,
            overlap_chars=chunk_overlap_chars,
        )
        try:
            chunk_summaries: list[dict[str, Any]] = []
            chunk_errors: list[str] = []
            for chunk_index, chunk_text in enumerate(chunks, start=1):
                parsed = self._call_llm_json(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    models=model_candidates,
                    prompt=_chunk_ingestion_prompt(
                        chunk_text=chunk_text,
                        chunk_index=chunk_index,
                        chunk_count=len(chunks),
                        source_name=source_name,
                        title=title,
                        topic=topic,
                    ),
                    timeout_seconds=timeout_seconds,
                    max_tokens=max_tokens,
                )
                if parsed:
                    parsed["chunk_index"] = chunk_index
                    parsed["chunk_count"] = len(chunks)
                    chunk_summaries.append(parsed)
                else:
                    chunk_errors.append(f"chunk {chunk_index}: llm returned non-json")

            if not chunk_summaries:
                reason = "; ".join(chunk_errors) or "all chunk extraction calls failed"
                return _fallback_generation(source_text, source_name, title, topic, reason)

            final_payload = self._call_llm_json(
                base_url=base_url,
                api_key=api_key,
                model=model,
                models=model_candidates,
                prompt=_final_ingestion_prompt(
                    chunk_summaries=chunk_summaries,
                    source_name=source_name,
                    title=title,
                    topic=topic,
                    max_chunk_summary_chars=chunk_summary_chars,
                ),
                timeout_seconds=timeout_seconds,
                max_tokens=max_tokens,
            )
            if final_payload:
                final_payload["llm_used"] = True
                final_payload["llm_error"] = "; ".join(chunk_errors)
                final_payload["ingestion_mode"] = "chunked_long_document"
                final_payload["chunk_count"] = len(chunks)
                final_payload["chunk_success_count"] = len(chunk_summaries)
                final_payload["chunk_summaries"] = chunk_summaries
                return final_payload

            fallback = _fallback_generation(
                _chunk_summaries_to_text(chunk_summaries, chunk_summary_chars),
                source_name,
                title,
                topic,
                "final synthesis returned non-json",
            )
            fallback["llm_used"] = False
            fallback["llm_error"] = "; ".join(
                [item for item in chunk_errors + ["final synthesis returned non-json"] if item]
            )
            fallback["ingestion_mode"] = "chunked_long_document"
            fallback["chunk_count"] = len(chunks)
            fallback["chunk_success_count"] = len(chunk_summaries)
            fallback["chunk_summaries"] = chunk_summaries
            return fallback
        except Exception as exc:
            fallback = _fallback_generation(source_text, source_name, title, topic, str(exc))
            fallback["ingestion_mode"] = "chunked_long_document"
            fallback["chunk_count"] = len(chunks)
            fallback["chunk_success_count"] = 0
            fallback["chunk_summaries"] = []
            return fallback

    def _call_llm_json(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        models: list[str] | None = None,
        prompt: str,
        timeout_seconds: int,
        max_tokens: int,
    ) -> dict[str, Any]:
        result = self._call_llm_chat(
            base_url=base_url,
            api_key=api_key,
            models=models or [model],
            prompt=prompt,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            response_format_json=True,
        )
        content = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        parsed = _extract_json_object(content)
        if parsed:
            parsed.setdefault("_llm_model_used", result.get("_model_used") or model)
        return parsed

    def _call_llm_chat(
        self,
        *,
        base_url: str,
        api_key: str,
        models: list[str],
        prompt: str,
        timeout_seconds: int,
        max_tokens: int,
        response_format_json: bool = False,
    ) -> dict[str, Any]:
        errors: list[str] = []
        candidates = [str(model).strip() for model in models if str(model).strip()]
        if not candidates:
            raise ValueError("no model candidates configured")
        for model in candidates:
            for use_json_mode in ([True, False] if response_format_json else [False]):
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是临床营养知识库入库助手，只输出 JSON。",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": max_tokens,
                }
                if use_json_mode:
                    payload["response_format"] = {"type": "json_object"}
                request = urllib.request.Request(
                    f"{base_url.rstrip('/')}/chat/completions",
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                        result = json.loads(response.read().decode("utf-8"))
                    if isinstance(result, dict):
                        result["_model_used"] = model
                        result["_json_mode_used"] = use_json_mode
                        return result
                except Exception as exc:
                    message = _llm_error_message(exc)
                    errors.append(f"{model}{' json' if use_json_mode else ''}: {message}")
                    if use_json_mode and _looks_like_json_mode_error(message):
                        continue
                    if _looks_like_model_quota_or_missing_error(message):
                        break
                    if use_json_mode:
                        continue
                    break
        raise RuntimeError("all model candidates failed: " + " | ".join(errors[-8:]))


def _model_candidates_for_stage(
    *,
    ingestion: dict[str, Any],
    llm_config: dict[str, Any],
    stage: str,
    primary_model: str,
) -> list[str]:
    configured = (
        ingestion.get(f"{stage}_model_candidates")
        or llm_config.get(f"{stage}_model_candidates")
        or ingestion.get("model_candidates")
        or llm_config.get("model_candidates")
    )
    candidates = _parse_model_candidates(configured)
    defaults = {
        "profile": ["deepseek-v4-flash", "qwen3.6-plus", "qwen-plus"],
        "extract": ["deepseek-v4-flash", "qwen3.6-plus", "qwen-plus"],
        "chunk": ["deepseek-v4-flash", "qwen3.6-plus", "qwen-plus"],
        "synthesis": ["qwen3.6-plus", "qwen-plus", "deepseek-v4-flash"],
        "review": ["qwen3.6-plus", "qwen-plus", "deepseek-v4-flash"],
    }.get(stage, [])
    ordered: list[str] = []
    for model in [primary_model, *candidates, *defaults]:
        clean = str(model or "").strip()
        if clean and clean not in ordered:
            ordered.append(clean)
    return ordered


def _parse_model_candidates(value: Any) -> list[str]:
    if isinstance(value, list):
        raw = value
    else:
        raw = re.split(r"[,，\n]+", str(value or ""))
    return [str(item).strip() for item in raw if str(item).strip()]


def _llm_error_message(exc: Exception) -> str:
    status = getattr(exc, "code", "")
    try:
        body = exc.read().decode("utf-8", errors="ignore")  # type: ignore[attr-defined]
    except Exception:
        body = ""
    message = str(exc)
    if status:
        message = f"HTTP {status}: {message}"
    if body:
        message = f"{message}; {body[:800]}"
    return message


def _looks_like_json_mode_error(message: str) -> bool:
    lowered = str(message or "").lower()
    return any(token in lowered for token in ("response_format", "json_object", "unsupported", "invalid parameter"))


def _looks_like_model_quota_or_missing_error(message: str) -> bool:
    lowered = str(message or "").lower()
    return any(token in lowered for token in ("401", "403", "404", "429", "quota", "not exist", "model", "no permission", "insufficient"))


def extract_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import pypdf
        except Exception as exc:
            raise ValueError("pypdf is required to ingest PDF files") from exc
        reader = pypdf.PdfReader(str(path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages).strip()
    if suffix in {".md", ".markdown", ".txt", ".csv", ".tsv", ".json"}:
        for encoding in ("utf-8", "utf-8-sig", "gb18030"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(errors="ignore")
    if suffix == ".docx":
        try:
            import docx
        except Exception as exc:
            raise ValueError("python-docx is required to ingest docx files") from exc
        document = docx.Document(str(path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()
    raise ValueError(f"unsupported ingestion file type: {suffix}")


def extract_document_pages_for_ingestion(path: Path) -> list[dict[str, Any]]:
    try:
        from core.clinical_nutrition.clinical_rag import extract_document_pages

        pages = extract_document_pages(path)
        return [
            {
                "page_number": int(getattr(page, "page_number", index) or index),
                "text": str(getattr(page, "text", "") or ""),
                "extraction_method": str(getattr(page, "extraction_method", "") or ""),
            }
            for index, page in enumerate(pages, start=1)
        ]
    except ValueError:
        raise
    except Exception:
        text = extract_document_text(path)
        return [{"page_number": 1, "text": text, "extraction_method": "text"}]


def _build_document_quality_report(pages: list[dict[str, Any]], source_name: str) -> dict[str, Any]:
    page_count = len(pages)
    char_counts = [len(str(page.get("text") or "").strip()) for page in pages]
    readable = sum(1 for count in char_counts if count >= 80)
    empty = sum(1 for count in char_counts if count == 0)
    low_text = sum(1 for count in char_counts if 0 < count < 80)
    total_chars = sum(char_counts)
    average = round(total_chars / page_count, 1) if page_count else 0.0
    issues: list[str] = []
    if page_count == 0:
        issues.append("文档没有可解析页。")
    if page_count and readable == 0:
        issues.append("所有页面都缺少足够可读文本，可能是扫描版 PDF。")
    elif page_count and readable / max(page_count, 1) < 0.35:
        issues.append("可读文本页面比例偏低，建议人工检查或 OCR。")
    if page_count and average < 60:
        issues.append("平均每页字符数过低。")
    likely_scanned = bool(page_count and readable == 0 and (empty + low_text) >= int(page_count * 0.8))
    needs_ocr = likely_scanned or (page_count > 0 and total_chars == 0)
    quality_status = "needs_ocr" if needs_ocr else ("needs_manual_check" if issues else "ok")
    return {
        "source_document": source_name,
        "page_count": page_count,
        "readable_page_count": readable,
        "empty_page_count": empty,
        "low_text_page_count": low_text,
        "average_chars_per_page": average,
        "likely_scanned": likely_scanned,
        "needs_ocr": needs_ocr,
        "quality_status": quality_status,
        "issues": issues,
    }


def _build_wiki_compiler_chunks(
    pages: list[dict[str, Any]],
    *,
    target_chars: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    section_stack: list[str] = []
    for page in pages:
        page_number = int(page.get("page_number") or 1)
        for paragraph_index, paragraph in enumerate(_split_ingestion_paragraphs(page.get("text") or ""), start=1):
            heading = _extract_ingestion_heading(paragraph)
            if heading:
                _update_ingestion_section_stack(section_stack, heading)
                block_type = "heading"
            else:
                block_type = _classify_source_block(paragraph, page_number)
            section_path = " > ".join(section_stack)
            units.append(
                {
                    "block_id": f"p{page_number}:b{paragraph_index}",
                    "page_number": page_number,
                    "text": paragraph,
                    "block_type": block_type,
                    "section_path": section_path,
                }
            )

    chunks: list[dict[str, Any]] = []
    buffer: list[dict[str, Any]] = []

    def emit() -> None:
        nonlocal buffer
        if not buffer:
            return
        text_parts = [str(item["text"]).strip() for item in buffer if str(item.get("text") or "").strip()]
        if not text_parts:
            buffer = []
            return
        pages_in_chunk = [int(item["page_number"]) for item in buffer]
        block_types = sorted({str(item.get("block_type") or "paragraph") for item in buffer})
        section_path = _last_non_empty([str(item.get("section_path") or "") for item in buffer])
        chunks.append(
            {
                "chunk_index": len(chunks) + 1,
                "section_path": section_path,
                "page_start": min(pages_in_chunk),
                "page_end": max(pages_in_chunk),
                "source_pages": sorted(set(pages_in_chunk)),
                "source_block_ids": [str(item["block_id"]) for item in buffer],
                "chunk_type": "mixed" if len(block_types) > 1 else block_types[0],
                "block_types": block_types,
                "text": "\n\n".join(text_parts).strip(),
            }
        )
        buffer = []

    for unit in units:
        current_len = sum(len(str(item.get("text") or "")) for item in buffer)
        unit_len = len(str(unit.get("text") or ""))
        starts_new_section = bool(unit.get("block_type") == "heading" and current_len >= int(target_chars * 0.6))
        if buffer and (current_len + unit_len > target_chars or starts_new_section):
            previous_text = "\n\n".join(str(item.get("text") or "") for item in buffer).strip()
            last_pages = [int(item["page_number"]) for item in buffer[-2:]]
            emit()
            overlap = _tail_text(previous_text, overlap_chars)
            if overlap:
                buffer = [
                    {
                        "block_id": f"overlap:{len(chunks)}",
                        "page_number": last_pages[-1] if last_pages else int(unit.get("page_number") or 1),
                        "text": overlap,
                        "block_type": "note",
                        "section_path": str(unit.get("section_path") or ""),
                    }
                ]
        buffer.append(unit)
    emit()
    return chunks or [
        {
            "chunk_index": 1,
            "section_path": "",
            "page_start": 1,
            "page_end": 1,
            "source_pages": [1],
            "source_block_ids": ["p1:b1"],
            "chunk_type": "paragraph",
            "block_types": ["paragraph"],
            "text": "",
        }
    ]


def _split_ingestion_paragraphs(text: str) -> list[str]:
    text = re.sub(r"\r\n?", "\n", str(text or ""))
    lines = [line.strip() for line in text.splitlines()]
    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in lines:
        if not line:
            if buffer:
                paragraphs.append("\n".join(buffer).strip())
                buffer = []
            continue
        if _looks_like_heading(line) and buffer:
            paragraphs.append("\n".join(buffer).strip())
            buffer = [line]
            continue
        if buffer and sum(len(item) for item in buffer) + len(line) > 900:
            paragraphs.append("\n".join(buffer).strip())
            buffer = [line]
            continue
        buffer.append(line)
    if buffer:
        paragraphs.append("\n".join(buffer).strip())
    return [item for item in paragraphs if item.strip()]


def _extract_ingestion_heading(paragraph: str) -> str:
    text = str(paragraph or "").strip()
    first_line = text.splitlines()[0].strip() if text else ""
    if _looks_like_heading(first_line):
        return first_line[:80]
    return ""


def _update_ingestion_section_stack(section_stack: list[str], heading: str) -> None:
    clean = re.sub(r"\s+", " ", str(heading or "").strip())
    if not clean:
        return
    if re.match(r"^第\s*[一二三四五六七八九十0-9]+\s*[章节部分]", clean):
        section_stack[:] = [clean]
    elif re.match(r"^[一二三四五六七八九十]+[、.]", clean):
        section_stack[:] = [clean]
    elif re.match(r"^[（(][一二三四五六七八九十0-9]+[）)]", clean):
        if section_stack:
            section_stack[:] = section_stack[:1] + [clean]
        else:
            section_stack[:] = [clean]
    elif clean.startswith("附录"):
        section_stack[:] = [clean]
    elif not section_stack or section_stack[-1] != clean:
        if len(section_stack) >= 2:
            section_stack[-1] = clean
        else:
            section_stack.append(clean)


def _classify_source_block(text: str, page_number: int) -> str:
    clean = str(text or "").strip()
    if re.search(r"^表\s*\d+(?:\.\d+)?", clean):
        return "table"
    if "食谱" in clean and ("总能量约" in clean or "早餐" in clean and "晚餐" in clean):
        return "recipe_plan"
    if page_number >= 60 and any(keyword in clean for keyword in ("主要材料", "制作方法", "用法用量", "注意事项")):
        return "therapeutic_recipe"
    if re.match(r"^(注|说明)[:：]", clean):
        return "note"
    if re.match(r"^[（(]?[0-9一二三四五六七八九十]+[）).、]", clean):
        return "list"
    return "paragraph"


def _wiki_compiler_chunk_prompt(
    *,
    chunk: dict[str, Any],
    chunk_count: int,
    source_name: str,
    title: str,
    topic: str,
) -> str:
    return f"""
你正在把一本较长的临床营养指南编译为多页 LLMWiki。请只处理当前 chunk，不要假装看过其他 chunk。

任务：
1. 输出严格 JSON，不要 Markdown 代码块。
2. 必须保留页码证据；每条关键结论、建议和安全边界都要带 source_pages。
3. 表格、食谱、食养方、MET、交换份等结构化内容不要改写成大段 Wiki 正文，只输出 structured_candidates 提示，后续会进结构化库。
4. 如果当前 chunk 只有目录、封面、参考文献，请在 skip_reason 中说明；不要硬编临床建议。
5. 不确定时保持原文边界，标记 needs_human_review。

输出 JSON：
{{
  "chunk_title": "...",
  "section_path": "{chunk.get('section_path') or ''}",
  "source_pages": {json.dumps(chunk.get('source_pages') or [], ensure_ascii=False)},
  "skip_reason": "",
  "key_conclusions": [
    {{"text": "核心结论", "source_pages": [1], "confidence": 0.8}}
  ],
  "clinical_recommendations": [
    {{"text": "可执行建议", "source_pages": [1], "applicable_to": "适用人群或边界"}}
  ],
  "safety_boundaries": [
    {{"text": "禁忌、红线或需要转诊的情况", "source_pages": [1], "severity": "warn|block"}}
  ],
  "structured_candidates": [
    {{"type": "table|recipe_plan|therapeutic_recipe|activity_met|exchange_portion", "description": "...", "source_pages": [1]}}
  ],
  "rules": []
}}

资料文件：{source_name}
Wiki 标题：{title}
主题：{topic or "未指定"}
当前 chunk：{chunk.get('chunk_index')} / {chunk_count}
章节路径：{chunk.get('section_path') or "未识别"}
页码：p.{chunk.get('page_start')} - p.{chunk.get('page_end')}
块类型：{", ".join(chunk.get("block_types") or [])}

当前 chunk 原文：
{chunk.get('text') or ''}
""".strip()


def _fallback_chunk_summary(chunk: dict[str, Any], source_name: str) -> dict[str, Any]:
    text = re.sub(r"\s+", " ", str(chunk.get("text") or "")).strip()
    source_pages = chunk.get("source_pages") or []
    structured_type = _structured_candidate_type(chunk, text)
    statements = _extract_guideline_statements(text, source_pages)
    structured = []
    if structured_type:
        structured.append(
            {
                "type": structured_type,
                "description": _compact_structured_description(text),
                "source_pages": source_pages,
            }
        )
    if structured_type in {"recipe_plan", "therapeutic_recipe", "exchange_portion", "activity_met"}:
        key_conclusions = []
    elif structured_type and not statements:
        key_conclusions = []
    else:
        key_conclusions = statements or [
            {"text": text[:420], "source_pages": source_pages, "confidence": 0.45}
        ] if text else []
    return {
        "chunk_title": chunk.get("section_path") or f"{source_name} p.{chunk.get('page_start')}",
        "section_path": chunk.get("section_path") or "",
        "source_pages": source_pages,
        "skip_reason": _page_skip_reason_for_text(text),
        "key_conclusions": key_conclusions,
        "clinical_recommendations": [],
        "safety_boundaries": [],
        "structured_candidates": structured,
        "rules": [],
        "fallback_generated": True,
    }


def _structured_candidate_type(chunk: dict[str, Any], text: str) -> str:
    block_types = set(str(item) for item in (chunk.get("block_types") or []))
    if "therapeutic_recipe" in block_types or any(keyword in text for keyword in ("主要材料", "制作方法", "用法用量")):
        return "therapeutic_recipe"
    if "recipe_plan" in block_types or ("食谱" in text and "总能量约" in text):
        return "recipe_plan"
    if "table" in block_types or re.search(r"表\s*\d+(?:\.\d+)?", text):
        if "MET" in text or "代谢当量" in text:
            return "activity_met"
        if "交换表" in text or "每份" in text:
            return "exchange_portion"
        return "table"
    return ""


def _compact_structured_description(text: str) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    anchors = []
    for pattern in (
        r"表\s*\d+(?:\.\d+)?[^。；;\n]{0,80}",
        r"[春夏秋冬]季食谱\s*\d+（总能量约\s*\d+kcal）",
        r"主要材料[:：][^。；;\n]{0,120}",
        r"[^\s。；;]{1,20}MET[^\s。；;]{0,60}",
    ):
        anchors.extend(match.group(0).strip() for match in re.finditer(pattern, clean))
    if anchors:
        return "；".join(dict.fromkeys(anchors[:5]))
    return clean[:260]


def _extract_guideline_statements(text: str, source_pages: list[int]) -> list[dict[str, Any]]:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return []
    keywords = (
        "建议",
        "推荐",
        "应",
        "不宜",
        "避免",
        "限制",
        "控制",
        "肥胖",
        "BMI",
        "腰围",
        "减重",
        "体重",
        "能量",
        "运动",
        "睡眠",
        "饮酒",
        "高能量",
    )
    statements = []
    for sentence in _split_guideline_sentences(clean):
        if len(sentence) < 18 or len(sentence) > 260:
            continue
        if "目 录" in sentence or "目录" in sentence or "......" in sentence:
            continue
        if "宏量营养素占总能量比" in sentence:
            continue
        if not any(keyword in sentence for keyword in keywords):
            continue
        if _looks_like_table_row(sentence):
            continue
        statements.append(
            {
                "text": sentence,
                "source_pages": source_pages,
                "confidence": 0.55,
            }
        )
        if len(statements) >= 5:
            break
    return statements


def _split_guideline_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；;])\s*|(?<=\.)\s+", text)
    sentences = []
    for part in parts:
        clean = part.strip(" ，,；;。")
        if clean:
            sentences.append(clean)
    return sentences


def _looks_like_table_row(sentence: str) -> bool:
    digit_count = len(re.findall(r"\d", sentence))
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", sentence))
    if digit_count >= 12 and digit_count > cjk_count * 0.35:
        return True
    if len(re.findall(r"\d+(?:\.\d+)?", sentence)) >= 6:
        return True
    return False


def _normalize_wiki_chunk_summary(payload: dict[str, Any], chunk: dict[str, Any]) -> dict[str, Any]:
    source_pages = _safe_page_list(payload.get("source_pages") or chunk.get("source_pages") or [])
    normalized = {
        "chunk_index": chunk.get("chunk_index"),
        "chunk_count": chunk.get("chunk_count"),
        "chunk_title": str(payload.get("chunk_title") or chunk.get("section_path") or f"chunk-{chunk.get('chunk_index')}").strip(),
        "section_path": str(payload.get("section_path") or chunk.get("section_path") or "").strip(),
        "page_start": int(chunk.get("page_start") or (source_pages[0] if source_pages else 1)),
        "page_end": int(chunk.get("page_end") or (source_pages[-1] if source_pages else 1)),
        "source_pages": source_pages,
        "source_block_ids": chunk.get("source_block_ids") or [],
        "chunk_type": chunk.get("chunk_type") or "paragraph",
        "block_types": chunk.get("block_types") or [],
        "skip_reason": str(payload.get("skip_reason") or "").strip(),
        "key_conclusions": _normalize_evidence_items(payload.get("key_conclusions"), source_pages),
        "clinical_recommendations": _normalize_evidence_items(payload.get("clinical_recommendations"), source_pages),
        "safety_boundaries": _normalize_evidence_items(payload.get("safety_boundaries"), source_pages),
        "structured_candidates": _normalize_evidence_items(payload.get("structured_candidates"), source_pages),
        "rules": payload.get("rules") if isinstance(payload.get("rules"), list) else [],
    }
    normalized["wiki_excluded"] = bool(
        normalized["structured_candidates"]
        and not normalized["key_conclusions"]
        and not normalized["clinical_recommendations"]
        and not normalized["safety_boundaries"]
    )
    if not normalized["skip_reason"] and not (
        normalized["key_conclusions"]
        or normalized["clinical_recommendations"]
        or normalized["safety_boundaries"]
        or normalized["structured_candidates"]
    ):
        normalized["skip_reason"] = "LLM 未抽取到可用知识，需人工复核"
    return normalized


def _normalize_evidence_items(raw: Any, fallback_pages: list[int]) -> list[dict[str, Any]]:
    items = raw if isinstance(raw, list) else []
    normalized = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("description") or item.get("content") or "").strip()
            if not text:
                continue
            copy = dict(item)
            copy["text"] = text
            copy["source_pages"] = _safe_page_list(copy.get("source_pages") or fallback_pages)
            normalized.append(copy)
        else:
            text = str(item or "").strip()
            if text:
                normalized.append({"text": text, "source_pages": list(fallback_pages)})
    return normalized


def _build_wiki_pages_from_summaries(
    chunk_summaries: list[dict[str, Any]],
    *,
    title: str,
    source_name: str,
    related_rag_document_id: str,
    related_structured_document_id: str,
    review_status: str,
) -> list[dict[str, Any]]:
    buckets = _wiki_page_buckets(title=title, source_name=source_name)
    grouped: dict[str, list[dict[str, Any]]] = {slug: [] for slug in buckets}
    for summary in chunk_summaries:
        slug = _classify_wiki_page(summary)
        grouped.setdefault(slug, []).append(summary)

    pages: list[dict[str, Any]] = []
    for slug, meta in buckets.items():
        summaries = grouped.get(slug) or []
        if not summaries:
            continue
        source_pages = sorted(
            {
                page
                for summary in summaries
                for page in _safe_page_list(summary.get("source_pages") or [])
            }
        )
        body = _render_wiki_page_body(meta["title"], summaries, source_pages)
        markdown = _render_wiki_page_markdown(
            page_title=meta["title"],
            body=body,
            source_name=source_name,
            source_pages=source_pages,
            review_status=review_status,
            coverage_id=f"{_slugify(source_name)}:{slug}",
            related_rag_document_id=related_rag_document_id,
            related_structured_document_id=related_structured_document_id,
        )
        pages.append(
            {
                "slug": slug,
                "title": meta["title"],
                "source_pages": source_pages,
                "markdown": markdown,
            }
        )
    return pages


def _wiki_page_buckets(*, title: str = "", source_name: str = "") -> dict[str, dict[str, str]]:
    topic = _infer_wiki_topic(title=title, source_name=source_name)
    return {
        "overview": {"title": f"{topic}指南总览与适用边界"},
        "principles": {"title": f"{topic}营养管理原则"},
        "energy-control": {"title": "能量控制与膳食安排"},
        "food-selection": {"title": "食物选择与餐次建议"},
        "exercise-sleep": {"title": "运动、睡眠与行为管理"},
        "safe-weight-loss": {"title": "安全目标与指标监测"},
        "structured-tables": {"title": "结构化表格、交换份与指标索引"},
        "regional-recipes": {"title": "食谱与结构化菜谱索引"},
        "tcm-diet-therapy": {"title": "中医食养与食药物质"},
    }


def _infer_wiki_topic(*, title: str = "", source_name: str = "") -> str:
    text = _clean_ingestion_display_text(title) or _clean_source_stem_title(Path(source_name).stem)
    text = re.sub(r"Wiki\s*总索引$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"(食养指南|膳食指南|指南|标准|共识|规范)$", "", text).strip(" _-")
    if not text:
        text = "临床营养"
    return text[:30]


def _classify_wiki_page(summary: dict[str, Any]) -> str:
    pages = _safe_page_list(summary.get("source_pages") or [])
    structured_types = {
        str(item.get("type") or "")
        for item in (summary.get("structured_candidates") or [])
        if isinstance(item, dict)
    }
    if summary.get("wiki_excluded"):
        if "recipe_plan" in structured_types:
            return "regional-recipes"
        if "therapeutic_recipe" in structured_types:
            return "tcm-diet-therapy"
        if structured_types & {"table", "activity_met", "exchange_portion"}:
            return "structured-tables"
    if pages:
        first_page = min(pages)
        if first_page <= 2:
            return "overview"
        if first_page <= 5:
            return "principles"
        if first_page <= 8:
            return "energy-control"
    text = " ".join(
        [
            str(summary.get("chunk_title") or ""),
            str(summary.get("section_path") or ""),
            json.dumps(summary.get("key_conclusions") or [], ensure_ascii=False),
            json.dumps(summary.get("clinical_recommendations") or [], ensure_ascii=False),
        ]
    )
    rules = [
        ("safe-weight-loss", ("减重速度", "每周约 0.5kg", "每周0.5kg", "每月 2", "每月2", "5%-10%", "5%～10%", "平台期", "自我监测", "血尿酸", "尿酸水平", "诊断标准")),
        ("exercise-sleep", ("运动", "身体活动", "MET", "睡眠", "久坐", "作息")),
        ("food-selection", ("食物选择", "主食", "谷薯", "蔬菜", "水果", "肉", "奶", "油盐", "限酒", "嘌呤", "低嘌呤", "高嘌呤", "动物内脏", "海鲜", "果糖", "饮酒", "足量饮水")),
        ("energy-control", ("能量", "热量", "kcal", "膳食安排", "三餐", "加餐", "低能量")),
        ("tcm-diet-therapy", ("中医对肥胖", "中医对高尿酸", "中医对痛风", "不同证型", "证型", "湿浊证", "湿热证", "痰瘀证", "脾肾亏虚证", "食养方", "药膳")),
        ("principles", ("原则", "推荐", "平衡膳食", "饮食行为")),
    ]
    for slug, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return slug
    if pages:
        first_page = min(pages)
        if first_page <= 12:
            return "exercise-sleep"
        if first_page <= 20:
            return "food-selection"
        if first_page >= 63:
            return "tcm-diet-therapy"
    return "overview"


def _render_wiki_page_body(title: str, summaries: list[dict[str, Any]], source_pages: list[int]) -> str:
    conclusions = _collect_wiki_items(summaries, "key_conclusions", limit=18)
    recommendations = _collect_wiki_items(summaries, "clinical_recommendations", limit=22)
    safety = _collect_wiki_items(summaries, "safety_boundaries", limit=16)
    structured = _collect_wiki_items(summaries, "structured_candidates", limit=12)
    parts = [f"# {title}", ""]
    if not conclusions and structured:
        parts.extend(
            [
                "## 核心结论",
                "",
                "本页对应内容以结构化知识库为主，不在 Wiki 中复写完整表格、菜谱或食养方。回答具体菜谱、交换份、BMI/MET 或食养方做法时，应优先调用结构化查询工具，并保留来源页码。",
                "",
            ]
        )
    else:
        parts.extend(["## 核心结论", "", _items_to_markdown(conclusions) or "本页暂无自动抽取结论，需人工复核。", ""])
    parts.extend(["## 可执行建议", "", _items_to_markdown(recommendations) or "本页暂无独立可执行建议，需结合相邻章节与结构化库使用。", ""])
    if safety:
        parts.extend(["## 安全边界", "", _items_to_markdown(safety), ""])
    if structured:
        parts.extend(
            [
                "## 结构化库提示",
                "",
                "下列内容应优先查询结构化知识库，不建议只依赖 Wiki 正文复述：",
                "",
                _items_to_markdown(structured),
                "",
            ]
        )
    parts.extend(["## 来源页码", "", _format_page_ranges(source_pages) or "待人工补充。", ""])
    return "\n".join(parts).strip()


def _collect_wiki_items(
    summaries: list[dict[str, Any]],
    field: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    seen = set()
    collected = []
    for summary in summaries:
        if summary.get("skip_reason") and field != "structured_candidates":
            continue
        for item in summary.get(field) or []:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            key = re.sub(r"\s+", "", text)[:80]
            if key in seen:
                continue
            seen.add(key)
            collected.append(item)
            if len(collected) >= limit:
                return collected
    return collected


def _items_to_markdown(items: list[dict[str, Any]]) -> str:
    lines = []
    for item in items:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        pages = _format_page_ranges(_safe_page_list(item.get("source_pages") or []))
        suffix = f"（来源：{pages}）" if pages and "来源" not in text else ""
        lines.append(f"- {text}{suffix}")
    return "\n".join(lines)


def _render_wiki_page_markdown(
    *,
    page_title: str,
    body: str,
    source_name: str,
    source_pages: list[int],
    review_status: str,
    coverage_id: str,
    related_rag_document_id: str,
    related_structured_document_id: str,
) -> str:
    frontmatter = {
        "title": page_title,
        "kb_layer": "llmwiki",
        "evidence_level": "uploaded_document",
        "last_reviewed_at": datetime.now(timezone.utc).date().isoformat(),
        "review_status": review_status,
        "source_document": source_name,
        "source_pages": json.dumps(source_pages, ensure_ascii=False),
        "coverage_id": coverage_id,
        "related_rag_document_id": related_rag_document_id,
        "related_structured_document_id": related_structured_document_id,
    }
    lines = ["---"]
    lines.extend(f"{key}: {value}" for key, value in frontmatter.items())
    lines.extend(["---", "", re.sub(r"^---.*?---\s*", "", body, flags=re.DOTALL).strip()])
    return "\n".join(lines).strip() + "\n"


def _build_wiki_index_markdown(
    *,
    title: str,
    source_name: str,
    wiki_pages: list[dict[str, Any]],
    coverage_report: dict[str, Any],
    related: dict[str, str],
) -> str:
    covered = len(coverage_report.get("covered_pages") or [])
    total = int(coverage_report.get("total_pages") or 0)
    uncovered = coverage_report.get("uncovered_pages") or []
    frontmatter = {
        "title": f"{title} Wiki 总索引",
        "kb_layer": "llmwiki",
        "evidence_level": "uploaded_document",
        "last_reviewed_at": datetime.now(timezone.utc).date().isoformat(),
        "review_status": "draft",
        "source_document": source_name,
        "source_pages": json.dumps(_all_source_page_numbers_from_report(coverage_report), ensure_ascii=False),
        "coverage_id": f"{_slugify(source_name)}:index",
        "related_rag_document_id": related.get("rag_document_id", ""),
        "related_structured_document_id": related.get("structured_document_id", ""),
    }
    lines = ["---"]
    lines.extend(f"{key}: {value}" for key, value in frontmatter.items())
    lines.extend(
        [
            "---",
            "",
            f"# {title} Wiki 总索引",
            "",
            f"> 来源文件：{source_name}",
            "",
            "## 页面目录",
            "",
        ]
    )
    for page in wiki_pages:
        lines.append(
            f"- [{page.get('title')}]({page.get('slug')}.md)：来源页码 {_format_page_ranges(page.get('source_pages') or []) or '待复核'}"
        )
    lines.extend(
        [
            "",
            "## 覆盖率",
            "",
            f"- 总页数：{total}",
            f"- 已覆盖页：{covered}",
            f"- 未覆盖页：{_format_page_ranges(uncovered) or '无'}",
            f"- RAG 文档：{related.get('rag_document_id') or '未关联'}",
            f"- 结构化知识文档：{related.get('structured_document_id') or '未关联'}",
            "",
            "## 使用规则",
            "",
            "- 原则性知识优先读本 Wiki；具体原文证据查 RAG；表格、食谱、食养方、MET、交换份查结构化知识库。",
            "- 本索引仍是草案，必须人工确认后才进入正式 Wiki。",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _build_coverage_report(
    *,
    project_root: Path,
    config: dict[str, Any],
    source_path: Path,
    source_pages: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    wiki_pages: list[dict[str, Any]],
    related: dict[str, str],
    ingestion_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    all_pages = _all_source_page_numbers(source_pages)
    wiki_covered = sorted(
        {
            page
            for wiki_page in wiki_pages
            for page in _safe_page_list(wiki_page.get("source_pages") or [])
        }
    )
    rag_covered = _query_rag_covered_pages(project_root, config, related.get("rag_document_id", ""))
    structured_covered = _query_structured_covered_pages(
        project_root,
        config,
        related.get("structured_document_id", ""),
    )
    skipped = {
        page["page_number"]: reason
        for page in source_pages
        if (reason := _page_skip_reason_for_text(str(page.get("text") or "")))
    }
    # Incorporate ingestion plan block routing
    plan_routes: dict[int, str] = {}
    plan_needs_review: set[int] = set()
    plan_skip_reasons: dict[int, str] = {}
    if ingestion_plan:
        for block in ingestion_plan.get("blocks") or []:
            store_in = str(block.get("should_store_in") or "")
            for p in range(int(block.get("page_start") or 0), int(block.get("page_end") or 0) + 1):
                if store_in and store_in != "skip":
                    plan_routes[p] = store_in
                if store_in == "needs_review":
                    plan_needs_review.add(p)
                if store_in == "skip":
                    plan_skip_reasons[p] = str(block.get("skip_reason") or "入库计划标记为跳过")
    for page, reason in plan_skip_reasons.items():
        skipped.setdefault(page, reason)
    covered_set = set(wiki_covered) | set(rag_covered) | set(structured_covered) | set(skipped) | set(plan_routes)
    uncovered = [page for page in all_pages if page not in covered_set]
    page_routes = []
    for page in all_pages:
        routes = []
        if page in wiki_covered:
            routes.append("wiki")
        if page in rag_covered:
            routes.append("rag")
        if page in structured_covered:
            routes.append("structured")
        if page in skipped:
            routes.append("skipped")
        if page in plan_routes and plan_routes[page] not in routes:
            routes.append(plan_routes[page])
        if page in plan_needs_review:
            routes.append("needs_review")
        page_routes.append(
            {
                "page": page,
                "routes": routes,
                "skip_reason": skipped.get(page, ""),
            }
        )
    return {
        "source_document": source_path.name,
        "total_pages": len(all_pages),
        "wiki_covered_pages": wiki_covered,
        "rag_covered_pages": rag_covered,
        "structured_covered_pages": structured_covered,
        "skipped_pages": [{"page": page, "reason": reason} for page, reason in sorted(skipped.items())],
        "covered_pages": sorted(covered_set),
        "uncovered_pages": uncovered,
        "page_routes": page_routes,
        "chunk_count": len(chunks),
        "wiki_page_count": len(wiki_pages),
        "profiler_block_count": len((ingestion_plan or {}).get("blocks") or []),
        "needs_review_pages": sorted(plan_needs_review),
    }


def _deterministic_wiki_review(
    *,
    wiki_pages: list[dict[str, Any]],
    coverage_report: dict[str, Any],
    chunk_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    issues = []
    recommendations = []
    uncovered = coverage_report.get("uncovered_pages") or []
    if uncovered:
        issues.append(f"存在未覆盖页：{_format_page_ranges(uncovered)}")
    if len(wiki_pages) < 5:
        issues.append("Wiki 页面数量少于 5，可能没有按主题拆分。")
    if not any(summary.get("structured_candidates") for summary in chunk_summaries):
        recommendations.append("未发现结构化内容提示，请复核表格、食谱、食养方是否已进入结构化库。")
    needs_review_pages = coverage_report.get("needs_review_pages") or []
    if needs_review_pages:
        issues.append(f"存在 needs_review 页（{len(needs_review_pages)} 页），需人工复核结构化抽取结果。")
    if not issues:
        recommendations.append("覆盖率检查通过；仍建议人工抽查后半部分、附录和页码引用。")
    return {
        "overall_status": "needs_human_review" if issues else "passed",
        "review_method": "deterministic_coverage_check",
        "confidence": 0.72 if issues else 0.82,
        "issues": issues,
        "recommendations": recommendations,
        "reviewed_at": _utc_now(),
    }


def _compact_coverage_for_review(coverage_report: dict[str, Any]) -> dict[str, Any]:
    skipped_pages = [
        int(item.get("page"))
        for item in coverage_report.get("skipped_pages") or []
        if isinstance(item, dict) and item.get("page")
    ]
    route_covered_pages = set(coverage_report.get("wiki_covered_pages") or [])
    route_covered_pages.update(coverage_report.get("rag_covered_pages") or [])
    route_covered_pages.update(coverage_report.get("structured_covered_pages") or [])
    skipped_only = [page for page in skipped_pages if page not in route_covered_pages]
    return {
        "source_document": coverage_report.get("source_document", ""),
        "total_pages": coverage_report.get("total_pages", 0),
        "covered_page_count": len(coverage_report.get("covered_pages") or []),
        "uncovered_pages": coverage_report.get("uncovered_pages") or [],
        "wiki_covered_ranges": _format_page_ranges(coverage_report.get("wiki_covered_pages") or []),
        "rag_covered_ranges": _format_page_ranges(coverage_report.get("rag_covered_pages") or []),
        "structured_covered_ranges": _format_page_ranges(coverage_report.get("structured_covered_pages") or []),
        "skipped_only_ranges": _format_page_ranges(skipped_only),
        "skipped_overlap_note": "skipped_only_ranges 只表示未进入 wiki/rag/structured 的跳过页；已被其他路线覆盖的页不算遗漏。",
        "chunk_count": coverage_report.get("chunk_count", 0),
        "wiki_page_count": coverage_report.get("wiki_page_count", 0),
    }


def _compact_markdown_excerpt(markdown: Any, max_chars: int) -> str:
    text = str(markdown or "")
    text = re.sub(r"^---\s*.*?\s*---", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text
    head_chars = max(80, int(max_chars * 0.7))
    tail_chars = max(40, max_chars - head_chars - 8)
    return f"{text[:head_chars]} ... {text[-tail_chars:]}"


def _has_review_placeholder(markdown: Any) -> bool:
    text = str(markdown or "")
    markers = (
        "暂无自动抽取结论",
        "暂无独立可执行建议",
        "需要人工复核",
        "待人工补充",
    )
    return any(marker in text for marker in markers)


def _wiki_compiler_review_prompt(
    *,
    source_name: str,
    title: str,
    wiki_pages: list[dict[str, Any]],
    coverage_report: dict[str, Any],
    chunk_summaries: list[dict[str, Any]],
) -> str:
    coverage_digest = _compact_coverage_for_review(coverage_report)
    page_digest = [
        {
            "slug": page.get("slug"),
            "title": page.get("title"),
            "source_pages": page.get("source_pages"),
            "char_count": len(str(page.get("markdown") or "")),
            "has_review_placeholder": _has_review_placeholder(page.get("markdown")),
            "excerpt": _compact_markdown_excerpt(page.get("markdown"), 360),
        }
        for page in wiki_pages
    ]
    compact_chunks = [
        {
            "chunk_index": item.get("chunk_index"),
            "source_pages": item.get("source_pages"),
            "section_path": item.get("section_path"),
            "skip_reason": item.get("skip_reason"),
            "key_count": len(item.get("key_conclusions") or []),
            "recommendation_count": len(item.get("clinical_recommendations") or []),
            "structured_count": len(item.get("structured_candidates") or []),
            "safety_count": len(item.get("safety_boundaries") or []),
            "wiki_excluded": bool(item.get("wiki_excluded")),
        }
        for item in chunk_summaries
    ]
    return f"""
请复核这次 LLMWiki 长文编译草案。你不是重新写 Wiki，而是判断是否达到了商用级入库草案标准。

检查重点：
1. 是否只整理开头、遗漏后半部分、附录、表格说明或页码。
2. 是否每个核心建议都有来源页码。
3. 是否把表格、菜谱、MET、食养方等结构化内容错误塞成大段 Wiki 正文。注意：这些结构化内容不应逐行复写到 Wiki；只要覆盖率报告显示已进入 structured，且 Wiki 有索引页提示调用结构化库，就不算遗漏。
4. 是否有明显需要人工确认的抽取瑕疵。
5. 复核通过也不能自动发布，只能进入人工确认。

判定口径：
- 只有出现未覆盖页、核心章节空洞、明显遗漏后半部分或附录、核心建议没有来源页码、结构化内容既没入 structured 又没在 Wiki 标注索引、严重乱码或明显幻觉时，才标记 needs_human_review。
- 表格编号、页码映射、抽样校对这类“上线前人工抽查建议”，如果没有证据显示已经错误，请写入 recommendations，不要写入 issues，也不要因此阻塞复核通过。

只输出 JSON：
{{
  "overall_status": "passed|needs_human_review",
  "confidence": 0.0,
  "issues": ["..."],
  "recommendations": ["..."],
  "structural_content_notes": ["..."]
}}

来源文件：{source_name}
标题：{title}
覆盖率摘要：
{json.dumps(coverage_digest, ensure_ascii=False)}

Wiki 页面摘要：
{json.dumps(page_digest, ensure_ascii=False)}

chunk 摘要清单：
{json.dumps(compact_chunks, ensure_ascii=False)[:7000]}
""".strip()


def _chunk_ingestion_prompt(
    *,
    chunk_text: str,
    chunk_index: int,
    chunk_count: int,
    source_name: str,
    title: str,
    topic: str,
) -> str:
    return f"""
你正在整理一份较长的临床营养资料。请只基于当前分块内容，提取这一块真正提到的信息，不要假装看过全文。

要求：
1. 只输出 JSON，不要 Markdown 包裹。
2. 这是第 {chunk_index}/{chunk_count} 块。你只总结当前块出现的事实、建议、禁忌、阈值、公式、表格结论、食谱信息或食养方。
3. 如果当前块没有某类信息，就留空，不要编。
4. rules 只提取强安全提示、禁忌、必须转诊或明确阈值，不要把普通建议写成规则。
5. 用简洁中文，方便后续汇总。

输出 JSON 结构：
{{
  "chunk_title": "...",
  "coverage": ["本块覆盖的主题1", "主题2"],
  "facts": [
    "关键事实或阈值",
    "关键公式或换算"
  ],
  "agent_notes_markdown": "## 本块要点\\n- ...",
  "rules": [
    {{
      "rule_id": "DRAFT_...",
      "severity": "warn|block",
      "category": "...",
      "description": "...",
      "if": {{"all": []}},
      "then": {{"action": "warn", "message": "...", "recommendation": "..."}},
      "evidence": {{"source_title": "{source_name}", "evidence_level": "draft"}}
    }}
  ]
}}

资料文件：{source_name}
用户给的标题：{title or "未提供"}
主题：{topic or "未提供"}
当前分块：{chunk_index}/{chunk_count}

当前分块正文：
{chunk_text}
""".strip()


def _final_ingestion_prompt(
    *,
    chunk_summaries: list[dict[str, Any]],
    source_name: str,
    title: str,
    topic: str,
    max_chunk_summary_chars: int,
) -> str:
    chunk_digest = _chunk_summaries_to_text(chunk_summaries, max_chunk_summary_chars)
    return f"""
下面是同一份长文档按分块提炼后的结果。请基于这些分块提炼结果，生成最终的 LLMWiki 页面草案和“需要人工审核的安全规则草案”。

要求：
1. 只输出 JSON，不要 Markdown 包裹。
2. wiki.markdown_body 必须是中文 Markdown，面向“个性化临床营养师 Agent”检索使用。
3. 这是长文整理，不要只写摘要，要尽量覆盖全文的重要章节、附录、表格结论、阈值、公式和适用边界。
4. Wiki 内容至少要组织出：核心结论、适用边界、关键建议、禁忌/红线、证据来源、使用注意；如果材料里有判定标准、换算表、示例食谱、活动系数、公式，也要纳入。
5. rules 只提取强安全提示或禁忌，不要把普通建议写成规则。
6. 规则默认 review_status=draft_needs_review，不是线上生效规则。
7. 如果分块信息彼此不一致，优先保守表述，并在使用注意中提示人工复核。

输出 JSON 结构：
{{
  "wiki": {{
    "title": "...",
    "evidence_level": "guideline|expert_consensus|paper|uploaded_document|unknown",
    "review_status": "draft",
    "markdown_body": "..."
  }},
  "rules": [
    {{
      "rule_id": "DRAFT_...",
      "severity": "warn|block",
      "category": "...",
      "description": "...",
      "if": {{"all": []}},
      "then": {{"action": "warn", "message": "...", "recommendation": "..."}},
      "evidence": {{"source_title": "{source_name}", "evidence_level": "draft"}}
    }}
  ]
}}

资料文件：{source_name}
用户给的标题：{title or "未提供"}
主题：{topic or "未提供"}

分块提炼结果：
{chunk_digest}
""".strip()


def _build_source_excerpt(source_text: str, max_chars: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", str(source_text or "")).strip()
    if len(text) <= max_chars:
        return text

    head_chars = int(max_chars * 0.65)
    tail_chars = int(max_chars * 0.15)
    keyword_chars = max_chars - head_chars - tail_chars - 800
    keyword_excerpt = _keyword_windows(
        text,
        [
            "原则",
            "建议",
            "禁忌",
            "红线",
            "风险",
            "不宜",
            "避免",
            "限制",
            "推荐",
            "附录",
            "食物选择",
            "食谱",
            "安全",
        ],
        max(2000, keyword_chars),
    )
    parts = [
        "【文档开头】\n" + text[:head_chars].strip(),
        "【关键片段】\n" + keyword_excerpt.strip(),
        "【文档结尾】\n" + text[-tail_chars:].strip(),
    ]
    return "\n\n".join(part for part in parts if part.strip())[:max_chars]


def _keyword_windows(text: str, keywords: list[str], max_chars: int) -> str:
    windows: list[str] = []
    used_ranges: list[tuple[int, int]] = []
    for keyword in keywords:
        for match in re.finditer(re.escape(keyword), text):
            start = max(0, match.start() - 500)
            end = min(len(text), match.end() + 1200)
            if any(not (end < used_start or start > used_end) for used_start, used_end in used_ranges):
                continue
            used_ranges.append((start, end))
            windows.append(text[start:end].strip())
            if sum(len(item) for item in windows) >= max_chars:
                return "\n\n---\n\n".join(windows)[:max_chars]
            break
    return "\n\n---\n\n".join(windows)[:max_chars]


def _split_document_for_ingestion(
    source_text: str,
    *,
    target_chars: int,
    overlap_chars: int,
) -> list[str]:
    text = re.sub(r"\r\n?", "\n", str(source_text or ""))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return [""]
    if len(text) <= target_chars:
        return [text]

    paragraphs = [item.strip() for item in text.split("\n\n") if item.strip()]
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        paragraph_len = len(paragraph)
        is_heading = _looks_like_heading(paragraph)
        if current_parts and (
            current_len + paragraph_len + 2 > target_chars
            or (is_heading and current_len >= int(target_chars * 0.65))
        ):
            chunk_text = "\n\n".join(current_parts).strip()
            if chunk_text:
                chunks.append(chunk_text)
            overlap_text = ""
            if overlap_chars > 0 and chunk_text:
                overlap_text = chunk_text[-overlap_chars:].strip()
            current_parts = [part for part in [overlap_text, paragraph] if part]
            current_len = sum(len(part) for part in current_parts) + max(0, len(current_parts) - 1) * 2
            continue
        current_parts.append(paragraph)
        current_len += paragraph_len + (2 if current_parts[:-1] else 0)

    final_chunk = "\n\n".join(current_parts).strip()
    if final_chunk:
        chunks.append(final_chunk)
    return chunks or [text]


def _looks_like_heading(paragraph: str) -> bool:
    text = str(paragraph or "").strip()
    if not text or len(text) > 80:
        return False
    if re.fullmatch(r"\d+", text):
        return True
    patterns = [
        r"^[一二三四五六七八九十]+[、.]",
        r"^[（(][一二三四五六七八九十0-9]+[）)]",
        r"^附录\s*[0-9一二三四五六七八九十]+",
        r"^表\s*[0-9]+(?:\.[0-9]+)?",
        r"^第\s*[0-9一二三四五六七八九十]+\s*[章节部分]",
    ]
    return any(re.match(pattern, text) for pattern in patterns)


def _chunk_summaries_to_text(chunk_summaries: list[dict[str, Any]], max_chars: int) -> str:
    parts: list[str] = []
    for item in chunk_summaries:
        chunk_index = item.get("chunk_index", "?")
        chunk_title = str(item.get("chunk_title") or "").strip() or f"chunk-{chunk_index}"
        coverage = item.get("coverage") if isinstance(item.get("coverage"), list) else []
        facts = item.get("facts") if isinstance(item.get("facts"), list) else []
        notes = str(item.get("agent_notes_markdown") or "").strip()
        rules = item.get("rules") if isinstance(item.get("rules"), list) else []
        block = [
            f"## Chunk {chunk_index}: {chunk_title}",
            f"覆盖主题：{', '.join(str(part) for part in coverage if str(part).strip())}",
            "关键事实：",
            "\n".join(f"- {str(fact).strip()}" for fact in facts if str(fact).strip()) or "- 无",
            "块内备注：",
            notes or "无",
            "候选规则：",
            json.dumps(rules, ensure_ascii=False),
        ]
        text = "\n".join(block).strip()[:max_chars]
        parts.append(text)
    return "\n\n".join(parts).strip()


def _last_non_empty(items: list[str]) -> str:
    for item in reversed(items):
        clean = str(item or "").strip()
        if clean:
            return clean
    return ""


def _tail_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    clean = str(text or "").strip()
    if len(clean) <= max_chars:
        return clean
    return clean[-max_chars:].strip()


def _safe_page_list(value: Any) -> list[int]:
    raw_items = value if isinstance(value, list) else [value]
    pages = []
    for item in raw_items:
        try:
            page = int(item)
        except (TypeError, ValueError):
            continue
        if page > 0:
            pages.append(page)
    return sorted(set(pages))


def _all_source_page_numbers(pages: list[dict[str, Any]]) -> list[int]:
    return sorted({int(page.get("page_number") or index) for index, page in enumerate(pages, start=1)})


def _all_source_page_numbers_from_report(report: dict[str, Any]) -> list[int]:
    total = int(report.get("total_pages") or 0)
    if total > 0:
        return list(range(1, total + 1))
    pages = set()
    for key in ("wiki_covered_pages", "rag_covered_pages", "structured_covered_pages", "covered_pages"):
        pages.update(_safe_page_list(report.get(key) or []))
    return sorted(pages)


def _format_page_ranges(pages: Any) -> str:
    page_numbers = _safe_page_list(pages)
    if not page_numbers:
        return ""
    ranges = []
    start = prev = page_numbers[0]
    for page in page_numbers[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append(f"p.{start}" if start == prev else f"pp.{start}-{prev}")
        start = prev = page
    ranges.append(f"p.{start}" if start == prev else f"pp.{start}-{prev}")
    return "、".join(ranges)


def _page_skip_reason_for_text(text: str) -> str:
    clean = re.sub(r"\s+", "", str(text or ""))
    if not clean:
        return "空白页或无法抽取文本"
    if any(keyword in clean[:120] for keyword in ("目录", "目次")):
        return "目录页"
    if clean.count("..") >= 8 or clean.count("…") >= 8:
        return "目录页"
    if len(clean) < 80 and any(keyword in clean for keyword in ("目录", "目次", "封面")):
        return "目录/封面页"
    if clean.startswith("参考文献") or clean.startswith("主要参考文献"):
        return "参考文献页"
    if len(clean) < 120 and re.search(r"^(图书在版|版权|编委会|前言)", clean):
        return "出版信息或前言页"
    return ""


def _resolve_project_path(project_root: Path, configured: Any, default: str) -> Path:
    raw = str(configured or default)
    path = Path(raw)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _find_related_rag_document_id(project_root: Path, config: dict[str, Any], source_path: Path) -> str:
    rag_config = config.get("clinical_rag") or {}
    db_path = _resolve_project_path(project_root, rag_config.get("db_path"), "data/clinical_rag.db")
    if not db_path.exists():
        return ""
    stored_path = _relative_to(project_root, source_path)
    try:
        with sqlite3.connect(db_path) as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                """
                SELECT document_id
                FROM rag_documents
                WHERE stored_path = ? OR original_name = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (stored_path, source_path.name),
            ).fetchone()
            return str(row["document_id"]) if row else ""
    except sqlite3.Error:
        return ""


def _find_related_structured_document_id(project_root: Path, config: dict[str, Any], source_path: Path) -> str:
    structured_config = config.get("clinical_knowledge") or {}
    db_path = _resolve_project_path(
        project_root,
        structured_config.get("db_path"),
        "data/clinical_knowledge.db",
    )
    if not db_path.exists():
        return ""
    stored_path = _relative_to(project_root, source_path)
    try:
        with sqlite3.connect(db_path) as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                """
                SELECT document_id
                FROM source_documents
                WHERE stored_path = ? OR original_name = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (stored_path, source_path.name),
            ).fetchone()
            return str(row["document_id"]) if row else ""
    except sqlite3.Error:
        return ""


def _query_rag_covered_pages(project_root: Path, config: dict[str, Any], document_id: str) -> list[int]:
    if not document_id:
        return []
    rag_config = config.get("clinical_rag") or {}
    db_path = _resolve_project_path(project_root, rag_config.get("db_path"), "data/clinical_rag.db")
    if not db_path.exists():
        return []
    try:
        with sqlite3.connect(db_path) as db:
            rows = db.execute(
                "SELECT page_number FROM rag_pages WHERE document_id = ? ORDER BY page_number",
                (document_id,),
            ).fetchall()
            return [int(row[0]) for row in rows]
    except sqlite3.Error:
        return []


def _query_structured_covered_pages(project_root: Path, config: dict[str, Any], document_id: str) -> list[int]:
    if not document_id:
        return []
    structured_config = config.get("clinical_knowledge") or {}
    db_path = _resolve_project_path(
        project_root,
        structured_config.get("db_path"),
        "data/clinical_knowledge.db",
    )
    if not db_path.exists():
        return []
    pages = set()
    queries = [
        "SELECT page_start, page_end FROM guide_tables WHERE document_id = ?",
        "SELECT page_start, page_end FROM food_exchange_portions WHERE document_id = ?",
        "SELECT page_start, page_end FROM recipe_plans WHERE document_id = ?",
        "SELECT page_start, page_end FROM therapeutic_recipes WHERE document_id = ?",
        "SELECT page_start, page_end FROM activity_mets WHERE document_id = ?",
    ]
    try:
        with sqlite3.connect(db_path) as db:
            for query in queries:
                try:
                    rows = db.execute(query, (document_id,)).fetchall()
                except sqlite3.Error:
                    continue
                for start, end in rows:
                    start_int = int(start or 1)
                    end_int = int(end or start_int)
                    pages.update(range(start_int, end_int + 1))
    except sqlite3.Error:
        return []
    return sorted(pages)


def _safe_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _clean_ingestion_display_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    question_marks = text.count("?") + text.count("？")
    if "????" in text or question_marks >= max(3, int(len(text) * 0.35)):
        return ""
    if text.lower() in {"undefined", "null", "none"}:
        return ""
    return text


def _clean_source_stem_title(stem: Any) -> str:
    text = _clean_ingestion_display_text(stem)
    text = re.sub(r"^\d{8}T\d{6}Z[_-]*", "", text)
    text = text.strip(" _-")
    return text or _clean_ingestion_display_text(stem) or "uploaded clinical document"


def _clean_wiki_publish_title(value: Any) -> str:
    text = _clean_ingestion_display_text(value) or "uploaded clinical document"
    text = re.sub(r"(?:[-_ ]*wiki[-_ ]*总索引)+$", "", text, flags=re.IGNORECASE).strip(" -_")
    text = re.sub(r"(?:\s*Wiki\s*总索引)+$", "", text, flags=re.IGNORECASE).strip(" -_")
    return text or "uploaded clinical document"


def _remove_prior_wiki_index_entries(existing: str, *, title: str, source_name: str) -> str:
    if not existing.strip():
        return existing
    title_key = _wiki_index_dedupe_key(title)
    source_key = source_name.strip().lower()
    kept = []
    for line in existing.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ["):
            line_key = _wiki_index_dedupe_key(stripped)
            source_matches = bool(source_key and source_key in stripped.lower())
            title_matches = bool(title_key and title_key in line_key)
            if source_matches or title_matches:
                continue
        kept.append(line)
    return "\n".join(kept).rstrip() + ("\n" if kept else "")


def _wiki_index_dedupe_key(value: Any) -> str:
    text = _clean_ingestion_display_text(value)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"-v\d{14}(?:-\d+)?", " ", text)
    text = re.sub(r"(?:[-_ ]*wiki[-_ ]*总索引)+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -_").lower()


def _fallback_generation(
    source_text: str,
    source_name: str,
    title: str,
    topic: str,
    reason: str,
) -> dict[str, Any]:
    fallback_title = title or topic or Path(source_name).stem
    excerpt = source_text[:5000].strip()
    return {
        "llm_used": False,
        "llm_error": reason,
        "wiki": {
            "title": fallback_title,
            "evidence_level": "uploaded_document",
            "review_status": "draft",
            "markdown_body": (
                f"# {fallback_title}\n\n"
                f"> 来源文件：{source_name}\n\n"
                "## 待整理内容\n\n"
                f"{excerpt}\n\n"
                "## 入库备注\n\n"
                "本页面由系统回退生成，尚未经过大模型整理，请人工编辑后再作为正式知识页使用。\n"
            ),
        },
        "rules": [],
    }


def _normalize_wiki_markdown(
    *,
    payload: dict[str, Any],
    fallback_title: str,
    source_name: str,
    source_text: str,
) -> str:
    title = str(payload.get("title") or fallback_title or "uploaded clinical nutrition note").strip()
    body = str(payload.get("markdown_body") or "").strip()
    if not body:
        body = _fallback_generation(source_text, source_name, title, "")["wiki"]["markdown_body"]
    body = re.sub(r"^---.*?---\s*", "", body, flags=re.DOTALL).strip()
    frontmatter = {
        "title": title,
        "kb_layer": "llmwiki",
        "evidence_level": str(payload.get("evidence_level") or "uploaded_document"),
        "last_reviewed_at": datetime.now(timezone.utc).date().isoformat(),
        "review_status": str(payload.get("review_status") or "draft"),
        "source_document": source_name,
    }
    lines = ["---"]
    lines.extend(f"{key}: {value}" for key, value in frontmatter.items())
    lines.extend(["---", "", body])
    return "\n".join(lines).strip() + "\n"


def _normalize_rule_drafts(raw_rules: Any, source_name: str) -> dict[str, Any]:
    rules = []
    if isinstance(raw_rules, list):
        for idx, rule in enumerate(raw_rules, start=1):
            if not isinstance(rule, dict):
                continue
            rule_id = str(rule.get("rule_id") or f"DRAFT_{_slugify(source_name).upper()}_{idx}").strip()
            rule["rule_id"] = re.sub(r"[^A-Z0-9_]+", "_", rule_id.upper()).strip("_")
            rule.setdefault("severity", "warn")
            rule.setdefault("category", "draft_clinical_nutrition")
            rule.setdefault("description", "")
            rule.setdefault("if", {"all": []})
            rule.setdefault("then", {"action": "warn", "message": "", "recommendation": ""})
            rule.setdefault("evidence", {"source_title": source_name, "evidence_level": "draft"})
            rule["review_status"] = "draft_needs_review"
            rules.append(rule)
    return {"version": "draft", "source_document": source_name, "rules": rules}


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw).strip()
    candidates = [raw]
    candidates.extend(_json_object_candidates(raw))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _json_object_candidates(raw: str) -> list[str]:
    candidates = []
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    starts = [index for index, char in enumerate(raw) if char == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(raw)):
            char = raw[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(raw[start : index + 1])
                    break
    return candidates


def _frontmatter_title(markdown: str) -> str:
    match = re.search(r"^title:\s*(.+)$", markdown, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _slugify(text: str) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff_.-]+", "-", value)
    value = value.strip(".-")
    if not value:
        value = f"uploaded-{uuid.uuid4().hex[:8]}"
    return value[:80]


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 1000):
        candidate = path.with_name(f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError("cannot create unique target path")


def _unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.name}-{index}")
        if not candidate.exists():
            return candidate
    raise ValueError("cannot create unique target directory")


def _relative_to(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _read_wiki_pages(draft_dir: Path) -> list[dict[str, Any]]:
    pages_dir = draft_dir / "pages"
    if not pages_dir.exists():
        return []
    pages = []
    for path in sorted(pages_dir.glob("*.md")):
        markdown = _read_text(path)
        pages.append(
            {
                "slug": path.stem,
                "path": str(path.relative_to(draft_dir)).replace("\\", "/"),
                "title": _frontmatter_title(markdown) or path.stem,
                "source_pages": _frontmatter_source_pages(markdown),
                "review_status": _frontmatter_value(markdown, "review_status"),
                "markdown": markdown,
            }
        )
    return pages


def _frontmatter_value(markdown: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", markdown, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _frontmatter_source_pages(markdown: str) -> list[int]:
    raw = _frontmatter_value(markdown, "source_pages")
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = re.findall(r"\d+", raw)
    return _safe_page_list(payload)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
