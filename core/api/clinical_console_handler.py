from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from aiohttp import web

from config.config_loader import get_project_dir, merge_configs, read_config
from core.api.base_handler import BaseHandler
from core.clinical_nutrition.clinical_rag import ClinicalRAGService
from core.clinical_nutrition.knowledge_ingestion import KnowledgeIngestionService
from core.clinical_nutrition.nutrition_targets import estimate_daily_nutrition_targets
from core.clinical_nutrition.structured_knowledge import StructuredKnowledgeStore
from core.providers.memory.clinical_ltm.health_profile import (
    HealthProfileStore,
    analyze_blood_glucose_readings,
)
from core.providers.memory.clinical_ltm.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    SHORT_TERM_SUMMARY_SYSTEM_PROMPT,
)
from core.utils.conversation_history_store import ConversationHistoryStore
from core.utils.device_identity import normalize_device_user_id

TAG = __name__


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = {
    ".pdf",
    ".md",
    ".markdown",
    ".txt",
    ".doc",
    ".docx",
    ".csv",
    ".tsv",
    ".xls",
    ".xlsx",
    ".json",
}

SECRET_FIELD_NAMES = {"api_key", "access_token", "secret_key", "api_secret"}
PLACEHOLDER_SECRET_HINTS = ("你的", "填你的", "在这里填", "api key", "api密钥", "API密钥")


class ClinicalConsoleHandler(BaseHandler):
    """HTTP API and static page for the clinical nutrition agent console."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.project_root = Path(get_project_dir()).resolve()
        self.console_root = self.project_root / "console"
        self.console_html = self.console_root / "index.html"
        self.upload_root = (
            self.project_root / "data" / "knowledge_uploads" / "clinical-nutrition"
        )
        self.upload_manifest = self.upload_root / "upload_manifest.jsonl"
        self.history_store = ConversationHistoryStore(
            self.project_root / "data" / "console_history.db"
        )
        self._rag_tasks: dict[str, asyncio.Task] = {}

    async def handle_console(self, request: web.Request) -> web.StreamResponse:
        if request.path == "/console":
            raise web.HTTPFound("/console/")
        if not self.console_html.exists():
            return self._json_response(
                {"ok": False, "error": "console/index.html not found"},
                status=404,
            )
        response = web.FileResponse(self.console_html)
        self._add_cors_headers(response)
        self._add_no_cache_headers(response)
        return response

    async def handle_console_asset(self, request: web.Request) -> web.StreamResponse:
        relative = str(request.match_info.get("path") or "").strip().lstrip("/")
        if not relative:
            raise web.HTTPFound("/console/")

        asset_path = (self.console_root / relative).resolve()
        try:
            asset_path.relative_to(self.console_root.resolve())
        except ValueError:
            return self._json_response({"ok": False, "error": "invalid asset path"}, status=404)

        if not asset_path.exists() or not asset_path.is_file():
            return self._json_response({"ok": False, "error": "console asset not found"}, status=404)

        response = web.FileResponse(asset_path)
        self._add_cors_headers(response)
        self._add_no_cache_headers(response)
        return response

    def _add_no_cache_headers(self, response: web.StreamResponse) -> None:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    async def handle_summary(self, request: web.Request) -> web.Response:
        health_db = self._health_profile_db_path()
        ltm_db = self._ltm_db_path()
        powermem_db = self._powermem_db_path()
        food_db = self._food_db_path()
        rag_db = self._rag_db_path()
        clinical_knowledge_db = self._clinical_knowledge_db_path()
        rules_path = self.project_root / "knowledge_base" / "rules" / "clinical_safety_rules.json"
        rag_summary = self._rag_summary()
        clinical_knowledge_summary = self._clinical_knowledge_summary()
        wiki_root = self._wiki_root_path()
        wiki_drafts = self._knowledge_ingestion_service().list_drafts()

        payload = {
            "ok": True,
            "generated_at": _utc_now(),
            "paths": {
                "project_root": str(self.project_root),
                "upload_root": self._relative_path(self.upload_root),
                "console": "/console/",
            },
            "databases": {
                "health_profile": self._database_info(
                    health_db,
                    [
                        "health_profiles",
                        "health_profile_items",
                        "blood_glucose_readings",
                        "daily_nutrition_intakes",
                        "health_profile_review_items",
                    ],
                ),
                "long_term_memory": self._database_info(
                    ltm_db,
                    [
                        "ltm_working_memory",
                        "ltm_short_term_summary",
                        "ltm_memory_items",
                    ],
                ),
                "powermem": self._database_info(powermem_db, []),
                "food_nutrition": self._database_info(
                    food_db,
                    [
                        "source_documents",
                        "food_items",
                        "food_aliases",
                        "food_nutrients_per_100g",
                        "portion_units",
                        "food_risk_tags",
                    ],
                ),
                "clinical_rag": self._database_info(
                    rag_db,
                    [
                        "rag_documents",
                        "rag_pages",
                        "rag_chunks",
                        "rag_embeddings",
                        "rag_ingestion_jobs",
                    ],
                ),
                "clinical_knowledge": self._database_info(
                    clinical_knowledge_db,
                    [
                        "source_documents",
                        "guide_tables",
                        "guide_table_rows",
                        "recipe_plans",
                        "therapeutic_recipes",
                        "activity_mets",
                    ],
                ),
            },
            "knowledge": {
                "rag_db": self._relative_path(rag_db),
                "rag_documents": rag_summary["documents"],
                "rag_indexed_documents": rag_summary["indexed_documents"],
                "rag_chunks": rag_summary["chunks"],
                "rag_embeddings": rag_summary["embeddings"],
                "wiki_root": self._relative_path(wiki_root),
                "wiki_pages": self._count_wiki_pages(wiki_root),
                "wiki_drafts": len(wiki_drafts),
                "clinical_knowledge_db": self._relative_path(clinical_knowledge_db),
                "structured_tables": clinical_knowledge_summary["tables"],
                "structured_table_rows": clinical_knowledge_summary["table_rows"],
                "structured_recipe_plans": clinical_knowledge_summary["recipe_plans"],
                "structured_therapeutic_recipes": clinical_knowledge_summary["therapeutic_recipes"],
                "structured_activity_mets": clinical_knowledge_summary["activity_mets"],
                "uploaded_files": len(self._list_uploaded_files()),
            },
            "rules": self._rules_summary(rules_path),
            "users": self._list_users(limit=12),
        }
        return self._json_response(payload)

    async def handle_users(self, request: web.Request) -> web.Response:
        try:
            limit = _int_query(request, "limit", 50, minimum=1, maximum=200)
            users = self._list_users(limit=limit)
            return self._json_response({"ok": True, "users": users})
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_profile(self, request: web.Request) -> web.Response:
        user_id = self._user_id_from_request(request)
        if not user_id:
            return self._json_response(
                {"ok": False, "error": "missing user_id or device_id"},
                status=400,
            )
        return self._json_response(
            {"ok": True, "user_id": user_id, "profile": self._get_profile(user_id)}
        )

    async def handle_profile_review(self, request: web.Request) -> web.Response:
        user_id = self._user_id_from_request(request)
        if not user_id:
            return self._json_response(
                {"ok": False, "error": "missing user_id or device_id"},
                status=400,
            )
        status = str(request.query.get("status") or "pending").strip() or "pending"
        try:
            store = HealthProfileStore(self._health_profile_db_path())
            review_items = store.list_review_items_sync(user_id, status=status)
            return self._json_response(
                {"ok": True, "user_id": user_id, "status": status, "review_items": review_items}
            )
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_profile_review_resolve(self, request: web.Request) -> web.Response:
        review_id = str(request.match_info.get("review_id") or "").strip()
        if not review_id:
            return self._json_response({"ok": False, "error": "missing review_id"}, status=400)
        try:
            payload = await request.json()
        except Exception as exc:
            return self._json_response(
                {"ok": False, "error": f"invalid json payload: {exc}"},
                status=400,
            )
        try:
            decision = str((payload or {}).get("decision") or "").strip()
            store = HealthProfileStore(self._health_profile_db_path())
            review_item = store.resolve_review_item_sync(
                review_id,
                decision,
                resolved_by="console_admin",
            )
            return self._json_response({"ok": True, "review_item": review_item})
        except ValueError as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_memory(self, request: web.Request) -> web.Response:
        user_id = self._user_id_from_request(request)
        if not user_id:
            return self._json_response(
                {"ok": False, "error": "missing user_id or device_id"},
                status=400,
            )
        limit = _int_query(request, "limit", 50, minimum=1, maximum=200)
        return self._json_response(
            {
                "ok": True,
                "user_id": user_id,
                "limit": limit,
                "memory": self._get_memory(user_id, limit),
            }
        )

    async def handle_knowledge_files(self, request: web.Request) -> web.Response:
        return self._json_response(
            {
                "ok": True,
                "upload_root": self._relative_path(self.upload_root),
                "files": self._list_uploaded_files(),
            }
        )

    async def handle_knowledge_upload(self, request: web.Request) -> web.Response:
        try:
            reader = await request.multipart()
        except Exception as exc:
            return self._json_response(
                {"ok": False, "error": f"invalid multipart upload: {exc}"},
                status=400,
            )

        self.upload_root.mkdir(parents=True, exist_ok=True)
        uploaded: list[dict[str, Any]] = []

        while True:
            part = await reader.next()
            if part is None:
                break
            if not part.filename:
                continue

            safe_name = _safe_filename(part.filename)
            suffix = Path(safe_name).suffix.lower()
            if suffix not in ALLOWED_UPLOAD_SUFFIXES:
                return self._json_response(
                    {
                        "ok": False,
                        "error": f"unsupported file type: {suffix or '(none)'}",
                        "allowed_suffixes": sorted(ALLOWED_UPLOAD_SUFFIXES),
                    },
                    status=400,
                )

            stored_name = self._stored_upload_name(safe_name)
            target_path = self.upload_root / stored_name
            size = 0
            try:
                with target_path.open("wb") as output:
                    while True:
                        chunk = await part.read_chunk()
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > MAX_UPLOAD_BYTES:
                            output.close()
                            target_path.unlink(missing_ok=True)
                            return self._json_response(
                                {
                                    "ok": False,
                                    "error": "file is larger than 50MB",
                                    "filename": part.filename,
                                },
                                status=413,
                            )
                        output.write(chunk)
            except Exception as exc:
                target_path.unlink(missing_ok=True)
                return self._json_response(
                    {"ok": False, "error": f"upload failed: {exc}"},
                    status=500,
                )

            record = {
                "uploaded_at": _utc_now(),
                "original_name": Path(part.filename).name,
                "stored_name": stored_name,
                "relative_path": self._relative_path(target_path),
                "size_bytes": size,
                "content_type": part.headers.get("Content-Type", ""),
                "status": "raw_uploaded",
            }
            try:
                document = self._rag_service().register_document(
                    target_path,
                    original_name=record["original_name"],
                    content_type=record["content_type"],
                )
                record["document_id"] = document.get("document_id", "")
                record["rag_status"] = document.get("status", "")
            except Exception as exc:
                record["rag_status"] = "register_failed"
                record["rag_error"] = str(exc)
            self._append_upload_manifest(record)
            uploaded.append(record)

        if not uploaded:
            return self._json_response(
                {"ok": False, "error": "no files received"},
                status=400,
            )

        return self._json_response({"ok": True, "uploaded": uploaded})

    async def handle_rag_documents(self, request: web.Request) -> web.Response:
        try:
            service = self._rag_service()
            structured_reviews = self._structured_document_review_map()
            documents = []
            for document in service.list_documents():
                enriched = dict(document)
                review = structured_reviews.get(document.get("document_id") or "")
                if review:
                    enriched["structured_review"] = review
                documents.append(enriched)
            return self._json_response(
                {
                    "ok": True,
                    "db_path": self._relative_path(service.db_path),
                    "documents": documents,
                }
            )
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_rag_upload(self, request: web.Request) -> web.Response:
        return await self.handle_knowledge_upload(request)

    async def handle_rag_index(self, request: web.Request) -> web.Response:
        document_id = str(request.match_info.get("document_id") or "").strip()
        if not document_id:
            return self._json_response({"ok": False, "error": "missing document_id"}, status=400)
        try:
            service = self._rag_service()
            job = service.create_index_job(document_id)
            task = asyncio.create_task(
                asyncio.to_thread(self._run_rag_index_job, document_id, job["job_id"])
            )
            self._rag_tasks[job["job_id"]] = task
            task.add_done_callback(lambda _task, job_id=job["job_id"]: self._rag_tasks.pop(job_id, None))
            return self._json_response({"ok": True, "job": job})
        except ValueError as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_rag_job(self, request: web.Request) -> web.Response:
        job_id = str(request.match_info.get("job_id") or "").strip()
        if not job_id:
            return self._json_response({"ok": False, "error": "missing job_id"}, status=400)
        try:
            job = self._rag_service().get_job(job_id)
            if job is None:
                return self._json_response({"ok": False, "error": "job not found"}, status=404)
            return self._json_response({"ok": True, "job": job})
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_rag_chunks(self, request: web.Request) -> web.Response:
        document_id = str(request.match_info.get("document_id") or "").strip()
        if not document_id:
            return self._json_response({"ok": False, "error": "missing document_id"}, status=400)
        limit = _int_query(request, "limit", 200, minimum=1, maximum=1000)
        try:
            chunks = self._rag_service().list_chunks(document_id, limit=limit)
            return self._json_response({"ok": True, "document_id": document_id, "chunks": chunks})
        except ValueError as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=404)
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_rag_search(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception as exc:
            return self._json_response(
                {"ok": False, "error": f"invalid json payload: {exc}"},
                status=400,
            )
        question = str((payload or {}).get("question") or "").strip()
        if not question:
            return self._json_response({"ok": False, "error": "missing question"}, status=400)
        try:
            top_k = int((payload or {}).get("top_k") or 6)
        except (TypeError, ValueError):
            top_k = 6
        try:
            results = self._rag_service().search(question, top_k=top_k)
            return self._json_response({"ok": True, "question": question, "results": results})
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_rag_delete(self, request: web.Request) -> web.Response:
        document_id = str(request.match_info.get("document_id") or "").strip()
        if not document_id:
            return self._json_response({"ok": False, "error": "missing document_id"}, status=400)
        try:
            document = self._rag_service().delete_document(document_id)
            return self._json_response({"ok": True, "document": document})
        except ValueError as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=404)
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_structured_knowledge_review(self, request: web.Request) -> web.Response:
        document_id = str(request.match_info.get("document_id") or "").strip()
        if not document_id:
            return self._json_response({"ok": False, "error": "missing document_id"}, status=400)
        try:
            result = await asyncio.to_thread(self._run_structured_knowledge_review, document_id)
            return self._json_response({"ok": True, "review": result})
        except ValueError as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_structured_knowledge_approve(self, request: web.Request) -> web.Response:
        document_id = str(request.match_info.get("document_id") or "").strip()
        if not document_id:
            return self._json_response({"ok": False, "error": "missing document_id"}, status=400)
        try:
            result = self._structured_knowledge_store().approve_document(
                document_id,
                approved_by="console_admin",
            )
            return self._json_response({"ok": True, "review": result})
        except ValueError as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_structured_needs_review_resolve(self, request: web.Request) -> web.Response:
        review_id = str(request.match_info.get("review_id") or "").strip()
        if not review_id:
            return self._json_response({"ok": False, "error": "missing review_id"}, status=400)
        try:
            payload = await request.json()
        except Exception as exc:
            return self._json_response(
                {"ok": False, "error": f"invalid json payload: {exc}"},
                status=400,
            )
        try:
            status = str((payload or {}).get("status") or "").strip()
            reviewer_notes = str((payload or {}).get("reviewer_notes") or "").strip()
            item = self._structured_knowledge_store().resolve_needs_review_item(
                review_id,
                status=status,
                reviewer_notes=reviewer_notes,
            )
            return self._json_response({"ok": True, "review_item": item})
        except ValueError as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_knowledge_ingest(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception as exc:
            return self._json_response(
                {"ok": False, "error": f"invalid json payload: {exc}"},
                status=400,
            )

        try:
            source_path = self._resolve_ingestion_source(payload or {})
            response_payload: dict[str, Any] = {"ok": True}
            document: dict[str, Any] | None = None

            try:
                document = self._rag_service().register_document(
                    source_path,
                    original_name=source_path.name,
                    content_type="",
                )
                response_payload["document"] = document
            except Exception as exc:
                self.logger.bind(tag=TAG).error(f"Clinical RAG registration failed: {exc}")
                response_payload["rag_error"] = str(exc)

            try:
                draft = self._knowledge_ingestion_service().create_draft(
                    source_path=source_path,
                    title=str((payload or {}).get("title") or source_path.stem).strip(),
                    topic=str((payload or {}).get("topic") or "").strip(),
                )
                response_payload["draft"] = draft
            except Exception as exc:
                self.logger.bind(tag=TAG).error(f"Knowledge draft generation failed: {exc}")
                response_payload["draft_error"] = str(exc)

            if document and document.get("document_id"):
                try:
                    service = self._rag_service()
                    job = service.create_index_job(document["document_id"])
                    task = asyncio.create_task(
                        asyncio.to_thread(
                            self._run_rag_index_job,
                            document["document_id"],
                            job["job_id"],
                        )
                    )
                    self._rag_tasks[job["job_id"]] = task
                    task.add_done_callback(
                        lambda _task, job_id=job["job_id"]: self._rag_tasks.pop(job_id, None)
                    )
                    response_payload["job"] = job
                except Exception as exc:
                    self.logger.bind(tag=TAG).error(f"Clinical RAG indexing failed: {exc}")
                    response_payload["rag_error"] = str(exc)

            if response_payload.get("draft_error") and response_payload.get("rag_error"):
                response_payload["ok"] = False
                return self._json_response(response_payload, status=500)
            return self._json_response(response_payload)
        except ValueError as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_knowledge_ingestion_drafts(self, request: web.Request) -> web.Response:
        try:
            drafts = self._knowledge_ingestion_service().list_drafts()
            return self._json_response({"ok": True, "drafts": drafts})
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_knowledge_ingestion_detail(self, request: web.Request) -> web.Response:
        draft_id = str(request.match_info.get("draft_id") or "").strip()
        if not draft_id:
            return self._json_response({"ok": False, "error": "missing draft_id"}, status=400)
        try:
            draft = self._knowledge_ingestion_service().get_draft(draft_id)
            if draft is None:
                return self._json_response({"ok": False, "error": "draft not found"}, status=404)
            return self._json_response({"ok": True, "draft": draft})
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_knowledge_ingestion_approve(self, request: web.Request) -> web.Response:
        draft_id = str(request.match_info.get("draft_id") or "").strip()
        if not draft_id:
            return self._json_response({"ok": False, "error": "missing draft_id"}, status=400)
        try:
            draft = self._knowledge_ingestion_service().approve_draft(draft_id)
            return self._json_response({"ok": True, "draft": draft})
        except ValueError as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_knowledge_ingestion_review(self, request: web.Request) -> web.Response:
        """Re-run LLM review for a multi-page Wiki draft."""
        draft_id = str(request.match_info.get("draft_id") or "").strip()
        if not draft_id:
            return self._json_response({"ok": False, "error": "missing draft_id"}, status=400)
        try:
            service = self._knowledge_ingestion_service()
            draft = service.get_draft(draft_id)
            if not draft:
                return self._json_response({"ok": False, "error": "draft not found"}, status=404)

            ingestion = service.config.get("knowledge_ingestion") or {}
            llm_options = service._knowledge_ingestion_llm_options(ingestion, stage="review")
            review = service._review_wiki_compiler_output(
                llm_options=llm_options,
                source_name=str(draft.get("source_name") or ""),
                title=str(draft.get("title") or draft.get("source_name") or ""),
                wiki_pages=draft.get("wiki_pages") or [],
                coverage_report=draft.get("coverage_report") or {},
                chunk_summaries=draft.get("chunk_summaries") or [],
            )

            draft_dir = service._draft_dir(draft_id)
            _write_json(draft_dir / "llm_review.json", review)
            manifest_path = draft_dir / "manifest.json"
            manifest = _read_json(manifest_path, {})
            manifest["status"] = "reviewed"
            manifest["review_status"] = review.get("overall_status") or "needs_human_review"
            manifest["updated_at"] = _utc_now()
            _write_json(manifest_path, manifest)

            return self._json_response({"ok": True, "review": review, "draft": service.get_draft(draft_id)})
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_knowledge_ingestion_regenerate_plan(self, request: web.Request) -> web.Response:
        """Re-run document profiler on a draft's source document."""
        draft_id = str(request.match_info.get("draft_id") or "").strip()
        if not draft_id:
            return self._json_response({"ok": False, "error": "missing draft_id"}, status=400)
        try:
            service = self._knowledge_ingestion_service()
            draft = service.get_draft(draft_id)
            if not draft:
                return self._json_response({"ok": False, "error": "draft not found"}, status=404)

            source_path_str = draft.get("source_path") or ""
            source_path = service.project_root / source_path_str
            if not source_path.exists():
                return self._json_response({"ok": False, "error": "source file not found"}, status=404)

            from core.clinical_nutrition.knowledge_ingestion import (
                _build_document_quality_report,
                extract_document_pages_for_ingestion,
            )

            pages = extract_document_pages_for_ingestion(source_path)
            document_quality = _build_document_quality_report(pages, source_path.name)
            result = service._run_document_profiler(
                pages=pages,
                source_path=source_path,
                document_id=draft_id,
                quality_report=document_quality,
            )

            # Save updated plan to draft directory
            draft_dir = service._draft_dir(draft_id)
            _write_json(draft_dir / "document_quality.json", document_quality)
            _write_json(draft_dir / "document_profile.json", result.get("document_profile") or {})
            _write_json(draft_dir / "ingestion_plan.json", result.get("ingestion_plan") or {})
            _write_json(draft_dir / "structured_extraction.json", result.get("extraction_results") or {})

            # Update manifest
            manifest_path = draft_dir / "manifest.json"
            manifest = _read_json(manifest_path, {})
            manifest["profiler_used"] = result.get("profiler_used", False)
            manifest["profiler_error"] = result.get("profiler_error", "")
            manifest["status"] = "profiled"
            manifest["document_quality_status"] = document_quality.get("quality_status", "ok")
            manifest["document_profile_type"] = (result.get("document_profile") or {}).get("document_type", "")
            manifest["structured_extraction_stats"] = (
                result.get("extraction_results", {}).get("stats") or {}
            )
            manifest["needs_review_count"] = len(
                result.get("extraction_results", {}).get("needs_review") or []
            )
            manifest["updated_at"] = _utc_now()
            _write_json(manifest_path, manifest)

            return self._json_response({
                "ok": True,
                "ingestion_plan": result.get("ingestion_plan"),
                "document_quality": document_quality,
                "document_profile": result.get("document_profile"),
                "extraction_stats": result.get("extraction_results", {}).get("stats"),
                "needs_review_count": len(
                    result.get("extraction_results", {}).get("needs_review") or []
                ),
                "profiler_used": result.get("profiler_used", False),
            })
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_knowledge_ingestion_extract_structured(self, request: web.Request) -> web.Response:
        """Re-run structured extraction on a draft's ingestion plan blocks."""
        draft_id = str(request.match_info.get("draft_id") or "").strip()
        if not draft_id:
            return self._json_response({"ok": False, "error": "missing draft_id"}, status=400)
        try:
            service = self._knowledge_ingestion_service()
            draft = service.get_draft(draft_id)
            if not draft:
                return self._json_response({"ok": False, "error": "draft not found"}, status=404)

            ingestion_plan = draft.get("ingestion_plan") or {}
            if not ingestion_plan.get("blocks"):
                return self._json_response(
                    {"ok": False, "error": "no ingestion plan found; run regenerate-plan first"},
                    status=400,
                )

            source_path_str = draft.get("source_path") or ""
            source_path = service.project_root / source_path_str
            if not source_path.exists():
                return self._json_response({"ok": False, "error": "source file not found"}, status=404)

            from core.clinical_nutrition.document_profiler import DocumentProfiler
            from core.clinical_nutrition.ingestion_schemas import IngestionPlan as IngestionPlanSchema
            from core.clinical_nutrition.knowledge_ingestion import extract_document_pages_for_ingestion

            pages = extract_document_pages_for_ingestion(source_path)

            ingestion = service.config.get("knowledge_ingestion") or {}
            profile_llm_options = service._knowledge_ingestion_llm_options(ingestion, stage="extract")
            if not profile_llm_options:
                return self._json_response(
                    {"ok": False, "error": "no LLM configured for profile stage"},
                    status=400,
                )

            def llm_caller(prompt: str, max_tokens: int) -> dict[str, Any]:
                return service._call_llm_chat(
                    base_url=profile_llm_options["base_url"],
                    api_key=profile_llm_options["api_key"],
                    models=profile_llm_options.get("models") or [profile_llm_options["model"]],
                    prompt=prompt,
                    timeout_seconds=profile_llm_options["timeout_seconds"],
                    max_tokens=max_tokens,
                    response_format_json=True,
                )

            plan = IngestionPlanSchema.model_validate(ingestion_plan)
            profiler = DocumentProfiler(
                llm_caller=llm_caller,
                source_name=draft.get("source_name") or "",
                document_id=draft_id,
            )
            extraction_results = profiler.extract_structured_blocks(plan, pages)

            extraction_data = {
                "extracted": extraction_results["extracted"],
                "needs_review": [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in extraction_results["needs_review"]
                ],
                "stats": extraction_results["stats"],
            }

            draft_dir = service._draft_dir(draft_id)
            _write_json(draft_dir / "structured_extraction.json", extraction_data)

            manifest_path = draft_dir / "manifest.json"
            manifest = _read_json(manifest_path, {})
            manifest["status"] = "extracted"
            manifest["structured_extraction_stats"] = extraction_data["stats"]
            manifest["needs_review_count"] = len(extraction_data["needs_review"])
            manifest["updated_at"] = _utc_now()
            _write_json(manifest_path, manifest)

            return self._json_response({
                "ok": True,
                "stats": extraction_data["stats"],
                "needs_review_count": len(extraction_data["needs_review"]),
            })
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_knowledge_ingestion_needs_review(self, request: web.Request) -> web.Response:
        """List needs_review items for a draft."""
        draft_id = str(request.match_info.get("draft_id") or "").strip()
        if not draft_id:
            return self._json_response({"ok": False, "error": "missing draft_id"}, status=400)
        try:
            service = self._knowledge_ingestion_service()
            draft = service.get_draft(draft_id)
            if not draft:
                return self._json_response({"ok": False, "error": "draft not found"}, status=404)

            extraction = draft.get("structured_extraction") or {}
            needs_review = extraction.get("needs_review") or []

            # Also query from DB if structured ingestion has been run
            db_items = []
            try:
                store = self._structured_knowledge_store()
                with store._connect() as db:
                    rows = db.execute(
                        "SELECT * FROM needs_review WHERE document_id = ? ORDER BY page_start",
                        (draft_id,),
                    ).fetchall()
                    db_items = [dict(row) for row in rows]
            except Exception:
                pass

            return self._json_response({
                "ok": True,
                "draft_needs_review": needs_review,
                "db_needs_review": db_items,
                "total": len(needs_review) + len(db_items),
            })
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_rules(self, request: web.Request) -> web.Response:
        rules_path = self.project_root / "knowledge_base" / "rules" / "clinical_safety_rules.json"
        payload = _read_json(rules_path, default={"rules": []})
        rules = []
        for item in payload.get("rules", []):
            rules.append(
                {
                    "rule_id": item.get("rule_id", ""),
                    "severity": item.get("severity", ""),
                    "category": item.get("category", ""),
                    "description": item.get("description", ""),
                    "action": (item.get("then") or {}).get("action", ""),
                    "message": (item.get("then") or {}).get("message", ""),
                    "recommendation": (item.get("then") or {}).get("recommendation", ""),
                    "evidence": item.get("evidence") or {},
                }
            )
        return self._json_response(
            {
                "ok": True,
                "version": payload.get("version", ""),
                "count": len(rules),
                "rules": rules,
            }
        )

    async def handle_food_search(self, request: web.Request) -> web.Response:
        query = str(request.query.get("q", "")).strip()
        limit = _int_query(request, "limit", 8, minimum=1, maximum=20)
        if not query:
            return self._json_response({"ok": True, "query": query, "foods": []})

        db_path = self._food_db_path()
        if not db_path.exists():
            return self._json_response(
                {"ok": False, "error": "food nutrition database not found"},
                status=404,
            )
        try:
            foods = self._search_foods(db_path, query, limit)
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)
        return self._json_response({"ok": True, "query": query, "foods": foods})

    async def handle_meal_analyze(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception as exc:
            return self._json_response(
                {"ok": False, "error": f"invalid json payload: {exc}"},
                status=400,
            )

        meal_text = str((payload or {}).get("meal_text") or "").strip()
        if not meal_text:
            return self._json_response({"ok": False, "error": "missing meal_text"}, status=400)

        user_id = normalize_device_user_id(
            str(
                (payload or {}).get("user_id")
                or request.query.get("user_id")
                or request.query.get("device_id")
                or ""
            ).strip()
        )
        should_record = bool((payload or {}).get("record", False))
        db_path = self._food_db_path()
        if not db_path.exists():
            return self._json_response(
                {"ok": False, "error": "food nutrition database not found"},
                status=404,
            )

        try:
            from plugins_func.functions.analyze_meal_nutrition import (
                build_meal_nutrition_payload,
            )

            analysis = build_meal_nutrition_payload(db_path, meal_text)
            profile = self._get_profile(user_id) if user_id else None
            nutrition_targets = estimate_daily_nutrition_targets(
                profile or {"scalars": {}, "items": []}
            )
            record = None
            if should_record and user_id and analysis.get("resolved_items"):
                store = HealthProfileStore(self._health_profile_db_path())
                record = store.record_nutrition_intake_sync(
                    user_id,
                    meal_text=meal_text,
                    meal_label=_guess_console_meal_label(meal_text),
                    totals=analysis.get("totals") or {},
                    items=analysis.get("resolved_items") or [],
                    source="clinical_console_meal_analysis",
                )
                profile = self._get_profile(user_id)
            return self._json_response(
                {
                    "ok": True,
                    "user_id": user_id,
                    "recorded": bool(record and record.get("inserted")),
                    "record": record,
                    "analysis": analysis,
                    "nutrition_targets": nutrition_targets,
                    "profile": profile,
                }
            )
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_model_config_get(self, request: web.Request) -> web.Response:
        try:
            effective = self._load_effective_config()
            custom = self._load_custom_config()
            return self._json_response(
                {
                    "ok": True,
                    "config_path": self._relative_path(self._custom_config_path()),
                    "restart_required_after_save": True,
                    "model_config": self._model_config_payload(effective, custom),
                }
            )
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_model_config_save(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception as exc:
            return self._json_response(
                {"ok": False, "error": f"invalid json payload: {exc}"},
                status=400,
            )

        try:
            custom = self._load_custom_config()
            changes = self._apply_model_config_update(custom, payload or {})
            self._write_custom_config(custom)
            self._refresh_runtime_config_cache(custom)
            return self._json_response(
                {
                    "ok": True,
                    "changes": changes,
                    "config_path": self._relative_path(self._custom_config_path()),
                    "restart_required": True,
                    "message": "模型配置已保存到 data/.config.yaml，重启 xiaozhi-esp32-server 后完全生效。",
                }
            )
        except ValueError as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_agent_settings_get(self, request: web.Request) -> web.Response:
        try:
            effective = self._load_effective_config()
            custom = self._load_custom_config()
            return self._json_response(
                {
                    "ok": True,
                    "config_path": self._relative_path(self._custom_config_path()),
                    "restart_required_after_save": True,
                    "agent_settings": self._agent_settings_payload(effective, custom),
                }
            )
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_agent_settings_save(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception as exc:
            return self._json_response(
                {"ok": False, "error": f"invalid json payload: {exc}"},
                status=400,
            )

        try:
            custom = self._load_custom_config()
            changes = self._apply_agent_settings_update(custom, payload or {})
            self._write_custom_config(custom)
            self._refresh_runtime_config_cache(custom)
            return self._json_response(
                {
                    "ok": True,
                    "changes": changes,
                    "config_path": self._relative_path(self._custom_config_path()),
                    "restart_required": True,
                    "message": "Agent settings saved to data/.config.yaml. Restart xiaozhi-esp32-server to fully apply them.",
                }
            )
        except ValueError as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_history_list(self, request: web.Request) -> web.Response:
        try:
            limit = _int_query(request, "limit", 50, minimum=1, maximum=200)
            user_id = self._user_id_from_request(request)
            sessions = self.history_store.list_sessions(limit=limit, user_id=user_id)
            return self._json_response(
                {
                    "ok": True,
                    "limit": limit,
                    "user_id": user_id,
                    "sessions": sessions,
                }
            )
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_history_detail(self, request: web.Request) -> web.Response:
        session_id = str(request.match_info.get("session_id") or "").strip()
        if not session_id:
            return self._json_response({"ok": False, "error": "missing session_id"}, status=400)
        try:
            session = self.history_store.get_session(session_id)
            if session is None:
                return self._json_response({"ok": False, "error": "session not found"}, status=404)
            return self._json_response({"ok": True, "session": session})
        except Exception as exc:
            return self._json_response({"ok": False, "error": str(exc)}, status=500)

    def _json_response(self, payload: dict[str, Any], status: int = 200) -> web.Response:
        response = web.Response(
            text=json.dumps(payload, ensure_ascii=False, default=str),
            status=status,
            content_type="application/json",
        )
        self._add_cors_headers(response)
        return response

    def _user_id_from_request(self, request: web.Request) -> str:
        raw = (
            request.match_info.get("user_id")
            or request.query.get("user_id")
            or request.query.get("device_id")
            or ""
        )
        return normalize_device_user_id(str(raw).strip())

    def _health_profile_db_path(self) -> Path:
        memory_config = self.config.get("Memory", {}).get("clinical_ltm", {})
        return self._resolve_project_path(
            memory_config.get("health_profile_sqlite_path"),
            "data/clinical_health_profile.db",
        )

    def _ltm_db_path(self) -> Path:
        memory_config = self.config.get("Memory", {}).get("clinical_ltm", {})
        return self._resolve_project_path(
            memory_config.get("sqlite_path"),
            "data/clinical_ltm.db",
        )

    def _powermem_db_path(self) -> Path:
        memory_config = self.config.get("Memory", {}).get("clinical_ltm", {})
        powermem_config = (
            memory_config.get("powermem", {})
            .get("vector_store", {})
            .get("config", {})
        )
        return self._resolve_project_path(
            powermem_config.get("database_path"),
            "data/clinical_ltm_powermem.db",
        )

    def _food_db_path(self) -> Path:
        plugin_config = self.config.get("plugins", {}).get("search_food_nutrition", {})
        return self._resolve_project_path(
            plugin_config.get("db_path"),
            "data/clinical_foods.db",
        )

    def _rag_db_path(self) -> Path:
        rag_config = self.config.get("clinical_rag") or {}
        plugin_config = self.config.get("plugins", {}).get("search_clinical_rag", {})
        return self._resolve_project_path(
            rag_config.get("db_path") or plugin_config.get("db_path"),
            "data/clinical_rag.db",
        )

    def _clinical_knowledge_db_path(self) -> Path:
        knowledge_config = self.config.get("clinical_knowledge") or {}
        plugin_config = self.config.get("plugins", {}).get("search_clinical_structured_knowledge", {})
        return self._resolve_project_path(
            knowledge_config.get("db_path") or plugin_config.get("db_path"),
            "data/clinical_knowledge.db",
        )

    def _wiki_root_path(self) -> Path:
        plugin_config = self.config.get("plugins", {}).get("search_from_llmwiki", {})
        return self._resolve_project_path(
            plugin_config.get("wiki_root"),
            "knowledge_base/llmwiki/clinical-nutrition",
        )

    def _custom_config_path(self) -> Path:
        return self.project_root / "data" / ".config.yaml"

    def _load_custom_config(self) -> dict[str, Any]:
        path = self._custom_config_path()
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            return {}
        return payload

    def _load_effective_config(self) -> dict[str, Any]:
        default_config = read_config(str(self.project_root / "config.yaml"))
        return merge_configs(default_config, self._load_custom_config())

    def _write_custom_config(self, payload: dict[str, Any]) -> None:
        path = self._custom_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                payload,
                handle,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )

    def _refresh_runtime_config_cache(self, custom: dict[str, Any]) -> None:
        default_config = read_config(str(self.project_root / "config.yaml"))
        self.config = merge_configs(default_config, custom)
        try:
            from core.utils.cache.config import CacheType
            from core.utils.cache.manager import cache_manager

            cache_manager.delete(CacheType.CONFIG, "main_config")
        except Exception:
            pass

    def _model_config_payload(
        self,
        effective: dict[str, Any],
        custom: dict[str, Any],
    ) -> dict[str, Any]:
        llm_modules = sorted((effective.get("LLM") or {}).keys())
        selected_llm = (
            (effective.get("selected_module") or {}).get("LLM")
            or (llm_modules[0] if llm_modules else "")
        )
        llm_config = (effective.get("LLM") or {}).get(selected_llm, {})
        endpoint_field = _endpoint_field(llm_config)

        memory_config = (effective.get("Memory") or {}).get("clinical_ltm", {})
        powermem_config = memory_config.get("powermem") or {}
        powermem_llm = powermem_config.get("llm") or {}
        powermem_llm_config = powermem_llm.get("config") or {}
        embedder = powermem_config.get("embedder") or {}
        embedder_config = embedder.get("config") or {}
        vector_store = powermem_config.get("vector_store") or {}
        vector_config = vector_store.get("config") or {}
        mem0_config = memory_config.get("mem0") or {}
        knowledge_ingestion = effective.get("knowledge_ingestion") or {}
        ingestion_llm = knowledge_ingestion.get("llm") or {}
        ingestion_tasks = knowledge_ingestion.get("tasks") or {}
        rag_config = effective.get("clinical_rag") or {}
        rag_embedding = rag_config.get("embedding") or {}

        return {
            "main_llm": {
                "module": selected_llm,
                "available_modules": llm_modules,
                "type": llm_config.get("type", ""),
                "model_name": llm_config.get("model_name", ""),
                "endpoint_field": endpoint_field,
                "endpoint_url": llm_config.get(endpoint_field, ""),
                "temperature": llm_config.get("temperature", ""),
                "max_tokens": llm_config.get("max_tokens", ""),
                "api_key_configured": _secret_configured(llm_config.get("api_key")),
                "overridden_in_custom": selected_llm
                in ((custom.get("LLM") or {}) if isinstance(custom.get("LLM"), dict) else {}),
            },
            "powermem_llm": {
                "provider": powermem_llm.get("provider", ""),
                "model": powermem_llm_config.get("model", ""),
                "openai_base_url": powermem_llm_config.get("openai_base_url", ""),
                "api_key_configured": _secret_configured(powermem_llm_config.get("api_key")),
            },
            "embedding": {
                "provider": embedder.get("provider", ""),
                "model": embedder_config.get("model", ""),
                "openai_base_url": embedder_config.get("openai_base_url", ""),
                "embedding_dims": embedder_config.get(
                    "embedding_dims",
                    memory_config.get("embedding_dimensions", ""),
                ),
                "api_key_configured": _secret_configured(embedder_config.get("api_key")),
            },
            "mem0": {
                "mode": mem0_config.get("mode", ""),
                "host": mem0_config.get("host", ""),
                "api_key_configured": _secret_configured(mem0_config.get("api_key")),
            },
            "vector_store": {
                "provider": vector_store.get("provider", ""),
                "database_path": vector_config.get("database_path", ""),
                "collection_name": vector_config.get("collection_name", ""),
                "embedding_model_dims": vector_config.get("embedding_model_dims", ""),
            },
            "intent": self._intent_config_payload(effective),
            "knowledge_ingestion": {
                "enabled": bool(knowledge_ingestion.get("enabled", False)),
                "provider": ingestion_llm.get("provider", "openai"),
                "model": ingestion_llm.get("model", ""),
                "openai_base_url": ingestion_llm.get("openai_base_url", ""),
                "api_key_configured": _secret_configured(ingestion_llm.get("api_key")),
                "target_wiki_root": knowledge_ingestion.get(
                    "target_wiki_root",
                    self._relative_path(self._wiki_root_path()),
                ),
                "generate_llmwiki": bool(ingestion_tasks.get("generate_llmwiki", True)),
                "draft_rules": bool(ingestion_tasks.get("draft_rules", True)),
                "extract_citations": bool(ingestion_tasks.get("extract_citations", True)),
            },
            "clinical_rag": {
                "enabled": bool(rag_config.get("enabled", True)),
                "db_path": rag_config.get("db_path", "data/clinical_rag.db"),
                "upload_root": rag_config.get(
                    "upload_root",
                    "data/knowledge_uploads/clinical-nutrition",
                ),
                "chunk_chars": rag_config.get("chunk_chars", 700),
                "chunk_overlap_chars": rag_config.get("chunk_overlap_chars", 120),
                "top_k": rag_config.get("top_k", 6),
                "bm25_candidates": rag_config.get("bm25_candidates", 40),
                "vector_candidates": rag_config.get("vector_candidates", 40),
                "embedding_provider": rag_embedding.get("provider", embedder.get("provider", "")),
                "embedding_model": rag_embedding.get(
                    "model",
                    embedder_config.get("model", ""),
                ),
                "embedding_openai_base_url": rag_embedding.get(
                    "openai_base_url",
                    embedder_config.get("openai_base_url", ""),
                ),
                "embedding_dimensions": rag_embedding.get(
                    "dimensions",
                    embedder_config.get(
                        "embedding_dims",
                        memory_config.get("embedding_dimensions", ""),
                    ),
                ),
                "api_key_configured": _secret_configured(
                    rag_embedding.get("api_key") or embedder_config.get("api_key")
                ),
            },
            "vision": self._runtime_module_payload(effective, "VLLM"),
            "asr": self._runtime_module_payload(effective, "ASR"),
            "tts": self._runtime_module_payload(effective, "TTS"),
            "usage_inventory": self._model_usage_inventory(effective),
        }

    def _runtime_module_payload(
        self,
        effective: dict[str, Any],
        section: str,
    ) -> dict[str, Any]:
        modules = sorted((effective.get(section) or {}).keys())
        module = (
            (effective.get("selected_module") or {}).get(section)
            or (modules[0] if modules else "")
        )
        config = ((effective.get(section) or {}).get(module) or {})
        endpoint_field = _first_existing_key(
            config,
            ["base_url", "url", "api_url", "ws_url", "openai_base_url"],
        )
        return {
            "section": section,
            "module": module,
            "available_modules": modules,
            "type": config.get("type", ""),
            "model_name": config.get("model_name", ""),
            "model": config.get("model", ""),
            "endpoint_field": endpoint_field,
            "endpoint_url": config.get(endpoint_field, "") if endpoint_field else "",
            "voice": config.get("voice", ""),
            "speaker": config.get("speaker", ""),
            "cluster": config.get("cluster", ""),
            "resource_id": config.get("resource_id", ""),
            "appid": config.get("appid", config.get("app_id", "")),
            "api_key_configured": _secret_configured(config.get("api_key")),
            "access_token_configured": _secret_configured(config.get("access_token")),
            "secret_key_configured": _secret_configured(config.get("secret_key")),
            "api_secret_configured": _secret_configured(config.get("api_secret")),
        }

    def _intent_config_payload(self, effective: dict[str, Any]) -> dict[str, Any]:
        intent_modules = sorted((effective.get("Intent") or {}).keys())
        selected_intent = (
            (effective.get("selected_module") or {}).get("Intent")
            or (intent_modules[0] if intent_modules else "")
        )
        intent_config = ((effective.get("Intent") or {}).get(selected_intent) or {})
        return {
            "module": selected_intent,
            "available_modules": intent_modules,
            "type": intent_config.get("type", ""),
            "dedicated_llm": ((effective.get("Intent") or {}).get("intent_llm") or {}).get("llm", ""),
            "available_llm_modules": sorted((effective.get("LLM") or {}).keys()),
            "uses_main_llm": selected_intent == "function_call",
        }

    def _model_usage_inventory(self, effective: dict[str, Any]) -> list[dict[str, Any]]:
        selected = effective.get("selected_module") or {}
        memory_type = selected.get("Memory", "")
        intent_type = selected.get("Intent", "")
        selected_memory_name = selected.get("Memory", "")
        selected_memory_config = ((effective.get("Memory") or {}).get(selected_memory_name) or {})
        memory_llm = selected_memory_config.get("llm") or selected.get("LLM", "")
        clinical_ltm = (effective.get("Memory") or {}).get("clinical_ltm") or {}
        powermem = clinical_ltm.get("powermem") or {}
        embedder = powermem.get("embedder") or {}
        rag_config = effective.get("clinical_rag") or {}
        rag_embedding = rag_config.get("embedding") or {}

        return [
            {
                "key": "main_dialogue_llm",
                "name": "主对话回复",
                "config_path": f"selected_module.LLM / LLM.{selected.get('LLM', '')}",
                "model": selected.get("LLM", ""),
                "needs_model": True,
                "status": "直接调用",
                "note": "负责生成小智和用户的主对话回复，也是大多数回答链路的核心模型。",
            },
            {
                "key": "function_call",
                "name": "工具调用决策",
                "config_path": "selected_module.Intent=function_call",
                "model": selected.get("LLM", ""),
                "needs_model": True,
                "status": "跟随主 LLM" if intent_type == "function_call" else "未启用",
                "note": "负责判断什么时候要调用营养查询、知识库、规则等工具；当前跟随主 LLM。",
            },
            {
                "key": "intent_llm",
                "name": "独立意图识别",
                "config_path": "Intent.intent_llm.llm",
                "model": ((effective.get("Intent") or {}).get("intent_llm") or {}).get("llm", ""),
                "needs_model": True,
                "status": "启用" if intent_type == "intent_llm" else "备用配置",
                "note": "负责先做意图分类，再把请求路由到合适的处理流程；仅在独立意图模式下使用。",
            },
            {
                "key": "clinical_ltm_extractor",
                "name": "长期记忆抽取",
                "config_path": "Memory.clinical_ltm.llm / fallback selected_module.LLM",
                "model": memory_llm,
                "needs_model": True,
                "status": "启用" if memory_type == "clinical_ltm" else "未启用",
                "note": "负责把对话内容提炼成事实、事件和长期习惯，写入长期记忆与健康画像。",
            },
            {
                "key": "clinical_ltm_summary",
                "name": "长期记忆语义总结",
                "config_path": "Memory.clinical_ltm.llm + summary_* / fallback selected_module.LLM",
                "model": memory_llm,
                "needs_model": True,
                "status": "启用" if memory_type == "clinical_ltm" else "未启用",
                "note": "负责把多次短期事件归纳成更稳定的长期规律和用户特征。",
            },
            {
                "key": "powermem_llm",
                "name": "PowerMem 文本模型",
                "config_path": "Memory.clinical_ltm.powermem.llm",
                "model": (powermem.get("llm") or {}).get("config", {}).get("model", ""),
                "needs_model": True,
                "status": "已配置" if _secret_configured((powermem.get("llm") or {}).get("config", {}).get("api_key")) else "缺 API Key",
                "note": "负责 PowerMem 检索层里的文本理解、记忆整理和检索增强。",
            },
            {
                "key": "embedding",
                "name": "向量检索模型",
                "config_path": "Memory.clinical_ltm.powermem.embedder",
                "model": (embedder.get("config") or {}).get("model", embedder.get("provider", "")),
                "needs_model": True,
                "status": "mock" if embedder.get("provider") == "mock" else "已配置",
                "note": "负责把文本转成向量，供长期记忆和知识检索做相似度召回。",
            },
            {
                "key": "clinical_rag_embedding",
                "name": "Clinical RAG 向量模型",
                "config_path": "clinical_rag.embedding",
                "model": rag_embedding.get("model", (embedder.get("config") or {}).get("model", "")),
                "needs_model": True,
                "status": "启用" if rag_config.get("enabled", True) else "未启用",
                "note": "负责把上传文档分块后写入本地向量索引；对话时由 search_clinical_rag 检索并返回带页码的证据片段。",
            },
            {
                "key": "clinical_safety",
                "name": "临床安全红线",
                "config_path": "knowledge_base/rules/clinical_safety_rules.json",
                "model": "",
                "needs_model": False,
                "status": "规则引擎",
                "note": "负责在模型回答前先做药食冲突、过敏、肾病等临床安全拦截。",
            },
            {
                "key": "vision_vllm",
                "name": "视觉理解",
                "config_path": f"selected_module.VLLM / VLLM.{selected.get('VLLM', '')}",
                "model": selected.get("VLLM", ""),
                "needs_model": True,
                "status": "按需调用",
                "note": "负责图片理解、拍照识别或视觉分析相关能力。",
            },
            {
                "key": "asr",
                "name": "语音识别",
                "config_path": f"selected_module.ASR / ASR.{selected.get('ASR', '')}",
                "model": selected.get("ASR", ""),
                "needs_model": True,
                "status": "启动时初始化",
                "note": "负责把小智采集到的语音转成文字，作为后续对话和规则判断的输入。",
            },
            {
                "key": "tts",
                "name": "语音合成",
                "config_path": f"selected_module.TTS / TTS.{selected.get('TTS', '')}",
                "model": selected.get("TTS", ""),
                "needs_model": True,
                "status": "启动时初始化",
                "note": "负责把最终回答转成语音，回放给小智机器人或前端语音界面。",
            },
        ]

    def _agent_settings_payload(
        self,
        effective: dict[str, Any],
        custom: dict[str, Any],
    ) -> dict[str, Any]:
        ui_config = (effective.get("clinical_console_ui") or {})
        selected = effective.get("selected_module") or {}
        tts_module = selected.get("TTS", "")
        tts_runtime = self._runtime_module_payload(effective, "TTS")
        tts_config = ((effective.get("TTS") or {}).get(tts_module) or {})
        asr_module = selected.get("ASR", "")
        asr_runtime = self._runtime_module_payload(effective, "ASR")
        asr_config = ((effective.get("ASR") or {}).get(asr_module) or {})
        memory_config = (effective.get("Memory") or {}).get("clinical_ltm", {})
        voice_controls = _extract_voice_controls(tts_config)
        silence_ms = _first_numeric_value(
            asr_config,
            ["max_sentence_silence", "sentence_silence_ms", "silence_duration_ms"],
        )
        return {
            "assistant_name": ui_config.get("assistant_name", "小智营养师"),
            "language": ui_config.get("language", "普通话"),
            "prompt": effective.get("prompt", ""),
            "prompt_template": effective.get("prompt_template", ""),
            "prompt_template_content": self._prompt_template_content(effective),
            "prompt_template_content_overridden": bool(
                str(custom.get("prompt_template_content") or "").strip()
            ),
            "memory_prompts": self._memory_prompt_settings_payload(memory_config),
            "voice": {
                "module": tts_runtime.get("module", ""),
                "available_modules": tts_runtime.get("available_modules", []),
                "voice": tts_runtime.get("voice", ""),
                "speaker": tts_runtime.get("speaker", ""),
                "cluster": tts_runtime.get("cluster", ""),
                "resource_id": tts_runtime.get("resource_id", ""),
                "controls": voice_controls,
            },
            "asr": {
                "module": asr_runtime.get("module", ""),
                "available_modules": asr_runtime.get("available_modules", []),
                "recognition_silence_ms": silence_ms,
                "recognition_speed_preset": _recognition_speed_preset(silence_ms),
            },
            "short_term_memory": {
                "enabled": bool(memory_config.get("short_term_summary_enabled", True)),
                "max_chars": int(memory_config.get("short_term_summary_max_chars", 2000)),
                "max_tokens": int(memory_config.get("short_term_summary_max_tokens", 1200)),
                "temperature": float(memory_config.get("short_term_summary_temperature", 0.2)),
                "recent_messages": int(memory_config.get("short_term_recent_messages", 8)),
                "compact_trigger_messages": int(
                    memory_config.get("short_term_compact_trigger_messages", 18)
                ),
            },
            "mcp": {
                "endpoint": effective.get("mcp_endpoint", ""),
            },
            "custom_overrides_present": bool(custom),
        }

    def _apply_agent_settings_update(
        self,
        custom: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, int]:
        effective = self._load_effective_config()
        changes = {
            "ui": 0,
            "prompt": 0,
            "voice": 0,
            "asr": 0,
            "short_term_memory": 0,
            "memory_prompts": 0,
            "mcp": 0,
        }

        ui_config = _ensure_dict(custom, "clinical_console_ui")
        changes["ui"] += _set_if_present(
            ui_config, "assistant_name", payload.get("assistant_name")
        )
        changes["ui"] += _set_if_present(ui_config, "language", payload.get("language"))
        changes["prompt"] += _set_if_present(custom, "prompt", payload.get("prompt"))
        changes["prompt"] += _set_if_present(
            custom,
            "prompt_template_content",
            payload.get("prompt_template_content"),
        )

        voice_payload = payload.get("voice") or {}
        if isinstance(voice_payload, dict):
            changes["voice"] += self._apply_voice_settings_update(
                custom, effective, voice_payload
            )

        asr_payload = payload.get("asr") or {}
        if isinstance(asr_payload, dict):
            changes["asr"] += self._apply_asr_settings_update(
                custom, effective, asr_payload
            )

        stm_payload = payload.get("short_term_memory") or {}
        if isinstance(stm_payload, dict):
            changes["short_term_memory"] += self._apply_short_term_settings_update(
                custom, stm_payload
            )

        memory_prompts_payload = payload.get("memory_prompts") or {}
        if isinstance(memory_prompts_payload, dict):
            changes["memory_prompts"] += self._apply_memory_prompt_settings_update(
                custom,
                memory_prompts_payload,
            )

        mcp_payload = payload.get("mcp") or {}
        if isinstance(mcp_payload, dict):
            changes["mcp"] += _set_if_present(custom, "mcp_endpoint", mcp_payload.get("endpoint"))
        return changes

    def _apply_voice_settings_update(
        self,
        custom: dict[str, Any],
        effective: dict[str, Any],
        payload: dict[str, Any],
    ) -> int:
        changes = self._apply_runtime_module_config(custom, "TTS", payload)
        module = str(payload.get("module") or "").strip() or (
            (effective.get("selected_module") or {}).get("TTS") or ""
        )
        if not module:
            return changes

        module_config = _ensure_dict(_ensure_dict(custom, "TTS"), module)
        effective_module = ((effective.get("TTS") or {}).get(module) or {})
        control_fields = _extract_voice_controls(effective_module)
        controls_payload = payload.get("controls") or {}
        if not isinstance(controls_payload, dict):
            return changes

        rate = _coerce_float(controls_payload.get("rate"))
        pitch = _coerce_float(controls_payload.get("pitch"))
        volume = _coerce_float(controls_payload.get("volume"))
        if rate is not None:
            changes += _set_number_if_present(module_config, control_fields["rate_field"], rate)
        if pitch is not None:
            changes += _set_number_if_present(module_config, control_fields["pitch_field"], pitch)
        if volume is not None:
            changes += _set_number_if_present(module_config, control_fields["volume_field"], volume)
        return changes

    def _apply_asr_settings_update(
        self,
        custom: dict[str, Any],
        effective: dict[str, Any],
        payload: dict[str, Any],
    ) -> int:
        changes = self._apply_runtime_module_config(custom, "ASR", payload)
        module = str(payload.get("module") or "").strip() or (
            (effective.get("selected_module") or {}).get("ASR") or ""
        )
        if not module:
            return changes
        module_config = _ensure_dict(_ensure_dict(custom, "ASR"), module)
        effective_module = ((effective.get("ASR") or {}).get(module) or {})
        silence_field = _first_existing_key(
            effective_module,
            ["max_sentence_silence", "sentence_silence_ms", "silence_duration_ms"],
        ) or "max_sentence_silence"
        silence_value = payload.get("recognition_silence_ms")
        if silence_value in (None, ""):
            preset = str(payload.get("recognition_speed_preset") or "").strip().lower()
            silence_value = _recognition_silence_from_preset(preset)
        silence_ms = _coerce_positive_int(silence_value)
        if silence_ms is not None:
            changes += _set_if_changed(module_config, silence_field, silence_ms)
        return changes

    def _apply_short_term_settings_update(
        self,
        custom: dict[str, Any],
        payload: dict[str, Any],
    ) -> int:
        clinical_ltm = _ensure_clinical_ltm(custom)
        changes = 0
        changes += _set_bool_if_present(
            clinical_ltm, "short_term_summary_enabled", payload.get("enabled")
        )
        max_chars = _coerce_positive_int(payload.get("max_chars"))
        if max_chars is not None:
            changes += _set_if_changed(clinical_ltm, "short_term_summary_max_chars", max_chars)
        max_tokens = _coerce_positive_int(payload.get("max_tokens"))
        if max_tokens is not None:
            changes += _set_if_changed(clinical_ltm, "short_term_summary_max_tokens", max_tokens)
        temperature = _coerce_float(payload.get("temperature"))
        if temperature is not None:
            changes += _set_if_changed(clinical_ltm, "short_term_summary_temperature", temperature)
        recent_messages = _coerce_positive_int(payload.get("recent_messages"))
        if recent_messages is not None:
            changes += _set_if_changed(
                clinical_ltm, "short_term_recent_messages", recent_messages
            )
        compact_trigger = _coerce_positive_int(payload.get("compact_trigger_messages"))
        if compact_trigger is not None:
            changes += _set_if_changed(
                clinical_ltm,
                "short_term_compact_trigger_messages",
                compact_trigger,
            )
        return changes

    def _memory_prompt_settings_payload(self, memory_config: dict[str, Any]) -> dict[str, Any]:
        prompts = memory_config.get("prompts") if isinstance(memory_config.get("prompts"), dict) else {}
        long_term_prompt = (
            str(prompts.get("long_term_extraction_system_prompt") or "").strip()
            or EXTRACTION_SYSTEM_PROMPT
        )
        short_term_prompt = (
            str(prompts.get("short_term_summary_system_prompt") or "").strip()
            or SHORT_TERM_SUMMARY_SYSTEM_PROMPT
        )
        return {
            "long_term_extraction_system_prompt": long_term_prompt,
            "short_term_summary_system_prompt": short_term_prompt,
            "long_term_extraction_overridden": bool(
                str(prompts.get("long_term_extraction_system_prompt") or "").strip()
            ),
            "short_term_summary_overridden": bool(
                str(prompts.get("short_term_summary_system_prompt") or "").strip()
            ),
        }

    def _prompt_template_content(self, effective: dict[str, Any]) -> str:
        inline_template = str(effective.get("prompt_template_content") or "").strip()
        if inline_template:
            return inline_template
        template_path = str(effective.get("prompt_template") or "agent-base-prompt.txt").strip()
        path = Path(template_path)
        if not path.is_absolute():
            path = self.project_root / path
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def _apply_memory_prompt_settings_update(
        self,
        custom: dict[str, Any],
        payload: dict[str, Any],
    ) -> int:
        clinical_ltm = _ensure_clinical_ltm(custom)
        prompts = _ensure_dict(clinical_ltm, "prompts")
        changes = 0
        changes += _set_if_present(
            prompts,
            "long_term_extraction_system_prompt",
            payload.get("long_term_extraction_system_prompt"),
        )
        changes += _set_if_present(
            prompts,
            "short_term_summary_system_prompt",
            payload.get("short_term_summary_system_prompt"),
        )
        return changes

    def _apply_model_config_update(
        self,
        custom: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, int]:
        changes = {
            "main_llm": 0,
            "powermem_llm": 0,
            "embedding": 0,
            "mem0": 0,
            "vector_store": 0,
            "intent": 0,
            "knowledge_ingestion": 0,
            "clinical_rag": 0,
            "vision": 0,
            "asr": 0,
            "tts": 0,
        }

        main_llm = payload.get("main_llm") or {}
        if isinstance(main_llm, dict):
            changes["main_llm"] = self._apply_main_llm_config(custom, main_llm)

        powermem_llm = payload.get("powermem_llm") or {}
        if isinstance(powermem_llm, dict):
            changes["powermem_llm"] = self._apply_powermem_llm_config(
                custom, powermem_llm
            )

        embedding = payload.get("embedding") or {}
        if isinstance(embedding, dict):
            changes["embedding"] = self._apply_embedding_config(custom, embedding)

        mem0 = payload.get("mem0") or {}
        if isinstance(mem0, dict):
            changes["mem0"] = self._apply_mem0_config(custom, mem0)

        vector_store = payload.get("vector_store") or {}
        if isinstance(vector_store, dict):
            changes["vector_store"] = self._apply_vector_store_config(
                custom, vector_store
            )

        intent = payload.get("intent") or {}
        if isinstance(intent, dict):
            changes["intent"] = self._apply_intent_config(custom, intent)

        knowledge_ingestion = payload.get("knowledge_ingestion") or {}
        if isinstance(knowledge_ingestion, dict):
            changes["knowledge_ingestion"] = self._apply_knowledge_ingestion_config(
                custom, knowledge_ingestion
            )

        clinical_rag = payload.get("clinical_rag") or {}
        if isinstance(clinical_rag, dict):
            changes["clinical_rag"] = self._apply_clinical_rag_config(
                custom, clinical_rag
            )

        for payload_key, section in [("vision", "VLLM"), ("asr", "ASR"), ("tts", "TTS")]:
            runtime_payload = payload.get(payload_key) or {}
            if isinstance(runtime_payload, dict):
                changes[payload_key] = self._apply_runtime_module_config(
                    custom,
                    section,
                    runtime_payload,
                )
        return changes

    def _apply_main_llm_config(
        self,
        custom: dict[str, Any],
        payload: dict[str, Any],
    ) -> int:
        effective = self._load_effective_config()
        known_modules = set((effective.get("LLM") or {}).keys())
        module = str(payload.get("module") or "").strip()
        if not module:
            return 0
        if known_modules and module not in known_modules:
            raise ValueError(f"unknown LLM module: {module}")

        changes = 0
        selected_module = _ensure_dict(custom, "selected_module")
        if selected_module.get("LLM") != module:
            selected_module["LLM"] = module
            changes += 1

        llm_root = _ensure_dict(custom, "LLM")
        llm_config = _ensure_dict(llm_root, module)
        effective_llm = ((effective.get("LLM") or {}).get(module) or {})
        if "type" not in llm_config and effective_llm.get("type"):
            llm_config["type"] = effective_llm["type"]

        changes += _set_if_present(llm_config, "model_name", payload.get("model_name"))
        endpoint_field = str(payload.get("endpoint_field") or _endpoint_field(effective_llm))
        if endpoint_field not in {"base_url", "url"}:
            endpoint_field = "base_url"
        changes += _set_if_present(llm_config, endpoint_field, payload.get("endpoint_url"))
        changes += _set_optional_number(llm_config, "temperature", payload.get("temperature"))
        changes += _set_optional_int(llm_config, "max_tokens", payload.get("max_tokens"))
        changes += _set_secret_if_present(llm_config, "api_key", payload.get("api_key"))
        return changes

    def _apply_powermem_llm_config(
        self,
        custom: dict[str, Any],
        payload: dict[str, Any],
    ) -> int:
        powermem_llm = _ensure_powermem_section(custom, "llm")
        changes = 0
        changes += _set_if_present(powermem_llm, "provider", payload.get("provider"))
        config = _ensure_dict(powermem_llm, "config")
        changes += _set_if_present(config, "model", payload.get("model"))
        changes += _set_if_present(
            config, "openai_base_url", payload.get("openai_base_url")
        )
        changes += _set_secret_if_present(config, "api_key", payload.get("api_key"))
        return changes

    def _apply_embedding_config(
        self,
        custom: dict[str, Any],
        payload: dict[str, Any],
    ) -> int:
        clinical_ltm = _ensure_clinical_ltm(custom)
        embedder = _ensure_powermem_section(custom, "embedder")
        changes = 0
        changes += _set_if_present(embedder, "provider", payload.get("provider"))
        config = _ensure_dict(embedder, "config")
        changes += _set_if_present(config, "model", payload.get("model"))
        changes += _set_if_present(
            config, "openai_base_url", payload.get("openai_base_url")
        )
        changes += _set_secret_if_present(config, "api_key", payload.get("api_key"))
        dims = _coerce_positive_int(payload.get("embedding_dims"))
        if dims is not None:
            if config.get("embedding_dims") != dims:
                config["embedding_dims"] = dims
                changes += 1
            if clinical_ltm.get("embedding_dimensions") != dims:
                clinical_ltm["embedding_dimensions"] = dims
                changes += 1
        return changes

    def _apply_mem0_config(
        self,
        custom: dict[str, Any],
        payload: dict[str, Any],
    ) -> int:
        mem0 = _ensure_dict(_ensure_clinical_ltm(custom), "mem0")
        changes = 0
        changes += _set_if_present(mem0, "mode", payload.get("mode"))
        changes += _set_if_present(mem0, "host", payload.get("host"))
        changes += _set_secret_if_present(mem0, "api_key", payload.get("api_key"))
        return changes

    def _apply_vector_store_config(
        self,
        custom: dict[str, Any],
        payload: dict[str, Any],
    ) -> int:
        vector_store = _ensure_powermem_section(custom, "vector_store")
        changes = 0
        changes += _set_if_present(vector_store, "provider", payload.get("provider"))
        config = _ensure_dict(vector_store, "config")
        changes += _set_if_present(config, "database_path", payload.get("database_path"))
        changes += _set_if_present(config, "collection_name", payload.get("collection_name"))
        dims = _coerce_positive_int(payload.get("embedding_model_dims"))
        if dims is not None and config.get("embedding_model_dims") != dims:
            config["embedding_model_dims"] = dims
            changes += 1
        return changes

    def _apply_intent_config(
        self,
        custom: dict[str, Any],
        payload: dict[str, Any],
    ) -> int:
        effective = self._load_effective_config()
        known_modules = set((effective.get("Intent") or {}).keys())
        module = str(payload.get("module") or "").strip()
        changes = 0
        if module:
            if known_modules and module not in known_modules:
                raise ValueError(f"unknown Intent module: {module}")
            selected_module = _ensure_dict(custom, "selected_module")
            if selected_module.get("Intent") != module:
                selected_module["Intent"] = module
                changes += 1

        dedicated_llm = str(payload.get("dedicated_llm") or "").strip()
        if dedicated_llm:
            known_llm_modules = set((effective.get("LLM") or {}).keys())
            if known_llm_modules and dedicated_llm not in known_llm_modules:
                raise ValueError(f"unknown intent LLM module: {dedicated_llm}")
            intent_root = _ensure_dict(custom, "Intent")
            intent_llm = _ensure_dict(intent_root, "intent_llm")
            if intent_llm.get("llm") != dedicated_llm:
                intent_llm["llm"] = dedicated_llm
                changes += 1
        return changes

    def _apply_clinical_rag_config(
        self,
        custom: dict[str, Any],
        payload: dict[str, Any],
    ) -> int:
        rag = _ensure_dict(custom, "clinical_rag")
        embedding = _ensure_dict(rag, "embedding")
        changes = 0
        changes += _set_bool_if_present(rag, "enabled", payload.get("enabled"))
        changes += _set_if_present(rag, "db_path", payload.get("db_path"))
        changes += _set_if_present(rag, "upload_root", payload.get("upload_root"))
        changes += _set_optional_int(rag, "chunk_chars", payload.get("chunk_chars"))
        changes += _set_optional_int(
            rag,
            "chunk_overlap_chars",
            payload.get("chunk_overlap_chars"),
        )
        changes += _set_optional_int(rag, "top_k", payload.get("top_k"))
        changes += _set_optional_int(rag, "bm25_candidates", payload.get("bm25_candidates"))
        changes += _set_optional_int(
            rag,
            "vector_candidates",
            payload.get("vector_candidates"),
        )
        changes += _set_if_present(
            embedding,
            "provider",
            payload.get("embedding_provider"),
        )
        changes += _set_if_present(
            embedding,
            "model",
            payload.get("embedding_model"),
        )
        changes += _set_if_present(
            embedding,
            "openai_base_url",
            payload.get("embedding_openai_base_url"),
        )
        changes += _set_optional_int(
            embedding,
            "dimensions",
            payload.get("embedding_dimensions"),
        )
        changes += _set_secret_if_present(embedding, "api_key", payload.get("api_key"))
        return changes

    def _apply_knowledge_ingestion_config(
        self,
        custom: dict[str, Any],
        payload: dict[str, Any],
    ) -> int:
        ingestion = _ensure_dict(custom, "knowledge_ingestion")
        llm = _ensure_dict(ingestion, "llm")
        tasks = _ensure_dict(ingestion, "tasks")
        changes = 0

        changes += _set_bool_if_present(ingestion, "enabled", payload.get("enabled"))
        changes += _set_if_present(
            ingestion,
            "target_wiki_root",
            payload.get("target_wiki_root"),
        )
        changes += _set_if_present(llm, "provider", payload.get("provider"))
        changes += _set_if_present(llm, "model", payload.get("model"))
        changes += _set_if_present(llm, "openai_base_url", payload.get("openai_base_url"))
        changes += _set_secret_if_present(llm, "api_key", payload.get("api_key"))
        changes += _set_bool_if_present(
            tasks,
            "generate_llmwiki",
            payload.get("generate_llmwiki"),
        )
        changes += _set_bool_if_present(tasks, "draft_rules", payload.get("draft_rules"))
        changes += _set_bool_if_present(
            tasks,
            "extract_citations",
            payload.get("extract_citations"),
        )
        return changes

    def _apply_runtime_module_config(
        self,
        custom: dict[str, Any],
        section: str,
        payload: dict[str, Any],
    ) -> int:
        effective = self._load_effective_config()
        known_modules = set((effective.get(section) or {}).keys())
        module = str(payload.get("module") or "").strip()
        if not module:
            return 0
        if known_modules and module not in known_modules:
            raise ValueError(f"unknown {section} module: {module}")

        changes = 0
        selected_module = _ensure_dict(custom, "selected_module")
        if selected_module.get(section) != module:
            selected_module[section] = module
            changes += 1

        section_root = _ensure_dict(custom, section)
        module_config = _ensure_dict(section_root, module)
        effective_module = ((effective.get(section) or {}).get(module) or {})
        if "type" not in module_config and effective_module.get("type"):
            module_config["type"] = effective_module["type"]

        changes += _set_if_present(module_config, "model_name", payload.get("model_name"))
        changes += _set_if_present(module_config, "model", payload.get("model"))
        endpoint_field = str(payload.get("endpoint_field") or "").strip()
        if endpoint_field:
            changes += _set_if_present(
                module_config,
                endpoint_field,
                payload.get("endpoint_url"),
            )
        changes += _set_if_present(module_config, "voice", payload.get("voice"))
        changes += _set_if_present(module_config, "speaker", payload.get("speaker"))
        changes += _set_if_present(module_config, "cluster", payload.get("cluster"))
        changes += _set_if_present(module_config, "resource_id", payload.get("resource_id"))
        appid_key = "app_id" if "app_id" in effective_module else "appid"
        changes += _set_if_present(module_config, appid_key, payload.get("appid"))
        changes += _set_secret_if_present(module_config, "api_key", payload.get("api_key"))
        changes += _set_secret_if_present(
            module_config,
            "access_token",
            payload.get("access_token"),
        )
        changes += _set_secret_if_present(
            module_config,
            "secret_key",
            payload.get("secret_key"),
        )
        changes += _set_secret_if_present(
            module_config,
            "api_secret",
            payload.get("api_secret"),
        )
        return changes

    def _resolve_project_path(self, configured: Any, default: str) -> Path:
        raw = str(configured or default)
        path = Path(raw)
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    def _knowledge_ingestion_service(self) -> KnowledgeIngestionService:
        return KnowledgeIngestionService(
            project_root=self.project_root,
            config=self._load_effective_config(),
            logger=self.logger,
        )

    def _rag_service(self) -> ClinicalRAGService:
        return ClinicalRAGService(
            project_root=self.project_root,
            config=self._load_effective_config(),
            logger=self.logger,
        )

    def _structured_knowledge_store(self) -> StructuredKnowledgeStore:
        return StructuredKnowledgeStore(
            project_root=self.project_root,
            config=self._load_effective_config(),
        )

    def _run_structured_knowledge_review(self, document_id: str) -> dict[str, Any]:
        store = self._structured_knowledge_store()
        summary = store.document_review_summary(document_id)
        if not any(summary.get("counts", {}).values()):
            raise ValueError("this document has no structured extraction records to review")

        effective = self._load_effective_config()
        ingestion = effective.get("knowledge_ingestion") or {}
        service = self._knowledge_ingestion_service()
        llm_options = service._knowledge_ingestion_llm_options(ingestion, stage="review")
        if not llm_options:
            review_payload = {
                "llm_used": False,
                "overall_status": "needs_human_review",
                "approved_to_use": False,
                "confidence": 0,
                "issues": ["知识入库 LLM 未配置，无法自动复核。"],
                "recommendations": ["请先在模型设置里配置文档入库生成模型，或直接人工标记通过。"],
            }
            return store.set_document_review(
                document_id,
                review_status="llm_review_unavailable",
                review_method="none",
                review_summary=json.dumps(review_payload, ensure_ascii=False),
            )

        prompt = _structured_review_prompt(summary)
        try:
            review_payload = service._call_llm_json(
                base_url=llm_options["base_url"],
                api_key=llm_options["api_key"],
                model=llm_options["model"],
                models=llm_options.get("models"),
                prompt=prompt,
                timeout_seconds=llm_options["timeout_seconds"],
                max_tokens=llm_options["max_tokens"],
            )
        except Exception as exc:
            review_payload = {
                "llm_used": False,
                "overall_status": "review_failed",
                "approved_to_use": False,
                "confidence": 0,
                "issues": [str(exc)],
                "recommendations": ["请稍后重试 LLM 复核，或人工抽查后标记通过。"],
            }

        if not review_payload:
            review_payload = {
                "llm_used": False,
                "overall_status": "review_failed",
                "approved_to_use": False,
                "confidence": 0,
                "issues": ["LLM 没有返回可解析 JSON。"],
                "recommendations": ["请重试复核，或人工抽查后标记通过。"],
            }
        else:
            review_payload["llm_used"] = True

        approved_to_use = bool(review_payload.get("approved_to_use"))
        status = "llm_review_passed" if approved_to_use else "needs_review"
        return store.set_document_review(
            document_id,
            review_status=status,
            review_method=f"llm:{review_payload.get('_llm_model_used') or llm_options['model']}",
            review_summary=json.dumps(review_payload, ensure_ascii=False),
        )

    def _run_rag_index_job(self, document_id: str, job_id: str) -> None:
        ClinicalRAGService(
            project_root=self.project_root,
            config=self._load_effective_config(),
            logger=self.logger,
        ).index_document(document_id, job_id=job_id)

    def _resolve_ingestion_source(self, payload: dict[str, Any]) -> Path:
        raw = str(
            payload.get("relative_path")
            or payload.get("source_path")
            or payload.get("stored_name")
            or ""
        ).strip()
        if not raw:
            raise ValueError("missing source path")
        path = Path(raw)
        if not path.is_absolute():
            path = self.project_root / path
        path = path.resolve()
        try:
            path.relative_to(self.upload_root.resolve())
        except ValueError:
            try:
                path.relative_to(self.project_root.resolve())
            except ValueError as exc:
                raise ValueError("source path is outside project root") from exc
        if not path.exists() or not path.is_file():
            raise ValueError("source file not found")
        return path

    def _relative_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.project_root)).replace("\\", "/")
        except ValueError:
            return str(path)

    def _database_info(self, path: Path, tables: list[str]) -> dict[str, Any]:
        info: dict[str, Any] = {
            "path": self._relative_path(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "updated_at": _mtime(path),
            "tables": {},
        }
        if not path.exists():
            return info

        for table in tables:
            count = _sqlite_count(path, table)
            info["tables"][table] = count
        return info

    def _count_wiki_pages(self, root: Path) -> int:
        if not root.exists():
            return 0
        count = 0
        for path in root.rglob("*.md"):
            parts = {part.lower() for part in path.parts}
            if {"raw", "templates"} & parts:
                continue
            if path.name.lower() == "readme.md":
                continue
            count += 1
        return count

    def _rag_summary(self) -> dict[str, int]:
        db_path = self._rag_db_path()
        if not db_path.exists():
            return {
                "documents": 0,
                "indexed_documents": 0,
                "chunks": 0,
                "embeddings": 0,
            }
        return {
            "documents": _sqlite_count(db_path, "rag_documents"),
            "indexed_documents": _sqlite_count_where(db_path, "rag_documents", "status = 'indexed'"),
            "chunks": _sqlite_count(db_path, "rag_chunks"),
            "embeddings": _sqlite_count(db_path, "rag_embeddings"),
        }

    def _clinical_knowledge_summary(self) -> dict[str, int]:
        try:
            return StructuredKnowledgeStore(
                project_root=self.project_root,
                config=self._load_effective_config(),
            ).summary()
        except Exception:
            return {
                "documents": 0,
                "tables": 0,
                "table_rows": 0,
                "recipe_plans": 0,
                "therapeutic_recipes": 0,
                "activity_mets": 0,
            }

    def _structured_document_review_map(self) -> dict[str, dict[str, Any]]:
        path = self._clinical_knowledge_db_path()
        if not path.exists() or not _sqlite_table_exists(path, "source_documents"):
            return {}
        try:
            with sqlite3.connect(path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT document_id, review_status, review_method, review_summary,
                           reviewed_at, updated_at
                    FROM source_documents
                    """
                ).fetchall()
            result: dict[str, dict[str, Any]] = {}
            for row in rows:
                result[str(row["document_id"])] = {
                    "review_status": row["review_status"] or "auto_extracted",
                    "review_method": row["review_method"] or "",
                    "review_summary": row["review_summary"] or "",
                    "review_payload": _parse_json_maybe(row["review_summary"]),
                    "reviewed_at": row["reviewed_at"] or "",
                    "updated_at": row["updated_at"] or "",
                }
            return result
        except Exception:
            return {}

    def _rules_summary(self, rules_path: Path) -> dict[str, Any]:
        payload = _read_json(rules_path, default={"rules": []})
        rules = payload.get("rules", [])
        by_severity: dict[str, int] = {}
        by_category: dict[str, int] = {}
        for rule in rules:
            severity = str(rule.get("severity") or "unknown")
            category = str(rule.get("category") or "unknown")
            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_category[category] = by_category.get(category, 0) + 1
        return {
            "path": self._relative_path(rules_path),
            "exists": rules_path.exists(),
            "version": payload.get("version", ""),
            "count": len(rules),
            "by_severity": by_severity,
            "by_category": by_category,
        }

    def _list_users(self, limit: int = 50) -> list[dict[str, Any]]:
        users: dict[str, dict[str, Any]] = {}
        self._merge_users_from_health_profile(users)
        self._merge_users_from_memory(users)
        ordered = sorted(
            users.values(),
            key=lambda item: item.get("updated_at") or "",
            reverse=True,
        )
        return ordered[:limit]

    def _merge_users_from_health_profile(self, users: dict[str, dict[str, Any]]) -> None:
        path = self._health_profile_db_path()
        if not path.exists() or not _sqlite_table_exists(path, "health_profiles"):
            return
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT hp.user_id, hp.updated_at,
                       COUNT(hpi.item_id) AS item_count
                FROM health_profiles hp
                LEFT JOIN health_profile_items hpi ON hpi.user_id = hp.user_id
                GROUP BY hp.user_id, hp.updated_at
                ORDER BY hp.updated_at DESC
                LIMIT 200
                """
            ).fetchall()
        for row in rows:
            user = users.setdefault(
                row["user_id"],
                {"user_id": row["user_id"], "sources": [], "updated_at": ""},
            )
            user["health_profile_items"] = int(row["item_count"] or 0)
            user["sources"].append("health_profile")
            user["updated_at"] = max(user.get("updated_at") or "", row["updated_at"] or "")
        if not _sqlite_table_exists(path, "blood_glucose_readings"):
            return
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            glucose_rows = conn.execute(
                """
                SELECT user_id, MAX(measured_at) AS updated_at, COUNT(*) AS item_count
                FROM blood_glucose_readings
                GROUP BY user_id
                LIMIT 200
                """
            ).fetchall()
        for row in glucose_rows:
            user = users.setdefault(
                row["user_id"],
                {"user_id": row["user_id"], "sources": [], "updated_at": ""},
            )
            user["blood_glucose_readings"] = int(row["item_count"] or 0)
            user["sources"].append("blood_glucose")
            user["updated_at"] = max(user.get("updated_at") or "", row["updated_at"] or "")

    def _merge_users_from_memory(self, users: dict[str, dict[str, Any]]) -> None:
        path = self._ltm_db_path()
        if not path.exists():
            return
        queries = [
            (
                "ltm_memory_items",
                """
                SELECT user_id, MAX(updated_at) AS updated_at, COUNT(*) AS item_count
                FROM ltm_memory_items
                GROUP BY user_id
                LIMIT 200
                """,
                "memory_items",
                "long_term_memory",
            ),
            (
                "ltm_working_memory",
                """
                SELECT user_id, MAX(created_at) AS updated_at, COUNT(*) AS item_count
                FROM ltm_working_memory
                GROUP BY user_id
                LIMIT 200
                """,
                "working_turns",
                "working_memory",
            ),
            (
                "ltm_short_term_summary",
                """
                SELECT user_id, MAX(updated_at) AS updated_at, COUNT(*) AS item_count
                FROM ltm_short_term_summary
                GROUP BY user_id
                LIMIT 200
                """,
                "short_term_summaries",
                "short_term_summary",
            ),
        ]
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            for table, sql, count_key, source_name in queries:
                if not _sqlite_table_exists(path, table):
                    continue
                rows = conn.execute(sql).fetchall()
                for row in rows:
                    user = users.setdefault(
                        row["user_id"],
                        {"user_id": row["user_id"], "sources": [], "updated_at": ""},
                    )
                    user[count_key] = int(row["item_count"] or 0)
                    user["sources"].append(source_name)
                    user["updated_at"] = max(
                        user.get("updated_at") or "",
                        row["updated_at"] or "",
                    )

    def _get_profile(self, user_id: str) -> dict[str, Any]:
        path = self._health_profile_db_path()
        empty = {
            "user_id": user_id,
            "exists": False,
            "scalars": {},
            "items": [],
            "glucose_readings": [],
            "glucose_analysis": analyze_blood_glucose_readings([]),
            "nutrition_targets": estimate_daily_nutrition_targets({"scalars": {}, "items": []}),
            "nutrition_intake_series": [],
            "review_items": [],
        }
        if not path.exists() or not _sqlite_table_exists(path, "health_profiles"):
            return empty

        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            profile = conn.execute(
                "SELECT * FROM health_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if profile is None:
                return empty
            items = conn.execute(
                """
                SELECT category, name, status, value_json, source, evidence,
                       observed_at, updated_at, confidence, notes
                FROM health_profile_items
                WHERE user_id = ? AND status != 'deleted'
                ORDER BY category, updated_at DESC, name
                """,
                (user_id,),
            ).fetchall()
            glucose_rows = []
            if _sqlite_table_exists(path, "blood_glucose_readings"):
                glucose_rows = conn.execute(
                    """
                    SELECT reading_id, user_id, measured_at, reported_at, value_mmol_l,
                           measurement_type, meal_context, time_context, source, evidence,
                           confidence, notes, created_at
                    FROM blood_glucose_readings
                    WHERE user_id = ?
                    ORDER BY measured_at DESC, created_at DESC
                    LIMIT 60
                    """,
                    (user_id,),
                ).fetchall()

        scalar_fields = [
            "age_years",
            "sex",
            "height_cm",
            "weight_kg",
            "bmi",
            "activity_level",
            "nutrition_goal",
            "target_energy_kcal",
            "target_carbohydrate_g_per_meal",
            "target_protein_g_per_day",
            "target_fat_g_per_day",
            "notes",
        ]
        scalars = {
            field: profile[field]
            for field in scalar_fields
            if field in profile.keys() and profile[field] not in (None, "")
        }
        glucose_readings = [
            {
                "reading_id": row["reading_id"],
                "user_id": row["user_id"],
                "measured_at": row["measured_at"],
                "reported_at": row["reported_at"],
                "value_mmol_l": row["value_mmol_l"],
                "measurement_type": row["measurement_type"],
                "meal_context": row["meal_context"],
                "time_context": row["time_context"],
                "source": row["source"],
                "evidence": row["evidence"],
                "confidence": row["confidence"],
                "notes": row["notes"],
                "created_at": row["created_at"],
            }
            for row in glucose_rows
        ]
        result = {
            "user_id": user_id,
            "exists": True,
            "created_at": profile["created_at"],
            "updated_at": profile["updated_at"],
            "scalars": scalars,
            "items": [
                {
                    "category": row["category"],
                    "name": row["name"],
                    "status": row["status"],
                    "value": _loads_json(row["value_json"], {}),
                    "source": row["source"],
                    "evidence": row["evidence"],
                    "observed_at": row["observed_at"],
                    "updated_at": row["updated_at"],
                    "confidence": row["confidence"],
                    "notes": row["notes"],
                }
                for row in items
            ],
            "glucose_readings": glucose_readings,
            "glucose_analysis": analyze_blood_glucose_readings(glucose_readings),
            "review_items": self._get_profile_review_items(user_id, status="pending"),
        }
        store = HealthProfileStore(path)
        result["nutrition_intake_series"] = store.get_nutrition_intake_series_sync(user_id, days=30)
        result["nutrition_targets"] = estimate_daily_nutrition_targets(result)
        return result

    def _get_profile_review_items(self, user_id: str, status: str = "pending") -> list[dict[str, Any]]:
        path = self._health_profile_db_path()
        if not path.exists() or not _sqlite_table_exists(path, "health_profile_review_items"):
            return []
        store = HealthProfileStore(path)
        return store.list_review_items_sync(user_id, status=status)

    def _get_memory(self, user_id: str, limit: int) -> dict[str, Any]:
        path = self._ltm_db_path()
        result: dict[str, Any] = {
            "db_path": self._relative_path(path),
            "exists": path.exists(),
            "short_term_summary": None,
            "structured": [],
            "working": [],
        }
        if not path.exists():
            return result

        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            if _sqlite_table_exists(path, "ltm_short_term_summary"):
                summary_row = conn.execute(
                    """
                    SELECT summary, source_session_id, source_turn_count,
                           max_chars, created_at, updated_at, metadata_json
                    FROM ltm_short_term_summary
                    WHERE user_id = ?
                    """,
                    (user_id,),
                ).fetchone()
                if summary_row:
                    result["short_term_summary"] = {
                        "summary": summary_row["summary"],
                        "source_session_id": summary_row["source_session_id"],
                        "source_turn_count": summary_row["source_turn_count"],
                        "max_chars": summary_row["max_chars"],
                        "created_at": summary_row["created_at"],
                        "updated_at": summary_row["updated_at"],
                        "metadata": _loads_json(summary_row["metadata_json"], {}),
                    }

            if _sqlite_table_exists(path, "ltm_memory_items"):
                structured_rows = conn.execute(
                    """
                    SELECT memory_id, layer, entity, attribute, value, content,
                           source, observed_at, created_at, updated_at,
                           importance, weight, locked, evidence_json,
                           tags_json, metadata_json
                    FROM ltm_memory_items
                    WHERE user_id = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (user_id, limit),
                ).fetchall()
                result["structured"] = [
                    {
                        "memory_id": row["memory_id"],
                        "layer": row["layer"],
                        "entity": row["entity"],
                        "attribute": row["attribute"],
                        "value": row["value"],
                        "content": row["content"],
                        "source": row["source"],
                        "observed_at": row["observed_at"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "importance": row["importance"],
                        "weight": row["weight"],
                        "locked": bool(row["locked"]),
                        "evidence": _loads_json(row["evidence_json"], []),
                        "tags": _loads_json(row["tags_json"], []),
                        "metadata": _loads_json(row["metadata_json"], {}),
                    }
                    for row in structured_rows
                ]

            if _sqlite_table_exists(path, "ltm_working_memory"):
                working_rows = conn.execute(
                    """
                    SELECT turn_id, session_id, role, content, created_at, metadata_json
                    FROM ltm_working_memory
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (user_id, min(limit, 80)),
                ).fetchall()
                result["working"] = [
                    {
                        "turn_id": row["turn_id"],
                        "session_id": row["session_id"],
                        "role": row["role"],
                        "content": row["content"],
                        "created_at": row["created_at"],
                        "metadata": _loads_json(row["metadata_json"], {}),
                    }
                    for row in working_rows
                ]
        return result

    def _list_uploaded_files(self) -> list[dict[str, Any]]:
        if not self.upload_root.exists():
            return []
        manifest = self._read_upload_manifest()
        files = []
        for path in self.upload_root.rglob("*"):
            if not path.is_file() or path.name == self.upload_manifest.name:
                continue
            record = dict(manifest.get(path.name, {}))
            record.update(
                {
                    "stored_name": path.name,
                    "relative_path": self._relative_path(path),
                    "size_bytes": path.stat().st_size,
                    "updated_at": _mtime(path),
                    "suffix": path.suffix.lower(),
                }
            )
            files.append(record)
        files.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return files

    def _read_upload_manifest(self) -> dict[str, dict[str, Any]]:
        if not self.upload_manifest.exists():
            return {}
        records: dict[str, dict[str, Any]] = {}
        try:
            with self.upload_manifest.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if item.get("stored_name"):
                        records[item["stored_name"]] = item
        except (OSError, json.JSONDecodeError):
            return records
        return records

    def _append_upload_manifest(self, record: dict[str, Any]) -> None:
        self.upload_root.mkdir(parents=True, exist_ok=True)
        with self.upload_manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _stored_upload_name(self, safe_name: str) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = Path(safe_name).stem[:80].strip("._ ") or "document"
        suffix = Path(safe_name).suffix.lower()
        candidate = f"{stamp}_{stem}{suffix}"
        counter = 2
        while (self.upload_root / candidate).exists():
            candidate = f"{stamp}_{stem}_{counter}{suffix}"
            counter += 1
        return candidate

    def _search_foods(self, db_path: Path, query: str, limit: int) -> list[dict[str, Any]]:
        like = f"%{query}%"
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    fi.food_id,
                    fi.canonical_name,
                    fi.chinese_name,
                    fi.english_name,
                    fi.food_category,
                    fi.processing_level,
                    fn.energy_kcal,
                    fn.carbohydrate_g,
                    fn.protein_g,
                    fn.fat_g,
                    fn.dietary_fiber_g,
                    fn.sodium_mg,
                    fn.potassium_mg,
                    fn.phosphorus_mg,
                    fn.cholesterol_mg,
                    GROUP_CONCAT(pu.unit_name || '=' || ROUND(pu.grams, 1) || 'g', '; ') AS portions
                FROM food_items fi
                LEFT JOIN food_nutrients_per_100g fn ON fn.food_id = fi.food_id
                LEFT JOIN portion_units pu ON pu.food_id = fi.food_id
                WHERE
                    fi.canonical_name LIKE ?
                    OR fi.english_name LIKE ?
                    OR fi.chinese_name LIKE ?
                    OR EXISTS (
                        SELECT 1 FROM food_aliases fa
                        WHERE fa.food_id = fi.food_id AND fa.alias LIKE ?
                    )
                GROUP BY fi.food_id
                ORDER BY
                    CASE
                        WHEN fi.canonical_name = ? OR fi.chinese_name = ? THEN 0
                        WHEN fi.canonical_name LIKE ? OR fi.chinese_name LIKE ? THEN 1
                        WHEN EXISTS (
                            SELECT 1 FROM food_aliases fa
                            WHERE fa.food_id = fi.food_id AND fa.alias = ?
                        ) THEN 2
                        WHEN fi.english_name = ? THEN 3
                        WHEN fi.canonical_name LIKE ? THEN 4
                        ELSE 5
                    END,
                    CASE
                        WHEN LOWER(fi.canonical_name) LIKE '%dried%'
                             OR LOWER(fi.canonical_name) LIKE '%powder%'
                             OR fi.canonical_name LIKE '%粉%' THEN 2
                        WHEN fi.processing_level = 'raw_or_minimally_processed' THEN 0
                        WHEN fi.processing_level = 'processed' THEN 1
                        ELSE 1
                    END,
                    fi.canonical_name
                LIMIT ?
                """,
                (
                    like,
                    like,
                    like,
                    like,
                    query,
                    query,
                    f"{query}%",
                    f"{query}%",
                    query,
                    query,
                    f"{query}%",
                    limit,
                ),
            ).fetchall()

        foods = []
        for row in rows:
            foods.append(
                {
                    "food_id": row["food_id"],
                    "canonical_name": row["canonical_name"],
                    "chinese_name": row["chinese_name"],
                    "english_name": row["english_name"],
                    "food_category": row["food_category"],
                    "processing_level": row["processing_level"],
                    "nutrients_per_100g": {
                        "energy_kcal": row["energy_kcal"],
                        "carbohydrate_g": row["carbohydrate_g"],
                        "protein_g": row["protein_g"],
                        "fat_g": row["fat_g"],
                        "dietary_fiber_g": row["dietary_fiber_g"],
                        "sodium_mg": row["sodium_mg"],
                        "potassium_mg": row["potassium_mg"],
                        "phosphorus_mg": row["phosphorus_mg"],
                        "cholesterol_mg": row["cholesterol_mg"],
                    },
                    "portion_units": _parse_portion_units(row["portions"]),
                }
            )
        return foods


def _structured_review_prompt(summary: dict[str, Any]) -> str:
    safe_summary = {
        "document": {
            key: value
            for key, value in (summary.get("document") or {}).items()
            if key not in {"review_summary"}
        },
        "counts": summary.get("counts") or {},
        "review_status_counts": summary.get("review_status_counts") or {},
        "samples": summary.get("samples") or {},
    }
    return (
        "你是临床营养知识库结构化抽取质检员。请复核下面这次 PDF/文档结构化抽取结果，"
        "判断是否可以作为 Agent 的结构化知识库候选数据继续使用。你只检查抽取质量，"
        "不要重写原文，不要把普通建议升级为安全红线。\n\n"
        "请重点检查：\n"
        "1. 表格、BMI/MET、食谱、食养方是否看起来被正确分离；\n"
        "2. 数字、页码、标题、食材克数、宏量营养素是否存在明显错位；\n"
        "3. 是否有需要人工重点抽查的风险；\n"
        "4. 如果只是少量可接受瑕疵，可以 approved_to_use=true，但要列出注意事项。\n\n"
        "只输出严格 JSON，不要 Markdown。结构如下：\n"
        "{\n"
        '  "overall_status": "passed|needs_human_review|failed",\n'
        '  "approved_to_use": true,\n'
        '  "confidence": 0.0,\n'
        '  "issues": ["问题1"],\n'
        '  "recommendations": ["建议1"],\n'
        '  "spot_checks": [{"item": "要抽查的对象", "reason": "原因"}]\n'
        "}\n\n"
        f"待复核抽取摘要：\n{json.dumps(safe_summary, ensure_ascii=False, indent=2)}"
    )


def _bounded_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _parse_json_maybe(value: Any) -> Any:
    if not value:
        return None
    try:
        return json.loads(str(value))
    except Exception:
        return None


def _safe_filename(filename: str) -> str:
    name = Path(filename or "document").name
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name)
    name = re.sub(r"\s+", "_", name).strip("._")
    if not name:
        return "document.txt"
    return name[:140]


def _guess_console_meal_label(meal_text: str) -> str:
    text = str(meal_text or "")
    if any(token in text for token in ["早餐", "早饭", "早上"]):
        return "breakfast"
    if any(token in text for token in ["午餐", "午饭", "中饭", "中午"]):
        return "lunch"
    if any(token in text for token in ["晚餐", "晚饭", "晚上"]):
        return "dinner"
    if any(token in text for token in ["加餐", "夜宵", "零食"]):
        return "snack"
    return ""


def _ensure_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value


def _ensure_clinical_ltm(config: dict[str, Any]) -> dict[str, Any]:
    memory = _ensure_dict(config, "Memory")
    clinical_ltm = _ensure_dict(memory, "clinical_ltm")
    clinical_ltm.setdefault("type", "clinical_ltm")
    return clinical_ltm


def _ensure_powermem_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    clinical_ltm = _ensure_clinical_ltm(config)
    powermem = _ensure_dict(clinical_ltm, "powermem")
    return _ensure_dict(powermem, section)


def _endpoint_field(config: dict[str, Any]) -> str:
    if "base_url" in config:
        return "base_url"
    if "url" in config:
        return "url"
    return "base_url"


def _first_existing_key(config: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key in config:
            return key
    return keys[0] if keys else ""


def _first_numeric_value(config: dict[str, Any], keys: list[str]) -> float | int | None:
    for key in keys:
        if key not in config:
            continue
        value = _coerce_float(config.get(key))
        if value is not None:
            return int(value) if float(value).is_integer() else value
    return None


def _secret_configured(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    lowered = text.lower()
    return not any(hint.lower() in lowered for hint in PLACEHOLDER_SECRET_HINTS)


def _set_if_present(target: dict[str, Any], key: str, value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        value = value.strip()
    if value == "":
        return 0
    if target.get(key) == value:
        return 0
    target[key] = value
    return 1


def _set_if_changed(target: dict[str, Any], key: str, value: Any) -> int:
    if target.get(key) == value:
        return 0
    target[key] = value
    return 1


def _set_secret_if_present(target: dict[str, Any], key: str, value: Any) -> int:
    if value is None:
        return 0
    value = str(value).strip()
    if not value:
        return 0
    if target.get(key) == value:
        return 0
    target[key] = value
    return 1


def _coerce_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return None


def _set_bool_if_present(target: dict[str, Any], key: str, value: Any) -> int:
    parsed = _coerce_bool(value)
    if parsed is None:
        return 0
    if target.get(key) == parsed:
        return 0
    target[key] = parsed
    return 1


def _set_optional_number(target: dict[str, Any], key: str, value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if target.get(key) == number:
        return 0
    target[key] = number
    return 1


def _set_number_if_present(target: dict[str, Any], key: str, value: Any) -> int:
    if not key:
        return 0
    number = _coerce_float(value)
    if number is None:
        return 0
    if target.get(key) == number:
        return 0
    target[key] = number
    return 1


def _set_optional_int(target: dict[str, Any], key: str, value: Any) -> int:
    number = _coerce_positive_int(value)
    if number is None:
        return 0
    if target.get(key) == number:
        return 0
    target[key] = number
    return 1


def _coerce_positive_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def _coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_voice_controls(config: dict[str, Any]) -> dict[str, Any]:
    rate_field = _first_existing_key(
        config,
        ["speed_ratio", "speed_factor", "speed", "rate"],
    )
    pitch_field = _first_existing_key(
        config,
        ["pitch_ratio", "pitch_factor", "pitch"],
    )
    volume_field = _first_existing_key(
        config,
        ["volume_ratio", "volume", "volume_change_dB"],
    )
    return {
        "rate_field": rate_field,
        "pitch_field": pitch_field,
        "volume_field": volume_field,
        "rate": _coerce_float(config.get(rate_field)) or 1.0,
        "pitch": _coerce_float(config.get(pitch_field)) or 1.0,
        "volume": _coerce_float(config.get(volume_field)) or 1.0,
    }


def _recognition_speed_preset(value: Any) -> str:
    silence_ms = _coerce_positive_int(value)
    if silence_ms is None:
        return "normal"
    if silence_ms <= 300:
        return "fast"
    if silence_ms >= 700:
        return "stable"
    return "normal"


def _recognition_silence_from_preset(preset: str) -> int | None:
    mapping = {
        "fast": 250,
        "normal": 500,
        "stable": 800,
    }
    return mapping.get(str(preset or "").strip().lower())


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(
        microsecond=0
    ).isoformat()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _loads_json(text: str | None, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def _sqlite_count(db_path: Path, table: str) -> int | None:
    if not _sqlite_table_exists(db_path, table):
        return None
    with sqlite3.connect(db_path) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _sqlite_count_where(db_path: Path, table: str, where_sql: str) -> int:
    if not _sqlite_table_exists(db_path, table):
        return 0
    with sqlite3.connect(db_path) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where_sql}").fetchone()[0])


def _sqlite_table_exists(db_path: Path, table: str) -> bool:
    if not db_path.exists():
        return False
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            return row is not None
    except sqlite3.Error:
        return False


def _int_query(
    request: web.Request,
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(request.query.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _parse_portion_units(raw: str | None) -> list[dict[str, Any]]:
    if not raw:
        return []
    portions = []
    for part in str(raw).split(";"):
        if "=" not in part:
            continue
        unit, grams = part.split("=", 1)
        grams = grams.strip().removesuffix("g")
        try:
            gram_value: float | None = float(grams)
        except ValueError:
            gram_value = None
        portions.append({"unit_name": unit.strip(), "grams": gram_value})
    return portions
