from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import struct
import time
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.clinical_nutrition.structured_knowledge import (
    StructuredKnowledgeStore,
    extract_structured_blocks_for_rag,
)


DEFAULT_CHUNK_CHARS = 850
DEFAULT_CHUNK_OVERLAP_CHARS = 150
DEFAULT_TOP_K = 6
DEFAULT_BM25_CANDIDATES = 40
DEFAULT_VECTOR_CANDIDATES = 40
DEFAULT_MIN_READABLE_CHARS = 300
DEFAULT_MAX_EMPTY_PAGE_RATIO = 0.8

SUPPORTED_TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".csv", ".tsv", ".json"}
SUPPORTED_DOCUMENT_SUFFIXES = SUPPORTED_TEXT_SUFFIXES | {".pdf", ".docx"}


@dataclass
class ExtractedPage:
    page_number: int
    text: str
    extraction_method: str


@dataclass
class RAGChunk:
    chunk_index: int
    page_start: int
    page_end: int
    section_title: str
    text: str
    chunk_type: str = "paragraph"
    section_path: str = ""
    metadata_json: str = "{}"


class ClinicalRAGService:
    """Local clinical RAG store with page-aware ingestion and hybrid retrieval."""

    def __init__(self, *, project_root: Path, config: dict[str, Any], logger):
        self.project_root = Path(project_root).resolve()
        self.config = config or {}
        self.logger = logger
        self.rag_config = self._rag_config()
        self.db_path = self._resolve_project_path(
            self.rag_config.get("db_path"),
            "data/clinical_rag.db",
        )
        self.upload_root = self._resolve_project_path(
            self.rag_config.get("upload_root"),
            "data/knowledge_uploads/clinical-nutrition",
        )
        self._init_db()

    def register_document(
        self,
        source_path: Path,
        *,
        original_name: str = "",
        content_type: str = "",
    ) -> dict[str, Any]:
        source_path = Path(source_path).resolve()
        self._ensure_inside_project(source_path)
        if not source_path.exists() or not source_path.is_file():
            raise ValueError("source file not found")

        suffix = source_path.suffix.lower()
        if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
            raise ValueError(f"unsupported RAG document type: {suffix or '(none)'}")

        source_hash = _sha256_file(source_path)
        now = _utc_now()
        stored_path = _relative_to(self.project_root, source_path)
        original_name = original_name or source_path.name
        title = Path(original_name).stem or source_path.stem

        with self._connect() as db:
            existing = db.execute(
                "SELECT * FROM rag_documents WHERE source_hash = ?",
                (source_hash,),
            ).fetchone()
            if existing:
                db.execute(
                    """
                    UPDATE rag_documents
                    SET original_name = ?, stored_path = ?, content_type = ?, updated_at = ?
                    WHERE document_id = ?
                    """,
                    (
                        original_name,
                        stored_path,
                        content_type,
                        now,
                        existing["document_id"],
                    ),
                )
                return self._document_by_id(db, existing["document_id"])

            document_id = f"doc_{uuid.uuid4().hex}"
            db.execute(
                """
                INSERT INTO rag_documents (
                    document_id, source_hash, original_name, stored_path, file_type,
                    content_type, title, status, page_count, char_count, chunk_count,
                    embedded_count, error_message, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'uploaded', 0, 0, 0, 0, '', ?, ?)
                """,
                (
                    document_id,
                    source_hash,
                    original_name,
                    stored_path,
                    suffix,
                    content_type,
                    title,
                    now,
                    now,
                ),
            )
            return self._document_by_id(db, document_id)

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT *
                FROM rag_documents
                ORDER BY updated_at DESC, created_at DESC
                """
            ).fetchall()
            return [self._document_payload(row) for row in rows]

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM rag_documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            return self._document_payload(row) if row else None

    def delete_document(self, document_id: str) -> dict[str, Any]:
        with self._connect() as db:
            document = self._document_by_id(db, document_id)
            chunk_ids = [
                row["chunk_id"]
                for row in db.execute(
                    "SELECT chunk_id FROM rag_chunks WHERE document_id = ?",
                    (document_id,),
                ).fetchall()
            ]
            db.execute("DELETE FROM rag_pages WHERE document_id = ?", (document_id,))
            db.execute("DELETE FROM rag_chunks WHERE document_id = ?", (document_id,))
            db.execute("DELETE FROM rag_ingestion_jobs WHERE document_id = ?", (document_id,))
            for chunk_id in chunk_ids:
                db.execute("DELETE FROM rag_embeddings WHERE chunk_id = ?", (chunk_id,))
                db.execute("DELETE FROM rag_chunks_fts WHERE chunk_id = ?", (chunk_id,))
            db.execute("DELETE FROM rag_documents WHERE document_id = ?", (document_id,))
        try:
            StructuredKnowledgeStore(
                project_root=self.project_root,
                config=self.config,
            ).delete_document(document_id)
        except Exception as exc:
            if self.logger:
                self.logger.bind(tag=__name__).warning(f"Structured knowledge delete failed: {exc}")
        return document

    def create_index_job(self, document_id: str) -> dict[str, Any]:
        with self._connect() as db:
            self._document_by_id(db, document_id)
            job_id = f"job_{uuid.uuid4().hex}"
            now = _utc_now()
            db.execute(
                """
                INSERT INTO rag_ingestion_jobs (
                    job_id, document_id, status, stage, error, chunk_count,
                    embedded_count, created_at, updated_at
                )
                VALUES (?, ?, 'queued', 'queued', '', 0, 0, ?, ?)
                """,
                (job_id, document_id, now, now),
            )
            return self._job_by_id(db, job_id)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM rag_ingestion_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            return self._job_payload(row) if row else None

    def index_document(self, document_id: str, *, job_id: str | None = None) -> dict[str, Any]:
        job_id = job_id or self.create_index_job(document_id)["job_id"]
        try:
            with self._connect() as db:
                document = self._document_by_id(db, document_id)
            source_path = self._resolve_project_path(document["stored_path"], document["stored_path"])

            self._set_job_stage(job_id, document_id, "running", "extracting")
            pages = extract_document_pages(source_path)
            readable_chars = sum(len(page.text.strip()) for page in pages)
            empty_pages = sum(1 for page in pages if not page.text.strip())
            empty_ratio = empty_pages / max(len(pages), 1)
            if (
                source_path.suffix.lower() == ".pdf"
                and (
                    readable_chars < _safe_int(
                        self.rag_config.get("min_readable_chars"),
                        DEFAULT_MIN_READABLE_CHARS,
                        minimum=50,
                        maximum=5000,
                    )
                    or empty_ratio >= _safe_float(
                        self.rag_config.get("max_empty_page_ratio"),
                        DEFAULT_MAX_EMPTY_PAGE_RATIO,
                        minimum=0.2,
                        maximum=1.0,
                    )
                )
            ):
                error = (
                    "document appears to be a scanned PDF or image-only PDF; "
                    "OCR is not enabled, please upload a text-based PDF or OCR result"
                )
                self._replace_pages(document_id, pages)
                self._set_document_status(
                    document_id,
                    "needs_ocr",
                    page_count=len(pages),
                    char_count=readable_chars,
                    chunk_count=0,
                    embedded_count=0,
                    error_message=error,
                )
                self._finish_job(job_id, document_id, "needs_ocr", "extracting", error, 0, 0)
                return self.get_document(document_id) or {}

            if not readable_chars:
                raise ValueError("document has no readable text")

            self._replace_pages(document_id, pages)
            structured_error = ""
            try:
                StructuredKnowledgeStore(
                    project_root=self.project_root,
                    config=self.config,
                ).ingest_document(document, pages)
            except Exception as exc:
                structured_error = f"structured knowledge ingestion failed: {exc}"
                if self.logger:
                    self.logger.bind(tag=__name__).error(structured_error)
            self._set_job_stage(job_id, document_id, "running", "chunking")
            chunks = chunk_pages(
                pages,
                chunk_chars=_safe_int(
                    self.rag_config.get("chunk_chars"),
                    DEFAULT_CHUNK_CHARS,
                    minimum=300,
                    maximum=3000,
                ),
                overlap_chars=_safe_int(
                    self.rag_config.get("chunk_overlap_chars"),
                    DEFAULT_CHUNK_OVERLAP_CHARS,
                    minimum=0,
                    maximum=800,
                ),
            )
            if not chunks:
                raise ValueError("document produced no chunks")

            self._replace_chunks(document_id, chunks)
            self._set_document_status(
                document_id,
                "embedding",
                page_count=len(pages),
                char_count=readable_chars,
                chunk_count=len(chunks),
                embedded_count=0,
                error_message="",
            )

            self._set_job_stage(job_id, document_id, "running", "embedding", chunk_count=len(chunks))
            embedded_count = self._embed_document_chunks(document_id, chunks, job_id)
            if embedded_count == len(chunks):
                final_status = "indexed"
                error = structured_error
            elif embedded_count > 0:
                final_status = "indexed_partial_vector"
                error = "some chunks failed to embed"
            else:
                final_status = "indexed_lexical_only"
                error = "all chunks failed to embed; lexical search is still available"
            if structured_error:
                error = f"{error}; {structured_error}" if error else structured_error
            self._set_document_status(
                document_id,
                final_status,
                page_count=len(pages),
                char_count=readable_chars,
                chunk_count=len(chunks),
                embedded_count=embedded_count,
                error_message=error,
            )
            self._finish_job(job_id, document_id, final_status, final_status, error, len(chunks), embedded_count)
            return self.get_document(document_id) or {}
        except Exception as exc:
            message = str(exc)
            self._set_document_status(document_id, "failed", error_message=message)
            self._finish_job(job_id, document_id, "failed", "failed", message, 0, 0)
            if self.logger:
                self.logger.bind(tag=__name__).error(f"RAG indexing failed: {message}")
            return self.get_document(document_id) or {}

    def list_chunks(self, document_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as db:
            self._document_by_id(db, document_id)
            rows = db.execute(
                """
                SELECT chunk_id, document_id, chunk_index, page_start, page_end,
                       section_title, text, text_hash, token_count, created_at,
                       chunk_type, section_path, metadata_json
                FROM rag_chunks
                WHERE document_id = ?
                ORDER BY chunk_index
                LIMIT ?
                """,
                (document_id, max(1, min(int(limit or 200), 1000))),
            ).fetchall()
            return [dict(row) for row in rows]

    def search(self, question: str, *, top_k: int | None = None) -> list[dict[str, Any]]:
        question = str(question or "").strip()
        if not question:
            return []

        top_k = max(
            1,
            min(
                int(top_k or self.rag_config.get("top_k") or DEFAULT_TOP_K),
                12,
            ),
        )
        bm25_candidates = _safe_int(
            self.rag_config.get("bm25_candidates"),
            DEFAULT_BM25_CANDIDATES,
            minimum=5,
            maximum=120,
        )
        vector_candidates = _safe_int(
            self.rag_config.get("vector_candidates"),
            DEFAULT_VECTOR_CANDIDATES,
            minimum=5,
            maximum=120,
        )
        expanded_question = _expand_rag_question(question)
        lexical = self._lexical_search(expanded_question, bm25_candidates)
        vector = self._vector_search(expanded_question, vector_candidates)
        merged = self._merge_candidates(lexical, vector)
        for item in merged:
            item["score"] = _adjust_rag_candidate_score(item, question)
        merged.sort(key=lambda item: item["score"], reverse=True)
        return self._mmr_select(merged, top_k)

    def seed_markdown_directory(self, wiki_root: Path) -> dict[str, Any]:
        wiki_root = Path(wiki_root).resolve()
        if not wiki_root.exists():
            return {"ok": False, "error": "wiki root not found", "indexed": 0}

        indexed = 0
        skipped = 0
        for path in wiki_root.rglob("*.md"):
            lowered_parts = {part.lower() for part in path.parts}
            if {"raw", "templates"} & lowered_parts or path.name.lower() == "readme.md":
                skipped += 1
                continue
            document = self.register_document(path, original_name=path.name, content_type="text/markdown")
            if document.get("status") != "indexed":
                job = self.create_index_job(document["document_id"])
                result = self.index_document(document["document_id"], job_id=job["job_id"])
                if result.get("status") == "indexed":
                    indexed += 1
            else:
                skipped += 1
        return {"ok": True, "indexed": indexed, "skipped": skipped}

    def _rag_config(self) -> dict[str, Any]:
        rag = dict(self.config.get("clinical_rag") or {})
        plugin = (self.config.get("plugins") or {}).get("search_clinical_rag") or {}
        for key in ("db_path", "top_k", "bm25_candidates", "vector_candidates"):
            if key not in rag and key in plugin:
                rag[key] = plugin[key]
        memory = (self.config.get("Memory") or {}).get("clinical_ltm") or {}
        powermem = memory.get("powermem") or {}
        embedder = powermem.get("embedder") or {}
        embedder_config = embedder.get("config") or {}
        embedding = dict(rag.get("embedding") or {})
        embedding.setdefault("enabled", True)
        embedding.setdefault("provider", embedder.get("provider", "openai"))
        embedding.setdefault("model", embedder_config.get("model", "text-embedding-v4"))
        embedding.setdefault("api_key", embedder_config.get("api_key", ""))
        embedding.setdefault(
            "openai_base_url",
            embedder_config.get("openai_base_url")
            or _compatible_base_url(embedder_config.get("dashscope_base_url", "")),
        )
        embedding.setdefault(
            "dimensions",
            embedder_config.get("embedding_dims", memory.get("embedding_dimensions", "")),
        )
        rag["embedding"] = embedding
        return rag

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA journal_mode = WAL")
        return db

    def _init_db(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS rag_documents (
                    document_id TEXT PRIMARY KEY,
                    source_hash TEXT NOT NULL UNIQUE,
                    original_name TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    content_type TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'uploaded',
                    page_count INTEGER NOT NULL DEFAULT 0,
                    char_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    embedded_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS rag_pages (
                    page_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    extraction_method TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES rag_documents(document_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS rag_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    section_title TEXT NOT NULL DEFAULT '',
                    text TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES rag_documents(document_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS rag_embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector_blob BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(chunk_id) REFERENCES rag_chunks(chunk_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS rag_ingestion_jobs (
                    job_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    error TEXT NOT NULL DEFAULT '',
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    embedded_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES rag_documents(document_id) ON DELETE CASCADE
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    document_id UNINDEXED,
                    title,
                    section_title,
                    text,
                    tokenize='unicode61'
                );

                CREATE INDEX IF NOT EXISTS idx_rag_pages_document ON rag_pages(document_id, page_number);
                CREATE INDEX IF NOT EXISTS idx_rag_chunks_document ON rag_chunks(document_id, chunk_index);
                CREATE INDEX IF NOT EXISTS idx_rag_jobs_document ON rag_ingestion_jobs(document_id, updated_at);
                """
            )
            self._ensure_column(db, "rag_chunks", "chunk_type", "TEXT NOT NULL DEFAULT 'paragraph'")
            self._ensure_column(db, "rag_chunks", "section_path", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(db, "rag_chunks", "metadata_json", "TEXT NOT NULL DEFAULT '{}'")

    def _ensure_column(self, db: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {row["name"] for row in db.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _replace_pages(self, document_id: str, pages: list[ExtractedPage]) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM rag_pages WHERE document_id = ?", (document_id,))
            for page in pages:
                db.execute(
                    """
                    INSERT INTO rag_pages (
                        page_id, document_id, page_number, text, char_count, extraction_method
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"page_{uuid.uuid4().hex}",
                        document_id,
                        page.page_number,
                        page.text,
                        len(page.text),
                        page.extraction_method,
                    ),
                )

    def _replace_chunks(self, document_id: str, chunks: list[RAGChunk]) -> None:
        now = _utc_now()
        with self._connect() as db:
            old_chunk_ids = [
                row["chunk_id"]
                for row in db.execute(
                    "SELECT chunk_id FROM rag_chunks WHERE document_id = ?",
                    (document_id,),
                ).fetchall()
            ]
            for chunk_id in old_chunk_ids:
                db.execute("DELETE FROM rag_embeddings WHERE chunk_id = ?", (chunk_id,))
                db.execute("DELETE FROM rag_chunks_fts WHERE chunk_id = ?", (chunk_id,))
            db.execute("DELETE FROM rag_chunks WHERE document_id = ?", (document_id,))
            document = self._document_by_id(db, document_id)
            for chunk in chunks:
                chunk_id = f"chk_{uuid.uuid4().hex}"
                text_hash = hashlib.sha1(chunk.text.encode("utf-8")).hexdigest()
                token_count = len(_tokenize(chunk.text))
                db.execute(
                    """
                    INSERT INTO rag_chunks (
                        chunk_id, document_id, chunk_index, page_start, page_end,
                        section_title, text, text_hash, token_count, created_at,
                        chunk_type, section_path, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        document_id,
                        chunk.chunk_index,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.section_title,
                        chunk.text,
                        text_hash,
                        token_count,
                        now,
                        chunk.chunk_type,
                        chunk.section_path,
                        chunk.metadata_json,
                    ),
                )
                search_text = _fts_augmented_text(
                    "\n".join(
                        item
                        for item in [
                            document.get("title", ""),
                            chunk.section_path,
                            chunk.chunk_type,
                            chunk.section_title,
                            chunk.text,
                        ]
                        if item
                    )
                )
                db.execute(
                    """
                    INSERT INTO rag_chunks_fts (
                        chunk_id, document_id, title, section_title, text
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        document_id,
                        document.get("title", ""),
                        chunk.section_title,
                        search_text,
                    ),
                )

    def _embed_document_chunks(
        self,
        document_id: str,
        chunks: list[RAGChunk],
        job_id: str,
    ) -> int:
        embedding_config = self.rag_config.get("embedding") or {}
        if embedding_config.get("enabled") is False:
            return 0
        batch_size = _safe_int(
            embedding_config.get("batch_size"),
            16,
            minimum=1,
            maximum=64,
        )

        with self._connect() as db:
            rows = db.execute(
                """
                SELECT chunk_id, text
                FROM rag_chunks
                WHERE document_id = ?
                ORDER BY chunk_index
                """,
                (document_id,),
            ).fetchall()

        embedded_count = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            texts = [row["text"] for row in batch]
            try:
                vectors = embed_texts(texts, embedding_config)
            except Exception as exc:
                if self.logger:
                    self.logger.bind(tag=__name__).error(f"RAG embedding batch failed: {exc}")
                vectors = []
                for row in batch:
                    try:
                        single_text = str(row["text"] or "")
                        single_vectors = embed_texts([single_text[:6000]], embedding_config)
                        vectors.append(single_vectors[0] if single_vectors else None)
                    except Exception as single_exc:
                        vectors.append(None)
                        if self.logger:
                            self.logger.bind(tag=__name__).warning(
                                f"RAG embedding single chunk failed: {single_exc}"
                            )
            if len(vectors) != len(batch):
                if self.logger:
                    self.logger.bind(tag=__name__).error("RAG embedding batch size mismatch")
                continue
            with self._connect() as db:
                for row, vector in zip(batch, vectors):
                    if not vector:
                        continue
                    db.execute(
                        """
                        INSERT OR REPLACE INTO rag_embeddings (
                            chunk_id, provider, model, dimensions, vector_blob, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row["chunk_id"],
                            str(embedding_config.get("provider") or "openai"),
                            str(embedding_config.get("model") or ""),
                            len(vector),
                            _pack_vector(vector),
                            _utc_now(),
                        ),
                    )
                    embedded_count += 1
            self._set_job_stage(
                job_id,
                document_id,
                "running",
                "embedding",
                chunk_count=len(chunks),
                embedded_count=embedded_count,
            )
        return embedded_count

    def _lexical_search(self, question: str, limit: int) -> list[dict[str, Any]]:
        tokens = _tokenize(question)
        if not tokens:
            return []
        match_query = " OR ".join(f'"{token}"' for token in tokens[:16])
        rows: list[sqlite3.Row] = []
        with self._connect() as db:
            try:
                rows = db.execute(
                    """
                    SELECT c.*, d.title, d.original_name, d.stored_path,
                           bm25(rag_chunks_fts) AS rank
                    FROM rag_chunks_fts
                    JOIN rag_chunks c ON c.chunk_id = rag_chunks_fts.chunk_id
                    JOIN rag_documents d ON d.document_id = c.document_id
                    WHERE rag_chunks_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (match_query, limit),
                ).fetchall()
            except sqlite3.Error:
                like_terms = [f"%{token}%" for token in tokens[:5]]
                rows = []
                seen: set[str] = set()
                for like in like_terms:
                    fallback = db.execute(
                        """
                        SELECT c.*, d.title, d.original_name, d.stored_path, 0.0 AS rank
                        FROM rag_chunks c
                        JOIN rag_documents d ON d.document_id = c.document_id
                        WHERE c.text LIKE ? OR c.section_title LIKE ? OR d.title LIKE ?
                        ORDER BY c.chunk_index
                        LIMIT ?
                        """,
                        (like, like, like, limit),
                    ).fetchall()
                    for row in fallback:
                        if row["chunk_id"] not in seen:
                            seen.add(row["chunk_id"])
                            rows.append(row)
                        if len(rows) >= limit:
                            break
                    if len(rows) >= limit:
                        break

        results = []
        for rank_index, row in enumerate(rows):
            payload = self._candidate_payload(row)
            payload["lexical_score"] = 1.0 / (rank_index + 1)
            payload["vector_score"] = 0.0
            payload["score"] = payload["lexical_score"] * 0.55
            results.append(payload)
        return results

    def _vector_search(self, question: str, limit: int) -> list[dict[str, Any]]:
        embedding_config = self.rag_config.get("embedding") or {}
        if embedding_config.get("enabled") is False:
            return []
        try:
            query_vectors = embed_texts([question], embedding_config)
        except Exception as exc:
            if self.logger:
                self.logger.bind(tag=__name__).warning(f"RAG vector query failed: {exc}")
            return []
        if not query_vectors:
            return []
        query_vector = query_vectors[0]

        with self._connect() as db:
            rows = db.execute(
                """
                SELECT c.*, d.title, d.original_name, d.stored_path,
                       e.vector_blob, e.dimensions
                FROM rag_embeddings e
                JOIN rag_chunks c ON c.chunk_id = e.chunk_id
                JOIN rag_documents d ON d.document_id = c.document_id
                """
            ).fetchall()

        scored = []
        for row in rows:
            vector = _unpack_vector(row["vector_blob"], row["dimensions"])
            if len(vector) != len(query_vector):
                continue
            cosine = _cosine_similarity(query_vector, vector)
            payload = self._candidate_payload(row)
            payload["lexical_score"] = 0.0
            payload["vector_score"] = max(0.0, (cosine + 1.0) / 2.0)
            payload["score"] = payload["vector_score"] * 0.45
            scored.append(payload)
        scored.sort(key=lambda item: item["vector_score"], reverse=True)
        return scored[:limit]

    def _merge_candidates(
        self,
        lexical: list[dict[str, Any]],
        vector: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for item in lexical + vector:
            current = merged.setdefault(item["chunk_id"], dict(item))
            current["lexical_score"] = max(current.get("lexical_score", 0.0), item.get("lexical_score", 0.0))
            current["vector_score"] = max(current.get("vector_score", 0.0), item.get("vector_score", 0.0))
            current["score"] = current["lexical_score"] * 0.55 + current["vector_score"] * 0.45
        ranked = list(merged.values())
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked

    def _mmr_select(self, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        selected_tokens: list[set[str]] = []
        for candidate in candidates:
            tokens = set(_tokenize(candidate.get("text", "")))
            if selected_tokens:
                max_overlap = max(_jaccard(tokens, existing) for existing in selected_tokens)
                if max_overlap > 0.82:
                    continue
            selected.append(candidate)
            selected_tokens.append(tokens)
            if len(selected) >= top_k:
                break
        return selected

    def _candidate_payload(self, row: sqlite3.Row) -> dict[str, Any]:
        source_name = row["original_name"] or Path(row["stored_path"]).name
        page_start = int(row["page_start"] or 1)
        page_end = int(row["page_end"] or page_start)
        page_label = f"p.{page_start}" if page_start == page_end else f"pp.{page_start}-{page_end}"
        citation = f"{source_name} {page_label} [{row['chunk_id']}]"
        return {
            "document_id": row["document_id"],
            "chunk_id": row["chunk_id"],
            "title": row["title"] or Path(source_name).stem,
            "source_name": source_name,
            "stored_path": row["stored_path"],
            "page_start": page_start,
            "page_end": page_end,
            "section_title": row["section_title"] or "",
            "section_path": row["section_path"] or "",
            "chunk_type": row["chunk_type"] or "paragraph",
            "metadata": _loads_json(row["metadata_json"], {}),
            "text": row["text"],
            "citation": citation,
        }

    def _document_by_id(self, db: sqlite3.Connection, document_id: str) -> dict[str, Any]:
        row = db.execute(
            "SELECT * FROM rag_documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if not row:
            raise ValueError("RAG document not found")
        return self._document_payload(row)

    def _job_by_id(self, db: sqlite3.Connection, job_id: str) -> dict[str, Any]:
        row = db.execute(
            "SELECT * FROM rag_ingestion_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if not row:
            raise ValueError("RAG job not found")
        return self._job_payload(row)

    def _document_payload(self, row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        payload["db_path"] = _relative_to(self.project_root, self.db_path)
        return payload

    def _job_payload(self, row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        return dict(row)

    def _set_job_stage(
        self,
        job_id: str,
        document_id: str,
        status: str,
        stage: str,
        *,
        error: str = "",
        chunk_count: int | None = None,
        embedded_count: int | None = None,
    ) -> None:
        assignments = ["status = ?", "stage = ?", "error = ?", "updated_at = ?"]
        params: list[Any] = [status, stage, error, _utc_now()]
        if chunk_count is not None:
            assignments.append("chunk_count = ?")
            params.append(chunk_count)
        if embedded_count is not None:
            assignments.append("embedded_count = ?")
            params.append(embedded_count)
        params.append(job_id)
        with self._connect() as db:
            db.execute(
                f"UPDATE rag_ingestion_jobs SET {', '.join(assignments)} WHERE job_id = ?",
                params,
            )
            db.execute(
                "UPDATE rag_documents SET status = ?, updated_at = ? WHERE document_id = ?",
                (stage, _utc_now(), document_id),
            )

    def _finish_job(
        self,
        job_id: str,
        document_id: str,
        status: str,
        stage: str,
        error: str,
        chunk_count: int,
        embedded_count: int,
    ) -> None:
        with self._connect() as db:
            db.execute(
                """
                UPDATE rag_ingestion_jobs
                SET status = ?, stage = ?, error = ?, chunk_count = ?,
                    embedded_count = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (status, stage, error, chunk_count, embedded_count, _utc_now(), job_id),
            )
            db.execute(
                """
                UPDATE rag_documents
                SET status = ?, updated_at = ?
                WHERE document_id = ?
                """,
                (status, _utc_now(), document_id),
            )

    def _set_document_status(
        self,
        document_id: str,
        status: str,
        *,
        page_count: int | None = None,
        char_count: int | None = None,
        chunk_count: int | None = None,
        embedded_count: int | None = None,
        error_message: str | None = None,
    ) -> None:
        assignments = ["status = ?", "updated_at = ?"]
        params: list[Any] = [status, _utc_now()]
        for column, value in (
            ("page_count", page_count),
            ("char_count", char_count),
            ("chunk_count", chunk_count),
            ("embedded_count", embedded_count),
            ("error_message", error_message),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                params.append(value)
        params.append(document_id)
        with self._connect() as db:
            db.execute(
                f"UPDATE rag_documents SET {', '.join(assignments)} WHERE document_id = ?",
                params,
            )

    def _resolve_project_path(self, configured: Any, default: str) -> Path:
        raw = str(configured or default)
        path = Path(raw)
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    def _ensure_inside_project(self, path: Path) -> None:
        try:
            path.resolve().relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError("RAG source file must be inside project root") from exc


def extract_document_pages(path: Path) -> list[ExtractedPage]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import pypdf
        except Exception as exc:
            raise ValueError("pypdf is required to ingest PDF files") from exc
        reader = pypdf.PdfReader(str(path))
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            pages.append(
                ExtractedPage(
                    page_number=index,
                    text=_normalize_text(page.extract_text() or ""),
                    extraction_method="pypdf",
                )
            )
        return pages
    if suffix == ".docx":
        try:
            import docx
        except Exception as exc:
            raise ValueError("python-docx is required to ingest DOCX files") from exc
        document = docx.Document(str(path))
        text = "\n\n".join(paragraph.text for paragraph in document.paragraphs)
        return [ExtractedPage(1, _normalize_text(text), "python-docx")]
    if suffix in SUPPORTED_TEXT_SUFFIXES:
        return [ExtractedPage(1, _normalize_text(_read_text_any_encoding(path)), "text")]
    raise ValueError(f"unsupported RAG document type: {suffix or '(none)'}")


def chunk_pages(
    pages: list[ExtractedPage],
    *,
    chunk_chars: int,
    overlap_chars: int,
) -> list[RAGChunk]:
    special_blocks = extract_structured_blocks_for_rag(pages)
    special_recipe_pages = {
        page_number
        for block in special_blocks
        if block.block_type in {"recipe_plan", "therapeutic_recipe"}
        for page_number in range(block.page_start, block.page_end + 1)
    }

    paragraphs: list[tuple[str, int, str, str]] = []
    section_stack: list[str] = []
    for page in pages:
        if page.page_number in special_recipe_pages:
            continue
        for paragraph in _split_paragraphs(
            page.text,
            long_target=max(350, int(chunk_chars * 0.8)),
        ):
            heading = _extract_heading(paragraph)
            if heading:
                _update_section_stack(section_stack, heading)
                continue
            section_path = " > ".join(section_stack)
            section_title = section_stack[-1] if section_stack else ""
            paragraphs.append((paragraph, page.page_number, section_title, section_path))

    chunks: list[RAGChunk] = []
    buffer: list[str] = []
    buffer_pages: list[int] = []
    buffer_section = ""
    buffer_section_path = ""

    def emit() -> None:
        nonlocal buffer, buffer_pages, buffer_section, buffer_section_path
        text = "\n\n".join(item for item in buffer if item.strip()).strip()
        if not text:
            return
        chunks.append(
            RAGChunk(
                chunk_index=len(chunks),
                page_start=min(buffer_pages) if buffer_pages else 1,
                page_end=max(buffer_pages) if buffer_pages else 1,
                section_title=buffer_section,
                text=text,
                chunk_type="paragraph",
                section_path=buffer_section_path,
                metadata_json=json.dumps({"strategy": "section_paragraph_sentence"}, ensure_ascii=False),
            )
        )

    for paragraph, page_number, section, section_path in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        current_len = sum(len(item) for item in buffer)
        if buffer and current_len + len(paragraph) > chunk_chars:
            emit()
            overlap = _tail_text("\n\n".join(buffer), overlap_chars)
            buffer = [overlap] if overlap else []
            buffer_pages = [buffer_pages[-1]] if buffer_pages else []
            buffer_section = section or buffer_section
            buffer_section_path = section_path or buffer_section_path
        if not buffer_section:
            buffer_section = section
            buffer_section_path = section_path
        elif section and section != buffer_section and not buffer:
            buffer_section = section
            buffer_section_path = section_path
        buffer.append(paragraph)
        buffer_pages.append(page_number)

    if buffer:
        emit()

    for block in special_blocks:
        metadata = dict(block.metadata or {})
        metadata["strategy"] = "structured_block"
        chunks.append(
            RAGChunk(
                chunk_index=len(chunks),
                page_start=block.page_start,
                page_end=block.page_end,
                section_title=block.title,
                text=block.text,
                chunk_type=block.block_type,
                section_path=block.title,
                metadata_json=json.dumps(metadata, ensure_ascii=False),
            )
        )
    return chunks


def embed_texts(texts: list[str], embedding_config: dict[str, Any]) -> list[list[float]]:
    provider = str(embedding_config.get("provider") or "openai").lower()
    model = str(embedding_config.get("model") or "").strip()
    dimensions = _optional_int(
        embedding_config.get("dimensions")
        or embedding_config.get("embedding_dims")
        or embedding_config.get("embedding_dimensions")
    )
    if provider == "mock":
        dims = dimensions or 64
        return [_mock_embedding(text, dims) for text in texts]

    api_key = str(embedding_config.get("api_key") or "").strip()
    base_url = str(
        embedding_config.get("openai_base_url")
        or embedding_config.get("base_url")
        or _compatible_base_url(embedding_config.get("dashscope_base_url", ""))
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).strip()
    if not api_key:
        raise ValueError("RAG embedding API key is not configured")
    if not model:
        raise ValueError("RAG embedding model is not configured")

    payload: dict[str, Any] = {"model": model, "input": texts}
    if dimensions:
        payload["dimensions"] = dimensions

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/embeddings",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    retries = _safe_int(embedding_config.get("retries"), 3, minimum=1, maximum=5)
    timeout = _safe_int(embedding_config.get("timeout_seconds"), 60, minimum=5, maximum=300)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            data = result.get("data") or []
            vectors = [item.get("embedding") for item in data if isinstance(item, dict)]
            if len(vectors) != len(texts):
                raise ValueError("embedding response size mismatch")
            return [[float(value) for value in vector] for vector in vectors]
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.8 * (attempt + 1))
    raise ValueError(f"embedding request failed: {last_error}")


def _pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *[float(value) for value in vector])


def _unpack_vector(blob: bytes, dimensions: int) -> list[float]:
    if not blob or not dimensions:
        return []
    return list(struct.unpack(f"<{dimensions}f", blob))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _mock_embedding(text: str, dimensions: int) -> list[float]:
    buckets = [0.0] * dimensions
    for token in _tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        buckets[index] += sign
    norm = math.sqrt(sum(value * value for value in buckets)) or 1.0
    return [value / norm for value in buckets]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text_any_encoding(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _normalize_text(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_paragraphs(text: str, *, long_target: int = 600) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    parts = re.split(r"\n\s*\n", normalized)
    paragraphs = []
    for part in parts:
        part = part.strip()
        if len(part) > max(1000, long_target * 2):
            paragraphs.extend(_split_long_paragraph(part, long_target))
        elif part:
            paragraphs.append(part)
    return paragraphs


def _split_long_paragraph(text: str, target: int) -> list[str]:
    sentences = re.split(r"(?<=[。！？.!?；;])", text)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > target and current:
            parts.append(current.strip())
            current = sentence
        else:
            current += sentence
    if current.strip():
        parts.append(current.strip())
    return parts


def _extract_heading(paragraph: str) -> str:
    stripped = paragraph.strip()
    match = re.match(r"^#{1,6}\s+(.+)$", stripped)
    if match:
        return match.group(1).strip()[:160]
    if len(stripped) <= 80 and re.match(r"^(\d+[\.\、]|[一二三四五六七八九十]+[、.])", stripped):
        return stripped[:160]
    return ""


def _update_section_stack(stack: list[str], heading: str) -> None:
    heading = heading.strip()
    if not heading:
        return
    if heading.startswith("附录") or re.match(r"^[一二三四五六七八九十]+、", heading):
        stack[:] = [heading]
        return
    if re.match(r"^（[一二三四五六七八九十]+）", heading):
        if stack:
            stack[:] = [stack[0], heading]
        else:
            stack[:] = [heading]
        return
    if stack:
        stack[-1] = heading
    else:
        stack.append(heading)


def _tail_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:].strip()


def _tokenize(text: str) -> list[str]:
    lowered = str(text or "").lower()
    words = re.findall(r"[a-z0-9_+-]+", lowered)
    cjk_chars = [ch for ch in lowered if "\u4e00" <= ch <= "\u9fff"]
    bigrams = ["".join(cjk_chars[index : index + 2]) for index in range(max(0, len(cjk_chars) - 1))]
    seen: set[str] = set()
    tokens: list[str] = []
    for token in words + cjk_chars + bigrams:
        token = token.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= 80:
            break
    return tokens


def _fts_augmented_text(text: str) -> str:
    tokens = _tokenize(text)
    return f"{text}\n\n{' '.join(tokens)}"


def _expand_rag_question(question: str) -> str:
    text = str(question or "").strip()
    expansions: list[str] = []

    # ---- 肥胖相关 ----
    if any(kw in text for kw in ("少吃", "热量", "能量", "千卡", "kcal", "摄入")):
        expansions.append("成人肥胖 控制总能量摄入 每日能量摄入 降低 500 1000 kcal 限能量平衡膳食 安全减重")
    if any(kw in text for kw in ("运动", "活动", "锻炼", "多久")):
        expansions.append("多动少静 身体活动 睡眠充足 作息规律 运动建议")
    if any(kw in text for kw in ("减重速度", "减肥速度", "每月", "每周", "减重目标")):
        expansions.append("6个月 减少当前体重 5% 10% 每月 2 4kg 每周 0.5kg 平台期 自我监测")
    if any(kw in text for kw in ("三天", "七天", "一周", "十斤", "暴瘦", "速瘦", "瘦十斤", "快速减重")):
        expansions.append("短期内快速减重 减重速度过快 危及生命 快速反弹 每月 2 4kg 每周 0.5kg 不可急于求成")
    if any(kw in text for kw in ("BMI", "肥胖", "超重", "腰围")):
        expansions.append("成人肥胖判定标准 BMI 腰围 WC 中心型肥胖 体重分类")
    if any(kw in text for kw in ("腰围", "肚子", "腹型", "中心型", "中心性", "女生", "女性", "男生", "男性")):
        expansions.append("成年男性 WC 90cm 成年女性 WC 85cm 中心型肥胖 中心型肥胖前期")
    if any(kw in text for kw in ("食谱", "菜谱", "1600", "1200", "1400")):
        expansions.append("食谱 总能量 早餐 中餐 晚餐 加餐 宏量营养素")

    # ---- 糖尿病相关 ----
    if any(kw in text for kw in ("适用", "不适用", "范围", "人群")):
        expansions.append("范围 本标准适用 本标准不适用 特殊糖尿病人群 1型糖尿病儿童 妊娠糖尿病")
    if any(kw in text for kw in ("糖尿病", "血糖", "GI", "GL", "交换份")):
        expansions.append("糖尿病 医学营养治疗 膳食原则 交换份 GI GL 碳水化合物 血糖")
    if any(kw in text for kw in ("碳水", "碳水化合物", "供能比", "供能")):
        expansions.append("3.4 碳水化合物 占总能量 50% 60% 低 GI GL 食物")

    # ---- 高尿酸/痛风相关 ----
    if any(kw in text for kw in ("高尿酸", "痛风", "尿酸", "嘌呤")):
        expansions.append("高尿酸血症 痛风 嘌呤 低嘌呤 高嘌呤 血尿酸 饮水 限酒 果糖 动物内脏 海鲜")
    if any(kw in text for kw in ("饮水", "喝水", "喝够", "多少水", "饮酒", "喝酒", "啤酒", "白酒", "黄酒")):
        expansions.append("足量饮水 限制饮酒 酒精 啤酒 黄酒 白酒 尿酸排泄")

    # ---- 高血压相关 ----
    if any(kw in text for kw in ("高血压", "血压")) and any(kw in text for kw in ("生活方式", "除了吃药", "怎么改", "运动", "盐", "饮酒")):
        expansions.append("治疗性生活方式干预 限制钠盐 控制体重 限制长期饮酒 有氧运动 戒烟 健康睡眠")

    # ---- 食谱/做法类 ----
    if any(kw in text for kw in ("做法", "怎么做", "烹饪", "煲汤", "炖")):
        expansions.append("食养 食谱 做法 材料 烹饪 中医食疗")
    if any(kw in text for kw in ("铁皮石斛", "玉竹", "药膳", "食养", "食疗")):
        expansions.append("食养方 药食同源 中医食疗 材料 用量")

    # ---- 禁忌/不能吃类 ----
    if any(kw in text for kw in ("不能吃", "禁忌", "避免", "少吃", "忌口")):
        expansions.append("禁忌 避免 限制 不宜 减少摄入 忌口")
    if any(kw in text for kw in ("能吃", "可以吃", "推荐", "适合")):
        expansions.append("推荐 适宜 建议 可以食用 推荐食物")

    # ---- 蛋白质/脂肪/碳水具体数字 ----
    if any(kw in text for kw in ("蛋白质", "蛋白")):
        expansions.append("蛋白质 优质蛋白 蛋白质供能比 肾功能")
    if any(kw in text for kw in ("脂肪", "油脂", "胆固醇")):
        expansions.append("脂肪 脂肪酸 胆固醇 烹调油 供能比")
    if any(kw in text for kw in ("膳食纤维", "纤维")):
        expansions.append("膳食纤维 全谷物 蔬菜 水果 摄入量")

    # ---- 能量/热量类 ----
    if any(kw in text for kw in ("能量", "热量", "千卡", "kcal")):
        expansions.append("能量 热量 千卡 kcal 每日推荐摄入量 能量系数")

    # ---- 肾功能/并发症类 ----
    if any(kw in text for kw in ("肾", "肾脏", "肾功能")):
        expansions.append("肾功能不全 慢性肾脏病 蛋白质限制 钾 磷")
    if any(kw in text for kw in ("早餐", "晚餐", "加餐", "午餐")):
        expansions.append("早餐 午餐 晚餐 加餐 餐次分配 进餐时间")

    return " ".join([text] + expansions)


def _adjust_rag_candidate_score(item: dict[str, Any], question: str) -> float:
    score = float(item.get("score") or 0.0)
    text = str(item.get("text") or "")
    compact = re.sub(r"\s+", "", text)
    source = str(item.get("source_name") or "").lower()
    section = str(item.get("section_title") or "")
    section_compact = re.sub(r"\s+", "", section)
    chunk_type = str(item.get("chunk_type") or "")

    # 目录页/无效内容惩罚
    if (
        "目录" in compact[:200]
        or compact.count("..") >= 8
        or compact.count("…") >= 8
        or compact.count("·") >= 20
    ):
        score *= 0.18
    table_like_query = any(
        kw in question
        for kw in ("表", "含量", "多少mg", "mg/100", "嘌呤", "GI", "GL", "交换份", "能不能吃", "可以吃", "不能吃", "不宜", "禁忌", "海鲜", "内脏")
    )
    if chunk_type == "table" and not table_like_query:
        score *= 0.55
    obesity_intent = any(kw in question for kw in ("减肥", "减重", "瘦", "少吃", "控制体重", "热量", "能量", "肥胖", "超重"))
    obesity_energy_intent = any(kw in question for kw in ("少吃", "热量", "能量", "千卡", "kcal", "摄入"))
    rapid_weight_loss_intent = any(kw in question for kw in ("三天", "七天", "一周", "十斤", "暴瘦", "速瘦", "瘦十斤", "快速减重"))
    hypertension_lifestyle_intent = (
        any(kw in question for kw in ("高血压", "血压"))
        and any(kw in question for kw in ("生活方式", "除了吃药", "怎么改", "盐", "运动", "饮酒", "减重"))
    )
    if rapid_weight_loss_intent and chunk_type == "recipe_plan":
        score *= 0.25

    # 文档来源辅助函数：判断 chunk 是否来自相关文档
    def _source_matches(*keywords: str) -> bool:
        return any(kw in source for kw in keywords)

    # 章节标题加分辅助：section_title 匹配比正文匹配更有意义
    def _section_boost(keywords: tuple[str, ...], bonus: float) -> float:
        if any(kw in section_compact for kw in keywords):
            return bonus
        return 0.0

    # ---- 肥胖相关 ----
    if obesity_intent:
        if _source_matches("肥胖", "obesity", "weight"):
            if obesity_energy_intent and any(kw in compact for kw in ("控制总能量摄入", "500～1000kcal", "500-1000kcal", "限能量平衡膳食", "每日能量摄入")):
                score += 0.85
            if any(kw in compact for kw in ("安全减重", "达到并保持健康体重")):
                score += 0.25
            score += _section_boost(("能量", "减重", "体重", "肥胖"), 0.2)
        elif not any(kw in question for kw in ("糖尿病", "血糖", "高尿酸", "痛风", "血压", "高血压")):
            score *= 0.72
    if rapid_weight_loss_intent:
        if _source_matches("肥胖", "obesity", "weight"):
            if any(kw in compact for kw in ("短期内快速减重", "减重速度过快", "危及生命", "快速反弹", "每月减2～4kg", "每周0.5kg", "不可急于求成")):
                score += 1.0
    if any(kw in question for kw in ("运动", "活动", "锻炼", "多久")):
        if _source_matches("肥胖", "obesity", "weight"):
            if any(kw in text for kw in ("多动少静", "身体活动", "睡眠充足", "作息规律")):
                score += 0.3
            score += _section_boost(("运动", "身体活动", "锻炼"), 0.2)
    if any(kw in question for kw in ("减重速度", "减肥速度", "减重目标")):
        if _source_matches("肥胖", "obesity", "weight"):
            if any(kw in text for kw in ("6个月", "5%～10%", "5%-10%", "每月减2～4kg", "每周0.5kg")):
                score += 0.4
            score += _section_boost(("减重", "目标", "速度"), 0.2)
    if any(kw in question for kw in ("BMI", "肥胖", "超重", "腰围")):
        if _source_matches("肥胖", "obesity", "weight"):
            score += _section_boost(("诊断", "判定", "标准", "BMI", "腰围"), 0.3)
            if any(kw in question for kw in ("腰围", "肚子", "腹型", "中心型", "中心性", "女生", "女性", "男生", "男性")):
                if any(kw in compact for kw in ("成年女性WC≥85cm", "成年女性wc≥85cm", "女性WC≥85cm", "女性wc≥85cm", "中心型肥胖", "中心性肥胖", "WC≥90cm", "wc≥90cm")):
                    score += 0.9

    # ---- 糖尿病相关 ----
    if any(kw in question for kw in ("适用", "不适用", "范围", "人群")):
        if _source_matches("糖尿病", "diabetes"):
            if any(kw in compact for kw in ("范围", "本标准适用", "本标准不适用", "特殊糖尿病人群")):
                score += 0.5
    if any(kw in question for kw in ("糖尿病", "血糖", "GI", "GL", "交换份")):
        if _source_matches("糖尿病", "diabetes"):
            if any(kw in compact for kw in ("糖尿病", "血糖", "食物交换份", "碳水化合物", "GI", "GL")):
                score += 0.25
            score += _section_boost(("膳食", "营养", "原则", "GI", "GL"), 0.2)
    if any(kw in question for kw in ("碳水", "碳水化合物", "供能比", "供能")):
        if _source_matches("糖尿病", "diabetes"):
            if "碳水化合物" in compact and any(kw in compact for kw in ("50%一60%", "50%～60%", "50%-60%", "总能量")):
                score += 0.5
            score += _section_boost(("碳水", "宏量", "供能"), 0.2)

    # ---- 高尿酸/痛风相关 ----
    if any(kw in question for kw in ("高尿酸", "痛风", "尿酸", "嘌呤")):
        if _source_matches("hyperuricemia", "gout", "尿酸", "痛风"):
            if any(kw in compact for kw in ("高尿酸血症", "痛风", "嘌呤", "血尿酸")):
                score += 0.3
            if any(kw in compact for kw in ("动物内脏", "海鲜", "低嘌呤", "高嘌呤")):
                score += 0.2
            score += _section_boost(("嘌呤", "食物", "饮食", "限制"), 0.2)
            if any(kw in question for kw in ("无症状", "没症状", "没有症状", "体检", "需要管")):
                if any(kw in compact for kw in ("无症状高尿酸血症", "非同日2次", "非同日两次", "420μmol/L", "独立危险因素", "控制血尿酸")):
                    score += 0.55
    if any(kw in question for kw in ("诊断", "标准", "定义", "分期")):
        if _source_matches("hyperuricemia", "gout", "尿酸", "痛风"):
            if any(kw in compact for kw in ("非同日2次", "非同日两次", "420μmol/L", "420umol/L", "诊断为高尿酸血症")):
                score += 0.5
            if any(kw in compact for kw in ("定义与分期", "疾病特点与分型", "急性痛风性关节炎期", "慢性痛风性关节炎期")):
                score += 0.3
    if any(kw in question for kw in ("饮水", "喝水", "喝够", "多少水", "饮酒", "喝酒", "啤酒", "白酒", "黄酒")):
        if _source_matches("hyperuricemia", "gout", "尿酸", "痛风"):
            if any(kw in compact for kw in ("足量饮水", "限制饮酒", "酒精", "啤酒", "黄酒", "白酒", "尿酸排泄")):
                score += 0.4
            if any(kw in compact for kw in ("2000～3000mL", "2000~3000mL", "2000-3000mL", "尿量大于2000mL")):
                score += 0.4
            score += _section_boost(("饮水", "饮酒", "酒精", "液体"), 0.2)

    # ---- 高血压相关 ----
    if hypertension_lifestyle_intent:
        if _source_matches("高血压", "hypertension"):
            if any(kw in compact for kw in ("治疗性生活方式干预", "限制钠盐", "控制体重", "限制长期饮酒", "有氧运动", "健康睡眠")):
                score += 0.75
            score += _section_boost(("生活方式", "干预", "钠盐", "运动"), 0.25)

    # ---- 跨文档问题：不加 source 限制的通用加分（幅度更小）----
    # 食谱/做法类问题
    if any(kw in question for kw in ("食谱", "菜谱", "做法", "怎么吃")):
        score += _section_boost(("食谱", "菜谱", "食养", "做法"), 0.25)
    # 禁忌/不能吃类问题
    if any(kw in question for kw in ("不能吃", "禁忌", "避免", "少吃")):
        score += _section_boost(("禁忌", "避免", "限制", "不宜"), 0.2)

    return score


def _loads_json(value: str, default: Any) -> Any:
    try:
        return json.loads(value or "")
    except Exception:
        return default


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _safe_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _safe_float(value: Any, default: float, *, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _optional_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _compatible_base_url(base_url: Any) -> str:
    raw = str(base_url or "").strip()
    if not raw:
        return ""
    if raw.rstrip("/").endswith("/api/v1"):
        return raw.rstrip("/")[: -len("/api/v1")] + "/compatible-mode/v1"
    return raw


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _relative_to(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)
