from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MEAL_NAMES = ("早餐", "中餐", "午餐", "晚餐", "加餐")
RECIPE_PLAN_RE = re.compile(r"^(春季|夏季|秋季|冬季)食谱\s*(\d+)(?:（总能量约\s*(\d+)\s*kcal）)?")
THERAPEUTIC_RECIPE_RE = re.compile(r"^（([一二三四五六七八九十]+)）(.+?)[。.]?$")
SYNDROME_RE = re.compile(r"^[一二三四五六七八九十]+、(.+证)$")
PAREN_SYNDROME_RE = re.compile(r"^（[一二三四五六七八九十]+）(.+?证)[。.]?$")
NUMBERED_RECIPE_RE = re.compile(r"^\d+[.．]\s*(.+)$")


@dataclass
class StructuredBlock:
    block_type: str
    title: str
    text: str
    page_start: int
    page_end: int
    metadata: dict[str, Any]


class StructuredKnowledgeStore:
    """Stores guideline tables, recipes, exchange portions, and other structured evidence."""

    def __init__(self, *, project_root: Path, config: dict[str, Any] | None = None):
        self.project_root = Path(project_root).resolve()
        self.config = config or {}
        self.db_path = self._resolve_db_path()
        self._init_db()

    def ingest_document(self, document: dict[str, Any], pages: list[Any]) -> dict[str, Any]:
        document_id = str(document.get("document_id") or "").strip()
        if not document_id:
            raise ValueError("missing document_id for structured knowledge ingestion")

        source = self._source_payload(document, pages)
        blocks = extract_structured_blocks_for_rag(pages)
        tables = extract_tables(pages)
        recipe_plans = extract_recipe_plans(pages)
        therapeutic_recipes = extract_therapeutic_recipes(pages)

        with self._connect() as db:
            self._delete_document_rows(db, document_id)
            db.execute(
                """
                INSERT OR REPLACE INTO source_documents (
                    document_id, source_hash, original_name, stored_path, title,
                    page_count, char_count, review_status, review_method, review_summary,
                    reviewed_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    source["source_hash"],
                    source["original_name"],
                    source["stored_path"],
                    source["title"],
                    source["page_count"],
                    source["char_count"],
                    "auto_extracted",
                    "",
                    "",
                    "",
                    source["updated_at"],
                    source["updated_at"],
                ),
            )
            for table in tables:
                self._insert_table(db, document_id, table)
            for plan in recipe_plans:
                self._insert_recipe_plan(db, document_id, plan)
            for recipe in therapeutic_recipes:
                self._insert_therapeutic_recipe(db, document_id, recipe)

        return {
            "document_id": document_id,
            "blocks": len(blocks),
            "tables": len(tables),
            "recipe_plans": len(recipe_plans),
            "therapeutic_recipes": len(therapeutic_recipes),
            "db_path": _relative_to(self.project_root, self.db_path),
        }

    def delete_document(self, document_id: str) -> None:
        with self._connect() as db:
            self._delete_document_rows(db, document_id)

    def summary(self) -> dict[str, int]:
        with self._connect() as db:
            return {
                "documents": _count(db, "source_documents"),
                "tables": _count(db, "guide_tables"),
                "table_rows": _count(db, "guide_table_rows"),
                "food_exchange_portions": _count(db, "food_exchange_portions"),
                "recipe_plans": _count(db, "recipe_plans"),
                "therapeutic_recipes": _count(db, "therapeutic_recipes"),
                "activity_mets": _count(db, "activity_mets"),
            }

    def document_review_summary(self, document_id: str) -> dict[str, Any]:
        document_id = str(document_id or "").strip()
        if not document_id:
            raise ValueError("missing document_id")
        with self._connect() as db:
            document = db.execute(
                "SELECT * FROM source_documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            if not document:
                raise ValueError("structured knowledge document not found")
            return {
                "document": dict(document),
                "counts": {
                    "guide_tables": _count_where(db, "guide_tables", "document_id = ?", [document_id]),
                    "guide_table_rows": self._count_table_rows_for_document(db, document_id),
                    "food_exchange_portions": _count_where(db, "food_exchange_portions", "document_id = ?", [document_id]),
                    "recipe_plans": _count_where(db, "recipe_plans", "document_id = ?", [document_id]),
                    "recipe_meals": self._count_recipe_children_for_document(db, document_id, "recipe_meals"),
                    "recipe_dishes": self._count_recipe_children_for_document(db, document_id, "recipe_dishes"),
                    "recipe_ingredients": self._count_recipe_children_for_document(db, document_id, "recipe_ingredients"),
                    "therapeutic_recipes": _count_where(db, "therapeutic_recipes", "document_id = ?", [document_id]),
                    "activity_mets": _count_where(db, "activity_mets", "document_id = ?", [document_id]),
                },
                "review_status_counts": {
                    "guide_tables": self._status_counts(db, "guide_tables", document_id),
                    "recipe_plans": self._status_counts(db, "recipe_plans", document_id),
                    "therapeutic_recipes": self._status_counts(db, "therapeutic_recipes", document_id),
                    "activity_mets": self._status_counts(db, "activity_mets", document_id),
                },
                "samples": {
                    "tables": self._sample_tables(db, document_id),
                    "recipe_plans": self._sample_recipe_plans(db, document_id),
                    "therapeutic_recipes": self._sample_therapeutic_recipes(db, document_id),
                    "activity_mets": self._sample_activity_mets(db, document_id),
                },
            }

    def set_document_review(
        self,
        document_id: str,
        *,
        review_status: str,
        review_method: str,
        review_summary: str,
    ) -> dict[str, Any]:
        document_id = str(document_id or "").strip()
        if not document_id:
            raise ValueError("missing document_id")
        now = _utc_now()
        with self._connect() as db:
            row = db.execute(
                "SELECT 1 FROM source_documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            if not row:
                raise ValueError("structured knowledge document not found")
            db.execute(
                """
                UPDATE source_documents
                SET review_status = ?, review_method = ?, review_summary = ?,
                    reviewed_at = ?, updated_at = ?
                WHERE document_id = ?
                """,
                (
                    str(review_status or "reviewed"),
                    str(review_method or ""),
                    str(review_summary or ""),
                    now,
                    now,
                    document_id,
                ),
            )
        return self.document_review_summary(document_id)

    def approve_document(self, document_id: str, *, approved_by: str = "console_admin") -> dict[str, Any]:
        document_id = str(document_id or "").strip()
        if not document_id:
            raise ValueError("missing document_id")
        now = _utc_now()
        summary = f"Structured extraction approved by {approved_by} at {now}."
        with self._connect() as db:
            row = db.execute(
                "SELECT 1 FROM source_documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            if not row:
                raise ValueError("structured knowledge document not found")
            for table in (
                "guide_tables",
                "food_exchange_portions",
                "recipe_plans",
                "therapeutic_recipes",
                "activity_mets",
                "diagnostic_thresholds",
                "nutrition_targets",
                "safety_rule_candidates",
            ):
                db.execute(
                    f"UPDATE {table} SET review_status = 'approved' WHERE document_id = ?",
                    (document_id,),
                )
            db.execute(
                """
                UPDATE source_documents
                SET review_status = 'approved', review_method = 'human_console',
                    review_summary = ?, reviewed_at = ?, updated_at = ?
                WHERE document_id = ?
                """,
                (summary, now, now, document_id),
            )
        return self.document_review_summary(document_id)

    def resolve_needs_review_item(
        self,
        review_id: str,
        *,
        status: str,
        reviewer_notes: str = "",
    ) -> dict[str, Any]:
        review_id = str(review_id or "").strip()
        if not review_id:
            raise ValueError("missing review_id")
        normalized = str(status or "").strip().lower()
        if normalized not in {"approved", "discarded", "pending"}:
            raise ValueError("review status must be approved, discarded, or pending")
        now = _utc_now()
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM needs_review WHERE review_id = ?",
                (review_id,),
            ).fetchone()
            if row is None:
                raise ValueError("needs_review item not found")
            note = str(reviewer_notes or "").strip() or f"updated_at={now}"
            db.execute(
                """
                UPDATE needs_review
                SET review_status = ?, reviewer_notes = ?
                WHERE review_id = ?
                """,
                (normalized, note, review_id),
            )
            updated = db.execute(
                "SELECT * FROM needs_review WHERE review_id = ?",
                (review_id,),
            ).fetchone()
        return dict(updated)

    def ingest_from_plan(
        self,
        document: dict[str, Any],
        pages: list[Any],
        extraction_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Ingest structured data extracted via DocumentProfiler + schema validation.

        Args:
            document: Document metadata dict with document_id, original_name, etc.
            pages: List of page objects (for source_payload).
            extraction_results: Output from DocumentProfiler.extract_structured_blocks():
                {
                    "extracted": {block_type: [validated_dict, ...]},
                    "needs_review": [NeedsReviewItem, ...],
                    "stats": {block_type: count},
                }
        """
        document_id = str(document.get("document_id") or "").strip()
        if not document_id:
            raise ValueError("missing document_id for structured knowledge ingestion")

        source = self._source_payload(document, pages)
        extracted = extraction_results.get("extracted") or {}
        needs_review_items = extraction_results.get("needs_review") or []
        inserted_needs_review_count = len(needs_review_items)

        with self._connect() as db:
            self._delete_document_rows(db, document_id)
            # Upsert source document
            db.execute(
                """
                INSERT OR REPLACE INTO source_documents (
                    document_id, source_hash, original_name, stored_path, title,
                    page_count, char_count, review_status, review_method, review_summary,
                    reviewed_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    source["source_hash"],
                    source["original_name"],
                    source["stored_path"],
                    source["title"],
                    source["page_count"],
                    source["char_count"],
                    "auto_extracted",
                    "llm_schema_validated",
                    "",
                    "",
                    source["updated_at"],
                    source["updated_at"],
                ),
            )

            # Insert extracted items by type
            for block_type, items in extracted.items():
                for item in items:
                    try:
                        self._insert_extracted_item(db, document_id, block_type, item)
                    except Exception as exc:
                        source_pages = item.get("source_pages") or []
                        inserted_needs_review_count += 1
                        self._insert_needs_review(
                            db,
                            document_id,
                            {
                                "block_type": block_type,
                                "page_start": item.get("page_start") or (source_pages[0] if source_pages else 0),
                                "page_end": item.get("page_end") or (source_pages[-1] if source_pages else 0),
                                "raw_text": item.get("raw_text") or json.dumps(item, ensure_ascii=False),
                                "llm_output": json.dumps(item, ensure_ascii=False),
                                "schema_errors": str(exc),
                                "confidence": item.get("confidence") or 0.4,
                            },
                        )

            # Insert needs_review items
            for item in needs_review_items:
                self._insert_needs_review(db, document_id, item)
            rule_based_stats = self._insert_rule_based_items(db, document_id, pages)

        stats = dict(extraction_results.get("stats") or {})
        for key, value in rule_based_stats.items():
            stats[key] = int(stats.get(key) or 0) + int(value or 0)

        return {
            "document_id": document_id,
            "stats": stats,
            "needs_review_count": inserted_needs_review_count,
            "db_path": _relative_to(self.project_root, self.db_path),
        }

    def _insert_extracted_item(
        self,
        db: sqlite3.Connection,
        document_id: str,
        block_type: str,
        item: dict[str, Any],
    ) -> None:
        """Route a validated extraction item to the correct insert method."""
        if block_type in ("guide_table", "generic_table"):
            self._insert_table(db, document_id, item)
        elif block_type == "food_exchange_portion":
            self._insert_exchange_portion(db, document_id, item)
        elif block_type == "recipe_plan":
            self._insert_recipe_plan(db, document_id, item)
        elif block_type == "therapeutic_recipe":
            self._insert_therapeutic_recipe(db, document_id, item)
        elif block_type == "activity_met":
            self._insert_activity_met(db, document_id, item)
        elif block_type == "diagnostic_threshold":
            self._insert_diagnostic_threshold(db, document_id, item)
        elif block_type == "nutrition_target":
            self._insert_nutrition_target(db, document_id, item)
        elif block_type in ("safety_rule_candidate", "contraindication"):
            self._insert_safety_rule_candidate(db, document_id, item)
        else:
            # Fallback: store as generic table
            self._insert_table(db, document_id, item)

    def _insert_exchange_portion(self, db: sqlite3.Connection, document_id: str, item: dict[str, Any]) -> None:
        portion_id = _stable_id("fep", document_id, item.get("food_name"), item.get("page_start"))
        source_pages = item.get("source_pages") or []
        page_start = int(item.get("page_start") or (source_pages[0] if source_pages else 1))
        page_end = int(item.get("page_end") or (source_pages[-1] if source_pages else page_start))
        db.execute(
            """
            INSERT INTO food_exchange_portions (
                portion_id, document_id, food_name, exchange_group, serving_amount,
                energy_kcal, carbohydrate_g, protein_g, fat_g,
                page_start, page_end, raw_text, confidence, review_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                portion_id, document_id,
                str(item.get("food_name") or ""),
                str(item.get("exchange_group") or ""),
                str(item.get("serving_amount") or ""),
                _to_float(item.get("energy_kcal")),
                _to_float(item.get("carbohydrate_g")),
                _to_float(item.get("protein_g")),
                _to_float(item.get("fat_g")),
                page_start, page_end,
                str(item.get("raw_text") or ""),
                float(item.get("confidence") or 0.75),
                str(item.get("review_status") or "auto_extracted"),
            ),
        )

    def _insert_activity_met(self, db: sqlite3.Connection, document_id: str, item: dict[str, Any]) -> None:
        activity_id = _stable_id("met", document_id, item.get("activity_name"), item.get("page_start"))
        source_pages = item.get("source_pages") or []
        page_start = int(item.get("page_start") or (source_pages[0] if source_pages else 1))
        page_end = int(item.get("page_end") or (source_pages[-1] if source_pages else page_start))
        db.execute(
            """
            INSERT INTO activity_mets (
                activity_id, document_id, category, activity_name, met,
                intensity, page_start, page_end, raw_text, confidence, review_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                activity_id, document_id,
                str(item.get("category") or ""),
                str(item.get("activity_name") or ""),
                float(item.get("met") or 0),
                str(item.get("intensity") or ""),
                page_start, page_end,
                str(item.get("raw_text") or ""),
                float(item.get("confidence") or 0.85),
                str(item.get("review_status") or "auto_extracted"),
            ),
        )

    def _insert_diagnostic_threshold(self, db: sqlite3.Connection, document_id: str, item: dict[str, Any]) -> None:
        threshold_id = _stable_id("dth", document_id, item.get("indicator"), item.get("threshold"))
        now = _utc_now()
        source_pages = item.get("source_pages") or []
        page_start = int(item.get("page_start") or (source_pages[0] if source_pages else 1))
        page_end = int(item.get("page_end") or (source_pages[-1] if source_pages else page_start))
        db.execute(
            """
            INSERT INTO diagnostic_thresholds (
                threshold_id, document_id, indicator, threshold, unit, population,
                context, page_start, page_end, raw_text, confidence, review_status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                threshold_id, document_id,
                str(item.get("indicator") or ""),
                str(item.get("threshold") or ""),
                str(item.get("unit") or ""),
                str(item.get("population") or ""),
                str(item.get("context") or ""),
                page_start, page_end,
                str(item.get("raw_text") or ""),
                float(item.get("confidence") or 0.8),
                str(item.get("review_status") or "auto_extracted"),
                now,
            ),
        )

    def _insert_nutrition_target(self, db: sqlite3.Connection, document_id: str, item: dict[str, Any]) -> None:
        target_id = _stable_id("ntg", document_id, item.get("nutrient"), item.get("target_value"))
        now = _utc_now()
        source_pages = item.get("source_pages") or []
        page_start = int(item.get("page_start") or (source_pages[0] if source_pages else 1))
        page_end = int(item.get("page_end") or (source_pages[-1] if source_pages else page_start))
        db.execute(
            """
            INSERT INTO nutrition_targets (
                target_id, document_id, nutrient, target_value, population,
                context, page_start, page_end, raw_text, confidence, review_status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                target_id, document_id,
                str(item.get("nutrient") or ""),
                str(item.get("target_value") or ""),
                str(item.get("population") or ""),
                str(item.get("context") or ""),
                page_start, page_end,
                str(item.get("raw_text") or ""),
                float(item.get("confidence") or 0.8),
                str(item.get("review_status") or "auto_extracted"),
                now,
            ),
        )

    def _insert_safety_rule_candidate(self, db: sqlite3.Connection, document_id: str, item: dict[str, Any]) -> None:
        rule_id = _stable_id("src", document_id, item.get("trigger_condition"), item.get("page_start"))
        now = _utc_now()
        source_pages = item.get("source_pages") or []
        page_start = int(item.get("page_start") or (source_pages[0] if source_pages else 1))
        page_end = int(item.get("page_end") or (source_pages[-1] if source_pages else page_start))
        db.execute(
            """
            INSERT INTO safety_rule_candidates (
                rule_id, document_id, trigger_condition, risk_description,
                safety_recommendation, severity, page_start, page_end,
                raw_text, confidence, review_status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rule_id, document_id,
                str(item.get("trigger_condition") or ""),
                str(item.get("risk_description") or ""),
                str(item.get("safety_recommendation") or ""),
                str(item.get("severity") or "warn"),
                page_start, page_end,
                str(item.get("raw_text") or ""),
                float(item.get("confidence") or 0.7),
                str(item.get("review_status") or "auto_extracted"),
                now,
            ),
        )

    def _insert_rule_based_items(
        self,
        db: sqlite3.Connection,
        document_id: str,
        pages: list[Any],
    ) -> dict[str, int]:
        stats: dict[str, int] = {}
        for fallback in _rule_based_nutrition_targets(pages):
            self._insert_nutrition_target(db, document_id, fallback)
            stats["rule_based_nutrition_target"] = stats.get("rule_based_nutrition_target", 0) + 1
        return stats

    def _insert_needs_review(self, db: sqlite3.Connection, document_id: str, item: Any) -> None:
        """Insert a needs_review item (from NeedsReviewItem or dict)."""
        if hasattr(item, "model_dump"):
            d = item.model_dump()
        elif isinstance(item, dict):
            d = item
        else:
            return

        review_id = d.get("review_id") or _stable_id(
            "rev",
            document_id,
            d.get("block_id"),
            d.get("block_type"),
            d.get("page_start"),
            d.get("raw_text"),
            d.get("schema_errors"),
        )
        now = _utc_now()
        db.execute(
            """
            INSERT OR REPLACE INTO needs_review (
                review_id, document_id, block_id, block_type, section_path,
                page_start, page_end, raw_text, llm_output, schema_errors,
                confidence, review_status, reviewer_notes, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                review_id, document_id,
                str(d.get("block_id") or ""),
                str(d.get("block_type") or ""),
                str(d.get("section_path") or ""),
                int(d.get("page_start") or 0),
                int(d.get("page_end") or 0),
                str(d.get("raw_text") or ""),
                str(d.get("llm_output") or ""),
                str(d.get("schema_errors") or ""),
                float(d.get("confidence") or 0.5),
                str(d.get("review_status") or "pending"),
                str(d.get("reviewer_notes") or ""),
                now,
            ),
        )

    def _resolve_db_path(self) -> Path:
        knowledge_config = self.config.get("clinical_knowledge") or {}
        configured = knowledge_config.get("db_path") or "data/clinical_knowledge.db"
        path = Path(str(configured))
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

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
                CREATE TABLE IF NOT EXISTS source_documents (
                    document_id TEXT PRIMARY KEY,
                    source_hash TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    title TEXT NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    char_count INTEGER NOT NULL DEFAULT 0,
                    review_status TEXT NOT NULL DEFAULT 'auto_extracted',
                    review_method TEXT NOT NULL DEFAULT '',
                    review_summary TEXT NOT NULL DEFAULT '',
                    reviewed_at TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS guide_tables (
                    table_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    table_label TEXT NOT NULL,
                    title TEXT NOT NULL,
                    table_type TEXT NOT NULL DEFAULT 'generic',
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    raw_text TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.8,
                    review_status TEXT NOT NULL DEFAULT 'auto_extracted',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES source_documents(document_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS guide_table_rows (
                    row_id TEXT PRIMARY KEY,
                    table_id TEXT NOT NULL,
                    row_index INTEGER NOT NULL,
                    label TEXT NOT NULL DEFAULT '',
                    columns_json TEXT NOT NULL DEFAULT '{}',
                    raw_text TEXT NOT NULL,
                    FOREIGN KEY(table_id) REFERENCES guide_tables(table_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS food_exchange_portions (
                    portion_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    table_id TEXT,
                    food_name TEXT NOT NULL,
                    exchange_group TEXT NOT NULL DEFAULT '',
                    serving_amount TEXT NOT NULL DEFAULT '',
                    energy_kcal REAL,
                    carbohydrate_g REAL,
                    protein_g REAL,
                    fat_g REAL,
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    raw_text TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.75,
                    review_status TEXT NOT NULL DEFAULT 'auto_extracted',
                    FOREIGN KEY(document_id) REFERENCES source_documents(document_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS recipe_plans (
                    plan_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    season TEXT NOT NULL DEFAULT '',
                    plan_index INTEGER,
                    energy_kcal REAL,
                    protein_g REAL,
                    carbohydrate_g REAL,
                    fat_g REAL,
                    protein_pct REAL,
                    carbohydrate_pct REAL,
                    fat_pct REAL,
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    raw_text TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.8,
                    review_status TEXT NOT NULL DEFAULT 'auto_extracted',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES source_documents(document_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS recipe_meals (
                    meal_id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    meal_type TEXT NOT NULL,
                    sort_order INTEGER NOT NULL,
                    FOREIGN KEY(plan_id) REFERENCES recipe_plans(plan_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS recipe_dishes (
                    dish_id TEXT PRIMARY KEY,
                    meal_id TEXT NOT NULL,
                    dish_name TEXT NOT NULL,
                    raw_text TEXT NOT NULL,
                    sort_order INTEGER NOT NULL,
                    FOREIGN KEY(meal_id) REFERENCES recipe_meals(meal_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS recipe_ingredients (
                    ingredient_id TEXT PRIMARY KEY,
                    dish_id TEXT NOT NULL,
                    ingredient_name TEXT NOT NULL,
                    amount REAL,
                    unit TEXT NOT NULL DEFAULT '',
                    is_medicinal INTEGER NOT NULL DEFAULT 0,
                    raw_text TEXT NOT NULL,
                    FOREIGN KEY(dish_id) REFERENCES recipe_dishes(dish_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS therapeutic_recipes (
                    recipe_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    syndrome TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL,
                    ingredients_json TEXT NOT NULL DEFAULT '[]',
                    method TEXT NOT NULL DEFAULT '',
                    usage TEXT NOT NULL DEFAULT '',
                    cautions TEXT NOT NULL DEFAULT '',
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    raw_text TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.8,
                    review_status TEXT NOT NULL DEFAULT 'auto_extracted',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES source_documents(document_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS activity_mets (
                    activity_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    table_id TEXT,
                    category TEXT NOT NULL DEFAULT '',
                    activity_name TEXT NOT NULL,
                    met REAL NOT NULL,
                    intensity TEXT NOT NULL DEFAULT '',
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    raw_text TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.85,
                    review_status TEXT NOT NULL DEFAULT 'auto_extracted',
                    FOREIGN KEY(document_id) REFERENCES source_documents(document_id) ON DELETE CASCADE,
                    FOREIGN KEY(table_id) REFERENCES guide_tables(table_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS diagnostic_thresholds (
                    threshold_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    indicator TEXT NOT NULL,
                    threshold TEXT NOT NULL,
                    unit TEXT NOT NULL DEFAULT '',
                    population TEXT NOT NULL DEFAULT '',
                    context TEXT NOT NULL DEFAULT '',
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    raw_text TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.8,
                    review_status TEXT NOT NULL DEFAULT 'auto_extracted',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES source_documents(document_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS nutrition_targets (
                    target_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    nutrient TEXT NOT NULL,
                    target_value TEXT NOT NULL,
                    population TEXT NOT NULL DEFAULT '',
                    context TEXT NOT NULL DEFAULT '',
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    raw_text TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.8,
                    review_status TEXT NOT NULL DEFAULT 'auto_extracted',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES source_documents(document_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS safety_rule_candidates (
                    rule_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    trigger_condition TEXT NOT NULL,
                    risk_description TEXT NOT NULL,
                    safety_recommendation TEXT NOT NULL,
                    severity TEXT NOT NULL DEFAULT 'warn',
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    raw_text TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.7,
                    review_status TEXT NOT NULL DEFAULT 'auto_extracted',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES source_documents(document_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS needs_review (
                    review_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    block_id TEXT NOT NULL,
                    block_type TEXT NOT NULL,
                    section_path TEXT NOT NULL DEFAULT '',
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    raw_text TEXT NOT NULL,
                    llm_output TEXT NOT NULL DEFAULT '',
                    schema_errors TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    review_status TEXT NOT NULL DEFAULT 'pending',
                    reviewer_notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES source_documents(document_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_guide_table_rows_label ON guide_table_rows(label);
                CREATE INDEX IF NOT EXISTS idx_recipe_plans_title ON recipe_plans(title);
                CREATE INDEX IF NOT EXISTS idx_recipe_dishes_name ON recipe_dishes(dish_name);
                CREATE INDEX IF NOT EXISTS idx_therapeutic_recipes_title ON therapeutic_recipes(title);
                CREATE INDEX IF NOT EXISTS idx_activity_mets_name ON activity_mets(activity_name);
                CREATE INDEX IF NOT EXISTS idx_needs_review_status ON needs_review(review_status);
                """
            )
            self._ensure_column(db, "source_documents", "review_status", "TEXT NOT NULL DEFAULT 'auto_extracted'")
            self._ensure_column(db, "source_documents", "review_method", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(db, "source_documents", "review_summary", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(db, "source_documents", "reviewed_at", "TEXT NOT NULL DEFAULT ''")

    def _source_payload(self, document: dict[str, Any], pages: list[Any]) -> dict[str, Any]:
        now = _utc_now()
        def _page_text(page: Any) -> str:
            if isinstance(page, dict):
                return str(page.get("text") or "")
            return str(getattr(page, "text", "") or "")

        return {
            "source_hash": str(document.get("source_hash") or ""),
            "original_name": str(document.get("original_name") or ""),
            "stored_path": str(document.get("stored_path") or ""),
            "title": str(document.get("title") or document.get("original_name") or ""),
            "page_count": len(pages),
            "char_count": sum(len(_page_text(page)) for page in pages),
            "updated_at": now,
        }

    def _ensure_column(self, db: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        existing = {row["name"] for row in db.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _delete_document_rows(self, db: sqlite3.Connection, document_id: str) -> None:
        db.execute("DELETE FROM source_documents WHERE document_id = ?", (document_id,))

    def _count_table_rows_for_document(self, db: sqlite3.Connection, document_id: str) -> int:
        return int(
            db.execute(
                """
                SELECT COUNT(*)
                FROM guide_table_rows gtr
                JOIN guide_tables gt ON gt.table_id = gtr.table_id
                WHERE gt.document_id = ?
                """,
                (document_id,),
            ).fetchone()[0]
        )

    def _count_recipe_children_for_document(
        self,
        db: sqlite3.Connection,
        document_id: str,
        table: str,
    ) -> int:
        joins = {
            "recipe_meals": """
                SELECT COUNT(*)
                FROM recipe_meals rm
                JOIN recipe_plans rp ON rp.plan_id = rm.plan_id
                WHERE rp.document_id = ?
            """,
            "recipe_dishes": """
                SELECT COUNT(*)
                FROM recipe_dishes rd
                JOIN recipe_meals rm ON rm.meal_id = rd.meal_id
                JOIN recipe_plans rp ON rp.plan_id = rm.plan_id
                WHERE rp.document_id = ?
            """,
            "recipe_ingredients": """
                SELECT COUNT(*)
                FROM recipe_ingredients ri
                JOIN recipe_dishes rd ON rd.dish_id = ri.dish_id
                JOIN recipe_meals rm ON rm.meal_id = rd.meal_id
                JOIN recipe_plans rp ON rp.plan_id = rm.plan_id
                WHERE rp.document_id = ?
            """,
        }
        return int(db.execute(joins[table], (document_id,)).fetchone()[0])

    def _status_counts(self, db: sqlite3.Connection, table: str, document_id: str) -> dict[str, int]:
        rows = db.execute(
            f"""
            SELECT review_status, COUNT(*) AS count
            FROM {table}
            WHERE document_id = ?
            GROUP BY review_status
            """,
            (document_id,),
        ).fetchall()
        return {str(row["review_status"] or "unknown"): int(row["count"]) for row in rows}

    def _sample_tables(self, db: sqlite3.Connection, document_id: str) -> list[dict[str, Any]]:
        rows = db.execute(
            """
            SELECT table_label, title, table_type, page_start, page_end,
                   confidence, review_status, raw_text
            FROM guide_tables
            WHERE document_id = ?
            ORDER BY page_start, table_label
            LIMIT 8
            """,
            (document_id,),
        ).fetchall()
        return [
            {
                **dict(row),
                "raw_text": str(row["raw_text"] or "")[:1200],
            }
            for row in rows
        ]

    def _sample_recipe_plans(self, db: sqlite3.Connection, document_id: str) -> list[dict[str, Any]]:
        rows = db.execute(
            """
            SELECT title, season, energy_kcal, protein_g, carbohydrate_g, fat_g,
                   page_start, page_end, confidence, review_status, raw_text
            FROM recipe_plans
            WHERE document_id = ?
            ORDER BY page_start, title
            LIMIT 8
            """,
            (document_id,),
        ).fetchall()
        return [
            {
                **dict(row),
                "raw_text": str(row["raw_text"] or "")[:1200],
            }
            for row in rows
        ]

    def _sample_therapeutic_recipes(self, db: sqlite3.Connection, document_id: str) -> list[dict[str, Any]]:
        rows = db.execute(
            """
            SELECT syndrome, title, ingredients_json, method, usage, cautions,
                   page_start, page_end, confidence, review_status
            FROM therapeutic_recipes
            WHERE document_id = ?
            ORDER BY page_start, title
            LIMIT 8
            """,
            (document_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def _sample_activity_mets(self, db: sqlite3.Connection, document_id: str) -> list[dict[str, Any]]:
        rows = db.execute(
            """
            SELECT category, activity_name, met, intensity, page_start, page_end,
                   confidence, review_status
            FROM activity_mets
            WHERE document_id = ?
            ORDER BY category, activity_name
            LIMIT 8
            """,
            (document_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def _insert_table(self, db: sqlite3.Connection, document_id: str, table: dict[str, Any]) -> str:
        table_label = str(table.get("table_label") or table.get("label") or "").strip()
        table_id = _stable_id("tbl", document_id, table_label, table.get("title"), table.get("page_start"))
        now = _utc_now()
        db.execute(
            """
            INSERT INTO guide_tables (
                table_id, document_id, table_label, title, table_type, page_start, page_end,
                raw_text, confidence, review_status, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                table_id,
                document_id,
                table_label,
                str(table.get("title") or ""),
                str(table.get("table_type") or "generic"),
                int(table.get("page_start") or 1),
                int(table.get("page_end") or table.get("page_start") or 1),
                str(table.get("raw_text") or ""),
                float(table.get("confidence") or 0.8),
                str(table.get("review_status") or "auto_extracted"),
                json.dumps(table.get("metadata") or {}, ensure_ascii=False),
                now,
            ),
        )
        for index, row in enumerate(table.get("rows") or [], start=1):
            row_id = _stable_id("row", table_id, index, row.get("raw_text"))
            db.execute(
                """
                INSERT INTO guide_table_rows (
                    row_id, table_id, row_index, label, columns_json, raw_text
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    table_id,
                    index,
                    str(row.get("label") or ""),
                    json.dumps(row.get("columns") or {}, ensure_ascii=False),
                    str(row.get("raw_text") or ""),
                ),
            )
        if table.get("table_type") == "activity_met":
            for row in table.get("rows") or []:
                columns = row.get("columns") or {}
                met = _to_float(columns.get("met"))
                if met is None:
                    continue
                db.execute(
                    """
                    INSERT INTO activity_mets (
                        activity_id, document_id, table_id, category, activity_name, met,
                        intensity, page_start, page_end, raw_text, confidence, review_status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _stable_id("met", document_id, row.get("raw_text")),
                        document_id,
                        table_id,
                        str(columns.get("category") or ""),
                        str(row.get("label") or ""),
                        met,
                        str(columns.get("intensity") or ""),
                        int(table.get("page_start") or 1),
                        int(table.get("page_end") or table.get("page_start") or 1),
                        str(row.get("raw_text") or ""),
                        0.85,
                        "auto_extracted",
                    ),
                )
        for portion in table.get("exchange_portions") or []:
            db.execute(
                """
                INSERT INTO food_exchange_portions (
                    portion_id, document_id, table_id, food_name, exchange_group,
                    serving_amount, energy_kcal, carbohydrate_g, protein_g, fat_g,
                    page_start, page_end, raw_text, confidence, review_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _stable_id(
                        "portion",
                        document_id,
                        table_id,
                        portion.get("food_name"),
                        portion.get("serving_amount"),
                        portion.get("raw_text"),
                    ),
                    document_id,
                    table_id,
                    str(portion.get("food_name") or ""),
                    str(portion.get("exchange_group") or ""),
                    str(portion.get("serving_amount") or ""),
                    _to_float(portion.get("energy_kcal")),
                    _to_float(portion.get("carbohydrate_g")),
                    _to_float(portion.get("protein_g")),
                    _to_float(portion.get("fat_g")),
                    int(portion.get("page_start") or table.get("page_start") or 1),
                    int(portion.get("page_end") or table.get("page_end") or table.get("page_start") or 1),
                    str(portion.get("raw_text") or ""),
                    float(portion.get("confidence") or 0.68),
                    str(portion.get("review_status") or "auto_extracted"),
                ),
            )
        return table_id

    def _insert_recipe_plan(self, db: sqlite3.Connection, document_id: str, plan: dict[str, Any]) -> None:
        plan_id = _stable_id("plan", document_id, plan.get("title"), plan.get("page_start"))
        now = _utc_now()
        nutrition = plan.get("nutrition") or {}
        protein_g = nutrition.get("protein_g", plan.get("protein_g"))
        carbohydrate_g = nutrition.get("carbohydrate_g", plan.get("carbohydrate_g"))
        fat_g = nutrition.get("fat_g", plan.get("fat_g"))
        protein_pct = nutrition.get("protein_pct", plan.get("protein_pct"))
        carbohydrate_pct = nutrition.get("carbohydrate_pct", plan.get("carbohydrate_pct"))
        fat_pct = nutrition.get("fat_pct", plan.get("fat_pct"))
        db.execute(
            """
            INSERT INTO recipe_plans (
                plan_id, document_id, title, season, plan_index, energy_kcal,
                protein_g, carbohydrate_g, fat_g, protein_pct, carbohydrate_pct,
                fat_pct, page_start, page_end, raw_text, confidence, review_status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan_id,
                document_id,
                str(plan.get("title") or ""),
                str(plan.get("season") or ""),
                _to_int(plan.get("plan_index")),
                _to_float(plan.get("energy_kcal")),
                _to_float(protein_g),
                _to_float(carbohydrate_g),
                _to_float(fat_g),
                _to_float(protein_pct),
                _to_float(carbohydrate_pct),
                _to_float(fat_pct),
                int(plan.get("page_start") or 1),
                int(plan.get("page_end") or plan.get("page_start") or 1),
                str(plan.get("raw_text") or ""),
                float(plan.get("confidence") or 0.8),
                str(plan.get("review_status") or "auto_extracted"),
                now,
            ),
        )
        for meal_index, meal in enumerate(plan.get("meals") or [], start=1):
            meal_id = _stable_id("meal", plan_id, meal_index, meal.get("meal_type"))
            db.execute(
                "INSERT INTO recipe_meals (meal_id, plan_id, meal_type, sort_order) VALUES (?, ?, ?, ?)",
                (meal_id, plan_id, str(meal.get("meal_type") or ""), meal_index),
            )
            for dish_index, dish in enumerate(meal.get("dishes") or [], start=1):
                dish_id = _stable_id("dish", meal_id, dish_index, dish.get("raw_text"))
                db.execute(
                    """
                    INSERT INTO recipe_dishes (dish_id, meal_id, dish_name, raw_text, sort_order)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        dish_id,
                        meal_id,
                        str(dish.get("dish_name") or ""),
                        str(dish.get("raw_text") or ""),
                        dish_index,
                    ),
                )
                for ingredient_index, ingredient in enumerate(dish.get("ingredients") or [], start=1):
                    db.execute(
                        """
                        INSERT INTO recipe_ingredients (
                            ingredient_id, dish_id, ingredient_name, amount, unit, is_medicinal, raw_text
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            _stable_id("ing", dish_id, ingredient_index, ingredient.get("raw_text")),
                            dish_id,
                            str(ingredient.get("name") or ""),
                            _to_float(ingredient.get("amount")),
                            str(ingredient.get("unit") or ""),
                            1 if ingredient.get("is_medicinal") else 0,
                            str(ingredient.get("raw_text") or ""),
                        ),
                    )

    def _insert_therapeutic_recipe(self, db: sqlite3.Connection, document_id: str, recipe: dict[str, Any]) -> None:
        now = _utc_now()
        db.execute(
            """
            INSERT INTO therapeutic_recipes (
                recipe_id, document_id, syndrome, title, ingredients_json, method,
                usage, cautions, page_start, page_end, raw_text, confidence,
                review_status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _stable_id("ther", document_id, recipe.get("title"), recipe.get("page_start")),
                document_id,
                str(recipe.get("syndrome") or ""),
                str(recipe.get("title") or ""),
                json.dumps(recipe.get("ingredients") or [], ensure_ascii=False),
                str(recipe.get("method") or ""),
                str(recipe.get("usage") or ""),
                str(recipe.get("cautions") or ""),
                int(recipe.get("page_start") or 1),
                int(recipe.get("page_end") or recipe.get("page_start") or 1),
                str(recipe.get("raw_text") or ""),
                float(recipe.get("confidence") or 0.8),
                str(recipe.get("review_status") or "auto_extracted"),
                now,
            ),
        )


def extract_structured_blocks_for_rag(pages: list[Any]) -> list[StructuredBlock]:
    blocks: list[StructuredBlock] = []
    for table in extract_tables(pages):
        blocks.append(
            StructuredBlock(
                "table",
                str(table.get("title") or table.get("label") or "指南表格"),
                str(table.get("raw_text") or ""),
                int(table.get("page_start") or 1),
                int(table.get("page_end") or table.get("page_start") or 1),
                {"table_type": table.get("table_type"), "label": table.get("label")},
            )
        )
    for plan in extract_recipe_plans(pages):
        blocks.append(
            StructuredBlock(
                "recipe_plan",
                str(plan.get("title") or "食谱"),
                str(plan.get("raw_text") or ""),
                int(plan.get("page_start") or 1),
                int(plan.get("page_end") or plan.get("page_start") or 1),
                {
                    "season": plan.get("season"),
                    "energy_kcal": plan.get("energy_kcal"),
                    "nutrition": plan.get("nutrition") or {},
                },
            )
        )
    for recipe in extract_therapeutic_recipes(pages):
        blocks.append(
            StructuredBlock(
                "therapeutic_recipe",
                str(recipe.get("title") or "食养方"),
                str(recipe.get("raw_text") or ""),
                int(recipe.get("page_start") or 1),
                int(recipe.get("page_end") or recipe.get("page_start") or 1),
                {"syndrome": recipe.get("syndrome"), "ingredients": recipe.get("ingredients") or []},
            )
        )
    blocks.sort(key=lambda item: (item.page_start, item.page_end, item.title))
    return blocks


def extract_tables(pages: list[Any]) -> list[dict[str, Any]]:
    page_map = _page_map(pages)
    tables: list[dict[str, Any]] = []

    page8 = page_map.get(8, "")
    if "表 1 中国居民成人膳食能量需要量" in page8:
        raw = _slice_text(page8, "表 1 中国居民成人膳食能量需要量", "（二）少吃高能量食物")
        rows = []
        for label in ("成年男性", "成年女性"):
            match = re.search(rf"{label}\s+([0-9～~-]+)\s+([0-9～~-]+)\s+([0-9～~-]+)", raw)
            if match:
                rows.append(
                    {
                        "label": label,
                        "columns": {
                            "low_activity_kcal_per_day": match.group(1),
                            "moderate_activity_kcal_per_day": match.group(2),
                            "high_activity_kcal_per_day": match.group(3),
                        },
                        "raw_text": match.group(0),
                    }
                )
        tables.append(
            {
                "label": "表 1",
                "title": "中国居民成人膳食能量需要量",
                "table_type": "energy_requirement",
                "page_start": 8,
                "page_end": 8,
                "raw_text": raw,
                "rows": rows,
                "confidence": 0.9,
            }
        )

    page68 = page_map.get(68, "")
    if "表 5.1 成人体重分类" in page68:
        raw = _slice_text(page68, "表 5.1 成人体重分类", "表 5.2 成人中心型肥胖分类")
        rows = []
        patterns = [
            ("肥胖", r"肥胖\s+(BMI≥28\.0)"),
            ("超重", r"超重\s+(24\.0≤BMI＜28\.0)"),
            ("体重正常", r"体重正常\s+(18\.5≤BMI＜24\.0)"),
            ("体重过低", r"体重过低\s+(BMI＜18\.5)"),
        ]
        for label, pattern in patterns:
            match = re.search(pattern, raw)
            if match:
                rows.append({"label": label, "columns": {"criterion": match.group(1)}, "raw_text": match.group(0)})
        tables.append(
            {
                "label": "表 5.1",
                "title": "成人体重分类",
                "table_type": "bmi_classification",
                "page_start": 68,
                "page_end": 68,
                "raw_text": raw,
                "rows": rows,
                "confidence": 0.92,
            }
        )

    if "表 5.2 成人中心型肥胖分类" in page68:
        raw = _slice_text(page68, "表 5.2 成人中心型肥胖分类", "注：")
        rows = []
        for raw_line in raw.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("表") or line.startswith("分类"):
                continue
            if "腰围" in line:
                label = "中心型肥胖前期" if "前期" in line else "中心型肥胖"
                rows.append({"label": label, "columns": {"criterion": line}, "raw_text": line})
        tables.append(
            {
                "label": "表 5.2",
                "title": "成人中心型肥胖分类",
                "table_type": "waist_classification",
                "page_start": 68,
                "page_end": 68,
                "raw_text": raw,
                "rows": rows,
                "confidence": 0.88,
            }
        )

    met_raw = "\n".join([page_map.get(69, ""), page_map.get(70, "")])
    if "表 6.1 中国 18～64 岁健康成年人常见身体活动强度系数" in met_raw:
        raw = _slice_text(met_raw, "表 6.1 中国 18～64 岁健康成年人常见身体活动强度系数", "强度分类：")
        rows = _parse_met_rows(raw)
        tables.append(
            {
                "label": "表 6.1",
                "title": "中国 18～64 岁健康成年人常见身体活动强度系数",
                "table_type": "activity_met",
                "page_start": 69,
                "page_end": 70,
                "raw_text": raw,
                "rows": rows,
                "confidence": 0.86,
            }
        )

    tables.extend(_extract_diabetes_guideline_tables(page_map))
    tables.extend(_extract_hyperuricemia_guideline_tables(page_map))
    return tables


def _extract_diabetes_guideline_tables(page_map: dict[int, str]) -> list[dict[str, Any]]:
    full_text = "\n".join(page_map.values())
    if "糖尿" not in full_text and "Diabetes" not in full_text:
        return []

    tables: list[dict[str, Any]] = []
    page4 = page_map.get(4, "")
    if "表 1" in page4 and "糖尿" in page4 and "能量" in page4:
        raw = _slice_from_markers(page4, ["表 1"], ["3.2 脂肪", "3. 2 脂肪"])
        tables.append(
            {
                "label": "表 1",
                "title": "成人糖尿病患者每日能量供给量",
                "table_type": "diabetes_energy_requirement",
                "page_start": 4,
                "page_end": 4,
                "raw_text": raw or page4,
                "rows": _rows_from_numeric_lines(raw or page4, default_label="能量供给"),
                "confidence": 0.78,
            }
        )

    gi_raw = page_map.get(6, "")
    if "表 A.1" in gi_raw or ("常见 食物" in gi_raw and "GI" in gi_raw):
        raw = _slice_from_markers(gi_raw, ["表 A.1", "表 A. 1"], ["注 :"])
        tables.append(
            {
                "label": "表 A.1",
                "title": "常见食物 GI",
                "table_type": "glycemic_index",
                "page_start": 6,
                "page_end": 6,
                "raw_text": raw or gi_raw,
                "rows": _rows_from_numeric_lines(raw or gi_raw, default_label="食物GI"),
                "confidence": 0.72,
            }
        )

    gl_raw = page_map.get(7, "")
    if "表 B.1" in gl_raw or ("常见 食物" in gl_raw and "GL" in gl_raw):
        raw = _slice_from_markers(gl_raw, ["表 B.1", "表 B. 1"], ["注 :"])
        tables.append(
            {
                "label": "表 B.1",
                "title": "常见食物 GL",
                "table_type": "glycemic_load",
                "page_start": 7,
                "page_end": 7,
                "raw_text": raw or gl_raw,
                "rows": _rows_from_numeric_lines(raw or gl_raw, default_label="食物GL"),
                "confidence": 0.72,
            }
        )

    exchange_specs = [
        ("表 C.1", "食物交换份表", "exchange_summary", 8, 8, page_map.get(8, ""), ["表 C.1"], ["表 C.2"]),
        ("表 C.2", "等值谷薯类食物交换份", "food_exchange_portion", 8, 8, page_map.get(8, ""), ["表 C.2"], ["表 C.3"]),
        ("表 C.3", "等值豆乳类食物交换份", "food_exchange_portion", 8, 9, "\n".join([page_map.get(8, ""), page_map.get(9, "")]), ["表 C.3"], ["表 C.4"]),
        ("表 C.4", "等值水果类食物交换份", "food_exchange_portion", 9, 9, page_map.get(9, ""), ["表 C.4"], ["表 C.5"]),
        ("表 C.5", "等值蔬菜类食物交换份", "food_exchange_portion", 9, 9, page_map.get(9, ""), ["表 C.5"], ["表 C.6"]),
        ("表 C.6", "油脂及坚果类食物交换份", "food_exchange_portion", 9, 9, page_map.get(9, ""), ["表 C.6"], ["注 : 每 份 提供 :能 量", "表 C.7"]),
        ("表 C.7", "肉蛋类食物交换份", "food_exchange_portion", 10, 10, page_map.get(10, ""), ["表 C.7"], ["注 :"]),
    ]
    for label, title, table_type, page_start, page_end, text, starts, ends in exchange_specs:
        if not text or label not in text:
            continue
        raw = _slice_from_markers(text, starts, ends) or text
        rows = _rows_from_numeric_lines(raw, default_label=title)
        portions = _exchange_portions_from_text(
            raw,
            exchange_group=title,
            page_start=page_start,
            page_end=page_end,
        )
        tables.append(
            {
                "label": label,
                "title": title,
                "table_type": table_type,
                "page_start": page_start,
                "page_end": page_end,
                "raw_text": raw,
                "rows": rows,
                "exchange_portions": portions,
                "confidence": 0.74,
            }
        )

    page11 = page_map.get(11, "")
    if "表 D.1" in page11 or "推荐 交换 份 分 配 表" in page11:
        raw = _slice_from_markers(page11, ["表 D.1", "表 D. 1"], ["注 1"])
        tables.append(
            {
                "label": "表 D.1",
                "title": "常见糖尿病膳食推荐交换份分配表及营养素含量",
                "table_type": "diabetes_exchange_distribution",
                "page_start": 11,
                "page_end": 11,
                "raw_text": raw or page11,
                "rows": _rows_from_numeric_lines(raw or page11, default_label="交换份分配"),
                "confidence": 0.76,
            }
        )

    return tables


def _extract_hyperuricemia_guideline_tables(page_map: dict[int, str]) -> list[dict[str, Any]]:
    full_text = "\n".join(page_map.values())
    if "高尿酸血症" not in full_text and "痛风" not in full_text:
        return []

    tables: list[dict[str, Any]] = []
    page13 = page_map.get(13, "")
    if "表 1-1" in page13 and "嘌呤" in page13:
        raw = _slice_from_markers(page13, ["表 1-1"], ["a:"])
        tables.append(
            {
                "label": "表 1-1",
                "title": "常见食物按嘌呤含量分类",
                "table_type": "purine_classification",
                "page_start": 13,
                "page_end": 13,
                "raw_text": raw or page13,
                "rows": _rows_from_numeric_lines(raw or page13, default_label="嘌呤含量分类"),
                "confidence": 0.82,
            }
        )

    purine_raw = "\n".join(page_map.get(page, "") for page in range(14, 20))
    if "表 1-2" in purine_raw and "嘌呤含量表" in purine_raw:
        raw = _slice_from_markers(purine_raw, ["表 1-2"], ["b:数据来源"])
        tables.append(
            {
                "label": "表 1-2",
                "title": "常见食物嘌呤含量表",
                "table_type": "purine_content",
                "page_start": 14,
                "page_end": 19,
                "raw_text": raw or purine_raw,
                "rows": _food_numeric_pairs_from_text(raw or purine_raw, default_label="食物嘌呤"),
                "confidence": 0.72,
            }
        )

    page20 = page_map.get(20, "")
    if "表 2.1" in page20 and "推荐食物名单" in page20:
        raw = _slice_from_markers(page20, ["表 2.1"], ["二、不同证型"])
        tables.append(
            {
                "label": "表 2.1",
                "title": "成人高尿酸血症与痛风人群推荐食物名单",
                "table_type": "gout_food_selection",
                "page_start": 20,
                "page_end": 20,
                "raw_text": raw or page20,
                "rows": _rows_from_text_lines(raw or page20, default_label="食物选择"),
                "confidence": 0.74,
            }
        )

    page21 = page_map.get(21, "")
    if "表 2.2" in page21 and "推荐食药物质" in page21:
        raw = _slice_from_markers(page21, ["表 2.2"], [])
        tables.append(
            {
                "label": "表 2.2",
                "title": "不同证型推荐食药物质及新食品原料",
                "table_type": "tcm_food_medicine_by_syndrome",
                "page_start": 21,
                "page_end": 21,
                "raw_text": raw or page21,
                "rows": _rows_from_text_lines(raw or page21, default_label="证型食药物质"),
                "confidence": 0.7,
            }
        )

    exchange_specs = [
        ("表 4.1", "谷薯类食物交换表", "food_exchange_portion", 63, 63, page_map.get(63, ""), ["表 4.1"], ["表 4.2"]),
        ("表 4.2", "蔬菜类食物交换表", "food_exchange_portion", 64, 64, page_map.get(64, ""), ["表 4.2"], ["表 4.3"]),
        ("表 4.3", "水果类食物交换表", "food_exchange_portion", 65, 65, page_map.get(65, ""), ["表 4.3"], ["表 4.4"]),
        ("表 4.4", "肉蛋水产品类食物交换表", "food_exchange_portion", 66, 66, page_map.get(66, ""), ["表 4.4"], ["表 4.5"]),
        ("表 4.5", "坚果类食物交换表", "food_exchange_portion", 67, 67, page_map.get(67, ""), ["表 4.5"], ["表 4.6"]),
        ("表 4.6", "大豆、乳及其制品食物交换表", "food_exchange_portion", 67, 67, page_map.get(67, ""), ["表 4.6"], ["表 4.7"]),
        ("表 4.7", "调味料类盐含量交换表", "salt_exchange", 68, 68, page_map.get(68, ""), ["表 4.7"], []),
    ]
    for label, title, table_type, page_start, page_end, text, starts, ends in exchange_specs:
        if not text or label not in text:
            continue
        raw = _slice_from_markers(text, starts, ends) or text
        tables.append(
            {
                "label": label,
                "title": title,
                "table_type": table_type,
                "page_start": page_start,
                "page_end": page_end,
                "raw_text": raw,
                "rows": _rows_from_numeric_lines(raw, default_label=title),
                "exchange_portions": _exchange_portions_from_text(
                    raw,
                    exchange_group=title,
                    page_start=page_start,
                    page_end=page_end,
                )
                if table_type == "food_exchange_portion"
                else [],
                "confidence": 0.7,
            }
        )

    return tables


def _slice_from_markers(text: str, starts: list[str], ends: list[str]) -> str:
    start_index = -1
    for marker in starts:
        candidate = text.find(marker)
        if candidate >= 0 and (start_index < 0 or candidate < start_index):
            start_index = candidate
    if start_index < 0:
        return ""
    end_index = len(text)
    for marker in ends:
        candidate = text.find(marker, start_index + 1)
        if candidate >= 0:
            end_index = min(end_index, candidate)
    return text[start_index:end_index].strip()


def _rows_from_numeric_lines(raw: str, *, default_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in str(raw or "").splitlines():
        clean = re.sub(r"\s+", " ", line).strip()
        if len(clean) < 4:
            continue
        if clean.startswith(("注", "WS/", "WSV/", "附 录")):
            continue
        if not re.search(r"\d", clean):
            continue
        label = _row_label_from_line(clean) or default_label
        rows.append(
            {
                "label": label[:80],
                "columns": {"raw": clean},
                "raw_text": clean,
            }
        )
    return rows[:80]


def _rows_from_text_lines(raw: str, *, default_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in str(raw or "").splitlines():
        clean = re.sub(r"\s+", " ", line).strip()
        if len(clean) < 4:
            continue
        if clean.startswith(("表", "食物类别", "证型", "注", "a:", "b:")):
            continue
        rows.append(
            {
                "label": _row_label_from_line(clean) or default_label,
                "columns": {"raw": clean},
                "raw_text": clean,
            }
        )
    return rows[:120]


def _food_numeric_pairs_from_text(raw: str, *, default_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_category = ""
    skip_words = {"单位", "食物", "嘌呤", "含量", "类别", "制品", "附录", "数据来源"}
    for raw_line in str(raw or "").splitlines():
        clean = re.sub(r"\s+", " ", raw_line).strip()
        if not clean:
            continue
        if not re.search(r"\d", clean):
            current_category = clean if len(clean) <= 30 else current_category
            continue
        if any(clean.startswith(word) for word in ("表", "（单位", "b:", "a:")):
            continue
        for match in re.finditer(r"([\u4e00-\u9fffA-Za-z0-9（）()\[\]、,，.·%-]{1,40})\s+(\d+(?:\.\d+)?)", clean):
            name = match.group(1).strip(" ，,、")
            value = match.group(2)
            if not name or name in skip_words:
                continue
            rows.append(
                {
                    "label": name,
                    "columns": {
                        "purine_mg_per_100g": value,
                        "category": current_category,
                    },
                    "raw_text": clean,
                }
            )
            if len(rows) >= 800:
                return rows
    return rows


def _row_label_from_line(line: str) -> str:
    text = re.sub(r"\s+", " ", str(line or "")).strip()
    text = re.split(r"\d", text, maxsplit=1)[0].strip(" |,，、:-")
    text = re.sub(r"^(表\s*[A-Z]?\.\d+|食品|食物|组别|类别|g|kJ|kcal)\s*", "", text).strip()
    return text or ""


def _exchange_portions_from_text(
    raw: str,
    *,
    exchange_group: str,
    page_start: int,
    page_end: int,
) -> list[dict[str, Any]]:
    portions: list[dict[str, Any]] = []
    for line in str(raw or "").splitlines():
        clean = re.sub(r"\s+", " ", line).strip()
        if not clean or clean.startswith(("注", "表", "食品", "食物", "g ")):
            continue
        for match in re.finditer(r"([\u4e00-\u9fffA-Za-z0-9、,，（）() ]{1,70})\s+(\d{1,4})(?=\s|$)", clean):
            food_name = re.sub(r"\s+", "", match.group(1)).strip(" ,，、|")
            if not food_name or len(food_name) < 2:
                continue
            if any(skip in food_name for skip in ("每份", "提供", "能量", "蛋白质", "脂肪", "碳水", "重量")):
                continue
            amount = match.group(2)
            portions.append(
                {
                    "food_name": food_name[:80],
                    "exchange_group": exchange_group,
                    "serving_amount": f"{amount} g",
                    "energy_kcal": 90.0,
                    "carbohydrate_g": None,
                    "protein_g": None,
                    "fat_g": None,
                    "page_start": page_start,
                    "page_end": page_end,
                    "raw_text": clean,
                    "confidence": 0.68,
                    "review_status": "auto_extracted",
                }
            )
    return portions[:160]


def extract_recipe_plans(pages: list[Any]) -> list[dict[str, Any]]:
    records = _line_records(pages)
    spans: list[tuple[int, int, re.Match[str]]] = []
    for index, item in enumerate(records):
        match = RECIPE_PLAN_RE.match(item["line"].strip())
        if match:
            spans.append((index, item["page"], match))

    plans = []
    for position, (start, page_start, match) in enumerate(spans):
        end = spans[position + 1][0] if position + 1 < len(spans) else len(records)
        lines: list[str] = []
        page_end = page_start
        for item in records[start:end]:
            if item["line"].strip().startswith("附录 4"):
                break
            lines.append(item["line"])
            page_end = item["page"]
        raw = "\n".join(lines).strip()
        if not raw:
            continue
        title = match.group(0)
        nutrition = _parse_recipe_nutrition(raw)
        energy_kcal = _to_float(match.group(3)) if match.lastindex and match.group(3) else _to_float(nutrition.get("energy_kcal"))
        plans.append(
            {
                "title": title,
                "season": match.group(1),
                "plan_index": int(match.group(2)),
                "energy_kcal": energy_kcal,
                "page_start": page_start,
                "page_end": page_end,
                "raw_text": raw,
                "nutrition": nutrition,
                "meals": _parse_recipe_meals(raw),
                "confidence": 0.82,
                "review_status": "auto_extracted",
            }
        )
    return plans


def extract_therapeutic_recipes(pages: list[Any]) -> list[dict[str, Any]]:
    records = []
    in_appendix = False
    for item in _line_records(pages):
        line = item["line"].strip()
        if "附录 5" in line or "食养方举例" in line:
            in_appendix = True
            continue
        if in_appendix:
            records.append(item)
    recipes: list[dict[str, Any]] = []
    syndrome = ""
    current: dict[str, Any] | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal current, buffer
        if not current or not buffer:
            current = None
            buffer = []
            return
        raw = "\n".join(buffer).strip()
        current.update(_parse_therapeutic_fields(raw))
        current["raw_text"] = raw
        current["ingredients"] = _parse_ingredients(current.get("ingredients_text") or "")
        current["confidence"] = 0.82
        current["review_status"] = "auto_extracted"
        recipes.append(current)
        current = None
        buffer = []

    for item in records:
        line = item["line"].strip()
        if not line or line.startswith("附录"):
            continue
        syndrome_match = SYNDROME_RE.match(line) or PAREN_SYNDROME_RE.match(line)
        if syndrome_match:
            flush()
            syndrome = syndrome_match.group(1)
            continue
        recipe_match = THERAPEUTIC_RECIPE_RE.match(line) or NUMBERED_RECIPE_RE.match(line)
        if recipe_match and syndrome:
            flush()
            title = (recipe_match.group(2) if recipe_match.lastindex and recipe_match.lastindex >= 2 else recipe_match.group(1)).strip()
            current = {
                "syndrome": syndrome,
                "title": title,
                "page_start": item["page"],
                "page_end": item["page"],
            }
            buffer = [line]
            continue
        if current:
            current["page_end"] = item["page"]
            buffer.append(line)
    flush()
    return recipes


def search_structured_knowledge(db_path: Path, query: str, limit: int = 6) -> list[dict[str, Any]]:
    query = str(query or "").strip()
    if not query or not db_path.exists():
        return []
    tokens = _query_tokens(query)
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        candidates: list[dict[str, Any]] = []
        candidates.extend(_search_table_rows(db, tokens, limit * 3))
        candidates.extend(_search_exchange_portions(db, tokens, limit * 3))
        candidates.extend(_search_recipe_plans(db, tokens, limit * 3))
        candidates.extend(_search_therapeutic_recipes(db, tokens, limit * 3))
        candidates.extend(_search_activity_mets(db, tokens, limit * 3))
        candidates.extend(_search_diagnostic_thresholds(db, tokens, limit * 3))
        candidates.extend(_search_nutrition_targets(db, tokens, limit * 3))
    for item in candidates:
        item["score"] = _score_text(item.get("search_text", ""), tokens)
        item["score"] = _adjust_structured_score(item, query)
    candidates.sort(key=lambda item: item.get("score", 0), reverse=True)
    unique: list[dict[str, Any]] = []
    seen = set()
    for item in candidates:
        key = (item.get("type"), item.get("id"))
        if key in seen or item.get("score", 0) <= 0:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique


def _adjust_structured_score(item: dict[str, Any], query: str) -> float:
    score = float(item.get("score") or 0.0)
    lowered = str(query or "").lower()
    text = " ".join(
        [
            str(item.get("title") or ""),
            str(item.get("source_name") or ""),
            str(item.get("content") or ""),
            str(item.get("raw_text") or ""),
        ]
    )
    item_type = str(item.get("type") or "")
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    label = str(metadata.get("label") or "").strip()
    table_type = str(metadata.get("table_type") or "").strip()
    query_food = _specific_food_query_term(query)
    if query_food and item_type == "table_row":
        if table_type == "purine_content":
            if label == query_food:
                score += 40.0
            elif query_food in label or label in query_food:
                score += 18.0
            elif query_food in text:
                score += 30.0
        elif table_type in {"purine_classification", "gout_food_selection"}:
            if label in {"表", "类别", "食物", "推荐食物"} or str(item.get("raw_text") or "").startswith("表 "):
                score *= 0.35
            if len(query_food) <= 4 and any(word in lowered for word in ("含量", "多少", "mg", "每100")):
                score *= 0.15
    if any(word in lowered for word in ("高尿酸", "痛风", "嘌呤", "尿酸")):
        if any(word in text for word in ("高尿酸", "痛风", "嘌呤", "尿酸", "hyperuricemia", "gout")):
            score += 3.0
        if any(word in text for word in ("成人肥胖", "减肥", "体重管理")) and not any(
            word in text for word in ("高尿酸", "痛风", "嘌呤", "尿酸")
        ):
            score -= 3.0
        if item_type in {"table_row", "food_exchange_portion"} and any(word in lowered for word in ("表", "含量", "多少", "食物", "嘌呤")):
            score += 1.2
    food_safety_query = any(
        word in lowered
        for word in (
            "能不能吃",
            "可以吃",
            "能吃",
            "不宜",
            "禁忌",
            "少吃",
            "限制",
            "海鲜",
            "内脏",
            "动物内脏",
            "肥肉",
            "啤酒",
            "黄酒",
        )
    )
    if food_safety_query:
        if item_type == "table_row" and table_type in {
            "gout_food_selection",
            "purine_content",
            "purine_classification",
        }:
            score += 14.0
            if any(word in text for word in ("不宜食物", "不宜", "动物内脏", "海鲜", "啤酒", "黄酒")):
                score += 10.0
        if item_type == "recipe_plan" and not any(word in lowered for word in ("食谱", "菜谱", "一日三餐", "早餐", "午餐", "晚餐")):
            score *= 0.25

    query_topic = re.sub(r"(怎么做|怎么煮|如何做|做法|配方|推荐|有哪些|是什么|多少)", "", str(query or "")).strip()
    title = str(item.get("title") or "").strip()
    action_topic = _query_topic_without_action_words(query)
    if len(query_topic) >= 2 and title:
        if title == query_topic:
            score += 4.0
        elif query_topic in title:
            score += 1.5
    if len(action_topic) >= 2 and title:
        if title == action_topic:
            score += 20.0
        elif title in str(query or "") or action_topic in title:
            score += 10.0
        if item_type == "therapeutic_recipe" and (
            title == action_topic or title in str(query or "") or action_topic in title
        ):
            score += 25.0
        if item_type == "table_row" and any(word in lowered for word in ("怎么做", "怎么煮", "做法", "配方")):
            if action_topic not in text and title not in str(query or ""):
                score *= 0.2
    if "糖尿病" in lowered:
        if "糖尿病" in text:
            score += 2.5
        if "成人肥胖" in text:
            score -= 2.0
    if "交换份" in lowered or "一份" in lowered:
        if item_type in {"table_row", "food_exchange_portion"}:
            score += 1.8
        if item_type == "recipe_plan" and not any(word in lowered for word in ("食谱", "菜谱", "餐谱")):
            score *= 0.25
    if any(word in lowered for word in ("gi", "血糖生成指数")) and "GI" in text:
        score += 1.5
    if any(word in lowered for word in ("gl", "血糖负荷")) and "GL" in text:
        score += 1.5
    salt_query = any(word in lowered for word in ("盐", "钠", "限盐", "低盐", "最多"))
    if item_type == "diagnostic_threshold":
        blood_pressure_query = any(word in lowered for word in ("血压", "高血压", "诊断高血压"))
        bmi_query = any(word in lowered for word in ("bmi", "肥胖", "超重", "体重指数"))
        if blood_pressure_query:
            score += 35.0
            if any(word in text for word in ("诊室血压", "非同日", "140", "90")):
                score += 25.0
        elif bmi_query:
            if any(word in text for word in ("BMI", "体重指数", "≥28")):
                score += 10.0
            else:
                score *= 0.2
        if blood_pressure_query and any(word in text for word in ("收缩压", "舒张压")):
            score += 12.0
        if salt_query:
            score *= 0.25
    if item_type == "nutrition_target":
        if salt_query:
            score += 55.0
        if any(word in text for word in ("<5", "5 g", "2 g", "食盐", "氯化钠", "钠")):
            score += 12.0
    return score


def _search_diagnostic_thresholds(db: sqlite3.Connection, tokens: list[str], limit: int) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT dt.*, sd.original_name
        FROM diagnostic_thresholds dt
        JOIN source_documents sd ON sd.document_id = dt.document_id
        ORDER BY dt.page_start, dt.indicator
        """
    ).fetchall()
    result = []
    for row in rows:
        content = (
            f"{row['indicator']}: {row['threshold']} {row['unit']}".strip()
            + (f"；适用人群: {row['population']}" if row["population"] else "")
            + (f"；场景: {row['context']}" if row["context"] else "")
        )
        search_text = " ".join(
            [
                row["indicator"],
                row["threshold"],
                row["unit"],
                row["population"],
                row["context"],
                row["raw_text"],
                row["original_name"],
                content,
            ]
        )
        result.append(
            {
                "type": "diagnostic_threshold",
                "id": row["threshold_id"],
                "title": row["indicator"],
                "source_name": row["original_name"],
                "citation": _citation(row["original_name"], row["page_start"], row["page_end"], row["threshold_id"]),
                "content": content,
                "raw_text": row["raw_text"],
                "search_text": search_text,
                "metadata": {
                    "indicator": row["indicator"],
                    "threshold": row["threshold"],
                    "unit": row["unit"],
                    "population": row["population"],
                    "context": row["context"],
                },
            }
        )
    return result


def _search_nutrition_targets(db: sqlite3.Connection, tokens: list[str], limit: int) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT nt.*, sd.original_name
        FROM nutrition_targets nt
        JOIN source_documents sd ON sd.document_id = nt.document_id
        ORDER BY nt.page_start, nt.nutrient
        """
    ).fetchall()
    result = []
    for row in rows:
        content = (
            f"{row['nutrient']}: {row['target_value']}"
            + (f"；适用人群: {row['population']}" if row["population"] else "")
            + (f"；场景: {row['context']}" if row["context"] else "")
        )
        search_text = " ".join(
            [
                row["nutrient"],
                row["target_value"],
                row["population"],
                row["context"],
                row["raw_text"],
                row["original_name"],
                content,
            ]
        )
        result.append(
            {
                "type": "nutrition_target",
                "id": row["target_id"],
                "title": row["nutrient"],
                "source_name": row["original_name"],
                "citation": _citation(row["original_name"], row["page_start"], row["page_end"], row["target_id"]),
                "content": content,
                "raw_text": row["raw_text"],
                "search_text": search_text,
                "metadata": {
                    "nutrient": row["nutrient"],
                    "target_value": row["target_value"],
                    "population": row["population"],
                    "context": row["context"],
                },
            }
        )
    return result


def _specific_food_query_term(query: str) -> str:
    text = re.sub(r"\s+", "", str(query or ""))
    for token in (
        "嘌呤",
        "含量",
        "多少",
        "每100克",
        "每100g",
        "mg/100g",
        "mg",
        "高尿酸",
        "痛风",
        "尿酸",
        "查询",
        "查一下",
        "是多少",
        "有多少",
        "能不能吃",
        "可以吃",
        "能吃",
        "的",
        "吗",
        "？",
        "?",
    ):
        text = text.replace(token, "")
    return text if len(text) >= 2 else ""


def _query_topic_without_action_words(query: str) -> str:
    text = re.sub(r"\s+", "", str(query or ""))
    for token in (
        "怎么做",
        "怎么煮",
        "如何做",
        "做法",
        "配方",
        "推荐",
        "有哪些",
        "是什么",
        "多少",
        "查询",
        "查一下",
        "的",
        "吗",
        "？",
        "?",
    ):
        text = text.replace(token, "")
    return text if len(text) >= 2 else ""


def _search_exchange_portions(db: sqlite3.Connection, tokens: list[str], limit: int) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT fep.*, sd.original_name
        FROM food_exchange_portions fep
        JOIN source_documents sd ON sd.document_id = fep.document_id
        ORDER BY fep.page_start, fep.food_name
        """
    ).fetchall()
    result = []
    for row in rows:
        content = (
            f"{row['exchange_group']}：{row['food_name']} 每交换份约 {row['serving_amount']}"
            + (f"，约 {row['energy_kcal']:g} kcal" if row["energy_kcal"] is not None else "")
        )
        search_text = " ".join(
            [
                row["food_name"],
                row["exchange_group"],
                row["serving_amount"],
                row["raw_text"],
                content,
            ]
        )
        result.append(
            {
                "type": "food_exchange_portion",
                "id": row["portion_id"],
                "title": row["exchange_group"] or "食物交换份",
                "source_name": row["original_name"],
                "citation": _citation(row["original_name"], row["page_start"], row["page_end"], row["portion_id"]),
                "content": content,
                "raw_text": row["raw_text"],
                "search_text": search_text,
                "metadata": {
                    "food_name": row["food_name"],
                    "serving_amount": row["serving_amount"],
                    "energy_kcal": row["energy_kcal"],
                },
            }
        )
    return result


def _search_table_rows(db: sqlite3.Connection, tokens: list[str], limit: int) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT gt.table_id, gt.title, gt.table_label, gt.table_type, gt.page_start, gt.page_end,
               gtr.row_id, gtr.label, gtr.columns_json, gtr.raw_text,
               sd.original_name
        FROM guide_table_rows gtr
        JOIN guide_tables gt ON gt.table_id = gtr.table_id
        JOIN source_documents sd ON sd.document_id = gt.document_id
        ORDER BY gt.page_start, gtr.row_index
        """
    ).fetchall()
    result = []
    for row in rows:
        columns = _loads_json(row["columns_json"], {})
        content = _format_table_row_content(
            table_label=row["table_label"],
            title=row["title"],
            label=row["label"],
            columns=columns,
        )
        search_text = " ".join(
            [row["title"], row["table_label"], row["table_type"], row["label"], row["raw_text"], json.dumps(columns, ensure_ascii=False)]
        )
        result.append(
            {
                "type": "table_row",
                "id": row["row_id"],
                "title": row["title"],
                "source_name": row["original_name"],
                "citation": _citation(row["original_name"], row["page_start"], row["page_end"], row["row_id"]),
                "content": content,
                "raw_text": row["raw_text"],
                "search_text": search_text,
                "metadata": {
                    "table_type": row["table_type"],
                    "table_label": row["table_label"],
                    "label": row["label"],
                    "columns": columns,
                },
            }
        )
    return result


def _format_table_row_content(
    *,
    table_label: str,
    title: str,
    label: str,
    columns: dict[str, Any],
) -> str:
    if "purine_mg_per_100g" in columns:
        value = columns.get("purine_mg_per_100g")
        category = columns.get("category")
        parts = [f"{label}: 嘌呤 {value} mg/100g"]
        if category:
            parts.append(f"类别: {category}")
        return "; ".join(parts)
    if "serving_amount" in columns:
        return f"{label}: {columns.get('serving_amount')}"
    compact = "; ".join(
        f"{key}: {value}" for key, value in columns.items() if value not in (None, "")
    )
    prefix = f"{table_label} {title}: {label}".strip()
    return f"{prefix}; {compact}" if compact else prefix


def _search_recipe_plans(db: sqlite3.Connection, tokens: list[str], limit: int) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT rp.*, sd.original_name,
               group_concat(DISTINCT rm.meal_type) AS meal_types,
               group_concat(DISTINCT rd.dish_name) AS dish_names,
               group_concat(DISTINCT ri.ingredient_name) AS ingredient_names
        FROM recipe_plans rp
        JOIN source_documents sd ON sd.document_id = rp.document_id
        LEFT JOIN recipe_meals rm ON rm.plan_id = rp.plan_id
        LEFT JOIN recipe_dishes rd ON rd.meal_id = rm.meal_id
        LEFT JOIN recipe_ingredients ri ON ri.dish_id = rd.dish_id
        GROUP BY rp.plan_id
        ORDER BY rp.page_start
        """
    ).fetchall()
    result = []
    for row in rows:
        search_text = " ".join(
            [
                row["title"],
                row["season"],
                str(row["energy_kcal"] or ""),
                row["meal_types"] or "",
                row["dish_names"] or "",
                row["ingredient_names"] or "",
                row["raw_text"],
                row["original_name"],
                _domain_tags_for_text(f"{row['original_name']} {row['raw_text']}"),
            ]
        )
        result.append(
            {
                "type": "recipe_plan",
                "id": row["plan_id"],
                "title": row["title"],
                "source_name": row["original_name"],
                "citation": _citation(row["original_name"], row["page_start"], row["page_end"], row["plan_id"]),
                "content": _format_recipe_plan(row),
                "raw_text": row["raw_text"],
                "search_text": search_text,
                "metadata": {
                    "season": row["season"],
                    "energy_kcal": row["energy_kcal"],
                    "protein_g": row["protein_g"],
                    "carbohydrate_g": row["carbohydrate_g"],
                    "fat_g": row["fat_g"],
                },
            }
        )
    return result


def _search_therapeutic_recipes(db: sqlite3.Connection, tokens: list[str], limit: int) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT tr.*, sd.original_name
        FROM therapeutic_recipes tr
        JOIN source_documents sd ON sd.document_id = tr.document_id
        ORDER BY tr.page_start
        """
    ).fetchall()
    result = []
    for row in rows:
        search_text = " ".join([row["syndrome"], row["title"], row["ingredients_json"], row["method"], row["usage"], row["cautions"]])
        result.append(
            {
                "type": "therapeutic_recipe",
                "id": row["recipe_id"],
                "title": row["title"],
                "source_name": row["original_name"],
                "citation": _citation(row["original_name"], row["page_start"], row["page_end"], row["recipe_id"]),
                "content": _format_therapeutic_recipe(row),
                "raw_text": row["raw_text"],
                "search_text": search_text,
                "metadata": {"syndrome": row["syndrome"], "ingredients": _loads_json(row["ingredients_json"], [])},
            }
        )
    return result


def _search_activity_mets(db: sqlite3.Connection, tokens: list[str], limit: int) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT am.*, sd.original_name
        FROM activity_mets am
        JOIN source_documents sd ON sd.document_id = am.document_id
        ORDER BY am.category, am.activity_name
        """
    ).fetchall()
    result = []
    for row in rows:
        search_text = " ".join([row["category"], row["activity_name"], str(row["met"]), row["intensity"], "运动 MET 活动强度"])
        result.append(
            {
                "type": "activity_met",
                "id": row["activity_id"],
                "title": "身体活动强度系数",
                "source_name": row["original_name"],
                "citation": _citation(row["original_name"], row["page_start"], row["page_end"], row["activity_id"]),
                "content": f"{row['activity_name']}：MET {row['met']:g}，强度 {row['intensity']}，类别 {row['category']}",
                "raw_text": row["raw_text"],
                "search_text": search_text,
                "metadata": {"met": row["met"], "intensity": row["intensity"], "category": row["category"]},
            }
        )
    return result


def _parse_met_rows(raw: str) -> list[dict[str, Any]]:
    rows = []
    category = ""
    category_names = {"不活动/休息", "家务性劳动", "交通性活动", "休闲性活动", "职业性活动"}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("表") or line.startswith("活动类别"):
            continue
        if line in category_names:
            category = line
            continue
        match = re.match(r"(.+?)\s+([0-9]+(?:\.[0-9]+)?)\s+(静态行为|低|中|高)$", line)
        if not match:
            continue
        rows.append(
            {
                "label": match.group(1).strip(),
                "columns": {
                    "category": category,
                    "met": float(match.group(2)),
                    "intensity": match.group(3),
                },
                "raw_text": line,
            }
        )
    return rows


def _parse_recipe_nutrition(raw: str) -> dict[str, Any]:
    nutrition: dict[str, Any] = {}
    compact_raw = re.sub(r"\s+", "", raw)
    match = re.search(
        r"本食谱提供能量约为\s*([0-9]+)\s*kcal.*?蛋白质\s*([0-9.]+)\s*g，碳水化合物\s*([0-9.]+)\s*g.*?脂肪\s*([0-9.]+)\s*g",
        raw,
        flags=re.S,
    )
    if match:
        nutrition.update(
            {
                "energy_kcal": float(match.group(1)),
                "protein_g": float(match.group(2)),
                "carbohydrate_g": float(match.group(3)),
                "fat_g": float(match.group(4)),
            }
        )
    else:
        ranged = re.search(
            r"本食谱可提供能量.*?总值为\s*([0-9.]+)(?:[～~-]([0-9.]+))?\s*kcal.*?"
            r"蛋白质\s*([0-9.]+)(?:[～~-]([0-9.]+))?\s*g.*?"
            r"碳水化合物\s*([0-9.]+)(?:[～~-]([0-9.]+))?\s*g.*?"
            r"脂肪\s*([0-9.]+)(?:[～~-]([0-9.]+))?\s*g",
            compact_raw,
            flags=re.S,
        )
        if ranged:
            nutrition.update(
                {
                    "energy_kcal": _midpoint_number(ranged.group(1), ranged.group(2)),
                    "protein_g": _midpoint_number(ranged.group(3), ranged.group(4)),
                    "carbohydrate_g": _midpoint_number(ranged.group(5), ranged.group(6)),
                    "fat_g": _midpoint_number(ranged.group(7), ranged.group(8)),
                    "energy_range": _format_range(ranged.group(1), ranged.group(2), "kcal"),
                    "protein_range": _format_range(ranged.group(3), ranged.group(4), "g"),
                    "carbohydrate_range": _format_range(ranged.group(5), ranged.group(6), "g"),
                    "fat_range": _format_range(ranged.group(7), ranged.group(8), "g"),
                }
            )
    pct = re.search(r"蛋白质\s*([0-9.]+)%.*?碳水化合物\s*([0-9.]+)%.*?脂肪\s*([0-9.]+)%", raw, flags=re.S)
    if pct:
        nutrition.update(
            {
                "protein_pct": float(pct.group(1)),
                "carbohydrate_pct": float(pct.group(2)),
                "fat_pct": float(pct.group(3)),
            }
        )
    return nutrition


def _midpoint_number(first: str | None, second: str | None = None) -> float | None:
    left = _to_float(first)
    right = _to_float(second)
    if left is None:
        return right
    if right is None:
        return left
    return round((left + right) / 2, 2)


def _format_range(first: str | None, second: str | None, unit: str) -> str:
    left = str(first or "").strip()
    right = str(second or "").strip()
    if not left:
        return ""
    return f"{left}～{right}{unit}" if right else f"{left}{unit}"


def _parse_recipe_meals(raw: str) -> list[dict[str, Any]]:
    meals: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in raw.splitlines()[1:]:
        line = raw_line.strip()
        if not line or line.startswith("注：") or line.startswith("2.") or line.startswith("宏量营养素"):
            continue
        if line.startswith("油、盐"):
            continue
        meal_type = ""
        content = ""
        for name in MEAL_NAMES:
            if line == name:
                meal_type = name
                content = ""
                break
            if line.startswith(name + " "):
                meal_type = name
                content = line[len(name) :].strip()
                break
        if meal_type:
            current = {"meal_type": meal_type, "dishes": []}
            meals.append(current)
            if content:
                current["dishes"].append(_parse_dish(content))
            continue
        if current and not line.startswith("本食谱提供能量"):
            current["dishes"].append(_parse_dish(line))
    return meals


def _parse_dish(line: str) -> dict[str, Any]:
    dish_name = line.split("（", 1)[0].strip()
    return {
        "dish_name": dish_name or line[:40],
        "raw_text": line,
        "ingredients": _parse_ingredients(line),
    }


def _parse_therapeutic_fields(raw: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    fields["ingredients_text"] = _field_between(raw, "主要材料：", ["制作方法：", "用法用量：", "注意："])
    fields["method"] = _field_between(raw, "制作方法：", ["用法用量：", "注意："])
    fields["usage"] = _field_between(raw, "用法用量：", ["注意："])
    fields["cautions"] = _field_between(raw, "注意：", [])
    if not any(fields.values()):
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if len(lines) >= 2:
            fields["ingredients_text"] = lines[1]
            remaining = "\n".join(lines[2:]).strip()
            fields["method"] = remaining
            if "代茶饮用" in remaining:
                fields["usage"] = "代茶饮用"
            elif "佐餐食用" in remaining:
                fields["usage"] = "佐餐食用"
            elif "早餐食用" in remaining:
                fields["usage"] = "早餐食用"
            cautions = []
            for marker in ("孕妇慎用", "孕妇、哺乳期妇女不宜食用", "慎用", "不宜食用"):
                if marker in raw and marker not in cautions:
                    cautions.append(marker)
            fields["cautions"] = "；".join(cautions)
    return fields


def _parse_ingredients(text: str) -> list[dict[str, Any]]:
    clean = text.replace("【干】", "干").replace("（", "，").replace("）", "，")
    parts = re.split(r"[，,、；;]", clean)
    ingredients = []
    for part in parts:
        part = part.strip(" 。.")
        match = re.search(r"(.+?)([0-9]+(?:\.[0-9]+)?)\s*(g|mL|ml|克|毫升|枚|片|个|次)", part)
        if not match:
            continue
        name = match.group(1).strip()
        if not name or name in {"主要材料", "材料"}:
            continue
        ingredients.append(
            {
                "name": name.strip("*"),
                "amount": float(match.group(2)),
                "unit": match.group(3),
                "is_medicinal": "*" in part,
                "raw_text": part,
            }
        )
    return ingredients


def _field_between(text: str, start: str, end_markers: list[str]) -> str:
    start_index = text.find(start)
    if start_index < 0:
        return ""
    start_index += len(start)
    end_index = len(text)
    for marker in end_markers:
        index = text.find(marker, start_index)
        if index >= 0:
            end_index = min(end_index, index)
    return text[start_index:end_index].strip()


def _line_records(pages: list[Any]) -> list[dict[str, Any]]:
    records = []
    for page in pages:
        page_number = _page_number_of(page)
        for line in _page_text_of(page).splitlines():
            clean = line.strip()
            if clean:
                records.append({"page": page_number, "line": clean})
    return records


def _page_map(pages: list[Any]) -> dict[int, str]:
    return {_page_number_of(page): _page_text_of(page) for page in pages}


def _page_number_of(page: Any) -> int:
    if isinstance(page, dict):
        return int(page.get("page_number") or page.get("page") or 1)
    return int(getattr(page, "page_number", 1) or 1)


def _page_text_of(page: Any) -> str:
    if isinstance(page, dict):
        return str(page.get("text") or "")
    return str(getattr(page, "text", "") or "")


def _rule_based_nutrition_targets(pages: list[Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for page in pages:
        page_number = _page_number_of(page)
        text = _page_text_of(page)
        compact = re.sub(r"\s+", "", text)
        if (
            "高血压" in compact
            and "限制钠盐摄入" in compact
            and ("钠的摄入量<2g/d" in compact or "钠的摄入量首先减少30%" in compact)
        ):
            for nutrient, value, raw in (
                ("钠", "<2 g/d", "建议钠的摄入量<2 g/d。"),
                ("食盐/氯化钠", "<5 g/d", "建议氯化钠摄入量<5 g/d。"),
            ):
                key = (nutrient, value)
                if key in seen:
                    continue
                seen.add(key)
                targets.append(
                    {
                        "nutrient": nutrient,
                        "target_value": value,
                        "population": "高血压患者",
                        "context": "减少钠盐摄入、增加钾摄入",
                        "page_start": page_number,
                        "page_end": page_number,
                        "raw_text": raw,
                        "confidence": 0.92,
                        "review_status": "auto_extracted_rule",
                    }
                )
    return targets


def _slice_text(text: str, start: str, end: str = "") -> str:
    start_index = text.find(start)
    if start_index < 0:
        return ""
    end_index = text.find(end, start_index + len(start)) if end else -1
    if end_index < 0:
        end_index = len(text)
    return text[start_index:end_index].strip()


def _query_tokens(query: str) -> list[str]:
    lowered = str(query or "").lower()
    words = re.findall(r"[a-z0-9_.+-]+|[\u4e00-\u9fff]+", lowered)
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", lowered)
    cjk_bigrams = [
        run[index : index + 2]
        for run in cjk_runs
        for index in range(max(0, len(run) - 1))
    ]
    cjk_trigrams = [
        run[index : index + 3]
        for run in cjk_runs
        for index in range(max(0, len(run) - 2))
    ]
    numbers = re.findall(r"\d+(?:\.\d+)?", lowered)
    extra: list[str] = []
    synonym_groups = {
        "减肥": ["肥胖", "体重管理", "食谱", "菜谱"],
        "减重": ["减肥", "肥胖", "体重管理"],
        "食谱": ["菜谱", "食谱", "餐谱"],
        "菜谱": ["食谱", "菜谱", "餐谱"],
        "bmi": ["BMI", "体重分类", "肥胖", "超重"],
        "肥胖": ["BMI", "体重分类", "减重", "体重管理"],
        "met": ["MET", "身体活动", "活动强度", "运动"],
        "运动": ["身体活动", "活动强度", "MET"],
        "跳绳": ["跳绳", "休闲性活动", "高"],
        "冬季": ["冬季"],
        "食养方": ["食养方", "主要材料", "制作方法", "用法用量"],
        "怎么做": ["制作方法", "主要材料", "用法用量"],
        "糖尿病": ["糖尿病", "血糖", "GI", "GL", "交换份", "碳水"],
        "交换份": ["交换份", "每份", "食物交换", "90kcal"],
        "一份": ["交换份", "每份", "重量"],
        "gi": ["GI", "血糖生成指数", "低GI", "高GI"],
        "gl": ["GL", "血糖负荷", "低GL", "高GL"],
        "碳水": ["碳水化合物", "碳水", "交换份"],
        "1600": ["1600", "交换份", "分配表"],
        "高尿酸": ["高尿酸血症", "痛风", "嘌呤", "尿酸", "限酒", "低嘌呤"],
        "痛风": ["痛风", "高尿酸血症", "嘌呤", "尿酸", "限酒", "动物内脏", "海鲜"],
        "嘌呤": ["嘌呤", "嘌呤含量", "高嘌呤", "低嘌呤", "mg/100g"],
        "尿酸": ["尿酸", "高尿酸血症", "痛风", "嘌呤"],
        "动物内脏": ["动物内脏", "肝", "肾", "心", "高嘌呤"],
        "海鲜": ["海鲜", "水产", "鱼", "虾", "蟹", "贝", "高嘌呤"],
        "饮酒": ["饮酒", "限酒", "啤酒", "黄酒", "白酒", "痛风"],
    }
    for trigger, synonyms in synonym_groups.items():
        if trigger.lower() in lowered:
            extra.extend(synonyms)
    seen: set[str] = set()
    tokens: list[str] = []
    for token in words + cjk_trigrams + cjk_bigrams + numbers + extra:
        token = token.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _score_text(text: str, tokens: list[str]) -> float:
    lowered = text.lower()
    compact_lowered = re.sub(r"\s+", "", lowered)
    score = 0.0
    for token in tokens:
        token = token.lower().strip()
        if not token:
            continue
        is_ascii = bool(re.fullmatch(r"[a-z0-9_.+-]+", token))
        is_cjk = bool(re.fullmatch(r"[\u4e00-\u9fff]+", token))
        if is_cjk and len(token) == 1:
            continue
        count = lowered.count(token)
        if count <= 0 and is_cjk:
            count = compact_lowered.count(token)
        if count <= 0:
            continue
        if any(ch.isdigit() for ch in token):
            weight = 4.0
        elif token in {"bmi", "met"}:
            weight = 5.0
        elif is_ascii:
            weight = 2.5
        elif len(token) >= 3:
            weight = 3.0
        else:
            weight = 1.6
        score += min(count, 4) * weight
    return score


def _domain_tags_for_text(text: str) -> str:
    value = str(text or "")
    tags: list[str] = []
    if any(word in value for word in ("高尿酸", "痛风", "嘌呤", "尿酸", "hyperuricemia", "gout")):
        tags.extend(["高尿酸", "痛风", "嘌呤", "尿酸", "低嘌呤", "限酒"])
    if any(word in value for word in ("成人肥胖", "肥胖", "减肥", "减重", "体重管理", "obesity")):
        tags.extend(["肥胖", "减肥", "减重", "体重管理"])
    if any(word in value for word in ("糖尿病", "血糖", "GI", "GL", "diabetes")):
        tags.extend(["糖尿病", "血糖", "GI", "GL", "交换份"])
    return " ".join(dict.fromkeys(tags))


def _format_recipe_plan(row: sqlite3.Row) -> str:
    parts = [str(row["title"] or "食谱")]
    if row["energy_kcal"] is not None:
        parts[0] = f"{row['title']}，约 {float(row['energy_kcal']):g} kcal"
    macros = []
    for label, key in [("蛋白质", "protein_g"), ("碳水", "carbohydrate_g"), ("脂肪", "fat_g")]:
        if row[key] is not None:
            macros.append(f"{label} {float(row[key]):g}g")
    if macros:
        parts.append("，".join(macros))
    return "；".join(parts)


def _format_therapeutic_recipe(row: sqlite3.Row) -> str:
    ingredients = _loads_json(row["ingredients_json"], [])
    ingredient_text = "、".join(item.get("raw_text", "") for item in ingredients[:8] if isinstance(item, dict))
    return f"{row['syndrome']}：{row['title']}；主要材料：{ingredient_text}；制作方法：{row['method']}；用法用量：{row['usage']}；注意：{row['cautions']}"


def _citation(source_name: str, page_start: int, page_end: int, item_id: str) -> str:
    page_label = f"p.{page_start}" if page_start == page_end else f"pp.{page_start}-{page_end}"
    return f"{source_name} {page_label} [{item_id}]"


def _loads_json(value: str, default: Any) -> Any:
    try:
        return json.loads(value or "")
    except Exception:
        return default


def _count(db: sqlite3.Connection, table: str) -> int:
    return int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _count_where(
    db: sqlite3.Connection,
    table: str,
    where_sql: str,
    params: list[Any] | tuple[Any, ...],
) -> int:
    return int(
        db.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {where_sql}",
            tuple(params),
        ).fetchone()[0]
    )


def _stable_id(prefix: str, *parts: Any) -> str:
    import hashlib

    raw = "|".join(str(part or "") for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:20]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative_to(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
