from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .models import MemoryLayer, RetrievedMemory, StructuredMemory, WorkingTurn


WORKING_MEMORY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ltm_working_memory (
    turn_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
"""

SHORT_TERM_SUMMARY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ltm_short_term_summary (
    user_id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    source_session_id TEXT NOT NULL DEFAULT '',
    source_turn_count INTEGER NOT NULL DEFAULT 0,
    max_chars INTEGER NOT NULL DEFAULT 2000,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
"""

MEMORY_ITEMS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ltm_memory_items (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    layer TEXT NOT NULL,
    entity TEXT NOT NULL,
    attribute TEXT NOT NULL,
    value TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    importance REAL NOT NULL,
    weight REAL NOT NULL,
    locked INTEGER NOT NULL DEFAULT 0,
    dedupe_key TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    tags_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    embedding_json TEXT NOT NULL DEFAULT '[]'
);
"""

MEMORY_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_ltm_memory_user_layer
ON ltm_memory_items(user_id, layer, updated_at DESC);
"""

DEDUPE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_ltm_memory_dedupe
ON ltm_memory_items(user_id, dedupe_key, layer);
"""


class PowerMemSQLiteStore:
    def __init__(self, db_path: str, embedding_dimensions: int = 256):
        self.db_path = Path(db_path)
        self.embedding_dimensions = embedding_dimensions
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(WORKING_MEMORY_TABLE_SQL)
            connection.execute(SHORT_TERM_SUMMARY_TABLE_SQL)
            connection.execute(MEMORY_ITEMS_TABLE_SQL)
            connection.execute(MEMORY_INDEX_SQL)
            connection.execute(DEDUPE_INDEX_SQL)
            connection.commit()

    async def save_working_memory(
        self,
        user_id: str,
        session_id: str,
        turns: list[WorkingTurn],
        keep_last: int,
    ) -> None:
        await asyncio.to_thread(
            self._save_working_memory_sync,
            user_id,
            session_id,
            turns,
            keep_last,
        )

    def _save_working_memory_sync(
        self,
        user_id: str,
        session_id: str,
        turns: list[WorkingTurn],
        keep_last: int,
    ) -> None:
        kept_turns = turns[-keep_last:]
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM ltm_working_memory WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            )
            for turn in kept_turns:
                connection.execute(
                    """
                    INSERT INTO ltm_working_memory (
                        turn_id, user_id, session_id, role, content, created_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        turn.turn_id,
                        turn.user_id,
                        turn.session_id,
                        turn.role,
                        turn.content,
                        turn.created_at.isoformat(),
                        json.dumps(turn.metadata, ensure_ascii=False),
                    ),
                )
            connection.commit()

    async def get_working_memory(
        self,
        user_id: str,
        session_id: str | None = None,
        limit: int = 12,
    ) -> list[WorkingTurn]:
        return await asyncio.to_thread(
            self._get_working_memory_sync,
            user_id,
            session_id,
            limit,
        )

    def _get_working_memory_sync(
        self,
        user_id: str,
        session_id: str | None,
        limit: int,
    ) -> list[WorkingTurn]:
        sql = """
        SELECT turn_id, user_id, session_id, role, content, created_at, metadata_json
        FROM ltm_working_memory
        WHERE user_id = ?
        """
        params: list[object] = [user_id]
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        rows = list(reversed(rows))
        return [
            WorkingTurn(
                turn_id=row["turn_id"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                created_at=datetime.fromisoformat(row["created_at"]),
                metadata=json.loads(row["metadata_json"] or "{}"),
            )
            for row in rows
        ]

    async def get_short_term_summary(self, user_id: str) -> dict | None:
        return await asyncio.to_thread(self._get_short_term_summary_sync, user_id)

    def _get_short_term_summary_sync(self, user_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT user_id, summary, source_session_id, source_turn_count,
                       max_chars, created_at, updated_at, metadata_json
                FROM ltm_short_term_summary
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "user_id": row["user_id"],
            "summary": row["summary"],
            "source_session_id": row["source_session_id"],
            "source_turn_count": row["source_turn_count"],
            "max_chars": row["max_chars"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
        }

    async def upsert_short_term_summary(
        self,
        *,
        user_id: str,
        summary: str,
        source_session_id: str,
        source_turn_count: int,
        max_chars: int,
        metadata: dict | None = None,
    ) -> dict | None:
        return await asyncio.to_thread(
            self._upsert_short_term_summary_sync,
            user_id,
            summary,
            source_session_id,
            source_turn_count,
            max_chars,
            metadata or {},
        )

    def _upsert_short_term_summary_sync(
        self,
        user_id: str,
        summary: str,
        source_session_id: str,
        source_turn_count: int,
        max_chars: int,
        metadata: dict,
    ) -> dict | None:
        now = datetime.utcnow().isoformat()
        clean_summary = str(summary or "").strip()[:max_chars]
        if not clean_summary:
            return self._get_short_term_summary_sync(user_id)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT created_at FROM ltm_short_term_summary WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            created_at = existing["created_at"] if existing else now
            connection.execute(
                """
                INSERT INTO ltm_short_term_summary (
                    user_id, summary, source_session_id, source_turn_count,
                    max_chars, created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    summary = excluded.summary,
                    source_session_id = excluded.source_session_id,
                    source_turn_count = excluded.source_turn_count,
                    max_chars = excluded.max_chars,
                    updated_at = excluded.updated_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    user_id,
                    clean_summary,
                    source_session_id or "",
                    int(source_turn_count or 0),
                    int(max_chars or 2000),
                    created_at,
                    now,
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
            connection.commit()
        return self._get_short_term_summary_sync(user_id)

    async def upsert_memories(self, memories: Iterable[StructuredMemory]) -> list[StructuredMemory]:
        return await asyncio.to_thread(self._upsert_memories_sync, list(memories))

    def _upsert_memories_sync(self, memories: list[StructuredMemory]) -> list[StructuredMemory]:
        if not memories:
            return []

        persisted: list[StructuredMemory] = []
        with self._connect() as connection:
            for memory in memories:
                existing = connection.execute(
                    """
                    SELECT * FROM ltm_memory_items
                    WHERE user_id = ? AND dedupe_key = ? AND layer = ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (memory.user_id, memory.dedupe_key, memory.layer.value),
                ).fetchone()

                if existing and memory.layer == MemoryLayer.FACTUAL:
                    evidence = json.loads(existing["evidence_json"] or "[]")
                    evidence.extend(item for item in memory.evidence if item not in evidence)
                    metadata = json.loads(existing["metadata_json"] or "{}")
                    if memory.value != existing["value"]:
                        conflicts = metadata.setdefault("conflicts", [])
                        conflicts.append(
                            {
                                "incoming_value": memory.value,
                                "observed_at": memory.observed_at.isoformat(),
                                "source": memory.source,
                            }
                        )
                    connection.execute(
                        """
                        UPDATE ltm_memory_items
                        SET updated_at = ?, evidence_json = ?, metadata_json = ?
                        WHERE memory_id = ?
                        """,
                        (
                            datetime.utcnow().isoformat(),
                            json.dumps(evidence, ensure_ascii=False),
                            json.dumps(metadata, ensure_ascii=False),
                            existing["memory_id"],
                        ),
                    )
                    updated_row = connection.execute(
                        "SELECT * FROM ltm_memory_items WHERE memory_id = ?",
                        (existing["memory_id"],),
                    ).fetchone()
                    persisted.append(self._row_to_memory(updated_row))
                    continue

                if existing:
                    merged_evidence = json.loads(existing["evidence_json"] or "[]")
                    for item in memory.evidence:
                        if item not in merged_evidence:
                            merged_evidence.append(item)
                    merged_metadata = json.loads(existing["metadata_json"] or "{}")
                    merged_metadata.update(memory.metadata or {})
                    connection.execute(
                        """
                        UPDATE ltm_memory_items
                        SET value = ?, content = ?, source = ?, observed_at = ?, updated_at = ?,
                            importance = ?, weight = ?, locked = ?, evidence_json = ?, tags_json = ?,
                            metadata_json = ?, embedding_json = ?
                        WHERE memory_id = ?
                        """,
                        (
                            memory.value,
                            memory.content,
                            memory.source,
                            memory.observed_at.isoformat(),
                            memory.updated_at.isoformat(),
                            memory.importance,
                            memory.weight,
                            1 if memory.locked else 0,
                            json.dumps(merged_evidence, ensure_ascii=False),
                            json.dumps(memory.tags, ensure_ascii=False),
                            json.dumps(merged_metadata, ensure_ascii=False),
                            json.dumps(memory.embedding, ensure_ascii=False),
                            existing["memory_id"],
                        ),
                    )
                    updated_row = connection.execute(
                        "SELECT * FROM ltm_memory_items WHERE memory_id = ?",
                        (existing["memory_id"],),
                    ).fetchone()
                    persisted.append(self._row_to_memory(updated_row))
                    continue

                connection.execute(
                    """
                    INSERT INTO ltm_memory_items (
                        memory_id, user_id, layer, entity, attribute, value, content, source,
                        observed_at, created_at, updated_at, importance, weight, locked,
                        dedupe_key, evidence_json, tags_json, metadata_json, embedding_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory.memory_id,
                        memory.user_id,
                        memory.layer.value,
                        memory.entity,
                        memory.attribute,
                        memory.value,
                        memory.content,
                        memory.source,
                        memory.observed_at.isoformat(),
                        memory.created_at.isoformat(),
                        memory.updated_at.isoformat(),
                        memory.importance,
                        memory.weight,
                        1 if memory.locked else 0,
                        memory.dedupe_key,
                        json.dumps(memory.evidence, ensure_ascii=False),
                        json.dumps(memory.tags, ensure_ascii=False),
                        json.dumps(memory.metadata, ensure_ascii=False),
                        json.dumps(memory.embedding, ensure_ascii=False),
                    ),
                )
                inserted_row = connection.execute(
                    "SELECT * FROM ltm_memory_items WHERE memory_id = ?",
                    (memory.memory_id,),
                ).fetchone()
                persisted.append(self._row_to_memory(inserted_row))
            connection.commit()
        return persisted

    async def list_recent_memories(
        self,
        user_id: str,
        layer: MemoryLayer | None = None,
        limit: int = 20,
    ) -> list[StructuredMemory]:
        return await asyncio.to_thread(self._list_recent_memories_sync, user_id, layer, limit)

    def _list_recent_memories_sync(
        self,
        user_id: str,
        layer: MemoryLayer | None,
        limit: int,
    ) -> list[StructuredMemory]:
        sql = """
        SELECT * FROM ltm_memory_items
        WHERE user_id = ?
        """
        params: list[object] = [user_id]
        if layer:
            sql += " AND layer = ?"
            params.append(layer.value)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._row_to_memory(row) for row in rows]

    async def apply_forgetting_curve(
        self,
        user_id: str,
        episodic_half_life_days: float,
        semantic_half_life_days: float,
        min_weight: float,
    ) -> None:
        await asyncio.to_thread(
            self._apply_forgetting_curve_sync,
            user_id,
            episodic_half_life_days,
            semantic_half_life_days,
            min_weight,
        )

    def _apply_forgetting_curve_sync(
        self,
        user_id: str,
        episodic_half_life_days: float,
        semantic_half_life_days: float,
        min_weight: float,
    ) -> None:
        now = datetime.utcnow()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT memory_id, layer, importance, weight, locked, updated_at
                FROM ltm_memory_items
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchall()

            for row in rows:
                if row["locked"]:
                    continue
                updated_at = datetime.fromisoformat(row["updated_at"])
                age_days = max((now - updated_at).total_seconds() / 86400, 0.0)
                half_life = (
                    semantic_half_life_days
                    if row["layer"] == MemoryLayer.SEMANTIC.value
                    else episodic_half_life_days
                )
                base_importance = float(row["importance"])
                decayed = base_importance * math.exp(-math.log(2) * age_days / max(half_life, 0.1))
                connection.execute(
                    "UPDATE ltm_memory_items SET weight = ? WHERE memory_id = ?",
                    (max(min_weight, decayed), row["memory_id"]),
                )
            connection.commit()

    async def search_memories(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int,
        min_weight: float,
    ) -> list[RetrievedMemory]:
        return await asyncio.to_thread(
            self._search_memories_sync,
            user_id,
            query_embedding,
            top_k,
            min_weight,
        )

    def _search_memories_sync(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int,
        min_weight: float,
    ) -> list[RetrievedMemory]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM ltm_memory_items
                WHERE user_id = ? AND weight >= ?
                """,
                (user_id, min_weight),
            ).fetchall()

        scored: list[RetrievedMemory] = []
        for row in rows:
            embedding = json.loads(row["embedding_json"] or "[]")
            cosine = self.cosine_similarity(query_embedding, embedding)
            layer = MemoryLayer(row["layer"])
            layer_bonus = {
                MemoryLayer.FACTUAL: 0.25,
                MemoryLayer.SEMANTIC: 0.18,
                MemoryLayer.EPISODIC: 0.12,
            }.get(layer, 0.0)
            score = cosine + float(row["weight"]) + layer_bonus
            scored.append(
                RetrievedMemory(
                    memory_id=row["memory_id"],
                    user_id=row["user_id"],
                    layer=layer,
                    content=row["content"],
                    source=row["source"],
                    observed_at=datetime.fromisoformat(row["observed_at"]),
                    weight=float(row["weight"]),
                    importance=float(row["importance"]),
                    locked=bool(row["locked"]),
                    score=score,
                    metadata=json.loads(row["metadata_json"] or "{}"),
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    async def clear_all_user_data(self, user_id: str) -> None:
        await asyncio.to_thread(self._clear_all_user_data_sync, user_id)

    def _clear_all_user_data_sync(self, user_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM ltm_working_memory WHERE user_id = ?", (user_id,))
            connection.execute("DELETE FROM ltm_short_term_summary WHERE user_id = ?", (user_id,))
            connection.execute("DELETE FROM ltm_memory_items WHERE user_id = ?", (user_id,))
            connection.commit()

    async def get_memory_by_id(self, memory_id: str) -> StructuredMemory | None:
        return await asyncio.to_thread(self._get_memory_by_id_sync, memory_id)

    def _get_memory_by_id_sync(self, memory_id: str) -> StructuredMemory | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ltm_memory_items WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    async def attach_powermem_index_id(self, memory_id: str, index_id: int) -> StructuredMemory | None:
        return await asyncio.to_thread(self._attach_powermem_index_id_sync, memory_id, index_id)

    def _attach_powermem_index_id_sync(self, memory_id: str, index_id: int) -> StructuredMemory | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT metadata_json FROM ltm_memory_items WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row is None:
                return None
            metadata = json.loads(row["metadata_json"] or "{}")
            metadata["powermem_memory_id"] = int(index_id)
            connection.execute(
                "UPDATE ltm_memory_items SET metadata_json = ? WHERE memory_id = ?",
                (json.dumps(metadata, ensure_ascii=False), memory_id),
            )
            connection.commit()
        return self._get_memory_by_id_sync(memory_id)

    @staticmethod
    def cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)

    def embed_text(self, text: str) -> list[float]:
        return hashed_embedding(text, self.embedding_dimensions)

    def _row_to_memory(self, row: sqlite3.Row) -> StructuredMemory:
        return StructuredMemory(
            memory_id=row["memory_id"],
            user_id=row["user_id"],
            layer=MemoryLayer(row["layer"]),
            entity=row["entity"],
            attribute=row["attribute"],
            value=row["value"],
            content=row["content"],
            source=row["source"],
            observed_at=datetime.fromisoformat(row["observed_at"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            importance=float(row["importance"]),
            weight=float(row["weight"]),
            locked=bool(row["locked"]),
            dedupe_key=row["dedupe_key"],
            evidence=json.loads(row["evidence_json"] or "[]"),
            tags=json.loads(row["tags_json"] or "[]"),
            metadata=json.loads(row["metadata_json"] or "{}"),
            embedding=json.loads(row["embedding_json"] or "[]"),
        )


def hashed_embedding(text: str, dimensions: int = 256) -> list[float]:
    vector = [0.0] * dimensions
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for offset in range(0, 8, 2):
            idx = int.from_bytes(digest[offset:offset + 2], "big") % dimensions
            sign = 1.0 if digest[offset] % 2 == 0 else -1.0
            vector[idx] += sign

    norm = math.sqrt(sum(item * item for item in vector))
    if not norm:
        return vector
    return [item / norm for item in vector]
