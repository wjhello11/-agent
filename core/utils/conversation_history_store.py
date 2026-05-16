from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CREATE_SESSIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS console_conversation_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    device_id TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    preview TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0,
    has_tool_calls INTEGER NOT NULL DEFAULT 0,
    has_vision INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    transcript_json TEXT NOT NULL DEFAULT '[]'
);
"""


class ConversationHistoryStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(CREATE_SESSIONS_TABLE_SQL)
            conn.commit()

    def upsert_session(
        self,
        *,
        session_id: str,
        user_id: str,
        device_id: str,
        title: str,
        preview: str,
        created_at: str,
        updated_at: str,
        message_count: int,
        has_tool_calls: bool,
        has_vision: bool,
        metadata: dict[str, Any] | None,
        transcript: list[dict[str, Any]] | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO console_conversation_sessions (
                    session_id, user_id, device_id, title, preview,
                    created_at, updated_at, message_count,
                    has_tool_calls, has_vision, metadata_json, transcript_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    device_id = excluded.device_id,
                    title = excluded.title,
                    preview = excluded.preview,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    message_count = excluded.message_count,
                    has_tool_calls = excluded.has_tool_calls,
                    has_vision = excluded.has_vision,
                    metadata_json = excluded.metadata_json,
                    transcript_json = excluded.transcript_json
                """,
                (
                    session_id,
                    user_id,
                    device_id,
                    title,
                    preview,
                    created_at,
                    updated_at,
                    int(message_count or 0),
                    1 if has_tool_calls else 0,
                    1 if has_vision else 0,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    json.dumps(transcript or [], ensure_ascii=False),
                ),
            )
            conn.commit()

    def list_sessions(self, *, limit: int = 50, user_id: str = "") -> list[dict[str, Any]]:
        sql = """
        SELECT session_id, user_id, device_id, title, preview, created_at, updated_at,
               message_count, has_tool_calls, has_vision, metadata_json
        FROM console_conversation_sessions
        """
        params: list[Any] = []
        if user_id:
            sql += " WHERE user_id = ?"
            params.append(user_id)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(int(limit))

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            {
                "session_id": row["session_id"],
                "user_id": row["user_id"],
                "device_id": row["device_id"],
                "title": row["title"],
                "preview": row["preview"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "message_count": int(row["message_count"] or 0),
                "has_tool_calls": bool(row["has_tool_calls"]),
                "has_vision": bool(row["has_vision"]),
                "metadata": _loads_json(row["metadata_json"], {}),
            }
            for row in rows
        ]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id, user_id, device_id, title, preview, created_at, updated_at,
                       message_count, has_tool_calls, has_vision, metadata_json, transcript_json
                FROM console_conversation_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "session_id": row["session_id"],
            "user_id": row["user_id"],
            "device_id": row["device_id"],
            "title": row["title"],
            "preview": row["preview"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "message_count": int(row["message_count"] or 0),
            "has_tool_calls": bool(row["has_tool_calls"]),
            "has_vision": bool(row["has_vision"]),
            "metadata": _loads_json(row["metadata_json"], {}),
            "messages": _loads_json(row["transcript_json"], []),
        }


def build_session_title(messages: list[dict[str, Any]]) -> str:
    first_user = next(
        (
            str(item.get("content", "")).strip()
            for item in messages
            if item.get("role") == "user" and str(item.get("content", "")).strip()
        ),
        "",
    )
    if not first_user:
        return "未命名会话"
    if len(first_user) <= 22:
        return first_user
    return first_user[:22].rstrip("，。！？,.!? ") + "..."


def build_session_preview(messages: list[dict[str, Any]]) -> str:
    for item in messages:
        content = str(item.get("content", "")).strip()
        if content:
            if len(content) <= 40:
                return content
            return content[:40].rstrip("，。！？,.!? ") + "..."
    return ""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _loads_json(raw: Any, default: Any) -> Any:
    if raw in (None, ""):
        return default
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default
