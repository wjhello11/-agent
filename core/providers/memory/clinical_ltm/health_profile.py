from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


PROFILE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS health_profiles (
    user_id TEXT PRIMARY KEY,
    age_years REAL,
    sex TEXT,
    height_cm REAL,
    weight_kg REAL,
    bmi REAL,
    activity_level TEXT,
    nutrition_goal TEXT,
    target_energy_kcal REAL,
    target_carbohydrate_g_per_meal REAL,
    target_protein_g_per_day REAL,
    target_fat_g_per_day REAL,
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'unknown',
    evidence_json TEXT NOT NULL DEFAULT '[]'
);
"""

PROFILE_ITEM_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS health_profile_items (
    item_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    category TEXT NOT NULL,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    value_json TEXT NOT NULL DEFAULT '{}',
    source TEXT NOT NULL DEFAULT 'unknown',
    evidence TEXT NOT NULL DEFAULT '',
    observed_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.75,
    notes TEXT NOT NULL DEFAULT '',
    UNIQUE(user_id, category, normalized_name),
    FOREIGN KEY(user_id) REFERENCES health_profiles(user_id) ON DELETE CASCADE
);
"""

PROFILE_ITEM_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_health_profile_items_user_category
ON health_profile_items(user_id, category, updated_at DESC);
"""

BLOOD_GLUCOSE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS blood_glucose_readings (
    reading_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    measured_at TEXT NOT NULL,
    reported_at TEXT NOT NULL,
    value_mmol_l REAL NOT NULL,
    measurement_type TEXT NOT NULL,
    meal_context TEXT NOT NULL DEFAULT '',
    time_context TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'unknown',
    evidence TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.85,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE(user_id, measured_at, measurement_type, value_mmol_l, evidence),
    FOREIGN KEY(user_id) REFERENCES health_profiles(user_id) ON DELETE CASCADE
);
"""

BLOOD_GLUCOSE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_blood_glucose_readings_user_time
ON blood_glucose_readings(user_id, measured_at DESC);
"""

NUTRITION_INTAKE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS daily_nutrition_intakes (
    intake_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    intake_date TEXT NOT NULL,
    meal_label TEXT NOT NULL DEFAULT '',
    meal_text TEXT NOT NULL DEFAULT '',
    energy_kcal REAL,
    carbohydrate_g REAL,
    protein_g REAL,
    fat_g REAL,
    dietary_fiber_g REAL,
    sodium_mg REAL,
    potassium_mg REAL,
    phosphorus_mg REAL,
    items_json TEXT NOT NULL DEFAULT '[]',
    source TEXT NOT NULL DEFAULT 'unknown',
    source_session_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE(user_id, occurred_at, meal_text, source),
    FOREIGN KEY(user_id) REFERENCES health_profiles(user_id) ON DELETE CASCADE
);
"""

NUTRITION_INTAKE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_daily_nutrition_intakes_user_date
ON daily_nutrition_intakes(user_id, intake_date DESC, occurred_at DESC);
"""

PROFILE_REVIEW_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS health_profile_review_items (
    review_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    field_type TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL,
    current_value_json TEXT NOT NULL DEFAULT '{}',
    proposed_value_json TEXT NOT NULL DEFAULT '{}',
    source TEXT NOT NULL DEFAULT 'unknown',
    evidence TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    resolution_action TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    resolved_at TEXT,
    resolved_by TEXT NOT NULL DEFAULT '',
    UNIQUE(user_id, field_type, category, name, proposed_value_json, status)
);
"""

PROFILE_REVIEW_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_health_profile_review_user_status
ON health_profile_review_items(user_id, status, updated_at DESC);
"""

GLUCOSE_TYPE_LABELS = {
    "fasting": "空腹",
    "pre_meal": "餐前",
    "post_meal": "餐后",
    "postprandial_2h": "餐后2小时",
    "bedtime": "睡前",
    "random": "随机",
}

GLUCOSE_TARGETS_MMOL_L = {
    "fasting": (3.9, 7.0),
    "pre_meal": (3.9, 7.0),
    "postprandial_2h": (3.9, 10.0),
    "post_meal": (3.9, 10.0),
    "bedtime": (3.9, 10.0),
}

GLUCOSE_VALUE_PATTERN = (
    r"(?P<value>\d{1,3}(?:\.\d+)?|[零〇一二三四五六七八九十两]{1,4}(?:点[零〇一二三四五六七八九])?)"
)

PROFILE_SCALAR_FIELDS = {
    "age_years",
    "sex",
    "height_cm",
    "weight_kg",
    "activity_level",
    "nutrition_goal",
    "target_energy_kcal",
    "target_carbohydrate_g_per_meal",
    "target_protein_g_per_day",
    "target_fat_g_per_day",
    "notes",
}

CONFLICT_REVIEW_SCALAR_FIELDS = {
    "age_years",
    "sex",
    "height_cm",
    "weight_kg",
    "target_energy_kcal",
    "target_carbohydrate_g_per_meal",
    "target_protein_g_per_day",
    "target_fat_g_per_day",
}

PROFILE_ITEM_CATEGORIES = {
    "disease",
    "medication",
    "allergy",
    "goal",
    "renal_function",
    "glucose_metric",
    "exercise",
    "dietary_restriction",
}

FIELD_LABELS = {
    "age_years": "年龄",
    "sex": "性别",
    "height_cm": "身高",
    "weight_kg": "体重",
    "bmi": "BMI",
    "activity_level": "活动水平",
    "nutrition_goal": "营养目标",
    "target_energy_kcal": "每日热量目标",
    "target_carbohydrate_g_per_meal": "每餐碳水目标",
    "target_protein_g_per_day": "每日蛋白质目标",
    "target_fat_g_per_day": "每日脂肪目标",
    "notes": "备注",
}

CATEGORY_LABELS = {
    "disease": "疾病/诊断",
    "medication": "用药",
    "allergy": "过敏",
    "goal": "健康目标",
    "renal_function": "肾功能",
    "glucose_metric": "血糖指标",
    "exercise": "运动习惯",
    "dietary_restriction": "饮食限制",
}

DISEASE_ALIASES = [
    ("2型糖尿病", ["2型糖尿病", "二型糖尿病", "type 2 diabetes", "t2dm"]),
    ("1型糖尿病", ["1型糖尿病", "一型糖尿病", "type 1 diabetes", "t1dm"]),
    ("妊娠期糖尿病", ["妊娠期糖尿病", "妊娠糖尿病"]),
    ("糖尿病", ["糖尿病", "diabetes"]),
    ("高血压", ["高血压", "hypertension"]),
    ("高脂血症", ["高脂血症", "高血脂", "血脂高", "hyperlipidemia"]),
    ("慢性肾脏病", ["慢性肾脏病", "肾功能不全", "肾病", "ckd", "chronic kidney disease"]),
    ("痛风", ["痛风", "gout"]),
    ("高尿酸血症", ["高尿酸血症", "高尿酸", "hyperuricemia"]),
    ("冠心病", ["冠心病", "冠状动脉粥样硬化性心脏病"]),
    ("脂肪肝", ["脂肪肝"]),
]

MEDICATION_ALIASES = [
    "硝苯地平",
    "非洛地平",
    "辛伐他汀",
    "阿托伐他汀",
    "瑞舒伐他汀",
    "二甲双胍",
    "胰岛素",
    "阿卡波糖",
    "达格列净",
    "恩格列净",
    "利拉鲁肽",
    "司美格鲁肽",
    "华法林",
    "头孢",
    "甲硝唑",
    "依那普利",
    "贝那普利",
    "卡托普利",
    "氯沙坦",
    "缬沙坦",
    "厄贝沙坦",
]

ALLERGY_ALIASES = [
    "花生",
    "牛奶",
    "鸡蛋",
    "大豆",
    "海鲜",
    "虾",
    "蟹",
    "鱼",
    "坚果",
    "麸质",
    "小麦",
]


@dataclass
class ProfileItem:
    category: str
    name: str
    value: dict[str, Any] = field(default_factory=dict)
    status: str = "active"
    source: str = "user_reported"
    evidence: str = ""
    observed_at: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 0.75
    notes: str = ""


@dataclass
class BloodGlucoseReading:
    value_mmol_l: float
    measurement_type: str = "random"
    meal_context: str = ""
    time_context: str = ""
    source: str = "user_reported"
    evidence: str = ""
    observed_at: datetime = field(default_factory=lambda: datetime.now(_local_tz()))
    reported_at: datetime = field(default_factory=lambda: datetime.now(_local_tz()))
    confidence: float = 0.85
    notes: str = ""


@dataclass
class ProfileUpdate:
    scalars: dict[str, Any] = field(default_factory=dict)
    scalar_evidence: dict[str, str] = field(default_factory=dict)
    items: list[ProfileItem] = field(default_factory=list)
    glucose_readings: list[BloodGlucoseReading] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.scalars and not self.items and not self.glucose_readings


class HealthProfileStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(PROFILE_TABLE_SQL)
            connection.execute(PROFILE_ITEM_TABLE_SQL)
            connection.execute(PROFILE_ITEM_INDEX_SQL)
            connection.execute(BLOOD_GLUCOSE_TABLE_SQL)
            connection.execute(BLOOD_GLUCOSE_INDEX_SQL)
            connection.execute(NUTRITION_INTAKE_TABLE_SQL)
            connection.execute(NUTRITION_INTAKE_INDEX_SQL)
            connection.execute(PROFILE_REVIEW_TABLE_SQL)
            connection.execute(PROFILE_REVIEW_INDEX_SQL)
            connection.commit()

    async def apply_update(self, user_id: str, update: ProfileUpdate) -> dict[str, int]:
        return await asyncio.to_thread(self.apply_update_sync, user_id, update)

    def apply_update_sync(self, user_id: str, update: ProfileUpdate) -> dict[str, int]:
        now = datetime.utcnow().isoformat()
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            self._ensure_profile_row(connection, user_id, now)
            scalar_count, review_count = self._apply_scalars(
                connection,
                user_id,
                update.scalars,
                update.scalar_evidence,
                now,
            )
            item_count = 0
            for item in update.items:
                applied, reviews = self._upsert_item(connection, user_id, item, now)
                item_count += applied
                review_count += reviews
            glucose_count = self._insert_glucose_readings(
                connection,
                user_id,
                update.glucose_readings,
                now,
            )
            self._refresh_bmi(connection, user_id, now)
            connection.commit()
        return {
            "scalar_count": scalar_count,
            "item_count": item_count,
            "glucose_count": glucose_count,
            "review_count": review_count,
        }

    async def get_profile(self, user_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.get_profile_sync, user_id)

    def get_profile_sync(self, user_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            profile = connection.execute(
                "SELECT * FROM health_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            items = connection.execute(
                """
                SELECT * FROM health_profile_items
                WHERE user_id = ? AND status != 'deleted'
                ORDER BY category, updated_at DESC, name
                """,
                (user_id,),
            ).fetchall()
            glucose_rows = self._fetch_glucose_rows(connection, user_id, limit=30)

        if profile is None:
            return {
                "user_id": user_id,
                "scalars": {},
                "items": [],
                "glucose_readings": [],
                "glucose_analysis": analyze_blood_glucose_readings([]),
                "nutrition_intake_series": [],
                "review_items": [],
            }
        glucose_readings = [_glucose_row_to_dict(row) for row in glucose_rows]
        return {
            "user_id": user_id,
            "scalars": _profile_row_to_scalars(profile),
            "items": [_item_row_to_dict(row) for row in items],
            "glucose_readings": glucose_readings,
            "glucose_analysis": analyze_blood_glucose_readings(glucose_readings),
            "nutrition_intake_series": self.get_nutrition_intake_series_sync(user_id),
            "review_items": self.list_review_items_sync(user_id, status="pending"),
            "created_at": profile["created_at"],
            "updated_at": profile["updated_at"],
        }

    async def build_prompt_context(self, user_id: str) -> str:
        profile = await self.get_profile(user_id)
        return format_health_profile_context(profile)

    async def list_review_items(self, user_id: str, status: str = "pending") -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.list_review_items_sync, user_id, status)

    def list_review_items_sync(self, user_id: str, status: str = "pending") -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM health_profile_review_items
                WHERE user_id = ? AND (? = 'all' OR status = ?)
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 100
                """,
                (user_id, status, status),
            ).fetchall()
        return [_review_row_to_dict(row) for row in rows]

    async def resolve_review_item(
        self,
        review_id: str,
        decision: str,
        resolved_by: str = "console_admin",
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self.resolve_review_item_sync,
            review_id,
            decision,
            resolved_by,
        )

    def resolve_review_item_sync(
        self,
        review_id: str,
        decision: str,
        resolved_by: str = "console_admin",
    ) -> dict[str, Any]:
        normalized_decision = str(decision or "").strip().lower()
        if normalized_decision not in {"accept", "reject"}:
            raise ValueError("decision must be accept or reject")
        now = datetime.utcnow().isoformat()
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            row = connection.execute(
                "SELECT * FROM health_profile_review_items WHERE review_id = ?",
                (review_id,),
            ).fetchone()
            if row is None:
                raise ValueError("review item not found")
            if row["status"] != "pending":
                return _review_row_to_dict(row)

            if normalized_decision == "accept":
                self._apply_review_resolution(connection, row, now)
            new_status = "accepted" if normalized_decision == "accept" else "rejected"
            connection.execute(
                """
                UPDATE health_profile_review_items
                SET status = ?, updated_at = ?, resolved_at = ?, resolved_by = ?
                WHERE review_id = ?
                """,
                (new_status, now, now, resolved_by, review_id),
            )
            connection.commit()
            updated = connection.execute(
                "SELECT * FROM health_profile_review_items WHERE review_id = ?",
                (review_id,),
            ).fetchone()
        return _review_row_to_dict(updated)

    async def clear_profile(self, user_id: str) -> None:
        await asyncio.to_thread(self.clear_profile_sync, user_id)

    def clear_profile_sync(self, user_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM health_profile_items WHERE user_id = ?", (user_id,))
            connection.execute("DELETE FROM blood_glucose_readings WHERE user_id = ?", (user_id,))
            connection.execute("DELETE FROM daily_nutrition_intakes WHERE user_id = ?", (user_id,))
            connection.execute("DELETE FROM health_profile_review_items WHERE user_id = ?", (user_id,))
            connection.execute("DELETE FROM health_profiles WHERE user_id = ?", (user_id,))
            connection.commit()

    async def record_nutrition_intake(
        self,
        user_id: str,
        *,
        meal_text: str,
        totals: dict[str, Any],
        items: list[dict[str, Any]] | None = None,
        occurred_at: str | None = None,
        meal_label: str = "",
        source: str = "meal_nutrition_tool",
        source_session_id: str = "",
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self.record_nutrition_intake_sync,
            user_id,
            meal_text=meal_text,
            totals=totals,
            items=items,
            occurred_at=occurred_at,
            meal_label=meal_label,
            source=source,
            source_session_id=source_session_id,
        )

    def record_nutrition_intake_sync(
        self,
        user_id: str,
        *,
        meal_text: str,
        totals: dict[str, Any],
        items: list[dict[str, Any]] | None = None,
        occurred_at: str | None = None,
        meal_label: str = "",
        source: str = "meal_nutrition_tool",
        source_session_id: str = "",
    ) -> dict[str, Any]:
        now = datetime.now(_local_tz()).isoformat(timespec="seconds")
        occurred = _normalize_intake_time(occurred_at) or now
        intake_date = occurred[:10]
        clean_totals = {
            key: _safe_float(totals.get(key))
            for key in (
                "energy_kcal",
                "carbohydrate_g",
                "protein_g",
                "fat_g",
                "dietary_fiber_g",
                "sodium_mg",
                "potassium_mg",
                "phosphorus_mg",
            )
        }
        digest_payload = json.dumps(clean_totals, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha1(
            f"{user_id}|{occurred[:16]}|{meal_text}|{digest_payload}|{source}".encode("utf-8")
        ).hexdigest()[:18]
        intake_id = f"{user_id}:intake:{digest}"
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            self._ensure_profile_row(connection, user_id, now)
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO daily_nutrition_intakes (
                    intake_id, user_id, occurred_at, intake_date, meal_label, meal_text,
                    energy_kcal, carbohydrate_g, protein_g, fat_g, dietary_fiber_g,
                    sodium_mg, potassium_mg, phosphorus_mg, items_json, source,
                    source_session_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    intake_id,
                    user_id,
                    occurred,
                    intake_date,
                    meal_label or "",
                    meal_text or "",
                    clean_totals["energy_kcal"],
                    clean_totals["carbohydrate_g"],
                    clean_totals["protein_g"],
                    clean_totals["fat_g"],
                    clean_totals["dietary_fiber_g"],
                    clean_totals["sodium_mg"],
                    clean_totals["potassium_mg"],
                    clean_totals["phosphorus_mg"],
                    json.dumps(items or [], ensure_ascii=False),
                    source or "unknown",
                    source_session_id or "",
                    now,
                ),
            )
            connection.commit()
        return {
            "intake_id": intake_id,
            "inserted": bool(cursor.rowcount),
            "occurred_at": occurred,
            "intake_date": intake_date,
            "totals": clean_totals,
        }

    async def get_nutrition_intake_series(self, user_id: str, days: int = 30) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.get_nutrition_intake_series_sync, user_id, days)

    def get_nutrition_intake_series_sync(self, user_id: str, days: int = 30) -> list[dict[str, Any]]:
        days = max(1, min(int(days or 30), 180))
        today = datetime.now(_local_tz()).date()
        start_date = today - timedelta(days=days - 1)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    intake_date,
                    COUNT(*) AS intake_count,
                    SUM(COALESCE(energy_kcal, 0)) AS energy_kcal,
                    SUM(COALESCE(carbohydrate_g, 0)) AS carbohydrate_g,
                    SUM(COALESCE(protein_g, 0)) AS protein_g,
                    SUM(COALESCE(fat_g, 0)) AS fat_g,
                    SUM(COALESCE(dietary_fiber_g, 0)) AS dietary_fiber_g,
                    SUM(COALESCE(sodium_mg, 0)) AS sodium_mg,
                    SUM(COALESCE(potassium_mg, 0)) AS potassium_mg,
                    SUM(COALESCE(phosphorus_mg, 0)) AS phosphorus_mg
                FROM daily_nutrition_intakes
                WHERE user_id = ? AND intake_date >= ?
                GROUP BY intake_date
                ORDER BY intake_date
                """,
                (user_id, start_date.isoformat()),
            ).fetchall()
        by_date = {row["intake_date"]: row for row in rows}
        series = []
        for offset in range(days):
            item_date = (start_date + timedelta(days=offset)).isoformat()
            row = by_date.get(item_date)
            series.append(
                {
                    "date": item_date,
                    "intake_count": int(row["intake_count"] or 0) if row else 0,
                    "energy_kcal": round(float(row["energy_kcal"] or 0), 1) if row else 0.0,
                    "carbohydrate_g": round(float(row["carbohydrate_g"] or 0), 1) if row else 0.0,
                    "protein_g": round(float(row["protein_g"] or 0), 1) if row else 0.0,
                    "fat_g": round(float(row["fat_g"] or 0), 1) if row else 0.0,
                    "dietary_fiber_g": round(float(row["dietary_fiber_g"] or 0), 1) if row else 0.0,
                    "sodium_mg": round(float(row["sodium_mg"] or 0), 1) if row else 0.0,
                    "potassium_mg": round(float(row["potassium_mg"] or 0), 1) if row else 0.0,
                    "phosphorus_mg": round(float(row["phosphorus_mg"] or 0), 1) if row else 0.0,
                }
            )
        return series

    def _ensure_profile_row(self, connection: sqlite3.Connection, user_id: str, now: str) -> None:
        connection.execute(
            """
            INSERT OR IGNORE INTO health_profiles (user_id, created_at, updated_at, source)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, now, now, "system"),
        )

    def _apply_scalars(
        self,
        connection: sqlite3.Connection,
        user_id: str,
        scalars: dict[str, Any],
        scalar_evidence: dict[str, str],
        now: str,
    ) -> tuple[int, int]:
        cleaned = {
            field_name: _normalize_scalar_value(field_name, value)
            for field_name, value in scalars.items()
            if field_name in PROFILE_SCALAR_FIELDS and value not in (None, "")
        }
        cleaned = {key: value for key, value in cleaned.items() if value not in (None, "")}
        if not cleaned:
            return 0, 0

        current = connection.execute(
            "SELECT * FROM health_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        review_count = 0
        accepted: dict[str, Any] = {}
        for key, value in cleaned.items():
            current_value = current[key] if current and key in current.keys() else None
            if _is_scalar_conflict(key, current_value, value):
                review_count += self._create_review_item(
                    connection=connection,
                    user_id=user_id,
                    field_type="scalar",
                    category="basic",
                    name=key,
                    current_value={"value": current_value},
                    proposed_value={"value": value},
                    source="user_reported",
                    evidence=scalar_evidence.get(key, ""),
                    reason=_scalar_conflict_reason(key, current_value, value),
                    resolution_action="apply_scalar",
                    now=now,
                )
            else:
                accepted[key] = value

        if not accepted:
            return 0, review_count

        evidence = json.loads(current["evidence_json"] or "[]") if current else []
        for key, value in accepted.items():
            evidence_item = f"{FIELD_LABELS.get(key, key)}={value}"
            if evidence_item not in evidence:
                evidence.append(evidence_item)

        set_clause = ", ".join(f"{key} = ?" for key in accepted)
        values = list(accepted.values())
        connection.execute(
            f"""
            UPDATE health_profiles
            SET {set_clause}, updated_at = ?, source = ?, evidence_json = ?
            WHERE user_id = ?
            """,
            [*values, now, "user_reported", json.dumps(evidence[-40:], ensure_ascii=False), user_id],
        )
        return len(accepted), review_count

    def _upsert_item(
        self,
        connection: sqlite3.Connection,
        user_id: str,
        item: ProfileItem,
        now: str,
    ) -> tuple[int, int]:
        if item.category not in PROFILE_ITEM_CATEGORIES:
            return 0, 0
        name = str(item.name or "").strip()
        if not name:
            return 0, 0
        normalized_name = normalize_name(name)
        if _is_category_negation_item(item):
            existing = self._active_items_for_category(connection, user_id, item.category)
            existing = [row for row in existing if not _is_negation_name(row["name"])]
            if existing:
                review_count = self._create_review_item(
                    connection=connection,
                    user_id=user_id,
                    field_type="item",
                    category=item.category,
                    name=item.name,
                    current_value={"items": [_item_row_to_dict(row) for row in existing]},
                    proposed_value={"status": "negate_category", "name": item.name},
                    source=item.source,
                    evidence=item.evidence,
                    reason=f"用户说没有{CATEGORY_LABELS.get(item.category, item.category)}，但档案中已有相关条目，需确认后再删除或停用。",
                    resolution_action="apply_category_negation",
                    now=now,
                )
                return 0, review_count

        negations = [
            row
            for row in self._active_items_for_category(connection, user_id, item.category)
            if _is_negation_name(row["name"])
        ]
        review_count = 0
        if negations and not _is_category_negation_item(item):
            review_count += self._create_review_item(
                connection=connection,
                user_id=user_id,
                field_type="item",
                category=item.category,
                name=item.name,
                current_value={"items": [_item_row_to_dict(row) for row in negations]},
                proposed_value={"status": "add_item", "name": item.name, "value": item.value},
                source=item.source,
                evidence=item.evidence,
                reason=f"新条目与“无已知{CATEGORY_LABELS.get(item.category, item.category)}”冲突，已先保守写入并等待确认。",
                resolution_action="confirm_item_addition",
                now=now,
            )

        item_id = f"{user_id}:{item.category}:{normalized_name}"
        observed_at = item.observed_at.isoformat()
        value_json = json.dumps(item.value or {}, ensure_ascii=False)
        connection.execute(
            """
            INSERT INTO health_profile_items (
                item_id, user_id, category, name, normalized_name, status, value_json,
                source, evidence, observed_at, created_at, updated_at, confidence, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, category, normalized_name) DO UPDATE SET
                name = excluded.name,
                status = excluded.status,
                value_json = excluded.value_json,
                source = excluded.source,
                evidence = CASE
                    WHEN excluded.evidence = '' THEN health_profile_items.evidence
                    WHEN health_profile_items.evidence = '' THEN excluded.evidence
                    ELSE health_profile_items.evidence || ' | ' || excluded.evidence
                END,
                observed_at = excluded.observed_at,
                updated_at = excluded.updated_at,
                confidence = excluded.confidence,
                notes = excluded.notes
            """,
            (
                item_id,
                user_id,
                item.category,
                name,
                normalized_name,
                item.status,
                value_json,
                item.source,
                item.evidence,
                observed_at,
                now,
                now,
                max(0.0, min(1.0, float(item.confidence))),
                item.notes,
            ),
        )
        return 1, review_count

    def _active_items_for_category(
        self,
        connection: sqlite3.Connection,
        user_id: str,
        category: str,
    ) -> list[sqlite3.Row]:
        return connection.execute(
            """
            SELECT *
            FROM health_profile_items
            WHERE user_id = ? AND category = ? AND status = 'active'
            ORDER BY updated_at DESC, name
            """,
            (user_id, category),
        ).fetchall()

    def _create_review_item(
        self,
        *,
        connection: sqlite3.Connection,
        user_id: str,
        field_type: str,
        category: str,
        name: str,
        current_value: dict[str, Any],
        proposed_value: dict[str, Any],
        source: str,
        evidence: str,
        reason: str,
        resolution_action: str,
        now: str,
    ) -> int:
        proposed_json = json.dumps(proposed_value or {}, ensure_ascii=False, sort_keys=True)
        current_json = json.dumps(current_value or {}, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha1(
            f"{user_id}|{field_type}|{category}|{name}|{proposed_json}|pending".encode("utf-8")
        ).hexdigest()[:18]
        review_id = f"{user_id}:review:{digest}"
        cursor = connection.execute(
            """
            INSERT INTO health_profile_review_items (
                review_id, user_id, field_type, category, name, current_value_json,
                proposed_value_json, source, evidence, reason, resolution_action,
                status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
            ON CONFLICT(user_id, field_type, category, name, proposed_value_json, status)
            DO UPDATE SET
                current_value_json = excluded.current_value_json,
                source = excluded.source,
                evidence = excluded.evidence,
                reason = excluded.reason,
                resolution_action = excluded.resolution_action,
                updated_at = excluded.updated_at
            """,
            (
                review_id,
                user_id,
                field_type,
                category,
                name,
                current_json,
                proposed_json,
                source or "unknown",
                evidence or "",
                reason or "",
                resolution_action,
                now,
                now,
            ),
        )
        return 1 if cursor.rowcount else 0

    def _apply_review_resolution(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        now: str,
    ) -> None:
        proposed = json.loads(row["proposed_value_json"] or "{}")
        action = row["resolution_action"]
        user_id = row["user_id"]
        if action == "apply_scalar":
            field_name = row["name"]
            if field_name not in PROFILE_SCALAR_FIELDS:
                return
            value = _normalize_scalar_value(field_name, proposed.get("value"))
            if value in (None, ""):
                return
            connection.execute(
                f"""
                UPDATE health_profiles
                SET {field_name} = ?, updated_at = ?, source = ?
                WHERE user_id = ?
                """,
                (value, now, "confirmed_review", user_id),
            )
            self._refresh_bmi(connection, user_id, now)
            return

        if action == "apply_category_negation":
            category = row["category"]
            if category not in PROFILE_ITEM_CATEGORIES:
                return
            connection.execute(
                """
                UPDATE health_profile_items
                SET status = 'inactive', updated_at = ?, notes = ?
                WHERE user_id = ? AND category = ? AND status = 'active'
                """,
                (now, "由健康档案确认机制停用", user_id, category),
            )
            item = ProfileItem(
                category=category,
                name=row["name"],
                value={"negates_category": True},
                source="confirmed_review",
                evidence=row["evidence"],
                confidence=0.95,
            )
            self._upsert_item(connection, user_id, item, now)

    def _insert_glucose_readings(
        self,
        connection: sqlite3.Connection,
        user_id: str,
        readings: list[BloodGlucoseReading],
        now: str,
    ) -> int:
        count = 0
        for reading in readings:
            if reading.value_mmol_l <= 0:
                continue
            measured_at = reading.observed_at.isoformat()
            reported_at = reading.reported_at.isoformat()
            reading_id = _glucose_reading_id(user_id, reading)
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO blood_glucose_readings (
                    reading_id, user_id, measured_at, reported_at, value_mmol_l,
                    measurement_type, meal_context, time_context, source, evidence,
                    confidence, notes, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reading_id,
                    user_id,
                    measured_at,
                    reported_at,
                    round(float(reading.value_mmol_l), 1),
                    reading.measurement_type or "random",
                    reading.meal_context or "",
                    reading.time_context or "",
                    reading.source,
                    reading.evidence,
                    max(0.0, min(1.0, float(reading.confidence))),
                    reading.notes,
                    now,
                ),
            )
            count += int(cursor.rowcount or 0)
        return count

    def _fetch_glucose_rows(
        self,
        connection: sqlite3.Connection,
        user_id: str,
        limit: int = 30,
    ) -> list[sqlite3.Row]:
        return connection.execute(
            """
            SELECT *
            FROM blood_glucose_readings
            WHERE user_id = ?
            ORDER BY measured_at DESC, created_at DESC
            LIMIT ?
            """,
            (user_id, max(1, int(limit))),
        ).fetchall()

    def _refresh_bmi(self, connection: sqlite3.Connection, user_id: str, now: str) -> None:
        row = connection.execute(
            "SELECT height_cm, weight_kg FROM health_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is None or row["height_cm"] is None or row["weight_kg"] is None:
            return
        height_m = float(row["height_cm"]) / 100.0
        if height_m <= 0:
            return
        bmi = float(row["weight_kg"]) / (height_m * height_m)
        connection.execute(
            "UPDATE health_profiles SET bmi = ?, updated_at = ? WHERE user_id = ?",
            (round(bmi, 1), now, user_id),
        )


def extract_health_profile_update(text: str, *, source: str = "user_reported") -> ProfileUpdate:
    text = str(text or "").strip()
    update = ProfileUpdate()
    if not text:
        return update

    update.scalars.update(_extract_scalar_fields(text))
    update.scalar_evidence.update({key: text[:180] for key in update.scalars})
    update.items.extend(_extract_diseases(text, source))
    update.items.extend(_extract_medications(text, source))
    update.items.extend(_extract_allergies(text, source))
    update.items.extend(_extract_goals(text, source))
    update.items.extend(_extract_activity(text, source))
    update.items.extend(_extract_renal_metrics(text, source))
    update.items.extend(_extract_glucose_metrics(text, source))
    update.glucose_readings.extend(_extract_blood_glucose_readings(text, source))
    update.items.extend(_extract_dietary_restrictions(text, source))
    update.items = _dedupe_profile_items(update.items)
    update.glucose_readings = _dedupe_glucose_readings(update.glucose_readings)
    return update


def merge_profile_updates(updates: list[ProfileUpdate]) -> ProfileUpdate:
    merged = ProfileUpdate()
    seen_items: set[tuple[str, str]] = set()
    seen_readings: set[str] = set()
    for update in updates:
        merged.scalars.update(update.scalars)
        merged.scalar_evidence.update(update.scalar_evidence)
        for item in update.items:
            key = (item.category, normalize_name(item.name))
            if key in seen_items:
                continue
            seen_items.add(key)
            merged.items.append(item)
        for reading in update.glucose_readings:
            key = _glucose_reading_key(reading)
            if key in seen_readings:
                continue
            seen_readings.add(key)
            merged.glucose_readings.append(reading)
    return merged


def format_health_profile_context(profile: dict[str, Any]) -> str:
    scalars = profile.get("scalars") or {}
    items = profile.get("items") or []
    glucose_readings = profile.get("glucose_readings") or []
    glucose_analysis = profile.get("glucose_analysis") or analyze_blood_glucose_readings(glucose_readings)
    if not scalars and not items and not glucose_readings:
        return "【Health Profile｜结构化健康档案】\n- 暂无结构化健康档案。"

    lines = ["【Health Profile｜结构化健康档案】"]
    scalar_lines = []
    for field_name in [
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
    ]:
        value = scalars.get(field_name)
        if value in (None, ""):
            continue
        scalar_lines.append(f"{FIELD_LABELS.get(field_name, field_name)}={_format_scalar(field_name, value)}")
    if scalar_lines:
        lines.append("- 基本信息: " + "；".join(scalar_lines))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if item.get("status") != "active":
            continue
        grouped.setdefault(item["category"], []).append(item)

    for category in [
        "disease",
        "medication",
        "allergy",
        "goal",
        "renal_function",
        "glucose_metric",
        "exercise",
        "dietary_restriction",
    ]:
        category_items = grouped.get(category) or []
        if not category_items:
            continue
        values = [_format_item(item) for item in category_items[:8]]
        lines.append(f"- {CATEGORY_LABELS.get(category, category)}: " + "；".join(values))

    if glucose_readings:
        recent_values = []
        for reading in glucose_readings[:8]:
            label = GLUCOSE_TYPE_LABELS.get(
                reading.get("measurement_type") or "random",
                reading.get("measurement_type") or "随机",
            )
            measured_at = _format_glucose_time(reading.get("measured_at", ""))
            recent_values.append(
                f"{measured_at} {label}={float(reading.get('value_mmol_l', 0)):g}mmol/L"
            )
        lines.append("- 最近血糖记录: " + "；".join(recent_values))

    alerts = glucose_analysis.get("alerts") or []
    if alerts:
        alert_text = "；".join(str(item.get("message") or "") for item in alerts[:3] if item.get("message"))
        if alert_text:
            lines.append(f"- 血糖时间序列提醒: {alert_text}")
    summary = glucose_analysis.get("summary")
    if summary:
        lines.append(f"- 血糖趋势摘要: {summary}")

    lines.append("- 使用规则: 涉及营养建议、药食相互作用、过敏、肾病、电解质、血糖目标时，优先使用本档案；缺失字段需要向用户确认，不要编造。")
    return "\n".join(lines)


def _extract_scalar_fields(text: str) -> dict[str, Any]:
    scalars: dict[str, Any] = {}

    age_match = re.search(r"(?:我)?(?:今年|年龄)?(?:是|为|大概|大约|约)?\s*(\d{1,3})\s*岁", text)
    if age_match:
        age = float(age_match.group(1))
        if 0 < age < 130:
            scalars["age_years"] = age

    if re.search(r"(?:我是|性别)?\s*(男|男性|男士|先生)", text):
        scalars["sex"] = "male"
    elif re.search(r"(?:我是|性别)?\s*(女|女性|女士)", text):
        scalars["sex"] = "female"

    height_match = re.search(
        r"(?:身高)(?:是|为|大概|大约|约)?\s*(\d+(?:\.\d+)?)\s*(cm|厘米|米|m)?",
        text,
        flags=re.IGNORECASE,
    )
    if height_match:
        value = float(height_match.group(1))
        unit = (height_match.group(2) or "cm").lower()
        if unit in {"米", "m"} and value < 3:
            value *= 100
        if 80 <= value <= 230:
            scalars["height_cm"] = round(value, 1)

    weight_match = re.search(
        r"(?:体重)(?:是|为|有|大概|大约|约|差不多)?\s*(\d+(?:\.\d+)?)\s*(kg|公斤|千克|斤)?",
        text,
        flags=re.IGNORECASE,
    )
    if not weight_match:
        weight_match = re.search(
            r"(?:我(?:现在)?|本人)?\s*(?:重|体重)?(?:是|为|有)?\s*(\d+(?:\.\d+)?)\s*(kg|公斤|千克|斤)\b",
            text,
            flags=re.IGNORECASE,
        )
    if weight_match:
        value = float(weight_match.group(1))
        unit = (weight_match.group(2) or "kg").lower()
        if unit == "斤":
            value *= 0.5
        if 20 <= value <= 300:
            scalars["weight_kg"] = round(value, 1)

    energy_match = re.search(r"(?:每日|每天|一天).{0,8}?(\d{3,4})\s*(?:千卡|大卡|kcal)", text, flags=re.IGNORECASE)
    if energy_match:
        scalars["target_energy_kcal"] = float(energy_match.group(1))

    carb_match = re.search(r"(?:每餐|一餐).{0,8}?碳水.{0,6}?(\d{1,3})\s*(?:克|g)", text, flags=re.IGNORECASE)
    if carb_match:
        scalars["target_carbohydrate_g_per_meal"] = float(carb_match.group(1))

    if "久坐" in text:
        scalars["activity_level"] = "sedentary"
    elif any(token in text for token in ["轻体力", "轻度活动", "轻量运动"]):
        scalars["activity_level"] = "light"
    elif any(token in text for token in ["中等强度", "中等活动", "规律运动"]):
        scalars["activity_level"] = "moderate"
    elif any(token in text for token in ["重体力", "高强度运动"]):
        scalars["activity_level"] = "high"

    goals = _goal_names(text)
    if goals:
        scalars["nutrition_goal"] = "、".join(goals[:4])

    return scalars


def _extract_diseases(text: str, source: str) -> list[ProfileItem]:
    items: list[ProfileItem] = []
    lowered = text.lower()
    has_specific_diabetes = any(
        alias in lowered
        for alias in ["2型糖尿病", "二型糖尿病", "1型糖尿病", "一型糖尿病", "妊娠期糖尿病", "妊娠糖尿病"]
    )
    for canonical, aliases in DISEASE_ALIASES:
        if canonical == "糖尿病" and has_specific_diabetes:
            continue
        if any(alias.lower() in lowered for alias in aliases):
            items.append(
                ProfileItem(
                    category="disease",
                    name=canonical,
                    source=source,
                    evidence=_clip_evidence(text, aliases),
                    confidence=0.86,
                )
            )
    return items


def _extract_medications(text: str, source: str) -> list[ProfileItem]:
    items: list[ProfileItem] = []
    for medication in MEDICATION_ALIASES:
        if medication.lower() in text.lower():
            items.append(
                ProfileItem(
                    category="medication",
                    name=medication,
                    source=source,
                    evidence=_clip_evidence(text, [medication]),
                    confidence=0.84,
                )
            )

    for match in re.findall(r"(?:正在吃|在吃|服用|用药|吃药是|吃的是)([^，,。；;\n]{2,40})", text):
        for name in _split_list_text(match):
            if len(name) < 2:
                continue
            items.append(
                ProfileItem(
                    category="medication",
                    name=name,
                    source=source,
                    evidence=match.strip(),
                    confidence=0.72,
                )
            )
    return items


def _extract_allergies(text: str, source: str) -> list[ProfileItem]:
    items: list[ProfileItem] = []
    if re.search(r"(?:我)?(?:没有|无|没)(?:任何|已知)?(?:食物|药物)?过敏|不过敏", text):
        items.append(
            ProfileItem(
                category="allergy",
                name="无已知过敏",
                value={"negates_category": True},
                source=source,
                evidence=_clip_evidence(text, ["过敏"]),
                confidence=0.88,
            )
        )
        return items
    for allergy in re.findall(r"(?:对|有)?([^，。；,;\n]{1,12})过敏", text):
        name = allergy.strip("我会也对有 ")
        if name:
            items.append(
                ProfileItem(
                    category="allergy",
                    name=name,
                    source=source,
                    evidence=f"{allergy}过敏",
                    confidence=0.9,
                )
            )
    for allergy in ALLERGY_ALIASES:
        if allergy in text and "过敏" in text:
            items.append(
                ProfileItem(
                    category="allergy",
                    name=allergy,
                    source=source,
                    evidence=_clip_evidence(text, [allergy]),
                    confidence=0.82,
                )
            )
    return items


def _extract_goals(text: str, source: str) -> list[ProfileItem]:
    return [
        ProfileItem(category="goal", name=name, source=source, evidence=_clip_evidence(text, [name]), confidence=0.78)
        for name in _goal_names(text)
    ]


def _goal_names(text: str) -> list[str]:
    mapping = [
        ("减重", ["减重", "减肥", "瘦下来", "降低体重"]),
        ("控糖", ["控糖", "控制血糖", "血糖稳定"]),
        ("增肌", ["增肌", "增加肌肉"]),
        ("降尿酸", ["降尿酸", "控制尿酸"]),
        ("低盐饮食", ["低盐", "控盐", "少盐"]),
        ("控制血压", ["控制血压", "降压"]),
        ("降低胆固醇", ["降胆固醇", "控制血脂", "降血脂"]),
    ]
    goals = []
    for canonical, aliases in mapping:
        if any(alias in text for alias in aliases):
            goals.append(canonical)
    return goals


def _extract_activity(text: str, source: str) -> list[ProfileItem]:
    items: list[ProfileItem] = []
    match = re.search(r"(?:每周|一周)(\d)\s*(?:次|回).{0,12}?(运动|跑步|快走|健身|游泳)", text)
    if match:
        name = f"每周运动{match.group(1)}次"
        items.append(
            ProfileItem(
                category="exercise",
                name=name,
                value={"frequency_per_week": int(match.group(1))},
                source=source,
                evidence=match.group(0),
                confidence=0.82,
            )
        )
    if "久坐" in text:
        items.append(ProfileItem(category="exercise", name="久坐", source=source, evidence="久坐", confidence=0.82))
    return items


def _extract_renal_metrics(text: str, source: str) -> list[ProfileItem]:
    items: list[ProfileItem] = []
    egfr_match = re.search(r"(?:eGFR|egfr|肾小球滤过率)\s*(?:是|=|约|大概)?\s*(\d+(?:\.\d+)?)", text)
    if egfr_match:
        value = float(egfr_match.group(1))
        items.append(
            ProfileItem(
                category="renal_function",
                name="eGFR",
                value={"value": value, "unit": "mL/min/1.73m2"},
                source=source,
                evidence=egfr_match.group(0),
                confidence=0.9,
            )
        )

    creatinine_match = re.search(r"(?:肌酐|Scr|scr)\s*(?:是|=|约|大概)?\s*(\d+(?:\.\d+)?)\s*(umol/L|μmol/L|mg/dL)?", text)
    if creatinine_match:
        unit = creatinine_match.group(2) or ""
        items.append(
            ProfileItem(
                category="renal_function",
                name="肌酐",
                value={"value": float(creatinine_match.group(1)), "unit": unit},
                source=source,
                evidence=creatinine_match.group(0),
                confidence=0.88,
            )
        )
    return items


def _extract_glucose_metrics(text: str, source: str) -> list[ProfileItem]:
    items: list[ProfileItem] = []
    patterns = [
        ("空腹血糖", r"空腹血糖\s*(?:是|=|约|大概)?\s*(\d+(?:\.\d+)?)\s*(mmol/L|毫摩尔)?"),
        ("餐后2小时血糖", r"(?:餐后2小时血糖|餐后两小时血糖|餐二血糖)\s*(?:是|=|约|大概)?\s*(\d+(?:\.\d+)?)\s*(mmol/L|毫摩尔)?"),
        ("糖化血红蛋白", r"(?:糖化血红蛋白|HbA1c|hba1c)\s*(?:是|=|约|大概)?\s*(\d+(?:\.\d+)?)\s*%?"),
    ]
    for name, pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        unit = "%" if name == "糖化血红蛋白" else (match.group(2) or "mmol/L")
        items.append(
            ProfileItem(
                category="glucose_metric",
                name=name,
                value={"value": float(match.group(1)), "unit": unit},
                source=source,
                evidence=match.group(0),
                confidence=0.9,
            )
        )
    return items


def _extract_blood_glucose_readings(text: str, source: str) -> list[BloodGlucoseReading]:
    if "血糖" not in text and not any(token in text for token in ["餐二", "空腹", "饭后", "餐后"]):
        return []
    readings: list[BloodGlucoseReading] = []
    patterns = [
        (
            r"(?P<label>空腹血糖|餐前血糖|饭前血糖|早餐前血糖|午餐前血糖|晚餐前血糖|"
            r"餐后2小时血糖|餐后两小时血糖|饭后2小时血糖|饭后两小时血糖|餐二血糖|"
            r"早餐后血糖|午餐后血糖|晚餐后血糖|餐后血糖|饭后血糖|睡前血糖|随机血糖|血糖)"
            r"\s*(?:是|为|=|约|大概|大约|刚测|测得|测出来|测了)?\s*"
            + GLUCOSE_VALUE_PATTERN
            + r"\s*(?P<unit>mmol/L|mmol\/l|毫摩尔|mg/dL|mg\/dl|毫克每分升)?"
        ),
        (
            r"(?P<label>空腹|餐前|饭前|早餐前|午餐前|晚餐前|"
            r"餐后2小时|餐后两小时|饭后2小时|饭后两小时|餐二|早餐后|午餐后|晚餐后|餐后|饭后|睡前)"
            r"[^，。；;]{0,10}血糖[^0-9零〇一二三四五六七八九十两]{0,8}"
            + GLUCOSE_VALUE_PATTERN
            + r"\s*(?P<unit>mmol/L|mmol\/l|毫摩尔|mg/dL|mg\/dl|毫克每分升)?"
        ),
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = _normalize_glucose_value(match.group("value"), match.group("unit"))
            if value is None:
                continue
            label = str(match.group("label") or "")
            measurement_type = _classify_glucose_measurement(label, text, match.start(), match.end())
            observed_at, time_context = _parse_glucose_observed_at(
                text,
                match.start(),
                match.start("value"),
            )
            readings.append(
                BloodGlucoseReading(
                    value_mmol_l=value,
                    measurement_type=measurement_type,
                    meal_context=_classify_glucose_meal_context(label, text, match.start(), match.end()),
                    time_context=time_context,
                    source=source,
                    evidence=match.group(0).strip(),
                    observed_at=observed_at,
                    confidence=0.92 if label != "血糖" else 0.82,
                )
            )
    return _dedupe_glucose_readings(readings)


def _extract_dietary_restrictions(text: str, source: str) -> list[ProfileItem]:
    restrictions = []
    mapping = [
        ("低盐", ["低盐", "少盐", "限盐"]),
        ("低嘌呤", ["低嘌呤", "少吃内脏", "控制嘌呤"]),
        ("控碳水", ["控碳水", "控制碳水", "少吃主食"]),
        ("低钾", ["低钾", "限钾"]),
        ("低磷", ["低磷", "限磷"]),
    ]
    for name, aliases in mapping:
        if any(alias in text for alias in aliases):
            restrictions.append(
                ProfileItem(
                    category="dietary_restriction",
                    name=name,
                    source=source,
                    evidence=_clip_evidence(text, aliases),
                    confidence=0.78,
                )
            )
    return restrictions


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", "", str(name or "").strip().lower())


def _dedupe_profile_items(items: list[ProfileItem]) -> list[ProfileItem]:
    deduped: list[ProfileItem] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (item.category, normalize_name(item.name))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _dedupe_glucose_readings(readings: list[BloodGlucoseReading]) -> list[BloodGlucoseReading]:
    deduped: list[BloodGlucoseReading] = []
    seen: set[str] = set()
    for reading in readings:
        key = _glucose_reading_key(reading)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(reading)
    return deduped


def _is_scalar_conflict(field_name: str, current_value: Any, proposed_value: Any) -> bool:
    if field_name not in CONFLICT_REVIEW_SCALAR_FIELDS:
        return False
    if current_value in (None, "") or proposed_value in (None, ""):
        return False
    if field_name == "sex":
        return str(current_value) != str(proposed_value)
    try:
        current_number = float(current_value)
        proposed_number = float(proposed_value)
    except (TypeError, ValueError):
        return str(current_value).strip() != str(proposed_value).strip()
    diff = abs(current_number - proposed_number)
    if field_name == "age_years":
        return diff > 1
    if field_name == "height_cm":
        return diff >= 2
    if field_name == "weight_kg":
        return diff >= 3
    if current_number == 0:
        return diff > 0
    return diff / abs(current_number) >= 0.2


def _scalar_conflict_reason(field_name: str, current_value: Any, proposed_value: Any) -> str:
    label = FIELD_LABELS.get(field_name, field_name)
    return f"{label}新值与现有档案差异较大：当前={current_value}，新上报={proposed_value}。为避免 ASR 或抽取误写，已进入待确认队列。"


def _is_category_negation_item(item: ProfileItem) -> bool:
    return bool((item.value or {}).get("negates_category")) or _is_negation_name(item.name)


def _is_negation_name(name: str) -> bool:
    normalized = normalize_name(name)
    return normalized in {"无已知过敏", "无过敏", "没有过敏", "无已知疾病", "无疾病", "未用药", "无用药"}


def _glucose_reading_key(reading: BloodGlucoseReading) -> str:
    return "|".join(
        [
            reading.observed_at.isoformat(timespec="minutes"),
            reading.measurement_type,
            f"{float(reading.value_mmol_l):.1f}",
        ]
    )


def _glucose_reading_id(user_id: str, reading: BloodGlucoseReading) -> str:
    digest = hashlib.sha1(f"{user_id}|{_glucose_reading_key(reading)}".encode("utf-8")).hexdigest()[:18]
    return f"{user_id}:glucose:{digest}"


def _normalize_intake_time(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_local_tz())
    else:
        parsed = parsed.astimezone(_local_tz())
    return parsed.isoformat(timespec="seconds")


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 3)


def _normalize_glucose_value(raw_value: str, raw_unit: str | None) -> float | None:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = _chinese_number_to_float(str(raw_value or ""))
        if value is None:
            return None
    unit = str(raw_unit or "").strip().lower()
    if unit in {"mg/dl", "毫克每分升"}:
        value = value / 18.0
    if not 1.0 <= value <= 35.0:
        return None
    return round(value, 1)


def _chinese_number_to_float(text: str) -> float | None:
    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if "点" in cleaned:
        integer_text, decimal_text = cleaned.split("点", 1)
        integer_value = _chinese_integer_to_int(integer_text, digits)
        if integer_value is None:
            return None
        decimal_digits = []
        for char in decimal_text:
            if char not in digits:
                return None
            decimal_digits.append(str(digits[char]))
        if not decimal_digits:
            return float(integer_value)
        return float(f"{integer_value}.{''.join(decimal_digits)}")
    integer_value = _chinese_integer_to_int(cleaned, digits)
    return float(integer_value) if integer_value is not None else None


def _chinese_integer_to_int(text: str, digits: dict[str, int]) -> int | None:
    cleaned = str(text or "").strip()
    if not cleaned:
        return 0
    if cleaned == "十":
        return 10
    if "十" in cleaned:
        left, right = cleaned.split("十", 1)
        if left == "":
            tens = 1
        elif left in digits:
            tens = digits[left]
        else:
            return None
        if right == "":
            ones = 0
        elif right in digits:
            ones = digits[right]
        else:
            return None
        return tens * 10 + ones
    value = 0
    for char in cleaned:
        if char not in digits:
            return None
        value = value * 10 + digits[char]
    return value


def _classify_glucose_measurement(label: str, text: str, start: int, end: int) -> str:
    window = f"{_clause_window(text, start, end)} {label}"
    if "空腹" in window:
        return "fasting"
    if any(token in window for token in ["餐前", "饭前", "早餐前", "午餐前", "晚餐前"]):
        return "pre_meal"
    if any(token in window for token in ["餐后2小时", "餐后两小时", "饭后2小时", "饭后两小时", "餐二"]):
        return "postprandial_2h"
    if any(token in window for token in ["餐后", "饭后", "早餐后", "午餐后", "晚餐后"]):
        return "post_meal"
    if "睡前" in window:
        return "bedtime"
    return "random"


def _classify_glucose_meal_context(label: str, text: str, start: int, end: int) -> str:
    window = f"{_clause_window(text, start, end)} {label}"
    if "早餐" in window:
        return "breakfast"
    if "午餐" in window or "中饭" in window:
        return "lunch"
    if "晚餐" in window or "晚饭" in window:
        return "dinner"
    return ""


def _parse_glucose_observed_at(text: str, match_start: int, value_start: int) -> tuple[datetime, str]:
    now = datetime.now(_local_tz())
    boundary = max(
        text.rfind("，", 0, match_start),
        text.rfind(",", 0, match_start),
        text.rfind("。", 0, match_start),
        text.rfind("；", 0, match_start),
        text.rfind(";", 0, match_start),
    )
    prefix = text[max(boundary + 1, match_start - 32):value_start]
    day_delta = 0
    if "前天" in prefix:
        day_delta = -2
    elif "昨天" in prefix or "昨日" in prefix:
        day_delta = -1

    time_matches = list(
        re.finditer(
            r"(?:(凌晨|早上|上午|中午|下午|晚上|今晚|今早)\s*)?(\d{1,2})(?:[:：点](\d{1,2})?)",
            prefix,
        )
    )
    if not time_matches:
        date_only = "今天" if "今天" in prefix else "昨天" if day_delta == -1 else "前天" if day_delta == -2 else ""
        return now, date_only

    match = time_matches[-1]
    period = match.group(1) or ""
    hour = int(match.group(2))
    minute = int(match.group(3) or 0)
    if period in {"下午", "晚上", "今晚"} and hour < 12:
        hour += 12
    if period == "中午" and hour < 11:
        hour += 12
    if hour > 23 or minute > 59:
        return now, match.group(0)
    observed = (now + timedelta(days=day_delta)).replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
    )
    return observed, match.group(0)


def _clause_window(text: str, start: int, end: int) -> str:
    left_candidates = [
        text.rfind(mark, 0, start)
        for mark in ["，", ",", "。", "；", ";", "\n"]
    ]
    right_candidates = [
        position
        for position in [text.find(mark, end) for mark in ["，", ",", "。", "；", ";", "\n"]]
        if position >= 0
    ]
    left = max(left_candidates) + 1
    right = min(right_candidates) if right_candidates else len(text)
    return text[left:right]


def _local_tz() -> ZoneInfo:
    name = os.getenv("CLINICAL_LOCAL_TIMEZONE", "Asia/Hong_Kong")
    try:
        return ZoneInfo(name)
    except Exception:
        return ZoneInfo("UTC")


def _normalize_scalar_value(field_name: str, value: Any) -> Any:
    if field_name in {
        "age_years",
        "height_cm",
        "weight_kg",
        "target_energy_kcal",
        "target_carbohydrate_g_per_meal",
        "target_protein_g_per_day",
        "target_fat_g_per_day",
    }:
        try:
            return round(float(value), 1)
        except (TypeError, ValueError):
            return None
    if field_name == "sex":
        text = str(value).strip().lower()
        if text in {"男", "男性", "male", "m"}:
            return "male"
        if text in {"女", "女性", "female", "f"}:
            return "female"
        return text
    return str(value).strip()


def _profile_row_to_scalars(row: sqlite3.Row) -> dict[str, Any]:
    scalars: dict[str, Any] = {}
    for key in [
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
    ]:
        if row[key] is not None and row[key] != "":
            scalars[key] = row[key]
    return scalars


def _item_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "item_id": row["item_id"],
        "category": row["category"],
        "name": row["name"],
        "status": row["status"],
        "value": json.loads(row["value_json"] or "{}"),
        "source": row["source"],
        "evidence": row["evidence"],
        "observed_at": row["observed_at"],
        "updated_at": row["updated_at"],
        "confidence": row["confidence"],
        "notes": row["notes"],
    }


def _glucose_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "reading_id": row["reading_id"],
        "user_id": row["user_id"],
        "measured_at": row["measured_at"],
        "reported_at": row["reported_at"],
        "value_mmol_l": row["value_mmol_l"],
        "measurement_type": row["measurement_type"],
        "measurement_label": GLUCOSE_TYPE_LABELS.get(row["measurement_type"], row["measurement_type"]),
        "meal_context": row["meal_context"],
        "time_context": row["time_context"],
        "source": row["source"],
        "evidence": row["evidence"],
        "confidence": row["confidence"],
        "notes": row["notes"],
        "created_at": row["created_at"],
    }


def _review_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "review_id": row["review_id"],
        "user_id": row["user_id"],
        "field_type": row["field_type"],
        "category": row["category"],
        "name": row["name"],
        "current_value": json.loads(row["current_value_json"] or "{}"),
        "proposed_value": json.loads(row["proposed_value_json"] or "{}"),
        "source": row["source"],
        "evidence": row["evidence"],
        "reason": row["reason"],
        "resolution_action": row["resolution_action"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "resolved_at": row["resolved_at"],
        "resolved_by": row["resolved_by"],
    }


def analyze_blood_glucose_readings(readings: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = [_normalize_reading_dict(item) for item in readings]
    normalized = [item for item in normalized if item]
    normalized.sort(key=lambda item: item["measured_at"] or "", reverse=True)
    if not normalized:
        return {
            "count": 0,
            "latest": None,
            "summary": "",
            "alerts": [],
            "by_type": {},
        }

    latest = normalized[0]
    alerts: list[dict[str, Any]] = []
    for index, item in enumerate(normalized[:8]):
        alert = _glucose_alert_for_reading(item, latest_only=index == 0)
        if alert and (index == 0 or alert.get("severity") == "urgent"):
            alerts.append(alert)

    by_type: dict[str, dict[str, Any]] = {}
    for item in normalized:
        bucket = by_type.setdefault(
            item["measurement_type"],
            {
                "label": GLUCOSE_TYPE_LABELS.get(item["measurement_type"], item["measurement_type"]),
                "count": 0,
                "min": item["value_mmol_l"],
                "max": item["value_mmol_l"],
                "sum": 0.0,
                "latest": item,
            },
        )
        bucket["count"] += 1
        bucket["sum"] += item["value_mmol_l"]
        bucket["min"] = min(bucket["min"], item["value_mmol_l"])
        bucket["max"] = max(bucket["max"], item["value_mmol_l"])
        if item["measured_at"] > bucket["latest"]["measured_at"]:
            bucket["latest"] = item

    for measurement_type, bucket in by_type.items():
        bucket["average"] = round(bucket["sum"] / max(bucket["count"], 1), 1)
        del bucket["sum"]
        recent_same_type = [item for item in normalized if item["measurement_type"] == measurement_type][:3]
        target = GLUCOSE_TARGETS_MMOL_L.get(measurement_type)
        if target and len(recent_same_type) >= 3 and all(item["value_mmol_l"] > target[1] for item in recent_same_type):
            alerts.append(
                {
                    "severity": "warn",
                    "code": "repeated_high",
                    "message": f"最近3次{GLUCOSE_TYPE_LABELS.get(measurement_type, measurement_type)}血糖都高于{target[1]:g}mmol/L，提示这一时段可能持续偏高。",
                    "recommendation": "建议结合饮食、运动、用药时间记录原因；如果持续多天偏高，应把记录给医生或营养师评估。",
                }
            )

    values = [item["value_mmol_l"] for item in normalized]
    summary_parts = [
        f"最近{len(normalized)}条记录",
        f"范围{min(values):g}-{max(values):g}mmol/L",
    ]
    if len(values) >= 3:
        summary_parts.append(f"均值{round(sum(values) / len(values), 1):g}mmol/L")

    unique_alerts = []
    seen_codes = set()
    for alert in alerts:
        key = (alert.get("code"), alert.get("message"))
        if key in seen_codes:
            continue
        seen_codes.add(key)
        unique_alerts.append(alert)

    return {
        "count": len(normalized),
        "latest": latest,
        "summary": "，".join(summary_parts),
        "alerts": unique_alerts[:5],
        "by_type": by_type,
    }


def _normalize_reading_dict(item: dict[str, Any]) -> dict[str, Any] | None:
    try:
        value = round(float(item.get("value_mmol_l")), 1)
    except (TypeError, ValueError):
        return None
    if not 1.0 <= value <= 35.0:
        return None
    return {
        **item,
        "value_mmol_l": value,
        "measurement_type": item.get("measurement_type") or "random",
        "measured_at": str(item.get("measured_at") or ""),
    }


def _glucose_alert_for_reading(
    reading: dict[str, Any],
    *,
    latest_only: bool = False,
) -> dict[str, Any] | None:
    value = float(reading["value_mmol_l"])
    measurement_type = reading.get("measurement_type") or "random"
    label = GLUCOSE_TYPE_LABELS.get(measurement_type, "随机")
    prefix = f"最新{label}" if latest_only else f"{_format_glucose_time(reading.get('measured_at', ''))} {label}".strip()
    if value < 3.0:
        return {
            "severity": "urgent",
            "code": "severe_low",
            "message": f"{prefix}血糖{value:g}mmol/L，属于明显低血糖风险。",
            "recommendation": "如果伴随出汗、心慌、手抖、意识不清等症状，应立即按医生教过的低血糖处理方式处理并寻求医疗帮助。",
        }
    if value <= 3.9:
        return {
            "severity": "urgent",
            "code": "low",
            "message": f"{prefix}血糖{value:g}mmol/L，低于常用低血糖提醒阈值3.9mmol/L。",
            "recommendation": "建议先确认测量是否准确；如有低血糖症状，及时处理并联系医生确认后续用药和饮食安排。",
        }
    if value >= 16.7:
        return {
            "severity": "urgent",
            "code": "very_high",
            "message": f"{prefix}血糖{value:g}mmol/L，数值明显偏高。",
            "recommendation": "建议复测确认，并观察是否有口渴、多尿、乏力、恶心等不适；若持续很高或有症状，应及时就医。",
        }
    target = GLUCOSE_TARGETS_MMOL_L.get(measurement_type)
    if target and value > target[1]:
        return {
            "severity": "warn",
            "code": "above_target",
            "message": f"{prefix}血糖{value:g}mmol/L，高于当前内置提醒目标{target[1]:g}mmol/L。",
            "recommendation": "建议回看上一餐碳水量、含糖饮料、运动和用药时间；目标范围需要按医生给你的个人目标调整。",
        }
    if latest_only and measurement_type == "random" and value >= 13.9:
        return {
            "severity": "warn",
            "code": "random_high",
            "message": f"最新随机血糖{value:g}mmol/L偏高。",
            "recommendation": "建议补充这是餐前、餐后几小时或睡前测的，方便判断是否超过你的目标。",
        }
    return None


def _format_scalar(field_name: str, value: Any) -> str:
    if field_name == "sex":
        return {"male": "男", "female": "女"}.get(str(value), str(value))
    if field_name == "height_cm":
        return f"{float(value):g}cm"
    if field_name == "weight_kg":
        return f"{float(value):g}kg"
    if field_name == "bmi":
        return f"{float(value):g}"
    if field_name == "target_energy_kcal":
        return f"{float(value):g}kcal"
    if field_name in {"target_carbohydrate_g_per_meal", "target_protein_g_per_day", "target_fat_g_per_day"}:
        return f"{float(value):g}g"
    if field_name == "age_years":
        return f"{float(value):g}岁"
    return str(value)


def _format_item(item: dict[str, Any]) -> str:
    value = item.get("value") or {}
    if "value" in value:
        unit = value.get("unit") or ""
        return f"{item['name']}={value['value']}{unit}"
    return str(item["name"])


def _format_glucose_time(value: str) -> str:
    text = str(value or "")
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return text[:16]
    return parsed.strftime("%m-%d %H:%M")


def _clip_evidence(text: str, aliases: list[str]) -> str:
    for alias in aliases:
        index = text.lower().find(alias.lower())
        if index >= 0:
            start = max(index - 20, 0)
            end = min(index + len(alias) + 30, len(text))
            return text[start:end].strip()
    return text[:120].strip()


def _split_list_text(text: str) -> list[str]:
    cleaned = re.sub(r"(这些药|这个药|药物|药|每天|现在|目前|长期|一直|还有|以及)", "、", text)
    parts = re.split(r"[、,，和+加及/ ]+", cleaned)
    return [part.strip(" 。；;，,") for part in parts if part.strip(" 。；;，,")]
