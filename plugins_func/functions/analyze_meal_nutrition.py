from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config.config_loader import get_project_dir
from config.logger import setup_logging
from core.clinical_nutrition.nutrition_targets import estimate_daily_nutrition_targets
from core.providers.memory.clinical_ltm.health_profile import HealthProfileStore
from core.utils.device_identity import normalize_device_user_id
from plugins_func.register import Action, ActionResponse, ToolType, register_function

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

NUTRIENT_COLUMNS = [
    ("energy_kcal", "热量", "kcal"),
    ("carbohydrate_g", "碳水", "g"),
    ("protein_g", "蛋白质", "g"),
    ("fat_g", "脂肪", "g"),
    ("dietary_fiber_g", "膳食纤维", "g"),
    ("sodium_mg", "钠", "mg"),
    ("potassium_mg", "钾", "mg"),
    ("phosphorus_mg", "磷", "mg"),
]

TARGET_COLUMNS = {
    "energy_kcal": "energy_kcal_target",
    "carbohydrate_g": "carbohydrate_g_target",
    "protein_g": "protein_g_target",
    "fat_g": "fat_g_target",
    "sodium_mg": "sodium_mg_target",
}

DAILY_TARGET_KEYS = {
    "energy_kcal": "energy_kcal",
    "carbohydrate_g": "carbohydrate_g_per_day",
    "protein_g": "protein_g_per_day",
    "fat_g": "fat_g_per_day",
}

CHINESE_DIGITS = {
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

QUANTITY_PATTERN = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百半]+)"
UNIT_PATTERN = (
    r"(?:毫升|ml|mL|ML|公斤|千克|kg|KG|克|g|G|斤|两|汤匙|茶匙|大勺|小勺|"
    r"勺子|勺|杯|碗|片|个|只|枚|颗|块|份|根|条|包|袋|盒|瓶)"
)

GRAM_UNITS = {
    "克": 1.0,
    "g": 1.0,
    "G": 1.0,
    "公斤": 1000.0,
    "千克": 1000.0,
    "kg": 1000.0,
    "KG": 1000.0,
    "斤": 500.0,
    "两": 50.0,
}

MILLILITER_UNITS = {"毫升", "ml", "mL", "ML"}

UNIT_ALIASES = {
    "杯": ["cup"],
    "碗": ["bowl", "cup"],
    "片": ["slice"],
    "个": ["piece", "unit", "egg", "whole", "medium", "large"],
    "只": ["piece", "unit", "egg", "whole", "medium", "large"],
    "枚": ["piece", "unit", "egg", "whole", "medium", "large"],
    "颗": ["piece", "unit", "whole", "medium", "large"],
    "块": ["piece", "block", "cube"],
    "份": ["serving", "portion"],
    "根": ["piece", "stick", "medium"],
    "条": ["piece", "stick"],
    "包": ["package", "packet"],
    "袋": ["package", "packet"],
    "盒": ["container", "carton"],
    "瓶": ["bottle"],
    "汤匙": ["tablespoon", "tbsp"],
    "大勺": ["tablespoon", "tbsp"],
    "茶匙": ["teaspoon", "tsp"],
    "小勺": ["teaspoon", "tsp"],
    "勺": ["tablespoon", "teaspoon", "tbsp", "tsp"],
    "勺子": ["tablespoon", "teaspoon", "tbsp", "tsp"],
}

GENERIC_PORTION_GRAMS = {
    "杯": 240.0,
    "碗": 150.0,
    "片": 30.0,
    "个": 50.0,
    "只": 50.0,
    "枚": 50.0,
    "颗": 50.0,
    "块": 100.0,
    "份": 100.0,
    "根": 100.0,
    "条": 100.0,
    "包": 100.0,
    "袋": 100.0,
    "盒": 250.0,
    "瓶": 500.0,
    "汤匙": 15.0,
    "大勺": 15.0,
    "勺": 10.0,
    "勺子": 10.0,
    "茶匙": 5.0,
    "小勺": 5.0,
}

SPECIFIC_PORTION_RULES = [
    (["鸡蛋", "鸭蛋", "egg"], {"个": 50.0, "只": 50.0, "枚": 50.0, "颗": 50.0}),
    (["牛奶", "牛乳", "豆浆", "豆奶", "milk", "soy milk"], {"杯": 250.0, "盒": 250.0, "瓶": 250.0}),
    (["酸奶", "yogurt"], {"杯": 180.0, "盒": 200.0, "瓶": 200.0}),
    (["面包", "吐司", "bread"], {"片": 30.0, "个": 60.0}),
    (["米饭", "饭"], {"碗": 150.0, "杯": 150.0, "份": 150.0}),
    (["粥"], {"碗": 250.0, "杯": 250.0, "份": 250.0}),
    (["馒头"], {"个": 80.0, "只": 80.0}),
    (["包子"], {"个": 100.0, "只": 100.0}),
    (["油条"], {"根": 70.0, "条": 70.0}),
    (["面条", "面"], {"碗": 250.0, "份": 250.0}),
    (["豆腐"], {"块": 100.0, "份": 100.0}),
    (["青菜", "菠菜", "西兰花", "蔬菜"], {"份": 100.0, "碗": 100.0}),
    (["瘦肉", "猪肉", "牛肉", "鸡肉", "肉"], {"份": 75.0, "块": 75.0}),
    (["苹果", "橙子", "香蕉", "梨"], {"个": 150.0, "只": 150.0, "根": 115.0}),
]

ANALYZE_MEAL_NUTRITION_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "analyze_meal_nutrition",
        "description": (
            "按份量估算一餐的总营养，适合处理“两个鸡蛋、一杯牛奶、两片面包”这类中文口语输入。"
            "会查询本地结构化营养库，换算每种食物克重，并汇总热量、碳水、蛋白质、脂肪、钠等。"
            "如果用户给出本餐目标，也会判断是否超出目标。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "meal_text": {
                    "type": "string",
                    "description": "一餐的自然语言描述，例如：两个鸡蛋、一杯牛奶、两片白面包。",
                },
                "energy_kcal_target": {
                    "type": "number",
                    "description": "可选，本餐热量目标，单位 kcal。",
                },
                "carbohydrate_g_target": {
                    "type": "number",
                    "description": "可选，本餐碳水目标，单位 g。",
                },
                "protein_g_target": {
                    "type": "number",
                    "description": "可选，本餐蛋白质目标，单位 g。",
                },
                "fat_g_target": {
                    "type": "number",
                    "description": "可选，本餐脂肪目标，单位 g。",
                },
                "sodium_mg_target": {
                    "type": "number",
                    "description": "可选，本餐钠目标，单位 mg。",
                },
                "record_intake": {
                    "type": "boolean",
                    "description": "可选，是否把本次分析写入用户每日摄入曲线。实际已经吃/正在吃的一餐填 true；只是计划、假设或单纯问营养值填 false。",
                },
                "occurred_at": {
                    "type": "string",
                    "description": "可选，实际进食时间，ISO 8601 格式；不填则使用当前时间。",
                },
            },
            "required": ["meal_text"],
        },
    },
}


@dataclass
class ParsedMealItem:
    raw_text: str
    food_name: str
    quantity: float
    unit: str | None
    explicit_grams: float | None = None


@dataclass
class ResolvedMealItem:
    parsed: ParsedMealItem
    food: sqlite3.Row
    grams: float
    portion_source: str
    nutrient_totals: dict[str, float]


@register_function(
    "analyze_meal_nutrition", ANALYZE_MEAL_NUTRITION_FUNCTION_DESC, ToolType.SYSTEM_CTL
)
def analyze_meal_nutrition(
    conn: "ConnectionHandler",
    meal_text=None,
    energy_kcal_target=None,
    carbohydrate_g_target=None,
    protein_g_target=None,
    fat_g_target=None,
    sodium_mg_target=None,
    record_intake=None,
    occurred_at=None,
):
    meal_text = str(meal_text or "").strip()
    if not meal_text:
        return ActionResponse(Action.RESPONSE, None, "请告诉我要分析的一餐内容。")

    db_path = _resolve_db_path(conn)
    if not db_path.exists():
        return ActionResponse(
            Action.RESPONSE,
            None,
            f"结构化营养库还没有初始化：{db_path}",
        )

    targets = _normalize_targets(
        {
            "energy_kcal": energy_kcal_target,
            "carbohydrate_g": carbohydrate_g_target,
            "protein_g": protein_g_target,
            "fat_g": fat_g_target,
            "sodium_mg": sodium_mg_target,
        }
    )

    try:
        parsed_items = parse_meal_text(meal_text)
        if not parsed_items:
            return ActionResponse(Action.RESPONSE, None, "我没有识别出这餐里包含哪些食物。")
        resolved, unresolved = analyze_items(db_path, parsed_items)
        user_id = _resolve_user_id(conn)
        profile = _load_health_profile(conn, user_id)
        nutrition_targets = estimate_daily_nutrition_targets(profile or {"scalars": {}, "items": []})
        meal_label = _guess_meal_label(meal_text)
        inferred_targets = _derive_meal_targets(nutrition_targets, meal_label)
        merged_targets = {**inferred_targets, **targets}
        should_record = _should_record_intake(meal_text, record_intake)
        intake_record = (
            _record_meal_intake(
                conn,
                meal_text,
                resolved,
                meal_label=meal_label,
                occurred_at=str(occurred_at or "").strip() or None,
            )
            if should_record
            else None
        )
        today_intake = _load_today_nutrition_intake(conn, user_id) if user_id else None
    except Exception as exc:
        logger.bind(tag=TAG).error(f"Analyze meal nutrition failed: {exc}")
        return ActionResponse(Action.RESPONSE, None, "分析整餐营养失败，请稍后再试。")

    if not resolved:
        missing = "、".join(item.food_name for item in unresolved) or meal_text
        return ActionResponse(
            Action.RESPONSE,
            None,
            f"结构化营养库暂时没有查到这餐里的食物：{missing}。",
        )

    context = _format_context(
        meal_text,
        resolved,
        unresolved,
        merged_targets,
        intake_record=intake_record,
        profile=profile,
        nutrition_targets=nutrition_targets,
        daily_targets=_daily_targets_from_payload(nutrition_targets),
        today_intake=today_intake,
        explicit_targets=targets,
        should_record=should_record,
    )
    return ActionResponse(Action.REQLLM, context, None)


def build_meal_nutrition_payload(db_path: Path, meal_text: str) -> dict[str, Any]:
    """Analyze a meal into serializable structured nutrition payload."""
    parsed_items = parse_meal_text(meal_text)
    if not parsed_items:
        return {
            "meal_text": meal_text,
            "parsed_items": [],
            "resolved_items": [],
            "unresolved_items": [],
            "totals": {},
        }
    resolved, unresolved = analyze_items(db_path, parsed_items)
    return _build_analysis_payload(meal_text, parsed_items, resolved, unresolved)


def _resolve_db_path(conn: "ConnectionHandler") -> Path:
    plugins = conn.config.get("plugins", {})
    plugin_config = plugins.get("analyze_meal_nutrition", {})
    fallback_config = plugins.get("search_food_nutrition", {})
    configured = plugin_config.get("db_path") or fallback_config.get("db_path") or "data/clinical_foods.db"
    db_path = Path(configured)
    if not db_path.is_absolute():
        db_path = Path(get_project_dir()) / db_path
    return db_path


def _resolve_health_profile_db_path(conn: "ConnectionHandler") -> Path:
    memory_config = conn.config.get("Memory", {}).get("clinical_ltm", {})
    configured = memory_config.get("health_profile_sqlite_path") or "data/clinical_health_profile.db"
    db_path = Path(configured)
    if not db_path.is_absolute():
        db_path = Path(get_project_dir()) / db_path
    return db_path


def _resolve_user_id(conn: "ConnectionHandler") -> str:
    memory = getattr(conn, "memory", None)
    if memory is not None and getattr(memory, "role_id", None):
        return str(memory.role_id)
    return normalize_device_user_id(str(getattr(conn, "user_id", "") or getattr(conn, "device_id", "") or ""))


def _record_meal_intake(
    conn: "ConnectionHandler",
    meal_text: str,
    resolved: list[ResolvedMealItem],
    *,
    meal_label: str = "",
    occurred_at: str | None = None,
) -> dict | None:
    user_id = _resolve_user_id(conn)
    if not user_id or not resolved:
        return None
    try:
        store = HealthProfileStore(_resolve_health_profile_db_path(conn))
        totals = _sum_totals(resolved)
        return store.record_nutrition_intake_sync(
            user_id,
            meal_text=meal_text,
            meal_label=meal_label or _guess_meal_label(meal_text),
            totals=totals,
            occurred_at=occurred_at,
            items=[
                {
                    "raw_text": item.parsed.raw_text,
                    "food_name": item.parsed.food_name,
                    "matched_food": item.food["canonical_name"],
                    "grams": round(item.grams, 1),
                    "portion_source": item.portion_source,
                    "nutrients": {
                        key: round(value, 1)
                        for key, value in item.nutrient_totals.items()
                    },
                }
                for item in resolved
            ],
            source="meal_nutrition_tool",
            source_session_id=str(getattr(conn, "session_id", "") or ""),
        )
    except Exception as exc:
        logger.bind(tag=TAG).warning(f"Failed to record meal nutrition intake: {exc}")
        return None


def _should_record_intake(meal_text: str, record_intake: Any) -> bool:
    if isinstance(record_intake, bool):
        return record_intake
    if isinstance(record_intake, str):
        normalized = record_intake.strip().lower()
        if normalized in {"true", "1", "yes", "y", "是", "已吃", "记录"}:
            return True
        if normalized in {"false", "0", "no", "n", "否", "不记录", "假设"}:
            return False

    text = str(meal_text or "")
    hypothetical_tokens = [
        "如果",
        "假如",
        "计划",
        "打算",
        "准备",
        "能不能",
        "可不可以",
        "适合吗",
        "推荐",
        "建议",
        "会不会",
    ]
    actual_tokens = [
        "吃了",
        "喝了",
        "刚吃",
        "刚喝",
        "本餐",
        "这餐",
        "这一餐",
        "早餐",
        "早饭",
        "午餐",
        "午饭",
        "晚餐",
        "晚饭",
        "加餐",
        "夜宵",
    ]
    if any(token in text for token in hypothetical_tokens) and not any(
        token in text for token in actual_tokens
    ):
        return False
    return True


def _load_health_profile(conn: "ConnectionHandler", user_id: str) -> dict[str, Any]:
    if not user_id:
        return {"user_id": "", "scalars": {}, "items": []}
    try:
        store = HealthProfileStore(_resolve_health_profile_db_path(conn))
        return store.get_profile_sync(user_id)
    except Exception as exc:
        logger.bind(tag=TAG).warning(f"Failed to load health profile for meal targets: {exc}")
        return {"user_id": user_id, "scalars": {}, "items": []}


def _load_today_nutrition_intake(conn: "ConnectionHandler", user_id: str) -> dict[str, Any] | None:
    if not user_id:
        return None
    try:
        store = HealthProfileStore(_resolve_health_profile_db_path(conn))
        series = store.get_nutrition_intake_series_sync(user_id, days=1)
        return series[-1] if series else None
    except Exception as exc:
        logger.bind(tag=TAG).warning(f"Failed to load daily nutrition intake: {exc}")
        return None


def _guess_meal_label(meal_text: str) -> str:
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


def parse_meal_text(meal_text: str) -> list[ParsedMealItem]:
    normalized = _normalize_meal_text(meal_text)
    fragments = _split_meal_fragments(normalized)
    parsed: list[ParsedMealItem] = []
    for fragment in fragments:
        item = _parse_fragment(fragment)
        if item and item.food_name:
            parsed.append(item)
    return parsed


def analyze_items(db_path: Path, parsed_items: list[ParsedMealItem]) -> tuple[list[ResolvedMealItem], list[ParsedMealItem]]:
    resolved: list[ResolvedMealItem] = []
    unresolved: list[ParsedMealItem] = []
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        for item in parsed_items:
            food = _search_food(db, item.food_name)
            if food is None:
                unresolved.append(item)
                continue
            grams, portion_source = _resolve_item_grams(db, item, food)
            nutrient_totals = _calculate_nutrients(food, grams)
            resolved.append(
                ResolvedMealItem(
                    parsed=item,
                    food=food,
                    grams=grams,
                    portion_source=portion_source,
                    nutrient_totals=nutrient_totals,
                )
            )
    return resolved, unresolved


def _normalize_meal_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"[，、。；;]+", "，", text)
    text = re.sub(r"(以及|还有|另外|外加|再加|加上|搭配|配上|配着|配了|和)", "，", text)
    text = re.sub(r"(早餐|早饭|午餐|中饭|晚餐|晚饭|加餐|这一餐|这餐|一餐)", "", text)
    text = re.sub(r"(我|今天|早上|中午|晚上|刚才|大概|大约|约|吃了|喝了|吃|喝|有)", "", text)
    return text.strip(" ，。；;")


def _split_meal_fragments(text: str) -> list[str]:
    fragments: list[str] = []
    for chunk in [item.strip() for item in re.split(r"[,，、;；\n]+", text) if item.strip()]:
        markers = list(
            re.finditer(
                rf"(?P<quantity>{QUANTITY_PATTERN})\s*(?P<unit>{UNIT_PATTERN})",
                chunk,
                flags=re.IGNORECASE,
            )
        )
        if len(markers) <= 1:
            fragments.append(chunk)
            continue
        for index, marker in enumerate(markers):
            start = marker.start()
            end = markers[index + 1].start() if index + 1 < len(markers) else len(chunk)
            fragment = chunk[start:end].strip(" ，。；;")
            if fragment:
                fragments.append(fragment)
    return fragments


def _parse_fragment(fragment: str) -> ParsedMealItem | None:
    fragment = fragment.strip(" ，。；;")
    if not fragment:
        return None

    explicit = re.match(
        rf"^(?P<quantity>{QUANTITY_PATTERN})\s*(?P<unit>{UNIT_PATTERN})\s*(?P<food>.+)$",
        fragment,
        flags=re.IGNORECASE,
    )
    if explicit:
        quantity = _parse_quantity(explicit.group("quantity"))
        unit = explicit.group("unit")
        food_name = _clean_food_name(explicit.group("food"))
        explicit_grams = _to_explicit_grams(quantity, unit, food_name)
        return ParsedMealItem(fragment, food_name, quantity, unit, explicit_grams)

    trailing = re.match(
        rf"^(?P<food>.+?)(?P<quantity>{QUANTITY_PATTERN})\s*(?P<unit>{UNIT_PATTERN})$",
        fragment,
        flags=re.IGNORECASE,
    )
    if trailing:
        quantity = _parse_quantity(trailing.group("quantity"))
        unit = trailing.group("unit")
        food_name = _clean_food_name(trailing.group("food"))
        explicit_grams = _to_explicit_grams(quantity, unit, food_name)
        return ParsedMealItem(fragment, food_name, quantity, unit, explicit_grams)

    return ParsedMealItem(fragment, _clean_food_name(fragment), 1.0, None, None)


def _parse_quantity(text: str) -> float:
    text = str(text or "").strip()
    if not text:
        return 1.0
    if text == "半":
        return 0.5
    if text.startswith("半"):
        return 0.5
    try:
        return float(text)
    except ValueError:
        pass

    if "半" in text:
        return _parse_quantity(text.replace("半", "")) + 0.5
    if text == "十":
        return 10.0
    if "百" in text:
        left, _, right = text.partition("百")
        hundreds = CHINESE_DIGITS.get(left, 1 if not left else 0) * 100
        return float(hundreds + int(_parse_quantity(right) if right else 0))
    if "十" in text:
        left, _, right = text.partition("十")
        tens = CHINESE_DIGITS.get(left, 1 if not left else 0) * 10
        ones = CHINESE_DIGITS.get(right, 0) if right else 0
        return float(tens + ones)
    return float(CHINESE_DIGITS.get(text, 1))


def _to_explicit_grams(quantity: float, unit: str, food_name: str) -> float | None:
    if unit in GRAM_UNITS:
        return quantity * GRAM_UNITS[unit]
    if unit in MILLILITER_UNITS:
        return quantity * _density_for_food(food_name)
    return None


def _density_for_food(food_name: str) -> float:
    lowered = food_name.lower()
    if any(token in lowered for token in ["牛奶", "牛乳", "milk"]):
        return 1.03
    return 1.0


def _clean_food_name(food_name: str) -> str:
    food_name = re.sub(r"(一个|一只|一枚|一颗|一份|大份|小份|中等|普通|熟的|生的)", "", food_name)
    return food_name.strip(" 的左右上下约，。；;")


def _search_food(db: sqlite3.Connection, food_name: str) -> sqlite3.Row | None:
    like = f"%{food_name}%"
    starts = f"{food_name}%"
    return db.execute(
        """
        SELECT
            fi.food_id,
            fi.canonical_name,
            fi.chinese_name,
            fi.english_name,
            fi.food_category,
            fi.processing_level,
            fi.default_edible_portion_g,
            fi.source_food_id,
            fn.energy_kcal,
            fn.carbohydrate_g,
            fn.protein_g,
            fn.fat_g,
            fn.dietary_fiber_g,
            fn.sodium_mg,
            fn.potassium_mg,
            fn.phosphorus_mg,
            sd.source_title,
            sd.source_org,
            sd.evidence_level,
            sd.version
        FROM food_items fi
        LEFT JOIN food_nutrients_per_100g fn ON fn.food_id = fi.food_id
        LEFT JOIN source_documents sd ON sd.source_id = fi.source_id
        WHERE
            fi.canonical_name LIKE ?
            OR fi.english_name LIKE ?
            OR fi.chinese_name LIKE ?
            OR EXISTS (
                SELECT 1 FROM food_aliases fa
                WHERE fa.food_id = fi.food_id AND fa.alias LIKE ?
            )
        ORDER BY
            CASE
                WHEN fi.canonical_name = ? THEN 0
                WHEN fi.chinese_name = ? THEN 0
                WHEN fi.english_name = ? THEN 0
                WHEN EXISTS (
                    SELECT 1 FROM food_aliases fa
                    WHERE fa.food_id = fi.food_id AND fa.alias = ?
                ) THEN 1
                WHEN fi.canonical_name LIKE ? THEN 2
                WHEN fi.chinese_name LIKE ? THEN 2
                WHEN fi.english_name LIKE ? THEN 2
                ELSE 3
            END,
            CASE
                WHEN sd.evidence_level = 'user_provided_food_composition_table' THEN 0
                WHEN sd.source_title LIKE '%USDA%' THEN 1
                ELSE 2
            END,
            CASE
                WHEN (? LIKE '%面包%' OR lower(?) LIKE '%bread%') AND lower(fi.canonical_name) LIKE 'bread,%' THEN 0
                WHEN (? LIKE '%面包%' OR lower(?) LIKE '%bread%') AND lower(fi.canonical_name) LIKE 'flour,%' THEN 2
                ELSE 1
            END,
            CASE
                WHEN fi.processing_level = 'raw_or_minimally_processed' THEN 0
                WHEN fi.processing_level = 'unspecified' THEN 1
                WHEN fi.processing_level = 'processed' THEN 2
                ELSE 3
            END,
            LENGTH(fi.canonical_name),
            fi.food_id
        LIMIT 1
        """,
        (
            like,
            like,
            like,
            like,
            food_name,
            food_name,
            food_name,
            food_name,
            starts,
            starts,
            starts,
            food_name,
            food_name,
            food_name,
            food_name,
        ),
    ).fetchone()


def _resolve_item_grams(db: sqlite3.Connection, item: ParsedMealItem, food: sqlite3.Row) -> tuple[float, str]:
    if item.explicit_grams is not None:
        unit_label = "毫升按约1g/ml换算" if item.unit in MILLILITER_UNITS else "用户给定重量"
        return item.explicit_grams, unit_label

    portion = _match_portion_unit(db, food["food_id"], item.unit)
    if portion is not None:
        grams_per_unit, unit_name = portion
        return item.quantity * grams_per_unit, f"数据库份量单位：{unit_name}"

    default_portion = _specific_portion_grams(item, food)
    if default_portion is not None:
        return item.quantity * default_portion, f"常见份量估算：1{item.unit or '份'}≈{default_portion:g}g"

    if food["default_edible_portion_g"]:
        grams = float(food["default_edible_portion_g"]) * item.quantity
        return grams, "数据库默认可食部重量"

    unit = item.unit or "份"
    grams = GENERIC_PORTION_GRAMS.get(unit, 100.0) * item.quantity
    return grams, f"通用份量估算：1{unit}≈{GENERIC_PORTION_GRAMS.get(unit, 100.0):g}g"


def _match_portion_unit(db: sqlite3.Connection, food_id: int, unit: str | None) -> tuple[float, str] | None:
    if not unit:
        return None
    aliases = UNIT_ALIASES.get(unit, [unit.lower()])
    rows = db.execute(
        """
        SELECT unit_name, grams, confidence
        FROM portion_units
        WHERE food_id = ?
        ORDER BY
            CASE WHEN confidence = 'usda_reported' THEN 0 ELSE 1 END,
            grams
        """,
        (food_id,),
    ).fetchall()
    for row in rows:
        unit_name = str(row["unit_name"] or "")
        lowered = unit_name.lower()
        if unit in unit_name or any(alias in lowered for alias in aliases):
            amount = _extract_portion_amount(unit_name)
            if amount <= 0:
                amount = 1.0
            return float(row["grams"]) / amount, unit_name
    return None


def _extract_portion_amount(unit_name: str) -> float:
    match = re.search(r"\d+(?:\.\d+)?", unit_name)
    if not match:
        return 1.0
    try:
        return float(match.group(0))
    except ValueError:
        return 1.0


def _specific_portion_grams(item: ParsedMealItem, food: sqlite3.Row) -> float | None:
    unit = item.unit or "份"
    haystack = " ".join(
        str(value or "")
        for value in [
            item.food_name,
            food["canonical_name"],
            food["chinese_name"],
            food["english_name"],
            food["food_category"],
        ]
    ).lower()
    for keywords, portions in SPECIFIC_PORTION_RULES:
        if any(keyword.lower() in haystack for keyword in keywords):
            if unit in portions:
                return portions[unit]
            if item.unit is None and "个" in portions:
                return portions["个"]
            if item.unit is None and "份" in portions:
                return portions["份"]
    return None


def _calculate_nutrients(food: sqlite3.Row, grams: float) -> dict[str, float]:
    multiplier = grams / 100.0
    totals: dict[str, float] = {}
    for column, _, _ in NUTRIENT_COLUMNS:
        value = food[column]
        if value is not None:
            totals[column] = float(value) * multiplier
    return totals


def _normalize_targets(raw_targets: dict[str, object]) -> dict[str, float]:
    targets: dict[str, float] = {}
    for key, value in raw_targets.items():
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            targets[key] = number
    return targets


def _derive_meal_targets(target_payload: dict[str, Any], meal_label: str = "") -> dict[str, float]:
    if not target_payload or not target_payload.get("available"):
        return {}
    effective = target_payload.get("effective") or {}
    divisor = 10.0 if meal_label == "snack" else 3.0
    targets: dict[str, float] = {}

    energy = _target_number(effective.get("energy_kcal"))
    if energy:
        targets["energy_kcal"] = energy / divisor

    carb_per_meal = _target_number(effective.get("carbohydrate_g_per_meal"))
    carb_per_day = _target_number(effective.get("carbohydrate_g_per_day"))
    if carb_per_meal:
        targets["carbohydrate_g"] = carb_per_meal
    elif carb_per_day:
        targets["carbohydrate_g"] = carb_per_day / divisor

    protein = _target_number(effective.get("protein_g_per_day"))
    if protein:
        targets["protein_g"] = protein / divisor

    fat = _target_number(effective.get("fat_g_per_day"))
    if fat:
        targets["fat_g"] = fat / divisor
    return {key: round(value, 1) for key, value in targets.items() if value > 0}


def _daily_targets_from_payload(target_payload: dict[str, Any]) -> dict[str, float]:
    if not target_payload or not target_payload.get("available"):
        return {}
    effective = target_payload.get("effective") or {}
    targets = {
        nutrient_key: _target_number(effective.get(profile_key))
        for nutrient_key, profile_key in DAILY_TARGET_KEYS.items()
    }
    return {key: round(value, 1) for key, value in targets.items() if value and value > 0}


def _target_number(raw: Any) -> float | None:
    if isinstance(raw, dict):
        raw = raw.get("value")
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _build_analysis_payload(
    meal_text: str,
    parsed_items: list[ParsedMealItem],
    resolved: list[ResolvedMealItem],
    unresolved: list[ParsedMealItem],
) -> dict[str, Any]:
    return {
        "meal_text": meal_text,
        "parsed_items": [
            {
                "raw_text": item.raw_text,
                "food_name": item.food_name,
                "quantity": item.quantity,
                "unit": item.unit,
                "explicit_grams": item.explicit_grams,
            }
            for item in parsed_items
        ],
        "resolved_items": [
            {
                "raw_text": item.parsed.raw_text,
                "food_name": item.parsed.food_name,
                "matched_food": item.food["canonical_name"],
                "chinese_name": item.food["chinese_name"],
                "english_name": item.food["english_name"],
                "food_category": item.food["food_category"],
                "grams": round(float(item.grams), 1),
                "portion_source": item.portion_source,
                "source": _format_source(item.food),
                "nutrients": {
                    key: round(float(value), 1)
                    for key, value in item.nutrient_totals.items()
                },
            }
            for item in resolved
        ],
        "unresolved_items": [
            {
                "raw_text": item.raw_text,
                "food_name": item.food_name,
                "quantity": item.quantity,
                "unit": item.unit,
            }
            for item in unresolved
        ],
        "totals": {
            key: round(value, 1)
            for key, value in _sum_totals(resolved).items()
        },
    }


def _format_context(
    meal_text: str,
    resolved: list[ResolvedMealItem],
    unresolved: list[ParsedMealItem],
    targets: dict[str, float],
    intake_record: dict | None = None,
    profile: dict[str, Any] | None = None,
    nutrition_targets: dict[str, Any] | None = None,
    daily_targets: dict[str, float] | None = None,
    today_intake: dict[str, Any] | None = None,
    explicit_targets: dict[str, float] | None = None,
    should_record: bool = True,
) -> str:
    totals = _sum_totals(resolved)
    lines = [
        f"# 整餐营养估算：{meal_text}",
        "以下结果来自本地结构化食物营养成分库，并按份量换算到本餐总量。回答时请说明这是估算值；不要编造缺失食物或缺失营养素。",
        "",
        "## 食物明细",
    ]

    for idx, item in enumerate(resolved, start=1):
        parsed = item.parsed
        food = item.food
        source = _format_source(food)
        lines.append(f"\n### {idx}. {parsed.raw_text}")
        lines.append(f"- 匹配食物: {food['canonical_name']}")
        lines.append(f"- 识别份量: {parsed.quantity:g}{parsed.unit or '份'}，折算约 {item.grams:.1f} g")
        lines.append(f"- 份量依据: {item.portion_source}")
        if source:
            lines.append(f"- 数据来源: {source}")
        for column, label, unit in NUTRIENT_COLUMNS:
            per100 = food[column]
            total = item.nutrient_totals.get(column)
            if per100 is not None and total is not None:
                lines.append(f"- {label}: {total:.1f} {unit}（{float(per100):g} {unit}/100g）")

    if unresolved:
        lines.append("\n## 未能匹配的食物")
        for item in unresolved:
            lines.append(f"- {item.raw_text}: 未在结构化营养库中找到可靠匹配，整餐总量未包含此项。")

    lines.append("\n## 本餐汇总")
    for column, label, unit in NUTRIENT_COLUMNS:
        if column in totals:
            lines.append(f"- {label}: {totals[column]:.1f} {unit}")
    if intake_record:
        lines.append(f"- 已记录到每日摄入曲线: {intake_record.get('intake_date')}")
    elif not should_record:
        lines.append("- 本次被视作计划/假设餐，未写入每日摄入曲线。")

    profile_flags = _profile_context_flags(profile or {})
    if profile_flags:
        lines.append("\n## 健康档案提示")
        for item in profile_flags:
            lines.append(f"- {item}")

    if targets:
        source_note = "用户本次明确给定目标" if explicit_targets else "健康档案/系统估算目标"
        lines.append(f"\n## 本餐与参考目标对比（{source_note}）")
        for column, target_key in TARGET_COLUMNS.items():
            if column not in targets or column not in totals:
                continue
            label, unit = _nutrient_label(column)
            target = targets[column]
            total = totals[column]
            percent = total / target * 100
            status = _target_status(total, target)
            lines.append(f"- {label}: {total:.1f}/{target:g} {unit}，约 {percent:.0f}%，{status}")
    else:
        lines.append("\n## 与目标对比")
        reason = (nutrition_targets or {}).get("reason") or "档案信息不足，暂时无法生成参考目标。"
        lines.append(f"- {reason}")

    if should_record and daily_targets and today_intake:
        lines.append("\n## 今日累计摄入与每日目标")
        for column, label, unit in NUTRIENT_COLUMNS:
            if column not in daily_targets:
                continue
            consumed = float(today_intake.get(column) or 0.0)
            target = daily_targets[column]
            percent = consumed / target * 100
            remaining = max(target - consumed, 0.0)
            status = "已超过每日目标" if consumed > target * 1.05 else "接近每日目标" if consumed >= target * 0.9 else f"还剩约 {remaining:.1f} {unit}"
            lines.append(f"- {label}: 今日累计 {consumed:.1f}/{target:g} {unit}，约 {percent:.0f}%，{status}")

    lines.append("\n## 回复要求")
    lines.append("- 用中文口语化回答，第一句先给本餐总热量、碳水、蛋白质、脂肪。")
    lines.append("- 如果已经记录到每日摄入曲线，要自然说明已记入今天的摄入；如果有今日累计与剩余额，简短提醒。")
    lines.append("- 如果用户有糖尿病、肾病、高血压、痛风、过敏等背景，应结合记忆和临床安全规则提醒，但不要替代医生诊疗。")
    return "\n".join(lines)


def _profile_context_flags(profile: dict[str, Any]) -> list[str]:
    items = profile.get("items") or []
    scalars = profile.get("scalars") or {}
    flags: list[str] = []
    for category, label in [
        ("disease", "疾病"),
        ("medication", "用药"),
        ("allergy", "过敏"),
        ("renal_function", "肾功能"),
        ("dietary_restriction", "饮食限制"),
    ]:
        names = [
            str(item.get("name") or "").strip()
            for item in items
            if str(item.get("category") or "") == category and str(item.get("status") or "active") == "active"
        ]
        names = [name for name in names if name and not name.startswith("无")]
        if names:
            flags.append(f"{label}: {'、'.join(names[:6])}")
    if scalars.get("target_carbohydrate_g_per_meal"):
        flags.append(f"档案每餐碳水目标: {scalars.get('target_carbohydrate_g_per_meal')} g")
    if scalars.get("nutrition_goal"):
        flags.append(f"营养目标: {scalars.get('nutrition_goal')}")
    return flags[:8]


def _sum_totals(items: list[ResolvedMealItem]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for item in items:
        for column, value in item.nutrient_totals.items():
            totals[column] = totals.get(column, 0.0) + value
    return totals


def _format_source(food: sqlite3.Row) -> str:
    parts = [food["source_title"], food["source_org"], food["version"]]
    return " / ".join(str(part) for part in parts if part)


def _nutrient_label(column: str) -> tuple[str, str]:
    for item_column, label, unit in NUTRIENT_COLUMNS:
        if item_column == column:
            return label, unit
    return column, ""


def _target_status(total: float, target: float) -> str:
    if total > target * 1.05:
        return "超出目标"
    if total > target:
        return "略高于目标，但在估算误差范围内"
    return "未超出目标"
