import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from config.config_loader import get_project_dir
from config.logger import setup_logging
from plugins_func.register import Action, ActionResponse, ToolType, register_function

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

SEARCH_FOOD_NUTRITION_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "search_food_nutrition",
        "description": "查询本地结构化食物营养成分库，适合获取每100克热量、碳水、蛋白质、脂肪、膳食纤维、钠、钾、磷、钙、胆固醇等客观数值。涉及具体食物营养数字时应优先调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "food_name": {
                    "type": "string",
                    "description": "要查询的食物名称，可以是中文或英文，例如鸡蛋、牛奶、白面包、apple、salmon。",
                },
                "limit": {
                    "type": "integer",
                    "description": "最多返回几条候选食物，默认3。",
                },
            },
            "required": ["food_name"],
        },
    },
}


@register_function(
    "search_food_nutrition", SEARCH_FOOD_NUTRITION_FUNCTION_DESC, ToolType.SYSTEM_CTL
)
def search_food_nutrition(conn: "ConnectionHandler", food_name=None, limit=3):
    food_name = str(food_name or "").strip()
    if not food_name:
        return ActionResponse(Action.RESPONSE, None, "请告诉我要查询的食物名称。")

    try:
        limit = max(1, min(int(limit or 3), 8))
    except (TypeError, ValueError):
        limit = 3

    db_path = _resolve_db_path(conn)
    if not db_path.exists():
        return ActionResponse(
            Action.RESPONSE,
            None,
            f"结构化营养库还没有初始化：{db_path}",
        )

    try:
        rows = _search_rows(db_path, food_name, limit)
    except Exception as exc:
        logger.bind(tag=TAG).error(f"Search food nutrition failed: {exc}")
        return ActionResponse(Action.RESPONSE, None, "查询结构化营养库失败，请稍后再试。")

    if not rows:
        return ActionResponse(
            Action.RESPONSE,
            None,
            f"结构化营养库暂时没有查到“{food_name}”。",
        )

    context = _format_context(food_name, rows)
    return ActionResponse(Action.REQLLM, context, None)


def _resolve_db_path(conn: "ConnectionHandler") -> Path:
    plugin_config = conn.config.get("plugins", {}).get("search_food_nutrition", {})
    configured = plugin_config.get("db_path", "data/clinical_foods.db")
    db_path = Path(configured)
    if not db_path.is_absolute():
        db_path = Path(get_project_dir()) / db_path
    return db_path


def _search_rows(db_path: Path, food_name: str, limit: int) -> list[sqlite3.Row]:
    like = f"%{food_name}%"
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        return list(
            db.execute(
                """
                SELECT
                    fi.food_id,
                    fi.canonical_name,
                    fi.chinese_name,
                    fi.english_name,
                    fi.food_category,
                    fi.processing_level,
                    fi.source_food_id,
                    fn.energy_kcal,
                    fn.carbohydrate_g,
                    fn.protein_g,
                    fn.fat_g,
                    fn.dietary_fiber_g,
                    fn.sodium_mg,
                    fn.potassium_mg,
                    fn.phosphorus_mg,
                    fn.calcium_mg,
                    fn.magnesium_mg,
                    fn.iron_mg,
                    fn.zinc_mg,
                    fn.selenium_ug,
                    fn.vitamin_a_ug,
                    fn.vitamin_c_mg,
                    fn.vitamin_e_mg,
                    fn.cholesterol_mg,
                    af.contains_gluten,
                    af.contains_peanut,
                    af.contains_tree_nut,
                    af.contains_crustacean,
                    af.contains_soy,
                    af.contains_dairy,
                    af.contains_egg,
                    sd.source_title,
                    sd.source_org,
                    sd.version
                FROM food_items fi
                LEFT JOIN food_nutrients_per_100g fn ON fn.food_id = fi.food_id
                LEFT JOIN allergen_flags af ON af.food_id = fi.food_id
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
                        WHEN EXISTS (
                            SELECT 1 FROM food_aliases fa
                            WHERE fa.food_id = fi.food_id AND fa.alias = ?
                        ) THEN 0
                        WHEN fi.canonical_name = ? THEN 0
                        WHEN fi.english_name = ? THEN 1
                        WHEN fi.canonical_name LIKE ? THEN 2
                        ELSE 3
                    END,
                    CASE
                        WHEN sd.evidence_level = 'user_provided_food_composition_table' THEN 0
                        WHEN sd.source_title LIKE '%USDA%' THEN 1
                        ELSE 2
                    END,
                    CASE
                        WHEN fi.processing_level = 'raw_or_minimally_processed' THEN 0
                        WHEN fi.processing_level = 'processed' THEN 1
                        ELSE 2
                    END,
                    fi.canonical_name
                LIMIT ?
                """,
                (like, like, like, like, food_name, food_name, food_name, f"{food_name}%", limit),
            )
        )


def _format_context(food_name: str, rows: list[sqlite3.Row]) -> str:
    lines = [
        f"# 结构化营养库查询结果：{food_name}",
        "以下数值来自本地 SQL 食物营养成分库，默认单位为每100克可食部。回答时不要编造缺失数值；如果需要换算到一份食物，请先说明估算依据。",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(f"\n## 候选 {idx}: {row['canonical_name']}")
        lines.append(f"- 分类: {row['food_category'] or '未知'}")
        if row["source_food_id"]:
            lines.append(f"- 来源食物ID: {row['source_food_id']}")
        nutrients = [
            ("热量", row["energy_kcal"], "kcal"),
            ("碳水", row["carbohydrate_g"], "g"),
            ("蛋白质", row["protein_g"], "g"),
            ("脂肪", row["fat_g"], "g"),
            ("膳食纤维", row["dietary_fiber_g"], "g"),
            ("钠", row["sodium_mg"], "mg"),
            ("钾", row["potassium_mg"], "mg"),
            ("磷", row["phosphorus_mg"], "mg"),
            ("钙", row["calcium_mg"], "mg"),
            ("镁", row["magnesium_mg"], "mg"),
            ("铁", row["iron_mg"], "mg"),
            ("锌", row["zinc_mg"], "mg"),
            ("硒", row["selenium_ug"], "ug"),
            ("维生素A", row["vitamin_a_ug"], "ug"),
            ("维生素C", row["vitamin_c_mg"], "mg"),
            ("维生素E", row["vitamin_e_mg"], "mg"),
            ("胆固醇", row["cholesterol_mg"], "mg"),
        ]
        for label, value, unit in nutrients:
            if value is not None:
                lines.append(f"- {label}: {float(value):g} {unit}/100g")
        allergens = _format_allergens(row)
        if allergens:
            lines.append(f"- 过敏原启发式标签: {', '.join(allergens)}")
        source = " / ".join(
            str(item)
            for item in [row["source_title"], row["source_org"], row["version"]]
            if item
        )
        if source:
            lines.append(f"- 来源: {source}")
    return "\n".join(lines)


def _format_allergens(row: sqlite3.Row) -> list[str]:
    mapping = [
        ("contains_gluten", "含麸质"),
        ("contains_peanut", "含花生"),
        ("contains_tree_nut", "含树坚果"),
        ("contains_crustacean", "含甲壳类"),
        ("contains_soy", "含大豆"),
        ("contains_dairy", "含乳制品"),
        ("contains_egg", "含蛋"),
    ]
    return [label for column, label in mapping if row[column]]
