from __future__ import annotations

import argparse
import csv
import io
import sqlite3
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "knowledge_base" / "structured" / "schema.sql"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "clinical_foods.db"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "usda"
DEFAULT_ZIP_NAME = "FoodData_Central_foundation_food_csv_2025-12-18.zip"
DEFAULT_USDA_URL = (
    "https://fdc.nal.usda.gov/fdc-datasets/"
    "FoodData_Central_foundation_food_csv_2025-12-18.zip"
)

NUTRIENT_PRIORITY = {
    "energy_kcal": ["1008", "2047", "2048"],
    "protein_g": ["1003"],
    "fat_g": ["1004"],
    "carbohydrate_g": ["1005", "1050", "2039"],
    "dietary_fiber_g": ["1079", "2033"],
    "sodium_mg": ["1093"],
    "potassium_mg": ["1092"],
    "phosphorus_mg": ["1091"],
    "calcium_mg": ["1087"],
    "cholesterol_mg": ["1253"],
}

SOURCE_TITLE = "USDA FoodData Central Foundation Foods"
SOURCE_VERSION = "2025-12-18 CSV"


@dataclass(frozen=True)
class ImportStats:
    foods: int
    nutrient_rows: int
    aliases: int
    portions: int
    risk_tags: int


def main() -> None:
    args = parse_args()
    db_path = Path(args.db).resolve()
    zip_path = Path(args.zip).resolve() if args.zip else DEFAULT_RAW_DIR / DEFAULT_ZIP_NAME

    if not zip_path.exists():
        if args.no_download:
            raise FileNotFoundError(f"USDA zip not found: {zip_path}")
        download_usda_zip(zip_path, args.url)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        ensure_schema_migrations(conn)
        stats = import_foundation_foods(
            conn,
            zip_path,
            limit=args.limit,
            replace_existing=not args.append,
        )
        conn.commit()

    print(
        "USDA Foundation Foods import complete: "
        f"db={db_path}, foods={stats.foods}, nutrients={stats.nutrient_rows}, "
        f"aliases={stats.aliases}, portions={stats.portions}, risk_tags={stats.risk_tags}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import USDA FoodData Central Foundation Foods into clinical_foods.db"
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    parser.add_argument("--zip", default="", help="Local USDA Foundation Foods CSV zip path.")
    parser.add_argument("--url", default=DEFAULT_USDA_URL, help="USDA Foundation Foods download URL.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of foods to import.")
    parser.add_argument("--no-download", action="store_true", help="Do not download the zip if missing.")
    parser.add_argument("--append", action="store_true", help="Append/update without clearing prior USDA rows.")
    return parser.parse_args()


def download_usda_zip(zip_path: Path, url: str) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading USDA Foundation Foods from {url}")
    urllib.request.urlretrieve(url, zip_path)


def ensure_schema_migrations(conn: sqlite3.Connection) -> None:
    food_columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(food_items)").fetchall()
    }
    if "source_food_id" not in food_columns:
        conn.execute("ALTER TABLE food_items ADD COLUMN source_food_id TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_food_items_source_food_id ON food_items(source_food_id)"
    )


def import_foundation_foods(
    conn: sqlite3.Connection,
    zip_path: Path,
    limit: int = 0,
    replace_existing: bool = True,
) -> ImportStats:
    with ZipFile(zip_path) as archive:
        names = {Path(name).name: name for name in archive.namelist() if Path(name).name}
        required = {
            "foundation_food.csv",
            "food.csv",
            "food_category.csv",
            "food_nutrient.csv",
            "food_portion.csv",
            "measure_unit.csv",
        }
        missing = sorted(required - set(names))
        if missing:
            raise ValueError(f"USDA zip is missing required files: {missing}")

        foundation_ids = {
            row["fdc_id"]
            for row in read_csv(archive, names["foundation_food.csv"])
            if row.get("fdc_id")
        }
        categories = {
            row["id"]: row["description"]
            for row in read_csv(archive, names["food_category.csv"])
        }
        measure_units = {
            row["id"]: row["name"]
            for row in read_csv(archive, names["measure_unit.csv"])
        }
        foods = [
            row
            for row in read_csv(archive, names["food.csv"])
            if row.get("fdc_id") in foundation_ids
        ]
        foods.sort(key=lambda item: item.get("description", ""))
        if limit > 0:
            foods = foods[:limit]
        selected_ids = {row["fdc_id"] for row in foods}
        nutrient_values = collect_nutrients(archive, names["food_nutrient.csv"], selected_ids)
        portions_by_food = collect_portions(
            archive,
            names["food_portion.csv"],
            selected_ids,
            measure_units,
        )

    source_id = upsert_source(conn)
    if replace_existing:
        conn.execute("DELETE FROM food_items WHERE source_id = ?", (source_id,))
    stats = ImportStats(foods=0, nutrient_rows=0, aliases=0, portions=0, risk_tags=0)
    stats_dict = stats.__dict__.copy()

    for food in foods:
        fdc_id = food["fdc_id"]
        description = clean_text(food.get("description"))
        category = categories.get(food.get("food_category_id"), "Uncategorized")
        food_id = upsert_food_item(
            conn=conn,
            source_id=source_id,
            fdc_id=fdc_id,
            description=description,
            category=category,
            publication_date=food.get("publication_date", ""),
        )
        stats_dict["foods"] += 1
        if upsert_nutrients(conn, food_id, source_id, nutrient_values.get(fdc_id, {})):
            stats_dict["nutrient_rows"] += 1
        stats_dict["aliases"] += upsert_aliases(conn, food_id, description)
        stats_dict["portions"] += upsert_portions(conn, food_id, source_id, portions_by_food.get(fdc_id, []))
        stats_dict["risk_tags"] += upsert_risk_tags(
            conn,
            food_id,
            source_id,
            description,
            nutrient_values.get(fdc_id, {}),
        )
        upsert_allergen_flags(conn, food_id, source_id, description)

    return ImportStats(**stats_dict)


def read_csv(archive: ZipFile, member: str) -> list[dict[str, str]]:
    with archive.open(member) as handle:
        wrapper = io.TextIOWrapper(handle, encoding="utf-8-sig", newline="")
        return list(csv.DictReader(wrapper))


def collect_nutrients(archive: ZipFile, member: str, selected_ids: set[str]) -> dict[str, dict[str, float]]:
    priority_index = {
        nutrient_id: (column, index)
        for column, nutrient_ids in NUTRIENT_PRIORITY.items()
        for index, nutrient_id in enumerate(nutrient_ids)
    }
    values: dict[str, dict[str, tuple[int, float]]] = {}
    for row in read_csv(archive, member):
        fdc_id = row.get("fdc_id")
        nutrient_id = row.get("nutrient_id")
        if fdc_id not in selected_ids or nutrient_id not in priority_index:
            continue
        amount = parse_float(row.get("amount"))
        if amount is None:
            continue
        column, priority = priority_index[nutrient_id]
        existing = values.setdefault(fdc_id, {}).get(column)
        if existing is None or priority < existing[0]:
            values[fdc_id][column] = (priority, amount)
    return {
        fdc_id: {column: amount for column, (_, amount) in row.items()}
        for fdc_id, row in values.items()
    }


def collect_portions(
    archive: ZipFile,
    member: str,
    selected_ids: set[str],
    measure_units: dict[str, str],
) -> dict[str, list[dict[str, str]]]:
    portions: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(archive, member):
        fdc_id = row.get("fdc_id")
        if fdc_id not in selected_ids:
            continue
        grams = parse_float(row.get("gram_weight"))
        if grams is None or grams <= 0:
            continue
        amount = clean_text(row.get("amount"))
        unit = measure_units.get(row.get("measure_unit_id"), "")
        description = clean_text(row.get("portion_description"))
        modifier = clean_text(row.get("modifier"))
        pieces = [piece for piece in [amount, unit, description, modifier] if piece]
        unit_name = " ".join(pieces).strip() or f"{grams:g} g serving"
        portions.setdefault(fdc_id, []).append(
            {"unit_name": unit_name[:160], "grams": str(grams)}
        )
    return portions


def upsert_source(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT source_id FROM source_documents WHERE source_title = ? AND version = ?",
        (SOURCE_TITLE, SOURCE_VERSION),
    ).fetchone()
    if row:
        return int(row[0])
    cursor = conn.execute(
        """
        INSERT INTO source_documents (
            source_title, source_org, publish_year, version, source_url,
            evidence_level, license_note, last_reviewed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            SOURCE_TITLE,
            "U.S. Department of Agriculture",
            2025,
            SOURCE_VERSION,
            "https://fdc.nal.usda.gov/download-datasets",
            "public_food_composition_database",
            "USDA FoodData Central data is a U.S. government work and is generally public domain; verify attribution requirements for your deployment.",
            "2026-04-25",
        ),
    )
    return int(cursor.lastrowid)


def upsert_food_item(
    *,
    conn: sqlite3.Connection,
    source_id: int,
    fdc_id: str,
    description: str,
    category: str,
    publication_date: str,
) -> int:
    row = conn.execute(
        "SELECT food_id FROM food_items WHERE canonical_name = ?",
        (description,),
    ).fetchone()
    if row:
        food_id = int(row[0])
        conn.execute(
            """
            UPDATE food_items
            SET english_name = ?, food_category = ?, source_food_id = ?, notes = ?,
                source_id = ?, updated_at = CURRENT_TIMESTAMP
            WHERE food_id = ?
            """,
            (
                description,
                category,
                fdc_id,
                f"USDA FDC ID: {fdc_id}; publication_date: {publication_date}",
                source_id,
                food_id,
            ),
        )
        return food_id

    cursor = conn.execute(
        """
        INSERT INTO food_items (
            canonical_name, english_name, food_category, processing_level,
            source_food_id, notes, source_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            description,
            description,
            category,
            infer_processing_level(description, category),
            fdc_id,
            f"USDA FDC ID: {fdc_id}; publication_date: {publication_date}",
            source_id,
        ),
    )
    return int(cursor.lastrowid)


def upsert_nutrients(conn: sqlite3.Connection, food_id: int, source_id: int, values: dict[str, float]) -> bool:
    if not values:
        return False
    columns = list(NUTRIENT_PRIORITY)
    payload = [values.get(column) for column in columns]
    conn.execute(
        f"""
        INSERT INTO food_nutrients_per_100g (
            food_id, {", ".join(columns)}, data_quality, source_id
        )
        VALUES (?, {", ".join("?" for _ in columns)}, ?, ?)
        ON CONFLICT(food_id, source_id) DO UPDATE SET
            {", ".join(f"{column} = excluded.{column}" for column in columns)},
            data_quality = excluded.data_quality,
            updated_at = CURRENT_TIMESTAMP
        """,
        [food_id, *payload, "usda_foundation_food", source_id],
    )
    return True


def upsert_aliases(conn: sqlite3.Connection, food_id: int, description: str) -> int:
    aliases = [description, description.lower(), *infer_chinese_aliases(description)]
    inserted = 0
    for alias in aliases:
        alias = clean_text(alias)
        if not alias:
            continue
        language = "zh" if any("\u4e00" <= char <= "\u9fff" for char in alias) else "en"
        before = conn.total_changes
        conn.execute(
            "INSERT OR IGNORE INTO food_aliases (food_id, alias, language) VALUES (?, ?, ?)",
            (food_id, alias, language),
        )
        inserted += conn.total_changes - before
    return inserted


def upsert_portions(
    conn: sqlite3.Connection,
    food_id: int,
    source_id: int,
    portions: list[dict[str, str]],
) -> int:
    inserted = 0
    for portion in portions[:8]:
        before = conn.total_changes
        conn.execute(
            """
            INSERT OR IGNORE INTO portion_units (
                food_id, unit_name, grams, confidence, source_id, notes
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                food_id,
                portion["unit_name"],
                parse_float(portion["grams"]),
                "usda_reported",
                source_id,
                "USDA FoodData Central food_portion.csv",
            ),
        )
        inserted += conn.total_changes - before
    return inserted


def upsert_allergen_flags(conn: sqlite3.Connection, food_id: int, source_id: int, description: str) -> None:
    lowered = description.lower()
    flags = {
        "contains_gluten": int(any(token in lowered for token in ["bread", "wheat", "pasta", "flour", "cracker"])),
        "contains_peanut": int("peanut" in lowered),
        "contains_tree_nut": int(any(token in lowered for token in ["almond", "walnut", "cashew", "pistachio", "pecan", "hazelnut"])),
        "contains_crustacean": int(any(token in lowered for token in ["shrimp", "crab", "lobster", "crayfish"])),
        "contains_soy": int(any(token in lowered for token in ["soy", "tofu", "edamame"])),
        "contains_dairy": int(is_dairy_description(lowered)),
        "contains_egg": int("egg" in lowered),
    }
    conn.execute(
        """
        INSERT INTO allergen_flags (
            food_id, contains_gluten, contains_peanut, contains_tree_nut,
            contains_crustacean, contains_soy, contains_dairy, contains_egg,
            cross_contamination_risk, source_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(food_id) DO UPDATE SET
            contains_gluten = excluded.contains_gluten,
            contains_peanut = excluded.contains_peanut,
            contains_tree_nut = excluded.contains_tree_nut,
            contains_crustacean = excluded.contains_crustacean,
            contains_soy = excluded.contains_soy,
            contains_dairy = excluded.contains_dairy,
            contains_egg = excluded.contains_egg,
            cross_contamination_risk = excluded.cross_contamination_risk,
            source_id = excluded.source_id,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            food_id,
            flags["contains_gluten"],
            flags["contains_peanut"],
            flags["contains_tree_nut"],
            flags["contains_crustacean"],
            flags["contains_soy"],
            flags["contains_dairy"],
            flags["contains_egg"],
            "Heuristic from USDA food description; verify packaged-food labels.",
            source_id,
        ),
    )


def upsert_risk_tags(
    conn: sqlite3.Connection,
    food_id: int,
    source_id: int,
    description: str,
    nutrients: dict[str, float],
) -> int:
    tags: list[tuple[str, str, str]] = []
    lowered = description.lower()
    if "grapefruit" in lowered:
        tags.append((
            "cyp3a4_interaction_food",
            "drug_food_interaction",
            "Grapefruit can interact with selected CYP3A4-metabolized medicines.",
        ))
    if (nutrients.get("sodium_mg") or 0) >= 400:
        tags.append(("high_sodium", "hypertension", "Sodium is at least 400 mg per 100 g."))
    if (nutrients.get("potassium_mg") or 0) >= 300:
        tags.append(("high_potassium", "chronic_kidney_disease", "Potassium is at least 300 mg per 100 g."))
    if (nutrients.get("phosphorus_mg") or 0) >= 250:
        tags.append(("high_phosphorus", "chronic_kidney_disease", "Phosphorus is at least 250 mg per 100 g."))

    inserted = 0
    for risk_tag, condition, rationale in tags:
        before = conn.total_changes
        conn.execute(
            """
            INSERT OR IGNORE INTO food_risk_tags (
                food_id, risk_tag, applies_to_condition, rationale, source_id
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (food_id, risk_tag, condition, rationale, source_id),
        )
        inserted += conn.total_changes - before
    return inserted


def infer_processing_level(description: str, category: str) -> str:
    lowered = f"{description} {category}".lower()
    if any(token in lowered for token in ["raw", "fresh"]):
        return "raw_or_minimally_processed"
    if any(token in lowered for token in ["canned", "frozen", "prepared", "commercial", "breaded"]):
        return "processed"
    return "unspecified"


def infer_chinese_aliases(description: str) -> list[str]:
    lowered = description.lower()
    aliases: list[str] = []
    if lowered.startswith("milk,"):
        aliases.extend(["牛奶"])
    if lowered.startswith("almond milk"):
        aliases.extend(["杏仁奶"])
    if lowered.startswith("oat milk"):
        aliases.extend(["燕麦奶"])
    if lowered.startswith("soy milk"):
        aliases.extend(["豆奶", "豆浆"])
    if lowered.startswith("egg, whole"):
        aliases.extend(["鸡蛋", "全蛋"])
    if lowered.startswith("egg, white"):
        aliases.extend(["蛋白", "鸡蛋清"])
    if lowered.startswith("egg, yolk"):
        aliases.extend(["蛋黄"])
    for token, values in CHINESE_ALIAS_RULES:
        if token in lowered:
            aliases.extend(values)
    return aliases


def is_dairy_description(lowered_description: str) -> bool:
    if lowered_description.startswith(("almond milk", "oat milk", "soy milk")):
        return False
    if lowered_description.startswith("milk,"):
        return True
    return any(
        token in lowered_description
        for token in ["cheese", "yogurt", "buttermilk", "cream", "butter", "whole milk"]
    )


def clean_text(value: str | None) -> str:
    return " ".join(str(value or "").replace("\ufeff", "").split()).strip()


def parse_float(value: str | None) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


CHINESE_ALIAS_RULES = [
    ("yogurt", ["酸奶"]),
    ("cheese", ["奶酪", "芝士"]),
    ("bread, white", ["白面包", "吐司"]),
    ("bread", ["面包"]),
    ("rice", ["米饭", "大米"]),
    ("oat", ["燕麦"]),
    ("wheat", ["小麦"]),
    ("apple", ["苹果"]),
    ("banana", ["香蕉"]),
    ("orange", ["橙子"]),
    ("grapefruit", ["西柚", "葡萄柚"]),
    ("peach", ["桃子"]),
    ("kiwifruit", ["猕猴桃", "奇异果"]),
    ("tomato", ["番茄", "西红柿"]),
    ("potato", ["土豆", "马铃薯"]),
    ("sweet potato", ["红薯", "甘薯"]),
    ("broccoli", ["西兰花"]),
    ("kale", ["羽衣甘蓝"]),
    ("spinach", ["菠菜"]),
    ("almond", ["杏仁"]),
    ("peanut", ["花生"]),
    ("tofu", ["豆腐"]),
    ("soy", ["大豆", "黄豆"]),
    ("chicken", ["鸡肉"]),
    ("beef", ["牛肉"]),
    ("pork", ["猪肉"]),
    ("salmon", ["三文鱼", "鲑鱼"]),
    ("shrimp", ["虾"]),
]


if __name__ == "__main__":
    main()
