from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.clinical_safety import ClinicalSafetyInterceptor, SafetyRuleEngine

STRUCTURED_SCHEMA = PROJECT_ROOT / "knowledge_base" / "structured" / "schema.sql"
RULES_PATH = PROJECT_ROOT / "knowledge_base" / "rules" / "clinical_safety_rules.json"
LLMWIKI_ROOT = PROJECT_ROOT / "knowledge_base" / "llmwiki" / "clinical-nutrition"
RAG_DB = PROJECT_ROOT / "tmp" / "validate_clinical_rag.db"
STRUCTURED_DB = PROJECT_ROOT / "data" / "clinical_foods.db"
CONSOLE_HTML = PROJECT_ROOT / "console" / "index.html"
CONSOLE_APP_JS = PROJECT_ROOT / "console" / "app.js"
CONNECTION_PY = PROJECT_ROOT / "core" / "connection.py"


def main() -> None:
    validate_ingestion_dependencies()
    validate_sqlite_schema()
    validate_rules_file()
    validate_rule_engine_sample()
    validate_safety_interceptor_sample()
    validate_daily_life_safety_rules()
    validate_clinical_rag_service()
    validate_structured_food_database_if_present()
    validate_meal_nutrition_analyzer_if_present()
    validate_vision_nutrition_analyzer_if_present()
    validate_health_profile_store()
    validate_nutrition_targets()
    validate_voice_health_profile_confirmation()
    validate_device_identity()
    validate_clinical_console()
    print("clinical knowledge base validation passed")


def validate_ingestion_dependencies() -> None:
    try:
        import pypdf  # noqa: F401
    except Exception as exc:
        raise AssertionError("PDF ingestion requires pypdf") from exc
    try:
        import docx  # noqa: F401
    except Exception as exc:
        raise AssertionError("DOCX ingestion requires python-docx") from exc


def validate_sqlite_schema() -> None:
    sql = STRUCTURED_SCHEMA.read_text(encoding="utf-8")
    with sqlite3.connect(":memory:") as conn:
        conn.executescript(sql)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    required = {
        "source_documents",
        "food_items",
        "food_aliases",
        "food_nutrients_per_100g",
        "glycemic_values",
        "allergen_flags",
        "portion_units",
        "food_risk_tags",
    }
    missing = required - tables
    if missing:
        raise AssertionError(f"Missing structured DB tables: {sorted(missing)}")


def validate_rules_file() -> None:
    payload = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    seen = set()
    for rule in payload.get("rules", []):
        rule_id = rule.get("rule_id")
        if not rule_id:
            raise AssertionError("Rule is missing rule_id")
        if rule_id in seen:
            raise AssertionError(f"Duplicate rule_id: {rule_id}")
        seen.add(rule_id)
        for key in ("severity", "category", "if", "then", "evidence"):
            if key not in rule:
                raise AssertionError(f"Rule {rule_id} is missing {key}")


def validate_rule_engine_sample() -> None:
    engine = SafetyRuleEngine(RULES_PATH)
    context = {
        "foods": ["西柚"],
        "medications": ["硝苯地平"],
        "diseases": [],
        "allergies": [],
    }
    findings = engine.evaluate(context)
    if not any(item.rule_id == "FDI_GRAPEFRUIT_CYP3A4_DRUGS" for item in findings):
        raise AssertionError("Expected grapefruit drug-food rule did not fire")


def validate_safety_interceptor_sample() -> None:
    interceptor = ClinicalSafetyInterceptor(RULES_PATH)
    result = interceptor.evaluate(
        query="我正在吃硝苯地平，早餐还能吃西柚吗？",
        memory_context="",
        dialogue_messages=[],
    )
    if not result.should_block:
        raise AssertionError("Expected safety interceptor to block grapefruit + nifedipine")
    if "西柚" not in result.response_text:
        raise AssertionError("Safety interceptor response should mention the matched food")


def validate_daily_life_safety_rules() -> None:
    interceptor = ClinicalSafetyInterceptor(RULES_PATH)
    samples = [
        ("我血糖3.5，现在可以去跑步吗？", "GLUCOSE_HYPOGLYCEMIA_VALUE", True),
        ("我有2型糖尿病，午餐可以喝一杯奶茶吗？", "DIABETES_SUGARY_DRINK", False),
        ("我怀孕了，可以喝生牛奶吗？", "PREGNANCY_HIGH_RISK_FOODS", True),
        ("我正在吃左氧氟沙星，可以喝牛奶吗？", "FDI_QUINOLONE_TETRACYCLINE_DAIRY_MINERALS", False),
        ("我对虾过敏，这个虾饺可以吃吗？", "ALLERGY_SHELLFISH_EXPOSURE", True),
        ("面包有点发霉，削掉还能吃吗？", "FOOD_SAFETY_SPOILED_OR_MOLDY", True),
    ]
    for query, expected_rule_id, expected_block in samples:
        result = interceptor.evaluate(query=query, memory_context="", dialogue_messages=[])
        if not any(item.rule_id == expected_rule_id for item in result.findings):
            raise AssertionError(f"Expected daily-life rule {expected_rule_id} did not fire for: {query}")
        if result.should_block != expected_block:
            raise AssertionError(
                f"Daily-life rule {expected_rule_id} block={result.should_block}, expected {expected_block}"
            )


def validate_clinical_rag_service() -> None:
    from core.clinical_nutrition.clinical_rag import ClinicalRAGService
    from plugins_func.functions.search_clinical_rag import search_clinical_rag

    class DummyLogger:
        def bind(self, **_kwargs):
            return self

        def error(self, _message):
            return None

        def warning(self, _message):
            return None

    if RAG_DB.exists():
        RAG_DB.unlink()
    tmp_dir = PROJECT_ROOT / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    source_path = tmp_dir / "validate_clinical_rag.md"
    source_path.write_text(
        "\n".join(
            [
                "# 糖尿病饮食验证资料",
                "",
                "2型糖尿病患者应控制含糖饮料，优先选择白水、无糖茶或无糖咖啡。",
                "",
                "## 午餐建议",
                "",
                "午餐主食需要估算碳水化合物总量，搭配优质蛋白和非淀粉类蔬菜。",
                "",
                "## 低血糖风险",
                "",
                "如果出现低血糖症状，应及时补充快速碳水并复测血糖。",
            ]
        ),
        encoding="utf-8",
    )
    config = {
        "clinical_rag": {
            "db_path": str(RAG_DB),
            "chunk_chars": 220,
            "chunk_overlap_chars": 40,
            "top_k": 3,
            "embedding": {
                "enabled": True,
                "provider": "mock",
                "model": "mock-embedding",
                "dimensions": 32,
            },
        },
        "plugins": {"search_clinical_rag": {"db_path": str(RAG_DB), "top_k": 3}},
    }
    service = ClinicalRAGService(project_root=PROJECT_ROOT, config=config, logger=DummyLogger())
    document = service.register_document(source_path, original_name="糖尿病饮食验证资料.md")
    job = service.create_index_job(document["document_id"])
    indexed = service.index_document(document["document_id"], job_id=job["job_id"])
    try:
        if indexed.get("status") != "indexed":
            raise AssertionError(f"Clinical RAG indexing failed: {indexed}")
        if indexed.get("chunk_count", 0) < 1:
            raise AssertionError("Clinical RAG should create chunks")
        if indexed.get("embedded_count") != indexed.get("chunk_count"):
            raise AssertionError("Clinical RAG should embed all chunks with mock embedder")
        results = service.search("糖尿病午餐怎么控制碳水", top_k=2)
        if not results:
            raise AssertionError("Clinical RAG search returned no results")
        if not results[0].get("citation") or not results[0].get("chunk_id"):
            raise AssertionError("Clinical RAG result should include citation and chunk_id")

        class DummyConnection:
            def __init__(self):
                self.config = config

        response = search_clinical_rag(DummyConnection(), question="糖尿病可以喝含糖饮料吗？", top_k=2)
        if response.action.name != "REQLLM":
            raise AssertionError("search_clinical_rag should return REQLLM when evidence exists")
        if "引用:" not in (response.result or ""):
            raise AssertionError("search_clinical_rag context should include citations")
    finally:
        try:
            source_path.unlink()
        except OSError:
            pass
        try:
            RAG_DB.unlink()
        except OSError:
            pass


def validate_structured_food_database_if_present() -> None:
    if not STRUCTURED_DB.exists():
        return
    with sqlite3.connect(STRUCTURED_DB) as conn:
        food_count = conn.execute("SELECT COUNT(*) FROM food_items").fetchone()[0]
        nutrient_count = conn.execute("SELECT COUNT(*) FROM food_nutrients_per_100g").fetchone()[0]
        source_count = conn.execute(
            "SELECT COUNT(*) FROM source_documents WHERE source_title LIKE '%USDA%'"
        ).fetchone()[0]
        searchable_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM food_items fi
            WHERE EXISTS (
                SELECT 1 FROM food_aliases fa WHERE fa.food_id = fi.food_id
            )
            """
        ).fetchone()[0]

    if food_count < 100:
        raise AssertionError(f"Structured food DB has too few food_items: {food_count}")
    if nutrient_count < 100:
        raise AssertionError(f"Structured food DB has too few nutrient rows: {nutrient_count}")
    if source_count < 1:
        raise AssertionError("Structured food DB is missing a USDA source document")
    if searchable_count < 100:
        raise AssertionError(f"Structured food DB has too few searchable aliases: {searchable_count}")


def validate_meal_nutrition_analyzer_if_present() -> None:
    from plugins_func.functions.analyze_meal_nutrition import analyze_items, parse_meal_text
    from core.providers.memory.clinical_ltm.health_profile import HealthProfileStore

    meal = "\u4e24\u4e2a\u9e21\u86cb\u3001\u4e00\u676f\u725b\u5976\u3001\u4e24\u7247\u767d\u9762\u5305"
    parsed = parse_meal_text(meal)
    if len(parsed) != 3:
        raise AssertionError(f"Meal analyzer parsed wrong item count: {parsed}")
    if [item.unit for item in parsed] != ["\u4e2a", "\u676f", "\u7247"]:
        raise AssertionError(f"Meal analyzer parsed wrong units: {parsed}")
    compact_voice_meal = "\u4e24\u4e2a\u9e21\u86cb\u4e00\u676f\u725b\u5976\u4e24\u7247\u767d\u9762\u5305"
    compact_parsed = parse_meal_text(compact_voice_meal)
    if len(compact_parsed) != 3:
        raise AssertionError(f"Meal analyzer should parse voice-style compact meal text: {compact_parsed}")

    if not STRUCTURED_DB.exists():
        return

    resolved, unresolved = analyze_items(STRUCTURED_DB, parsed)
    if unresolved:
        names = [item.food_name for item in unresolved]
        raise AssertionError(f"Meal analyzer failed to resolve foods: {names}")
    if len(resolved) != 3:
        raise AssertionError(f"Meal analyzer resolved wrong item count: {len(resolved)}")
    total_energy = sum(item.nutrient_totals.get("energy_kcal", 0.0) for item in resolved)
    if not 200 <= total_energy <= 1000:
        raise AssertionError(f"Meal analyzer total energy looks implausible: {total_energy}")
    bread_item = resolved[2]
    if str(bread_item.food["canonical_name"]).lower().startswith("flour,"):
        raise AssertionError("Meal analyzer matched white bread to flour instead of prepared bread")
    intake_db = PROJECT_ROOT / "tmp" / "validate_nutrition_intake.db"
    if intake_db.exists():
        intake_db.unlink()
    store = HealthProfileStore(intake_db)
    totals = {}
    for item in resolved:
        for key, value in item.nutrient_totals.items():
            totals[key] = totals.get(key, 0.0) + value
    record = store.record_nutrition_intake_sync(
        "validate-user",
        meal_text=meal,
        totals=totals,
        items=[{"food_name": item.parsed.food_name, "grams": item.grams} for item in resolved],
        occurred_at=None,
        meal_label="breakfast",
    )
    if not record.get("inserted"):
        raise AssertionError("Meal analyzer should persist nutrition intake records")
    series = store.get_nutrition_intake_series_sync("validate-user", days=7)
    if not any(item.get("intake_count") == 1 and item.get("energy_kcal", 0) > 0 for item in series):
        raise AssertionError("Nutrition intake daily series should include persisted meal totals")
    try:
        intake_db.unlink()
    except OSError:
        pass


def validate_vision_nutrition_analyzer_if_present() -> None:
    from core.clinical_nutrition.vision_nutrition import VisionNutritionAnalyzer

    class DummyLogger:
        def bind(self, **_kwargs):
            return self

        def warning(self, _message):
            return None

    payload = {
        "is_food_or_drink": True,
        "scene_notes": "\u684c\u4e0a\u6709\u4e00\u676f\u5976\u8336",
        "items": [
            {
                "name": "\u5976\u8336",
                "brand_or_variant": "Bawangchaji",
                "quantity": 1,
                "unit": "\u676f",
                "estimated_ml": 500,
                "confidence": 0.8,
                "risk_tags": ["\u542b\u7cd6\u996e\u6599"],
            }
        ],
        "uncertainties": ["\u672a\u77e5\u52a0\u7cd6\u91cf"],
    }
    profile = {
        "items": [{"category": "disease", "name": "\u0032\u578b\u7cd6\u5c3f\u75c5"}],
        "scalars": {},
    }
    analyzer = VisionNutritionAnalyzer(
        project_root=PROJECT_ROOT,
        config={},
        logger=DummyLogger(),
    )
    result = analyzer.build_response(
        vlm_raw_text=json.dumps(payload, ensure_ascii=False),
        user_question="\u8fd9\u4e2a\u6211\u53ef\u4ee5\u559d\u5417\uff1f",
        health_profile=profile,
        food_db_path=STRUCTURED_DB,
        rules_path=RULES_PATH,
    )
    if not result.structured.get("items"):
        raise AssertionError("Vision nutrition analyzer failed to parse structured VLM JSON")
    if "\u7cd6\u5c3f\u75c5" not in result.response_text:
        raise AssertionError("Vision nutrition analyzer missed diabetes high-sugar drink notice")
    if "Bawangchaji" not in result.response_text:
        raise AssertionError("Vision nutrition analyzer should preserve visual brand context")
    if "unresolved_items" not in result.nutrition:
        raise AssertionError("Vision nutrition analyzer should return nutrition payload metadata")


def validate_knowledge_ingestion_service() -> None:
    from core.clinical_nutrition.knowledge_ingestion import KnowledgeIngestionService

    class DummyLogger:
        def bind(self, **_kwargs):
            return self

        def warning(self, _message):
            return None

    tmp_dir = PROJECT_ROOT / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    source_path = tmp_dir / "validate_knowledge_ingestion.txt"
    source_path.write_text(
        "\n".join(
            [
                "\u0032\u578b\u7cd6\u5c3f\u75c5\u533b\u5b66\u8425\u517b\u6cbb\u7597\u8981\u70b9",
                "\u542b\u7cd6\u996e\u6599\u53ef\u5bfc\u81f4\u9910\u540e\u8840\u7cd6\u660e\u663e\u5347\u9ad8\u3002",
                "\u5982\u679c\u6b63\u5728\u4f7f\u7528\u534e\u6cd5\u6797\uff0c\u7ef4\u751f\u7d20K\u6444\u5165\u5e94\u4fdd\u6301\u7a33\u5b9a\u3002",
            ]
        ),
        encoding="utf-8",
    )
    service = KnowledgeIngestionService(
        project_root=PROJECT_ROOT,
        config={"knowledge_ingestion": {"enabled": False}},
        logger=DummyLogger(),
    )
    draft = service.create_draft(
        source_path=source_path,
        title="\u9a8c\u8bc1\u5165\u5e93\u8349\u6848",
        topic="\u7cd6\u5c3f\u75c5",
    )
    draft_id = str(draft.get("draft_id") or "")
    try:
        if draft.get("status") not in {"draft_fallback", "extracted", "reviewed"}:
            raise AssertionError(f"Knowledge ingestion draft status mismatch: {draft.get('status')}")
        if draft.get("llm_used") is not False:
            raise AssertionError("Knowledge ingestion fallback draft should report llm_used=False")
        if not draft.get("wiki_markdown", "").startswith("---"):
            raise AssertionError("Knowledge ingestion draft should produce LLMWiki frontmatter")
        if draft.get("ingestion_mode") == "wiki_compiler_v2":
            if not draft.get("wiki_pages"):
                raise AssertionError("Wiki compiler v2 draft should include wiki_pages")
            if not draft.get("coverage_report", {}).get("total_pages"):
                raise AssertionError("Wiki compiler v2 draft should include coverage_report")
            if "llm_review" not in draft:
                raise AssertionError("Wiki compiler v2 draft should include llm_review")
        if "rules" not in (draft.get("rules_draft") or {}):
            raise AssertionError("Knowledge ingestion draft should include rule draft payload")
        if not service.get_draft(draft_id):
            raise AssertionError("Knowledge ingestion draft should be reloadable from disk")
    finally:
        if draft_id:
            shutil.rmtree(service.draft_root / draft_id, ignore_errors=True)
        try:
            source_path.unlink()
        except OSError:
            pass


def validate_health_profile_store() -> None:
    from core.providers.memory.clinical_ltm.health_profile import (
        HealthProfileStore,
        analyze_blood_glucose_readings,
        extract_health_profile_update,
        format_health_profile_context,
    )

    db_path = PROJECT_ROOT / "tmp" / "validate_health_profile.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    text = (
        "\u621145\u5c81\uff0c\u7537\uff0c\u8eab\u9ad8170\u5398\u7c73\uff0c"
        "\u4f53\u91cd72\u516c\u65a4\uff0c\u67092\u578b\u7cd6\u5c3f\u75c5\u548c\u9ad8\u8840\u538b\uff0c"
        "\u6b63\u5728\u5403\u4e8c\u7532\u53cc\u80cd\u3001\u785d\u82ef\u5730\u5e73\uff0c"
        "\u5bf9\u82b1\u751f\u8fc7\u654f\u3002\u7a7a\u8179\u8840\u7cd67.2\uff0c"
        "\u7cd6\u5316\u8840\u7ea2\u86cb\u767d7.1%\uff0ceGFR 65\u3002"
        "\u76ee\u6807\u662f\u63a7\u7cd6\u51cf\u91cd\uff0c\u6bcf\u9910\u78b3\u6c34\u63a7\u5236\u572845\u514b\u3002"
    )
    store = HealthProfileStore(db_path)
    update = extract_health_profile_update(text)
    if update.scalars.get("age_years") != 45.0:
        raise AssertionError("Health profile extractor missed age")
    for phrase in [
        "\u6211\u7684\u4f53\u91cd\u662f60\u5343\u514b",
        "\u6211\u4f53\u91cd60\u516c\u65a4",
        "\u8eab\u9ad8\u662f170\u5398\u7c73\uff0c\u4f53\u91cd\u662f60\u516c\u65a4",
    ]:
        phrase_update = extract_health_profile_update(phrase)
        if phrase_update.scalars.get("weight_kg") != 60.0:
            raise AssertionError(f"Health profile extractor missed weight phrase: {phrase!r}")
    expected_items = {
        ("disease", "\u0032\u578b\u7cd6\u5c3f\u75c5"),
        ("disease", "\u9ad8\u8840\u538b"),
        ("medication", "\u4e8c\u7532\u53cc\u80cd"),
        ("medication", "\u785d\u82ef\u5730\u5e73"),
        ("allergy", "\u82b1\u751f"),
        ("renal_function", "eGFR"),
        ("glucose_metric", "\u7a7a\u8179\u8840\u7cd6"),
        ("glucose_metric", "\u7cd6\u5316\u8840\u7ea2\u86cb\u767d"),
    }
    actual_items = {(item.category, item.name) for item in update.items}
    missing = expected_items - actual_items
    if missing:
        raise AssertionError(f"Health profile extractor missed items: {sorted(missing)}")
    glucose_update = extract_health_profile_update(
        "\u4eca\u5929\u65e9\u4e0a8\u70b9\u7a7a\u8179\u8840\u7cd67.2\uff0c"
        "\u5348\u9910\u540e\u4e24\u5c0f\u65f6\u8840\u7cd610.5\uff0c"
        "\u665a\u4e0a\u7761\u524d\u8840\u7cd63.8"
    )
    if len(glucose_update.glucose_readings) != 3:
        raise AssertionError(
            f"Health profile extractor missed blood glucose readings: {glucose_update.glucose_readings}"
        )
    reading_types = {reading.measurement_type for reading in glucose_update.glucose_readings}
    if {"fasting", "postprandial_2h", "bedtime"} - reading_types:
        raise AssertionError(f"Blood glucose reading types are wrong: {reading_types}")
    voice_style_update = extract_health_profile_update(
        "\u4eca\u5929\u65e9\u4e0a8\u70b9\u7a7a\u8179\u8840\u7cd6\u4e03\u70b9\u4e8c\uff0c"
        "\u5348\u9910\u540e\u4e24\u5c0f\u65f6\u8840\u7cd6\u5341\u70b9\u4e94"
    )
    voice_values = [reading.value_mmol_l for reading in voice_style_update.glucose_readings]
    if voice_values != [7.2, 10.5]:
        raise AssertionError(f"Blood glucose extractor missed Chinese spoken numbers: {voice_values}")

    store.apply_update_sync("validate-user", update)
    store.apply_update_sync("validate-user", glucose_update)
    profile = store.get_profile_sync("validate-user")
    if profile["scalars"].get("bmi") != 24.9:
        raise AssertionError(f"Health profile BMI mismatch: {profile['scalars'].get('bmi')}")
    if len(profile.get("glucose_readings") or []) < 3:
        raise AssertionError("Blood glucose readings should be persisted in health profile DB")
    glucose_analysis = analyze_blood_glucose_readings(profile.get("glucose_readings") or [])
    if not any(item.get("code") == "low" for item in glucose_analysis.get("alerts") or []):
        raise AssertionError("Blood glucose time-series analysis should flag low glucose")
    conflict_update = extract_health_profile_update("\u6211\u7684\u4f53\u91cd\u662f60\u5343\u514b")
    conflict_stats = store.apply_update_sync("validate-user", conflict_update)
    if conflict_stats.get("review_count", 0) < 1:
        raise AssertionError("Health profile scalar conflict should create a pending review item")
    conflict_profile = store.get_profile_sync("validate-user")
    if conflict_profile["scalars"].get("weight_kg") != 72.0:
        raise AssertionError("Health profile scalar conflict should not overwrite trusted current value")
    pending_reviews = conflict_profile.get("review_items") or []
    weight_review = next((item for item in pending_reviews if item.get("name") == "weight_kg"), None)
    if not weight_review:
        raise AssertionError("Health profile conflict review should include weight_kg")
    store.resolve_review_item_sync(weight_review["review_id"], "accept")
    accepted_profile = store.get_profile_sync("validate-user")
    if accepted_profile["scalars"].get("weight_kg") != 60.0:
        raise AssertionError("Accepting a health profile review should apply the proposed scalar value")
    allergy_update = extract_health_profile_update("\u6211\u6ca1\u6709\u8fc7\u654f")
    allergy_stats = store.apply_update_sync("validate-user", allergy_update)
    if allergy_stats.get("review_count", 0) < 1:
        raise AssertionError("Health profile allergy negation should create a pending review when allergy exists")
    context = format_health_profile_context(profile)
    for required in ["Health Profile", "\u0032\u578b\u7cd6\u5c3f\u75c5", "\u82b1\u751f", "eGFR", "\u6700\u8fd1\u8840\u7cd6\u8bb0\u5f55"]:
        if required not in context:
            raise AssertionError(f"Health profile context missing: {required}")

    try:
        db_path.unlink()
    except OSError:
        pass


def validate_nutrition_targets() -> None:
    from core.clinical_nutrition.nutrition_targets import estimate_daily_nutrition_targets

    profile = {
        "scalars": {
            "sex": "female",
            "age_years": 45,
            "height_cm": 165,
            "weight_kg": 60,
            "activity_level": "轻体力",
        },
        "items": [{"category": "disease", "name": "\u0032\u578b\u7cd6\u5c3f\u75c5"}],
    }
    targets = estimate_daily_nutrition_targets(profile)
    if not targets.get("available"):
        raise AssertionError("Nutrition target estimator should work when weight is present")
    effective = targets.get("effective") or {}
    if effective.get("energy_kcal", {}).get("value", 0) <= 0:
        raise AssertionError("Nutrition target estimator should produce an energy target")
    if not targets.get("flags", {}).get("diabetes_adjusted_carbohydrate_ratio"):
        raise AssertionError("Nutrition target estimator should detect diabetes profile context")


def validate_device_identity() -> None:
    from core.utils.device_identity import normalize_device_user_id

    if normalize_device_user_id("3C:0F:02:D9:24:E0") != "3c0f02d924e0":
        raise AssertionError("Device identity should normalize uppercase MAC with colons")
    if normalize_device_user_id("3c0f02d924e0") != "3c0f02d924e0":
        raise AssertionError("Device identity should keep compact device IDs stable")
    if normalize_device_user_id("") != "":
        raise AssertionError("Empty device identity should stay empty")


def validate_voice_health_profile_confirmation() -> None:
    text = CONNECTION_PY.read_text(encoding="utf-8")
    required_markers = [
        "_maybe_resolve_health_profile_voice_review",
        "_maybe_reply_health_profile_voice_review",
        "_reply_health_profile_confirmation",
        "store.resolve_review_item",
        "store.list_review_items",
        'resolved_by="voice_user"',
        "确认更新",
        "忽略",
    ]
    for marker in required_markers:
        if marker not in text:
            raise AssertionError(f"Voice health profile confirmation is missing marker: {marker}")


def validate_clinical_console() -> None:
    from core.api.clinical_console_handler import ClinicalConsoleHandler, _safe_filename

    if _safe_filename("../bad/path.pdf") != "path.pdf":
        raise AssertionError("Clinical console upload filename sanitization is unsafe")
    if ClinicalConsoleHandler is None:
        raise AssertionError("Clinical console handler import failed")

    if not CONSOLE_HTML.exists():
        print("clinical console page is missing; skipped frontend marker validation")
        return

    html_text = CONSOLE_HTML.read_text(encoding="utf-8")
    js_text = CONSOLE_APP_JS.read_text(encoding="utf-8") if CONSOLE_APP_JS.exists() else ""
    text = f"{html_text}\n{js_text}"
    required_markers = [
        "/console/api/summary",
        "/console/api/users",
        "/console/api/agent-settings",
        "/console/api/knowledge/upload",
        "/console/api/profile",
        "/console/api/memory",
        "/console/api/rules",
        "/console/api/food",
        "/console/api/model-config",
        '/console/api/history',
        "/console/api/knowledge/ingest",
        "/console/api/rag/documents",
        "/console/api/rag/search",
        "short_term_memory.max_chars",
        "historyView",
        "buildHistoryBuckets",
        "renderModelSettingsPage",
        "ragDocuments",
        "data-ingest-file",
        "data-index-rag",
        "renderRagDocumentReview",
        "glucoseReadings",
        "formatGlucoseValue",
        "/console/api/profile-review",
        "data-resolve-review",
        "memory_prompts.long_term_extraction_system_prompt",
        "memory_prompts.short_term_summary_system_prompt",
        "prompt_template_content",
        "运行基础 Prompt 模板",
        "Prompt 结构",
        "nutritionTargets",
        "nutrition_intake_series",
        "renderNutritionIntakeChart",
        "每日摄入曲线",
        "/console/api/meal/analyze",
        "meal-analysis-form",
        "整餐营养计算",
        "写入当前设备用户的每日摄入曲线",
    ]
    for marker in required_markers:
        if marker not in text:
            raise AssertionError(f"Clinical console page is missing marker: {marker}")


def _read_frontmatter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    metadata = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return metadata
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return {}


if __name__ == "__main__":
    main()
