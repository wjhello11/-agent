"""
Pydantic schemas for structured clinical nutrition knowledge extraction.

Each schema maps to a database table in clinical_knowledge.db.
LLM output must pass schema validation before入库.
Failed validation goes to needs_review table.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ============================================================
# Ingestion Plan (document profiling output)
# ============================================================

class ChapterOutline(BaseModel):
    title: str
    page_start: int
    page_end: int
    section_path: str = ""


class IngestionBlock(BaseModel):
    block_id: str
    block_type: str  # narrative_guideline | recommendation | diagnostic_threshold | ...
    section_path: str = ""
    page_start: int
    page_end: int
    raw_text_hash: str = ""
    confidence: float = 0.7
    should_store_in: str = "rag"  # wiki | rag | structured | skip | needs_review
    skip_reason: str = ""
    extraction_status: str = "pending"  # pending | done | failed | needs_review


class DocumentQualityReport(BaseModel):
    source_document: str = ""
    page_count: int = 0
    readable_page_count: int = 0
    empty_page_count: int = 0
    low_text_page_count: int = 0
    average_chars_per_page: float = 0.0
    likely_scanned: bool = False
    needs_ocr: bool = False
    quality_status: str = "ok"  # ok | needs_manual_check | needs_ocr
    issues: list[str] = Field(default_factory=list)


class DocumentProfile(BaseModel):
    document_type: str = "other"
    knowledge_types: list[str] = Field(default_factory=list)
    source_document: str = ""
    page_count: int = 0
    quality_status: str = "ok"
    suggested_status: str = "profiled"
    confidence: float = 0.7
    summary: str = ""


class IngestionPlan(BaseModel):
    document_type: str = "other"
    knowledge_types: list[str] = Field(default_factory=list)
    chapter_outline: list[ChapterOutline] = Field(default_factory=list)
    blocks: list[IngestionBlock] = Field(default_factory=list)
    total_pages: int = 0
    source_document: str = ""


# ============================================================
# Structured extraction schemas (maps to DB tables)
# ============================================================

class GuideTableRow(BaseModel):
    label: str = ""
    columns: dict[str, str] = Field(default_factory=dict)
    raw_text: str = ""


class GuideTable(BaseModel):
    table_label: str = ""
    title: str
    table_type: str = "generic"  # generic | diagnostic_threshold | nutrition_target | food_exchange | activity_met
    page_start: int
    page_end: int
    raw_text: str = ""
    rows: list[GuideTableRow] = Field(default_factory=list)
    confidence: float = 0.8
    source_document: str = ""
    source_pages: list[int] = Field(default_factory=list)


class ExchangePortion(BaseModel):
    food_name: str
    exchange_group: str = ""
    serving_amount: str = ""
    energy_kcal: float | None = None
    carbohydrate_g: float | None = None
    protein_g: float | None = None
    fat_g: float | None = None
    page_start: int = 0
    page_end: int = 0
    raw_text: str = ""
    confidence: float = 0.75
    source_document: str = ""
    source_pages: list[int] = Field(default_factory=list)


class RecipeIngredient(BaseModel):
    ingredient_name: str
    amount: float | None = None
    unit: str = ""
    is_medicinal: bool = False
    raw_text: str = ""


class RecipeDish(BaseModel):
    dish_name: str
    raw_text: str = ""
    ingredients: list[RecipeIngredient] = Field(default_factory=list)


class RecipeMeal(BaseModel):
    meal_type: str  # 早餐 | 中餐 | 午餐 | 晚餐 | 加餐
    dishes: list[RecipeDish] = Field(default_factory=list)


class RecipePlan(BaseModel):
    title: str
    season: str = ""
    plan_index: int | None = None
    energy_kcal: float | None = None
    protein_g: float | None = None
    carbohydrate_g: float | None = None
    fat_g: float | None = None
    protein_pct: float | None = None
    carbohydrate_pct: float | None = None
    fat_pct: float | None = None
    meals: list[RecipeMeal] = Field(default_factory=list)
    page_start: int = 0
    page_end: int = 0
    raw_text: str = ""
    confidence: float = 0.8
    source_document: str = ""
    source_pages: list[int] = Field(default_factory=list)


class TherapeuticRecipe(BaseModel):
    syndrome: str = ""
    title: str
    ingredients: list[str] = Field(default_factory=list)
    method: str = ""
    usage: str = ""
    cautions: str = ""
    page_start: int = 0
    page_end: int = 0
    raw_text: str = ""
    confidence: float = 0.8
    source_document: str = ""
    source_pages: list[int] = Field(default_factory=list)


class ActivityMET(BaseModel):
    category: str = ""
    activity_name: str
    met: float
    intensity: str = ""
    page_start: int = 0
    page_end: int = 0
    raw_text: str = ""
    confidence: float = 0.85
    source_document: str = ""
    source_pages: list[int] = Field(default_factory=list)


# ============================================================
# New structured types (not in current DB, added by this upgrade)
# ============================================================

class DiagnosticThreshold(BaseModel):
    indicator: str  # e.g. "BMI", "腰围", "空腹血糖"
    threshold: str  # e.g. "≥28.0", "≥90cm", "≥7.0 mmol/L"
    unit: str = ""
    population: str = ""  # e.g. "成人", "男性"
    context: str = ""
    page_start: int = 0
    page_end: int = 0
    raw_text: str = ""
    confidence: float = 0.8
    source_document: str = ""
    source_pages: list[int] = Field(default_factory=list)


class NutritionTarget(BaseModel):
    nutrient: str  # e.g. "碳水化合物", "蛋白质", "膳食纤维"
    target_value: str  # e.g. "45%-60%", "15%-20%", "25-30g/d"
    population: str = ""
    context: str = ""
    page_start: int = 0
    page_end: int = 0
    raw_text: str = ""
    confidence: float = 0.8
    source_document: str = ""
    source_pages: list[int] = Field(default_factory=list)


class SafetyRuleCandidate(BaseModel):
    trigger_condition: str
    risk_description: str
    safety_recommendation: str
    severity: str = "warn"  # warn | danger | critical
    page_start: int = 0
    page_end: int = 0
    raw_text: str = ""
    confidence: float = 0.7
    source_document: str = ""
    source_pages: list[int] = Field(default_factory=list)


class GenericTableRow(BaseModel):
    table_title: str
    columns: dict[str, str] = Field(default_factory=dict)
    row_index: int = 0
    page_start: int = 0
    page_end: int = 0
    raw_text: str = ""
    source_document: str = ""
    source_pages: list[int] = Field(default_factory=list)


# ============================================================
# Needs Review (failed validation)
# ============================================================

class NeedsReviewItem(BaseModel):
    review_id: str = ""
    document_id: str = ""
    block_id: str = ""
    block_type: str = ""
    section_path: str = ""
    page_start: int = 0
    page_end: int = 0
    raw_text: str = ""
    llm_output: str = ""
    schema_errors: str = ""
    confidence: float = 0.5
    review_status: str = "pending"  # pending | resolved | discarded
    reviewer_notes: str = ""


# ============================================================
# Block type → Schema mapping
# ============================================================

BLOCK_TYPE_SCHEMA_MAP: dict[str, type[BaseModel]] = {
    "guide_table": GuideTable,
    "generic_table": GuideTable,
    "diagnostic_threshold": DiagnosticThreshold,
    "nutrition_target": NutritionTarget,
    "food_exchange_portion": ExchangePortion,
    "recipe_plan": RecipePlan,
    "therapeutic_recipe": TherapeuticRecipe,
    "activity_met": ActivityMET,
    "safety_rule_candidate": SafetyRuleCandidate,
    "contraindication": SafetyRuleCandidate,
}


def get_schema_for_block_type(block_type: str) -> type[BaseModel] | None:
    return BLOCK_TYPE_SCHEMA_MAP.get(block_type)
