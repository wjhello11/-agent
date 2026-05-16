# Structured Food Database

This layer stores objective facts that should not be generated from memory by
the LLM.

## Current Open Data Source

The structured layer currently supports two import pipelines:

1. **User-provided Chinese food composition Excel**
   This is preferred for Chinese user queries and local food names.
2. **USDA FoodData Central Foundation Foods**
   This gives us a legally usable baseline for common raw/minimally processed
   foods and selected prepared foods.

Default local files:

- Chinese Excel: `data/raw/china_food_composition/china_food_composition.xlsx`
- Raw download: `data/raw/usda/FoodData_Central_foundation_food_csv_2025-12-18.zip`
- SQLite database: `data/clinical_foods.db`

Import command:

```powershell
python scripts\import_china_food_composition_excel.py
python scripts\import_usda_foundation_foods.py
```

The importer writes:

- `food_items`
- `food_aliases`
- `food_nutrients_per_100g`
- `allergen_flags` using conservative description-based heuristics
- `portion_units`
- `food_risk_tags` for sodium, potassium, phosphorus and grapefruit-related cautions

The runtime tools read `data/clinical_foods.db`:

- `search_food_nutrition`: returns objective nutrient values per 100 g.
- `analyze_meal_nutrition`: parses a meal description, converts portions to grams,
  summarizes nutrients, and compares the result with optional meal targets.

Example meal analysis input:

```text
两个鸡蛋、一杯牛奶、两片白面包
```

The meal analyzer first uses `portion_units` when a database portion is available.
If a local Chinese food item has no portion record yet, it falls back to explicit,
labeled common estimates such as one egg as about 50 g, one cup of milk as about
250 g, and one bread slice as about 30 g.

## Structured Health Profile

The long-term memory layer also maintains a stable health profile database:

- SQLite database: `data/clinical_health_profile.db`
- Runtime context: injected into the `<memory>` block before each LLM call
- Tools:
  - `get_health_profile`
  - `update_health_profile`

The profile stores scalar fields such as age, sex, height, weight, BMI, activity
level and nutrition targets, plus multi-value items such as diseases,
medications, allergies, glucose metrics and renal function. This is separate
from long-term memory: memory remembers conversation details, while the health
profile gives the agent a stable clinical snapshot for repeated nutrition
decisions.

Identity model for the current XiaoZhi deployment is intentionally simple:
one device is one user. The server derives the profile user ID from the incoming
`device-id` header and normalizes MAC-like values such as `3C:0F:02:D9:24:E0`
to `3c0f02d924e0`.

## Future Source Families

Recommended source families:

- Chinese food composition tables, with legal data access.
- GI/GL references.
- Purine tables.
- Allergen label data.
- Clinician-reviewed portion conversion tables.

Run the schema against SQLite during local development:

```powershell
sqlite3 data/clinical_foods.db ".read knowledge_base/structured/schema.sql"
```
