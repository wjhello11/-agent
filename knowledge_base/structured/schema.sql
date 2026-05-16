PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS source_documents (
    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_title TEXT NOT NULL,
    source_org TEXT,
    publish_year INTEGER,
    version TEXT,
    source_url TEXT,
    evidence_level TEXT,
    license_note TEXT,
    last_reviewed_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS food_items (
    food_id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT NOT NULL UNIQUE,
    chinese_name TEXT,
    english_name TEXT,
    food_category TEXT NOT NULL,
    processing_level TEXT,
    source_food_id TEXT,
    default_edible_portion_g REAL,
    notes TEXT,
    source_id INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES source_documents(source_id)
);

CREATE TABLE IF NOT EXISTS food_aliases (
    alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
    food_id INTEGER NOT NULL,
    alias TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'zh',
    UNIQUE (food_id, alias),
    FOREIGN KEY (food_id) REFERENCES food_items(food_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS food_nutrients_per_100g (
    nutrient_id INTEGER PRIMARY KEY AUTOINCREMENT,
    food_id INTEGER NOT NULL,
    energy_kcal REAL,
    carbohydrate_g REAL,
    protein_g REAL,
    fat_g REAL,
    dietary_fiber_g REAL,
    sodium_mg REAL,
    potassium_mg REAL,
    phosphorus_mg REAL,
    calcium_mg REAL,
    magnesium_mg REAL,
    iron_mg REAL,
    manganese_mg REAL,
    zinc_mg REAL,
    copper_mg REAL,
    selenium_ug REAL,
    vitamin_a_ug REAL,
    beta_carotene_ug REAL,
    retinol_equivalent_ug REAL,
    thiamin_mg REAL,
    riboflavin_mg REAL,
    niacin_mg REAL,
    vitamin_c_mg REAL,
    vitamin_e_mg REAL,
    cholesterol_mg REAL,
    purine_mg REAL,
    data_quality TEXT NOT NULL DEFAULT 'unreviewed',
    source_id INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (food_id, source_id),
    FOREIGN KEY (food_id) REFERENCES food_items(food_id) ON DELETE CASCADE,
    FOREIGN KEY (source_id) REFERENCES source_documents(source_id)
);

CREATE TABLE IF NOT EXISTS glycemic_values (
    glycemic_id INTEGER PRIMARY KEY AUTOINCREMENT,
    food_id INTEGER NOT NULL,
    gi REAL,
    gl_per_serving REAL,
    serving_g REAL,
    test_method TEXT,
    population_note TEXT,
    data_quality TEXT NOT NULL DEFAULT 'unreviewed',
    source_id INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (food_id) REFERENCES food_items(food_id) ON DELETE CASCADE,
    FOREIGN KEY (source_id) REFERENCES source_documents(source_id)
);

CREATE TABLE IF NOT EXISTS allergen_flags (
    allergen_id INTEGER PRIMARY KEY AUTOINCREMENT,
    food_id INTEGER NOT NULL,
    contains_gluten INTEGER NOT NULL DEFAULT 0 CHECK (contains_gluten IN (0, 1)),
    contains_peanut INTEGER NOT NULL DEFAULT 0 CHECK (contains_peanut IN (0, 1)),
    contains_tree_nut INTEGER NOT NULL DEFAULT 0 CHECK (contains_tree_nut IN (0, 1)),
    contains_crustacean INTEGER NOT NULL DEFAULT 0 CHECK (contains_crustacean IN (0, 1)),
    contains_soy INTEGER NOT NULL DEFAULT 0 CHECK (contains_soy IN (0, 1)),
    contains_dairy INTEGER NOT NULL DEFAULT 0 CHECK (contains_dairy IN (0, 1)),
    contains_egg INTEGER NOT NULL DEFAULT 0 CHECK (contains_egg IN (0, 1)),
    cross_contamination_risk TEXT,
    source_id INTEGER,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (food_id),
    FOREIGN KEY (food_id) REFERENCES food_items(food_id) ON DELETE CASCADE,
    FOREIGN KEY (source_id) REFERENCES source_documents(source_id)
);

CREATE TABLE IF NOT EXISTS portion_units (
    portion_id INTEGER PRIMARY KEY AUTOINCREMENT,
    food_id INTEGER NOT NULL,
    unit_name TEXT NOT NULL,
    grams REAL NOT NULL CHECK (grams > 0),
    confidence TEXT NOT NULL DEFAULT 'estimated',
    source_id INTEGER,
    notes TEXT,
    UNIQUE (food_id, unit_name),
    FOREIGN KEY (food_id) REFERENCES food_items(food_id) ON DELETE CASCADE,
    FOREIGN KEY (source_id) REFERENCES source_documents(source_id)
);

CREATE TABLE IF NOT EXISTS food_risk_tags (
    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    food_id INTEGER NOT NULL,
    risk_tag TEXT NOT NULL,
    applies_to_condition TEXT,
    rationale TEXT,
    source_id INTEGER,
    UNIQUE (food_id, risk_tag, applies_to_condition),
    FOREIGN KEY (food_id) REFERENCES food_items(food_id) ON DELETE CASCADE,
    FOREIGN KEY (source_id) REFERENCES source_documents(source_id)
);

CREATE INDEX IF NOT EXISTS idx_food_items_category ON food_items(food_category);
CREATE INDEX IF NOT EXISTS idx_food_items_source_food_id ON food_items(source_food_id);
CREATE INDEX IF NOT EXISTS idx_food_aliases_alias ON food_aliases(alias);
CREATE INDEX IF NOT EXISTS idx_food_risk_tags_condition ON food_risk_tags(applies_to_condition);
CREATE INDEX IF NOT EXISTS idx_food_nutrients_food ON food_nutrients_per_100g(food_id);
CREATE INDEX IF NOT EXISTS idx_glycemic_values_food ON glycemic_values(food_id);
