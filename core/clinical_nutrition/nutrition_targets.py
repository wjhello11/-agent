from __future__ import annotations

from typing import Any


SOURCE_REFERENCES = [
    {
        "title": "NASEM Dietary Reference Intakes: macronutrient AMDR and protein RDA",
        "url": "https://nap.nationalacademies.org/read/10490/chapter/13",
    },
    {
        "title": "NIH ODS nutrient recommendations: protein RDA is 0.8 g/kg for adults",
        "url": "https://ods.od.nih.gov/HealthInformation/nutrientrecommendations.aspx",
    },
    {
        "title": "Mifflin-St Jeor resting metabolic rate equation",
        "url": "https://www.ncbi.nlm.nih.gov/books/NBK278991/table/diet-treatment-obes.table12est/",
    },
]


def estimate_daily_nutrition_targets(profile: dict[str, Any]) -> dict[str, Any]:
    """Estimate conservative daily macro targets from the structured profile.

    Explicit user/clinician targets in the profile always win. Estimates are a
    planning aid for console display, not a prescription.
    """

    scalars = (profile or {}).get("scalars") or {}
    items = (profile or {}).get("items") or []
    sex = _normalize_sex(scalars.get("sex"))
    age_years = _number(scalars.get("age_years"))
    height_cm = _number(scalars.get("height_cm"))
    weight_kg = _number(scalars.get("weight_kg"))
    activity_level = str(scalars.get("activity_level") or "").strip()
    nutrition_goal = str(scalars.get("nutrition_goal") or "").strip()

    if not weight_kg or weight_kg <= 0:
        return {
            "available": False,
            "reason": "缺少体重，暂时无法按体重估算每日目标。",
            "sources": SOURCE_REFERENCES,
        }

    has_diabetes = _has_item(items, "disease", ("糖尿病", "diabetes", "t2dm", "type 2"))
    has_renal_risk = _has_item(items, "disease", ("肾", "ckd", "kidney")) or _has_item(
        items,
        "renal_function",
        ("egfr", "肾"),
    )

    activity = _activity_bucket(activity_level)
    energy = _estimate_energy_kcal(
        sex=sex,
        age_years=age_years,
        height_cm=height_cm,
        weight_kg=weight_kg,
        activity=activity,
    )
    if _contains_any(nutrition_goal, ("减重", "减脂", "控制体重", "控糖减重")):
        energy *= 0.9
    elif _contains_any(nutrition_goal, ("增肌", "增重")):
        energy *= 1.05
    energy = _round_to_nearest(energy, 50)

    carb_ratio = 0.45 if has_diabetes else 0.50
    fat_ratio = 0.30
    protein_ratio = 0.20

    protein_g = energy * protein_ratio / 4.0
    protein_rda_g = weight_kg * 0.8
    if has_renal_risk:
        protein_g = protein_rda_g
    else:
        protein_g = max(protein_g, protein_rda_g)
    protein_g = min(protein_g, energy * 0.35 / 4.0)

    carbohydrate_g = energy * carb_ratio / 4.0
    remaining_for_fat = max(energy - protein_g * 4.0 - carbohydrate_g * 4.0, energy * fat_ratio)
    fat_g = remaining_for_fat / 9.0

    estimated = {
        "energy_kcal": _target_value(energy, "kcal", "estimated"),
        "carbohydrate_g_per_day": _target_value(_round_to_nearest(carbohydrate_g, 5), "g", "estimated"),
        "carbohydrate_g_per_meal": _target_value(_round_to_nearest(carbohydrate_g / 3.0, 5), "g", "estimated"),
        "protein_g_per_day": _target_value(_round_to_nearest(protein_g, 5), "g", "estimated"),
        "fat_g_per_day": _target_value(_round_to_nearest(fat_g, 5), "g", "estimated"),
    }

    explicit = {
        "energy_kcal": _explicit_target(scalars, "target_energy_kcal", "kcal"),
        "carbohydrate_g_per_meal": _explicit_target(scalars, "target_carbohydrate_g_per_meal", "g"),
        "protein_g_per_day": _explicit_target(scalars, "target_protein_g_per_day", "g"),
        "fat_g_per_day": _explicit_target(scalars, "target_fat_g_per_day", "g"),
    }
    if explicit["carbohydrate_g_per_meal"]:
        explicit["carbohydrate_g_per_day"] = _target_value(
            explicit["carbohydrate_g_per_meal"]["value"] * 3.0,
            "g",
            "profile",
        )
    else:
        explicit["carbohydrate_g_per_day"] = None

    effective = {
        key: explicit.get(key) or estimated.get(key)
        for key in estimated
    }

    return {
        "available": True,
        "method": "mifflin_st_jeor" if _can_use_mifflin(sex, age_years, height_cm, weight_kg) else "sex_weight_kcal_per_kg",
        "activity": activity,
        "flags": {
            "diabetes_adjusted_carbohydrate_ratio": has_diabetes,
            "renal_protein_conservative": has_renal_risk,
            "goal_adjustment": nutrition_goal or "",
        },
        "estimated": estimated,
        "explicit": {key: value for key, value in explicit.items() if value},
        "effective": effective,
        "notes": _target_notes(has_diabetes, has_renal_risk),
        "sources": SOURCE_REFERENCES,
    }


def _estimate_energy_kcal(
    *,
    sex: str,
    age_years: float | None,
    height_cm: float | None,
    weight_kg: float,
    activity: str,
) -> float:
    if _can_use_mifflin(sex, age_years, height_cm, weight_kg):
        sex_constant = 5 if sex == "male" else -161
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age_years + sex_constant
        return bmr * _activity_factor(activity)
    kcal_per_kg = {
        "sedentary": {"male": 27, "female": 25, "unknown": 26},
        "light": {"male": 30, "female": 27, "unknown": 28},
        "moderate": {"male": 33, "female": 30, "unknown": 30},
        "active": {"male": 35, "female": 32, "unknown": 33},
    }
    return weight_kg * kcal_per_kg.get(activity, kcal_per_kg["light"]).get(sex or "unknown", 28)


def _can_use_mifflin(
    sex: str,
    age_years: float | None,
    height_cm: float | None,
    weight_kg: float | None,
) -> bool:
    return bool(
        sex in {"male", "female"}
        and age_years
        and height_cm
        and weight_kg
        and 10 <= age_years <= 100
        and 120 <= height_cm <= 230
        and 25 <= weight_kg <= 250
    )


def _target_notes(has_diabetes: bool, has_renal_risk: bool) -> list[str]:
    notes = [
        "估算值用于控制台参考；真实目标应结合年龄、身高、活动量、疾病状态和医生/营养师建议调整。",
        "蛋白质参考成人 RDA 0.8 g/kg，并让宏量营养素比例落在 DRI AMDR 范围内。",
    ]
    if has_diabetes:
        notes.append("档案提示糖尿病，估算时把碳水比例取 AMDR 下半区；具体每餐碳水仍应个体化。")
    if has_renal_risk:
        notes.append("档案提示肾功能风险，蛋白质估算采用更保守的 0.8 g/kg；肾病分期需医生确认。")
    return notes


def _activity_bucket(value: str) -> str:
    normalized = str(value or "").lower()
    if _contains_any(normalized, ("久坐", "很少", "sedentary", "低")):
        return "sedentary"
    if _contains_any(normalized, ("中等", "moderate", "每周3", "每周 3", "运动")):
        return "moderate"
    if _contains_any(normalized, ("高", "重体力", "active", "每天", "大量")):
        return "active"
    return "light"


def _activity_factor(activity: str) -> float:
    return {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
    }.get(activity, 1.375)


def _normalize_sex(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"male", "m", "男", "男性"}:
        return "male"
    if text in {"female", "f", "女", "女性"}:
        return "female"
    return "unknown"


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _explicit_target(scalars: dict[str, Any], key: str, unit: str) -> dict[str, Any] | None:
    value = _number(scalars.get(key))
    if value is None:
        return None
    return _target_value(value, unit, "profile")


def _target_value(value: float, unit: str, source: str) -> dict[str, Any]:
    return {
        "value": round(float(value), 1),
        "unit": unit,
        "source": source,
    }


def _round_to_nearest(value: float, step: int) -> float:
    return round(float(value) / step) * step


def _has_item(items: list[dict[str, Any]], category: str, tokens: tuple[str, ...]) -> bool:
    for item in items or []:
        if str(item.get("category") or "") != category:
            continue
        haystack = " ".join(
            [
                str(item.get("name") or ""),
                str(item.get("evidence") or ""),
                str(item.get("value") or ""),
            ]
        ).lower()
        if any(token.lower() in haystack for token in tokens):
            return True
    return False


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    lowered = str(text or "").lower()
    return any(token.lower() in lowered for token in tokens)
