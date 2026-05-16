from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.clinical_safety.rule_engine import SafetyFinding, SafetyRuleEngine
from plugins_func.functions.analyze_meal_nutrition import (
    NUTRIENT_COLUMNS,
    ParsedMealItem,
    ResolvedMealItem,
    analyze_items,
)


@dataclass
class VisionNutritionResult:
    response_text: str
    structured: dict[str, Any] = field(default_factory=dict)
    nutrition: dict[str, Any] = field(default_factory=dict)
    safety_findings: list[dict[str, Any]] = field(default_factory=list)


class VisionNutritionAnalyzer:
    """Turns VLM food observations into deterministic nutrition and safety context."""

    def __init__(self, *, project_root: Path, config: dict[str, Any], logger):
        self.project_root = Path(project_root)
        self.config = config
        self.logger = logger

    def build_structured_prompt(
        self,
        *,
        user_question: str,
        health_profile_context: str = "",
    ) -> str:
        parts = [
            "你是个性化临床营养师 AI Agent 的视觉结构化模块。",
            "你的任务不是直接给最终建议，而是把图片中的食物或饮料转成严格 JSON，供后端营养计算和安全规则使用。",
            "请只输出一个 JSON 对象，不要输出 Markdown，不要解释。",
            "JSON schema:",
            """
{
  "is_food_or_drink": true,
  "scene_notes": "一句话描述画面",
  "items": [
    {
      "name": "食物或饮料通用名称，例如 奶茶、白米饭、鸡蛋",
      "brand_or_variant": "品牌或具体品类，不确定则为空",
      "quantity": 1,
      "unit": "杯/瓶/份/个/片/g/ml 等，不确定则为空",
      "estimated_grams": null,
      "estimated_ml": null,
      "confidence": 0.0,
      "risk_tags": ["含糖饮料", "高糖", "油炸", "酒精", "花生", "西柚"]
    }
  ],
  "uncertainties": ["无法确认是否加糖"],
  "answer_hint": "给主回答的简短提示"
}
""".strip(),
            "估算规则：看不清份量时给出合理范围内的保守估计；饮料可优先估算 ml；固体食物可估算 g。",
            "风险标签规则：奶茶、含糖茶饮、果汁、可乐等请标记为 含糖饮料；甜点请标记为 高糖；油条炸鸡等请标记为 油炸。",
        ]
        if health_profile_context:
            parts.extend(
                [
                    "以下是当前用户健康档案，只用于帮助识别风险标签，不要在 JSON 外解释：",
                    health_profile_context,
                ]
            )
        parts.append(f"用户问题：{user_question}")
        return "\n".join(parts)

    def build_response(
        self,
        *,
        vlm_raw_text: str,
        user_question: str,
        health_profile: dict[str, Any],
        food_db_path: Path,
        rules_path: Path,
    ) -> VisionNutritionResult:
        structured = _extract_json_object(vlm_raw_text)
        if not structured:
            return VisionNutritionResult(
                response_text=str(vlm_raw_text or "").strip() or "我没有看清楚图片里的食物或饮料。",
                structured={},
                nutrition={},
                safety_findings=[],
            )

        parsed_items = _structured_items_to_parsed_meal_items(structured)
        resolved: list[ResolvedMealItem] = []
        unresolved: list[ParsedMealItem] = []
        if parsed_items and food_db_path.exists():
            try:
                resolved, unresolved = analyze_items(food_db_path, parsed_items)
            except Exception as exc:
                self.logger.bind(tag=__name__).warning(f"视觉营养计算失败: {exc}")
                unresolved = parsed_items

        nutrition = _nutrition_payload(resolved, unresolved)
        safety_findings = self._evaluate_safety(
            structured=structured,
            health_profile=health_profile,
            rules_path=rules_path,
        )
        diabetes_notice = _diabetes_sugar_notice(structured, health_profile)
        response_text = _format_vision_response(
            structured=structured,
            nutrition=nutrition,
            findings=safety_findings,
            diabetes_notice=diabetes_notice,
            user_question=user_question,
        )
        return VisionNutritionResult(
            response_text=response_text,
            structured=structured,
            nutrition=nutrition,
            safety_findings=[_finding_to_dict(item) for item in safety_findings],
        )

    def _evaluate_safety(
        self,
        *,
        structured: dict[str, Any],
        health_profile: dict[str, Any],
        rules_path: Path,
    ) -> list[SafetyFinding]:
        if not rules_path.exists():
            return []
        foods = _food_terms_from_structured(structured)
        items = health_profile.get("items") or []
        context = {
            "foods": foods,
            "diseases": _profile_item_names(items, "disease"),
            "medications": _profile_item_names(items, "medication"),
            "allergies": _profile_item_names(items, "allergy"),
        }
        try:
            return SafetyRuleEngine(rules_path).evaluate(context)
        except Exception as exc:
            self.logger.bind(tag=__name__).warning(f"视觉安全规则评估失败: {exc}")
            return []


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw).strip()
    candidates = [raw]
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _structured_items_to_parsed_meal_items(payload: dict[str, Any]) -> list[ParsedMealItem]:
    parsed: list[ParsedMealItem] = []
    for item in payload.get("items") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("food_name") or "").strip()
        if not name:
            continue
        quantity = _to_float(item.get("quantity"), default=1.0)
        unit = str(item.get("unit") or "").strip() or None
        grams = _to_float(item.get("estimated_grams"), default=None)
        ml = _to_float(item.get("estimated_ml"), default=None)
        if grams is None and ml is not None:
            grams = ml * _density_for_visual_item(name, item)
        raw_text = _raw_item_text(name, quantity, unit, grams, item)
        parsed.append(
            ParsedMealItem(
                raw_text=raw_text,
                food_name=name,
                quantity=quantity,
                unit=unit,
                explicit_grams=grams,
            )
        )
    return parsed


def _raw_item_text(
    name: str,
    quantity: float,
    unit: str | None,
    grams: float | None,
    item: dict[str, Any],
) -> str:
    variant = str(item.get("brand_or_variant") or "").strip()
    label = f"{variant}{name}" if variant and variant not in name else name
    if grams:
        return f"{label}约{grams:g}g"
    if unit:
        return f"{quantity:g}{unit}{label}"
    return label


def _nutrition_payload(
    resolved: list[ResolvedMealItem],
    unresolved: list[ParsedMealItem],
) -> dict[str, Any]:
    totals: dict[str, float] = {}
    resolved_items = []
    for item in resolved:
        for column, value in item.nutrient_totals.items():
            totals[column] = totals.get(column, 0.0) + float(value)
        resolved_items.append(
            {
                "input": item.parsed.raw_text,
                "food_name": item.food["canonical_name"],
                "grams": round(float(item.grams), 1),
                "portion_source": item.portion_source,
                "nutrients": {
                    key: round(float(value), 1)
                    for key, value in item.nutrient_totals.items()
                },
            }
        )
    return {
        "resolved_items": resolved_items,
        "unresolved_items": [item.raw_text for item in unresolved],
        "totals": {key: round(value, 1) for key, value in totals.items()},
    }


def _format_vision_response(
    *,
    structured: dict[str, Any],
    nutrition: dict[str, Any],
    findings: list[SafetyFinding],
    diabetes_notice: str,
    user_question: str,
) -> str:
    items = structured.get("items") or []
    names = [_display_item_name(item) for item in items if isinstance(item, dict)]
    names = [name for name in names if name]
    scene = str(structured.get("scene_notes") or "").strip()
    totals = nutrition.get("totals") or {}
    unresolved = nutrition.get("unresolved_items") or []

    lines: list[str] = []
    if findings and any(item.severity == "block" or item.action == "block" for item in findings):
        lines.append("先拦一下，这里有明确的安全风险。")
    elif diabetes_notice:
        lines.append(diabetes_notice)
    else:
        lines.append("我先按图片做一个营养判断。")

    if names:
        lines.append(f"我看到的主要是：{'、'.join(names[:4])}。")
    elif scene:
        lines.append(f"画面看起来是：{scene}。")

    if totals:
        nutrient_bits = []
        for key, label, unit in NUTRIENT_COLUMNS:
            if key in totals and key in {"energy_kcal", "carbohydrate_g", "protein_g", "fat_g"}:
                nutrient_bits.append(f"{label}约{totals[key]:g}{unit}")
        if nutrient_bits:
            lines.append("按本地营养库粗算，" + "，".join(nutrient_bits) + "。")

    if unresolved:
        lines.append(
            "其中有些项目暂时没在结构化营养库里精确匹配到，所以营养数字会偏保守。"
        )

    for finding in findings[:2]:
        if finding.message:
            lines.append(finding.message)
        if finding.recommendation:
            lines.append(finding.recommendation)

    if diabetes_notice and not any(diabetes_notice in line for line in lines):
        lines.append(diabetes_notice)

    if _looks_like_can_i_have_it(user_question) and not findings and not diabetes_notice:
        lines.append("如果你没有相关过敏、用药冲突或控糖目标，少量一般可以；如果这是正餐的一部分，建议告诉我份量，我再帮你算得更准。")
    elif _looks_like_can_i_have_it(user_question) and diabetes_notice:
        lines.append("更稳的选择是无糖茶、水、无糖豆浆，或者把这杯换成低糖/无糖并控制份量。")

    return "".join(lines)


def _food_terms_from_structured(payload: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for item in payload.get("items") or []:
        if not isinstance(item, dict):
            continue
        for key in ("name", "brand_or_variant", "category"):
            value = str(item.get(key) or "").strip()
            if value:
                terms.append(value)
        for tag in item.get("risk_tags") or []:
            terms.append(str(tag))
    return _dedupe(terms)


def _display_item_name(item: dict[str, Any]) -> str:
    name = str(item.get("name") or "").strip()
    variant = str(item.get("brand_or_variant") or "").strip()
    if variant and name and name not in variant:
        return f"{variant}{name}"
    return variant or name


def _profile_item_names(items: list[dict[str, Any]], category: str) -> list[str]:
    return [
        str(item.get("name") or "").strip()
        for item in items
        if item.get("category") == category and str(item.get("name") or "").strip()
    ]


def _diabetes_sugar_notice(payload: dict[str, Any], profile: dict[str, Any]) -> str:
    items = profile.get("items") or []
    diseases = " ".join(_profile_item_names(items, "disease")).lower()
    scalars = profile.get("scalars") or {}
    has_diabetes = any(token in diseases for token in ["糖尿病", "diabetes"]) or bool(
        scalars.get("target_carbohydrate_g_per_meal")
    )
    if not has_diabetes:
        return ""
    terms = " ".join(_food_terms_from_structured(payload)).lower()
    sugar_tokens = [
        "奶茶",
        "含糖",
        "甜饮",
        "甜品",
        "果汁",
        "可乐",
        "sugar",
        "sweet",
        "milk tea",
        "dessert",
    ]
    if any(token in terms for token in sugar_tokens):
        return "从你的 2 型糖尿病档案看，这类含糖饮料或甜品不建议作为正餐饮品。"
    return ""


def _looks_like_can_i_have_it(question: str) -> bool:
    text = str(question or "")
    return any(token in text for token in ["能喝", "可以喝", "能吃", "可以吃", "适合", "午餐"])


def _finding_to_dict(finding: SafetyFinding) -> dict[str, Any]:
    return {
        "rule_id": finding.rule_id,
        "severity": finding.severity,
        "category": finding.category,
        "action": finding.action,
        "message": finding.message,
        "recommendation": finding.recommendation,
    }


def _to_float(value: Any, default: float | None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _density_for_visual_item(name: str, item: dict[str, Any]) -> float:
    haystack = " ".join(
        [
            name,
            str(item.get("brand_or_variant") or ""),
            " ".join(str(tag) for tag in item.get("risk_tags") or []),
        ]
    ).lower()
    if any(token in haystack for token in ["奶", "milk"]):
        return 1.03
    return 1.0


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result
