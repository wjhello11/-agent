from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rule_engine import SafetyFinding, SafetyRuleEngine


@dataclass(frozen=True)
class SafetyInterception:
    findings: list[SafetyFinding]
    should_block: bool
    response_text: str
    prompt_context: str
    extracted_context: dict[str, Any]


class ClinicalSafetyInterceptor:
    """
    Deterministic clinical safety gate.

    The first production version intentionally uses explicit clinical keywords instead
    of an LLM extractor, because drug-food/allergy contraindications should be stable,
    auditable, and fast enough to run before every LLM call.
    """

    def __init__(self, rules_path: str | Path):
        self.rule_engine = SafetyRuleEngine(rules_path)

    def evaluate(
        self,
        *,
        query: str,
        memory_context: str | None = None,
        dialogue_messages: list[Any] | None = None,
        health_profile: dict[str, Any] | None = None,
    ) -> SafetyInterception:
        extracted_context = extract_clinical_safety_context(
            query=query,
            memory_context=memory_context,
            dialogue_messages=dialogue_messages,
            health_profile=health_profile,
        )
        findings = self.rule_engine.evaluate(extracted_context)
        should_block = any(
            item.severity == "block" or item.action == "block"
            for item in findings
        )
        return SafetyInterception(
            findings=findings,
            should_block=should_block,
            response_text=format_block_response(findings) if should_block else "",
            prompt_context=format_prompt_context(findings),
            extracted_context=extracted_context,
        )


def extract_clinical_safety_context(
    *,
    query: str,
    memory_context: str | None = None,
    dialogue_messages: list[Any] | None = None,
    health_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    query_texts = [query or ""]
    memory_texts = [memory_context or ""]
    for message in dialogue_messages or []:
        role = None
        content = None
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
        if role and str(role).lower() != "user":
            continue
        if content:
            query_texts.append(str(content))

    query_combined = "\n".join(query_texts).lower()
    memory_combined = "\n".join(memory_texts).lower()
    combined = query_combined
    blood_glucose_mmol_l = _extract_blood_glucose_mmol_l(query_combined)
    context: dict[str, Any] = {
        "foods": _extract_aliases(query_combined, FOOD_ALIASES),
        "medications": _extract_aliases(query_combined, MEDICATION_ALIASES),
        "diseases": _extract_aliases(query_combined, DISEASE_ALIASES),
        "allergies": _extract_allergies(query_combined),
        "symptoms": _extract_aliases(query_combined, SYMPTOM_ALIASES),
        "activities": _extract_aliases(query_combined, ACTIVITY_ALIASES),
        "populations": _extract_aliases(query_combined, POPULATION_ALIASES),
    }
    _merge_context(context, _extract_background_context(memory_combined))
    _merge_context(context, _extract_health_profile_context(health_profile))
    if blood_glucose_mmol_l is not None:
        context["blood_glucose_mmol_l"] = round(blood_glucose_mmol_l, 2)
        context["blood_glucose_mg_dl"] = round(blood_glucose_mmol_l * 18.0, 1)
    return context


def format_block_response(findings: list[SafetyFinding]) -> str:
    block_findings = [
        item for item in findings if item.severity == "block" or item.action == "block"
    ]
    if not block_findings:
        return ""

    parts = ["我先帮你拦一下，这里有明确的临床安全风险。"]
    for item in block_findings[:2]:
        if item.message:
            parts.append(item.message)
        if item.recommendation:
            parts.append(item.recommendation)
    parts.append("如果这涉及正在服用的处方药、严重过敏或慢性病治疗，请优先按医生或药师的建议执行。")
    return "".join(parts)


def format_prompt_context(findings: list[SafetyFinding]) -> str:
    if not findings:
        return ""

    lines = [
        "<clinical_safety_rules>",
        "以下为强规则引擎命中的临床安全提示，回答时必须优先遵守，不得弱化或忽略：",
    ]
    for item in findings:
        lines.append(
            f"- {item.rule_id} [{item.severity}/{item.category}]: "
            f"{item.message} 建议：{item.recommendation}"
        )
    lines.append("</clinical_safety_rules>")
    return "\n".join(lines)


def _extract_aliases(text: str, alias_groups: dict[str, list[str]]) -> list[str]:
    matches: list[str] = []
    for canonical, aliases in alias_groups.items():
        if any(alias.lower() in text for alias in aliases):
            matches.append(canonical)
    return _dedupe(matches)


def _merge_context(target: dict[str, Any], extra: dict[str, Any]) -> None:
    for key, value in extra.items():
        if not value:
            continue
        if isinstance(value, list):
            target[key] = _dedupe(list(target.get(key) or []) + list(value))
        elif key not in target or target.get(key) in (None, ""):
            target[key] = value


def _extract_background_context(text: str) -> dict[str, Any]:
    if not text:
        return {}
    return {
        "medications": _extract_aliases(text, MEDICATION_ALIASES),
        "diseases": _extract_aliases(text, DISEASE_ALIASES),
        "allergies": _extract_allergies(text),
        "populations": _extract_aliases(text, POPULATION_ALIASES),
    }


def _extract_health_profile_context(profile: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    result: dict[str, Any] = {}
    items = profile.get("items") or []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("status") not in (None, "", "active"):
            continue
        category = str(item.get("category") or "").lower()
        name = str(item.get("name") or "").strip().lower()
        if not name:
            continue
        if category == "disease":
            result.setdefault("diseases", []).append(name)
        elif category == "medication":
            result.setdefault("medications", []).append(name)
        elif category == "allergy":
            result.setdefault("allergies", []).append(name)
        elif category == "dietary_restriction":
            result.setdefault("foods", []).append(name)

    scalars = profile.get("scalars") or {}
    sex = str(scalars.get("sex") or "").lower()
    age = scalars.get("age_years")
    if sex == "pregnant":
        result.setdefault("populations", []).append("pregnancy")
    try:
        if age is not None and float(age) < 18:
            result.setdefault("populations", []).append("child")
        elif age is not None and float(age) >= 65:
            result.setdefault("populations", []).append("older_adult")
    except (TypeError, ValueError):
        pass
    return {key: _dedupe(value) for key, value in result.items()}


def _extract_allergies(text: str) -> list[str]:
    matches: list[str] = []
    for aliases in ALLERGY_ALIASES.values():
        for alias in aliases:
            normalized_alias = alias.lower()
            if normalized_alias not in text:
                continue
            if _has_allergy_context(text, normalized_alias):
                matches.extend(aliases)
                break
    return _dedupe(matches)


def _extract_blood_glucose_mmol_l(text: str) -> float | None:
    if not any(marker in text for marker in ["血糖", "glucose", "blood sugar"]):
        return None

    patterns = [
        r"(?:血糖|glucose|blood sugar)[^\d]{0,12}(\d+(?:\.\d+)?)\s*(mmol/l|mmol|毫摩尔|mg/dl|mg|毫克)?",
        r"(\d+(?:\.\d+)?)\s*(mmol/l|mmol|毫摩尔|mg/dl|mg|毫克)?[^\n，。；,;]{0,12}(?:血糖|glucose|blood sugar)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = float(match.group(1))
        unit = (match.group(2) or "").lower()
        if unit in {"mg/dl", "mg", "毫克"} or value > 35:
            return value / 18.0
        return value
    return None


def _has_allergy_context(text: str, alias: str) -> bool:
    direct_patterns = [
        f"{alias}过敏",
        f"对{alias}过敏",
        f"{alias} allergy",
        f"allergic to {alias}",
    ]
    if any(pattern in text for pattern in direct_patterns):
        return True

    allergy_markers = ["过敏", "allergy", "allergic"]
    index = text.find(alias)
    if index < 0:
        return False
    window = text[max(0, index - 12): index + len(alias) + 12]
    return any(marker in window for marker in allergy_markers)


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


FOOD_ALIASES = {
    "grapefruit": ["grapefruit", "pomelo", "西柚", "葡萄柚"],
    "alcohol": ["alcohol", "beer", "wine", "baijiu", "酒精", "酒", "饮酒", "喝酒", "喝一点酒", "喝点酒", "小酌", "啤酒", "白酒", "红酒", "黄酒"],
    "spinach": ["spinach", "菠菜"],
    "pork_liver": ["pork liver", "猪肝", "动物肝脏"],
    "green_leafy_vegetable": ["green leafy", "绿叶菜", "深绿色蔬菜", "西兰花", "羽衣甘蓝"],
    "low_sodium_salt": ["low sodium salt", "potassium salt", "低钠盐", "高钾盐"],
    "peanut": ["peanut", "peanut butter", "花生", "花生酱"],
    "tree_nut": ["tree nut", "almond", "cashew", "walnut", "坚果", "杏仁", "腰果", "核桃"],
    "milk": ["milk", "dairy", "牛奶", "奶", "奶制品", "酸奶", "奶酪"],
    "egg": ["egg", "鸡蛋", "蛋黄", "蛋清"],
    "fish": ["fish", "鱼", "鱼肉"],
    "shellfish": ["shellfish", "shrimp", "crab", "clam", "虾", "蟹", "螃蟹", "贝类", "蛤蜊"],
    "wheat": ["wheat", "gluten", "小麦", "麸质", "面包", "面条"],
    "soy": ["soy", "soybean", "tofu", "豆浆", "大豆", "黄豆", "豆腐"],
    "sesame": ["sesame", "芝麻", "芝麻酱"],
    "organ_meat": ["organ meat", "动物内脏", "内脏"],
    "broth": ["broth", "浓肉汤", "肉汤", "高汤"],
    "anchovy": ["anchovy", "凤尾鱼"],
    "sardine": ["sardine", "沙丁鱼"],
    "sugary_drink": ["sugary drink", "sweet drink", "milk tea", "soda", "juice", "奶茶", "含糖饮料", "甜饮料", "可乐", "汽水", "果汁", "蜂蜜水", "甜咖啡"],
    "high_potassium_fruit": ["banana", "orange", "kiwi", "avocado", "香蕉", "橙子", "猕猴桃", "牛油果"],
    "high_potassium_vegetable": ["potato", "sweet potato", "tomato", "spinach", "土豆", "马铃薯", "红薯", "番茄", "西红柿", "菠菜"],
    "coconut_water": ["coconut water", "椰子水"],
    "dried_fruit": ["dried fruit", "raisin", "date", "果干", "葡萄干", "红枣", "枣干"],
    "cola_dark_soda": ["cola", "dark soda", "可乐", "深色汽水"],
    "processed_meat": ["processed meat", "ham", "sausage", "bacon", "加工肉", "火腿", "香肠", "腊肉", "培根", "午餐肉"],
    "instant_noodles": ["instant noodles", "方便面", "泡面"],
    "pickled_food": ["pickled", "pickle", "咸菜", "腌菜", "榨菜", "泡菜", "酱菜"],
    "raw_milk": ["raw milk", "unpasteurized milk", "生牛奶", "未巴氏杀菌奶"],
    "raw_seafood": ["raw seafood", "sashimi", "raw oyster", "生鱼片", "刺身", "生蚝"],
    "undercooked_egg": ["undercooked egg", "runny egg", "溏心蛋", "半熟蛋", "生鸡蛋"],
    "raw_sprouts": ["raw sprouts", "sprouts", "生豆芽", "芽菜"],
    "deli_meat": ["deli meat", "cold cuts", "冷切肉", "即食熟肉"],
    "aged_cheese": ["aged cheese", "blue cheese", "陈年奶酪", "蓝纹奶酪"],
    "fermented_food": ["fermented", "sauerkraut", "kimchi", "发酵食品", "泡菜", "豆豉", "纳豆"],
    "soy_sauce": ["soy sauce", "酱油"],
    "mineral_supplement": ["calcium", "iron", "zinc", "magnesium", "钙片", "铁剂", "锌片", "镁", "复合矿物质"],
    "coffee": ["coffee", "咖啡"],
    "spoiled_food": ["moldy", "spoiled", "expired", "发霉", "霉了", "变质", "馊了", "异味", "过期", "常温过夜"],
}

MEDICATION_ALIASES = {
    "simvastatin": ["simvastatin", "辛伐他汀"],
    "atorvastatin": ["atorvastatin", "阿托伐他汀"],
    "nifedipine": ["nifedipine", "硝苯地平"],
    "felodipine": ["felodipine", "非洛地平"],
    "cephalosporin": ["cephalosporin", "cefoperazone", "头孢", "头孢哌酮"],
    "metronidazole": ["metronidazole", "甲硝唑"],
    "warfarin": ["warfarin", "华法林"],
    "acei": ["acei", "普利", "依那普利", "贝那普利", "卡托普利"],
    "arb": ["arb", "沙坦", "氯沙坦", "缬沙坦", "厄贝沙坦"],
    "levothyroxine": ["levothyroxine", "左甲状腺素", "优甲乐", "雷替斯"],
    "quinolone": ["quinolone", "ciprofloxacin", "levofloxacin", "moxifloxacin", "喹诺酮", "环丙沙星", "左氧氟沙星", "莫西沙星"],
    "tetracycline": ["tetracycline", "doxycycline", "minocycline", "四环素", "多西环素", "米诺环素"],
    "maoi": ["maoi", "monoamine oxidase inhibitor", "单胺氧化酶抑制剂", "苯乙肼", "反苯环丙胺"],
    "linezolid": ["linezolid", "利奈唑胺"],
    "opioid": ["opioid", "morphine", "oxycodone", "tramadol", "阿片", "吗啡", "羟考酮", "曲马多"],
    "benzodiazepine": ["benzodiazepine", "diazepam", "alprazolam", "lorazepam", "苯二氮卓", "地西泮", "阿普唑仑", "劳拉西泮"],
    "sleeping_pill": ["sleeping pill", "zolpidem", "eszopiclone", "安眠药", "佐匹克隆", "唑吡坦"],
    "acetaminophen": ["acetaminophen", "paracetamol", "对乙酰氨基酚", "扑热息痛"],
    "insulin": ["insulin", "胰岛素"],
    "sulfonylurea": ["sulfonylurea", "glipizide", "gliclazide", "glimepiride", "磺脲", "格列吡嗪", "格列齐特", "格列美脲"],
}

DISEASE_ALIASES = {
    "ckd": ["ckd", "chronic kidney disease", "慢性肾脏病", "肾功能不全", "肾病"],
    "gout": ["gout", "痛风"],
    "hyperuricemia": ["hyperuricemia", "高尿酸血症", "高尿酸"],
    "diabetes": ["diabetes", "type 2 diabetes", "2型糖尿病", "二型糖尿病", "糖尿病"],
    "hypertension": ["hypertension", "high blood pressure", "高血压", "血压高"],
    "liver_disease": ["liver disease", "hepatic impairment", "肝病", "肝功能异常", "脂肪肝", "肝硬化"],
    "immunocompromised": ["immunocompromised", "免疫低下", "免疫抑制", "化疗", "器官移植"],
}

ALLERGY_ALIASES = {
    "peanut": ["peanut", "花生"],
    "tree_nut": ["tree nut", "almond", "cashew", "walnut", "坚果", "杏仁", "腰果", "核桃"],
    "milk": ["milk", "dairy", "牛奶", "奶"],
    "egg": ["egg", "鸡蛋", "蛋"],
    "fish": ["fish", "鱼"],
    "shellfish": ["shellfish", "shrimp", "crab", "虾", "蟹", "贝类"],
    "wheat": ["wheat", "gluten", "小麦", "麸质"],
    "soy": ["soy", "soybean", "大豆", "黄豆"],
    "sesame": ["sesame", "芝麻"],
}

SYMPTOM_ALIASES = {
    "hypoglycemia": ["hypoglycemia", "低血糖", "手抖", "冒汗", "出汗", "心慌", "头晕", "发软", "意识模糊", "昏迷"],
    "dka": ["ketone", "ketones", "ketoacidosis", "酮体", "酮症", "酮症酸中毒", "呕吐", "腹痛", "呼吸深快", "嗜睡", "脱水"],
    "anaphylaxis": ["anaphylaxis", "呼吸困难", "喉咙肿", "嘴唇肿", "脸肿", "全身风团", "严重过敏"],
}

ACTIVITY_ALIASES = {
    "exercise": ["exercise", "running", "swimming", "workout", "运动", "跑步", "游泳", "健身", "骑车"],
    "driving": ["driving", "drive", "开车", "驾驶"],
}

POPULATION_ALIASES = {
    "pregnancy": ["pregnancy", "pregnant", "怀孕", "孕妇", "孕期"],
    "child": ["child", "儿童", "孩子", "小孩"],
    "older_adult": ["older adult", "elderly", "老人", "老年人"],
}
