from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from config.config_loader import get_project_dir
from config.logger import setup_logging
from core.utils.device_identity import normalize_device_user_id
from core.providers.memory.clinical_ltm.health_profile import (
    HealthProfileStore,
    ProfileItem,
    ProfileUpdate,
    extract_health_profile_update,
    format_health_profile_context,
)
from plugins_func.register import Action, ActionResponse, ToolType, register_function

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

GET_HEALTH_PROFILE_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "get_health_profile",
        "description": "读取用户结构化健康档案，包括年龄、性别、身高体重、疾病、用药、过敏、目标、运动量、肾功能、血糖指标等稳定信息。",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}

UPDATE_HEALTH_PROFILE_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "update_health_profile",
        "description": (
            "新增或更新用户结构化健康档案。当用户明确说出年龄、性别、身高体重、疾病、用药、过敏、目标、"
            "运动量、肾功能、血糖指标等长期健康信息时调用。不要把临时饮食事件写成稳定档案。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "profile_text": {
                    "type": "string",
                    "description": "用户原话或需要抽取的健康档案文本，例如：我45岁，男，身高170，体重72公斤，有2型糖尿病，吃二甲双胍。",
                },
                "age_years": {"type": "number", "description": "年龄，单位岁。"},
                "sex": {"type": "string", "description": "性别，可填 male/female/男/女。"},
                "height_cm": {"type": "number", "description": "身高，单位 cm。"},
                "weight_kg": {"type": "number", "description": "体重，单位 kg。"},
                "activity_level": {
                    "type": "string",
                    "description": "活动水平，例如 sedentary/light/moderate/high 或中文描述。",
                },
                "nutrition_goal": {"type": "string", "description": "营养或健康目标摘要，例如控糖、减重、低盐。"},
                "target_energy_kcal": {"type": "number", "description": "每日热量目标，单位 kcal。"},
                "target_carbohydrate_g_per_meal": {"type": "number", "description": "每餐碳水目标，单位 g。"},
                "target_protein_g_per_day": {"type": "number", "description": "每日蛋白质目标，单位 g。"},
                "target_fat_g_per_day": {"type": "number", "description": "每日脂肪目标，单位 g。"},
                "diseases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "长期疾病或诊断列表，例如 2型糖尿病、高血压、慢性肾脏病。",
                },
                "medications": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "长期或当前用药列表，例如 二甲双胍、硝苯地平、华法林。",
                },
                "allergies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "过敏原列表，例如 花生、海鲜、牛奶。",
                },
                "goals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "健康目标列表，例如 控糖、减重、降尿酸。",
                },
                "dietary_restrictions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "稳定饮食限制列表，例如 低盐、低嘌呤、低钾。",
                },
                "egfr": {"type": "number", "description": "eGFR，单位 mL/min/1.73m2。"},
                "creatinine": {"type": "number", "description": "肌酐数值，单位不明确时只记录数值。"},
                "fasting_glucose": {"type": "number", "description": "空腹血糖，默认单位 mmol/L。"},
                "postprandial_2h_glucose": {"type": "number", "description": "餐后2小时血糖，默认单位 mmol/L。"},
                "hba1c": {"type": "number", "description": "糖化血红蛋白 HbA1c，单位%。"},
                "notes": {"type": "string", "description": "其他需要稳定保留的健康档案备注。"},
            },
        },
    },
}


@register_function("get_health_profile", GET_HEALTH_PROFILE_FUNCTION_DESC, ToolType.SYSTEM_CTL)
def get_health_profile(conn: "ConnectionHandler"):
    user_id = _get_user_id(conn)
    if not user_id:
        return ActionResponse(Action.RESPONSE, None, "当前还没有可绑定健康档案的用户或设备ID。")

    try:
        store = _get_profile_store(conn)
        profile = store.get_profile_sync(user_id)
        context = format_health_profile_context(profile)
    except Exception as exc:
        logger.bind(tag=TAG).error(f"Get health profile failed: {exc}")
        return ActionResponse(Action.RESPONSE, None, "读取健康档案失败，请稍后再试。")

    return ActionResponse(Action.REQLLM, context, None)


@register_function("update_health_profile", UPDATE_HEALTH_PROFILE_FUNCTION_DESC, ToolType.SYSTEM_CTL)
def update_health_profile(
    conn: "ConnectionHandler",
    profile_text: str | None = None,
    age_years: Any = None,
    sex: str | None = None,
    height_cm: Any = None,
    weight_kg: Any = None,
    activity_level: str | None = None,
    nutrition_goal: str | None = None,
    target_energy_kcal: Any = None,
    target_carbohydrate_g_per_meal: Any = None,
    target_protein_g_per_day: Any = None,
    target_fat_g_per_day: Any = None,
    diseases: Any = None,
    medications: Any = None,
    allergies: Any = None,
    goals: Any = None,
    dietary_restrictions: Any = None,
    egfr: Any = None,
    creatinine: Any = None,
    fasting_glucose: Any = None,
    postprandial_2h_glucose: Any = None,
    hba1c: Any = None,
    notes: str | None = None,
):
    user_id = _get_user_id(conn)
    if not user_id:
        return ActionResponse(Action.RESPONSE, None, "当前还没有可绑定健康档案的用户或设备ID。")

    update = extract_health_profile_update(str(profile_text or ""), source="tool_text")
    update.scalars.update(
        _clean_scalars(
            {
                "age_years": age_years,
                "sex": sex,
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "activity_level": activity_level,
                "nutrition_goal": nutrition_goal,
                "target_energy_kcal": target_energy_kcal,
                "target_carbohydrate_g_per_meal": target_carbohydrate_g_per_meal,
                "target_protein_g_per_day": target_protein_g_per_day,
                "target_fat_g_per_day": target_fat_g_per_day,
                "notes": notes,
            }
        )
    )
    update.items.extend(_items_from_list("disease", diseases))
    update.items.extend(_items_from_list("medication", medications))
    update.items.extend(_items_from_list("allergy", allergies))
    update.items.extend(_items_from_list("goal", goals))
    update.items.extend(_items_from_list("dietary_restriction", dietary_restrictions))
    update.items.extend(_metric_item("renal_function", "eGFR", egfr, "mL/min/1.73m2"))
    update.items.extend(_metric_item("renal_function", "肌酐", creatinine, ""))
    update.items.extend(_metric_item("glucose_metric", "空腹血糖", fasting_glucose, "mmol/L"))
    update.items.extend(_metric_item("glucose_metric", "餐后2小时血糖", postprandial_2h_glucose, "mmol/L"))
    update.items.extend(_metric_item("glucose_metric", "糖化血红蛋白", hba1c, "%"))

    if update.is_empty():
        return ActionResponse(Action.RESPONSE, None, "没有识别到需要写入健康档案的稳定信息。")

    try:
        store = _get_profile_store(conn)
        stats = store.apply_update_sync(user_id, update)
        profile = store.get_profile_sync(user_id)
        context = format_health_profile_context(profile)
    except Exception as exc:
        logger.bind(tag=TAG).error(f"Update health profile failed: {exc}")
        return ActionResponse(Action.RESPONSE, None, "更新健康档案失败，请稍后再试。")

    result = [
        "# 健康档案已更新",
        f"- 更新基础字段: {stats['scalar_count']} 个",
        f"- 更新档案项目: {stats['item_count']} 个",
        "",
        context,
    ]
    return ActionResponse(Action.REQLLM, "\n".join(result), None)


def _get_user_id(conn: "ConnectionHandler") -> str:
    memory = getattr(conn, "memory", None)
    role_id = getattr(memory, "role_id", None)
    if role_id:
        return str(role_id).strip()
    user_id = getattr(conn, "user_id", None)
    if user_id:
        return str(user_id).strip()
    return normalize_device_user_id(getattr(conn, "device_id", ""))


def _get_profile_store(conn: "ConnectionHandler") -> HealthProfileStore:
    memory = getattr(conn, "memory", None)
    store = getattr(memory, "health_profile_store", None)
    if store is not None:
        return store

    clinical_ltm_config = conn.config.get("Memory", {}).get("clinical_ltm", {})
    configured = clinical_ltm_config.get("health_profile_sqlite_path", "data/clinical_health_profile.db")
    db_path = Path(configured)
    if not db_path.is_absolute():
        db_path = Path(get_project_dir()) / db_path
    return HealthProfileStore(db_path)


def _clean_scalars(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value not in (None, "")}


def _items_from_list(category: str, values: Any) -> list[ProfileItem]:
    return [
        ProfileItem(category=category, name=item, source="tool_explicit", evidence=item, confidence=0.92)
        for item in _ensure_list(values)
        if item
    ]


def _metric_item(category: str, name: str, value: Any, unit: str) -> list[ProfileItem]:
    if value in (None, ""):
        return []
    try:
        number = float(value)
    except (TypeError, ValueError):
        return []
    return [
        ProfileItem(
            category=category,
            name=name,
            value={"value": number, "unit": unit},
            source="tool_explicit",
            evidence=f"{name}={number}{unit}",
            confidence=0.94,
        )
    ]


def _ensure_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = str(value).replace("，", "、").replace(",", "、").split("、")
    return [str(item).strip() for item in raw_items if str(item).strip()]
