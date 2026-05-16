from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import Any

from .models import (
    EpisodicMemory,
    ExtractionEnvelope,
    ExtractionResult,
    FactualMemory,
    MemoryLayer,
    SemanticMemory,
)
from .prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_TEMPLATE,
)


class ClinicalCognitiveExtractor:
    def __init__(self, config: dict[str, Any], llm, logger, powermem_index=None):
        self.config = config
        self.llm = llm
        self.logger = logger
        self.powermem_index = powermem_index
        prompts = config.get("prompts") if isinstance(config.get("prompts"), dict) else {}
        self.extraction_system_prompt = (
            str(prompts.get("long_term_extraction_system_prompt") or "").strip()
            or EXTRACTION_SYSTEM_PROMPT
        )
        self.extraction_user_template = (
            str(prompts.get("long_term_extraction_user_template") or "").strip()
            or EXTRACTION_USER_TEMPLATE
        )

    async def extract(self, envelope: ExtractionEnvelope) -> ExtractionResult:
        retrieval_hints = await self._get_powermem_hints(envelope.messages, envelope.user_id)
        if self.llm is not None:
            try:
                raw = await self._extract_with_llm(envelope, retrieval_hints)
                parsed = self._parse_llm_result(raw, envelope)
                if parsed is not None:
                    return parsed
            except Exception as exc:
                self.logger.bind(tag=__name__).warning(f"LLM 结构化抽取失败，回退规则抽取: {exc}")

        return self._fallback_extract(envelope, retrieval_hints)

    async def _extract_with_llm(
        self,
        envelope: ExtractionEnvelope,
        retrieval_hints: list[str],
    ) -> str:
        dialogue_lines = []
        for item in envelope.messages:
            role = item.get("role", "user")
            content = item.get("content", "").strip()
            if not content:
                continue
            dialogue_lines.append(f"{role}: {content}")

        user_prompt = self.extraction_user_template.format(
            user_id=envelope.user_id,
            session_id=envelope.session_id,
            now=envelope.generated_at.isoformat(),
            retrieval_hints="\n".join(f"- {item}" for item in retrieval_hints) or "- 无",
            dialogue="\n".join(dialogue_lines),
        )
        return await asyncio.to_thread(
            self.llm.response_no_stream,
            self.extraction_system_prompt,
            user_prompt,
            max_tokens=self.config.get("extract_max_tokens", 1200),
            temperature=self.config.get("extract_temperature", 0.1),
        )

    async def _get_powermem_hints(
        self,
        messages: list[dict[str, str]],
        user_id: str,
    ) -> list[str]:
        if not self.powermem_index or not getattr(self.powermem_index, "enabled", False):
            return []

        powermem_cfg = self.config.get("powermem", {})
        if not powermem_cfg.get("extraction_hints_enabled", True):
            return []

        query = self._build_hint_query(messages)
        if not query:
            return []

        limit = int(powermem_cfg.get("extraction_hints_limit", 6))
        max_chars = int(powermem_cfg.get("extraction_hints_max_chars", 1200))

        try:
            result = await self.powermem_index.search(
                query=query,
                user_id=user_id,
                limit=limit,
            )
        except Exception as exc:
            self.logger.bind(tag=__name__).warning(f"PowerMem hints 检索失败: {exc}")
            return []

        hints = []
        total_chars = 0
        for item in result.get("results") or []:
            text = item.get("memory") or item.get("content") or item.get("text") or ""
            text = str(text).strip()
            if not text:
                continue
            if total_chars + len(text) > max_chars:
                break
            hints.append(text)
            total_chars += len(text)

        return hints

    def _build_hint_query(self, messages: list[dict[str, str]]) -> str:
        user_texts = []
        for item in messages[-6:]:
            role = item.get("role", "user")
            content = str(item.get("content", "")).strip()
            if role == "user" and content:
                user_texts.append(content)
        query = " ".join(user_texts[-3:])
        return query[:500]

    def _parse_llm_result(
        self,
        raw_text: str,
        envelope: ExtractionEnvelope,
    ) -> ExtractionResult | None:
        payload = _extract_json_object(raw_text)
        if payload is None:
            return None

        observed_at = envelope.generated_at
        result = ExtractionResult(
            is_noise=bool(payload.get("is_noise", False)),
            noise_reason=str(payload.get("noise_reason", "")),
            raw_response=raw_text,
        )

        for item in payload.get("factual_memories", []):
            result.factual_memories.append(
                FactualMemory(
                    user_id=envelope.user_id,
                    entity=str(item.get("entity", "用户")),
                    attribute=str(item.get("attribute", "事实")),
                    value=str(item.get("value", "")).strip(),
                    content=str(item.get("content", "")).strip(),
                    source=str(item.get("source", "用户自述")),
                    observed_at=_parse_datetime(item.get("observed_at"), observed_at),
                    evidence=[str(item.get("content", "")).strip()],
                    dedupe_key=_build_dedupe_key(
                        MemoryLayer.FACTUAL,
                        str(item.get("entity", "用户")),
                        str(item.get("attribute", "事实")),
                        str(item.get("value", "")),
                    ),
                    tags=["clinical", "locked-fact"],
                )
            )

        for item in payload.get("episodic_memories", []):
            importance = float(item.get("importance", 0.62))
            result.episodic_memories.append(
                EpisodicMemory(
                    user_id=envelope.user_id,
                    entity=str(item.get("entity", "用户")),
                    attribute=str(item.get("attribute", "事件")),
                    value=str(item.get("value", "")).strip(),
                    content=str(item.get("content", "")).strip(),
                    source=str(item.get("source", "对话事件")),
                    observed_at=_parse_datetime(item.get("observed_at"), observed_at),
                    importance=max(0.1, min(1.0, importance)),
                    weight=max(0.1, min(1.0, importance)),
                    evidence=[str(item.get("content", "")).strip()],
                    dedupe_key=_build_dedupe_key(
                        MemoryLayer.EPISODIC,
                        str(item.get("entity", "用户")),
                        str(item.get("attribute", "事件")),
                        str(item.get("value", "")),
                    ),
                    tags=["clinical", "episode"],
                )
            )

        for item in payload.get("semantic_memories", []):
            importance = float(item.get("importance", 0.86))
            result.semantic_memories.append(
                SemanticMemory(
                    user_id=envelope.user_id,
                    entity=str(item.get("entity", "用户")),
                    attribute=str(item.get("attribute", "规律")),
                    value=str(item.get("value", "")).strip(),
                    content=str(item.get("content", "")).strip(),
                    source=str(item.get("source", "多轮总结")),
                    observed_at=_parse_datetime(item.get("observed_at"), observed_at),
                    importance=max(0.1, min(1.0, importance)),
                    weight=max(0.1, min(1.0, importance)),
                    evidence=[str(item.get("content", "")).strip()],
                    dedupe_key=_build_dedupe_key(
                        MemoryLayer.SEMANTIC,
                        str(item.get("entity", "用户")),
                        str(item.get("attribute", "规律")),
                        str(item.get("value", "")),
                    ),
                    tags=["clinical", "summary"],
                )
            )

        if not result.factual_memories and not result.episodic_memories and not result.semantic_memories:
            result.is_noise = True
            result.noise_reason = result.noise_reason or "没有抽取到具有医学营养价值的信息"
        return result

    def _fallback_extract(
        self,
        envelope: ExtractionEnvelope,
        retrieval_hints: list[str],
    ) -> ExtractionResult:
        joined_text = "\n".join(item.get("content", "") for item in envelope.messages)
        observed_at = envelope.generated_at
        result = ExtractionResult(raw_response="\n".join(retrieval_hints))

        medical_keywords = [
            "糖尿病",
            "过敏",
            "血糖",
            "医生",
            "医嘱",
            "药",
            "服用",
            "二甲双胍",
            "胰岛素",
            "降糖",
            "降压",
            "早餐",
            "午餐",
            "晚餐",
            "加餐",
            "碳水",
            "脂肪",
            "蛋白质",
            "牛奶",
            "鸡蛋",
            "米饭",
            "面包",
            "水果",
        ]
        if not any(keyword in joined_text for keyword in medical_keywords):
            result.is_noise = True
            result.noise_reason = "没有检测到医学营养相关关键词"
            return result

        diabetes_match = re.search(r"((?:1|一|2|二|妊娠)?型?糖尿病)", joined_text)
        if diabetes_match:
            value = diabetes_match.group(1).replace("一型", "1型").replace("二型", "2型")
            result.factual_memories.append(
                FactualMemory(
                    user_id=envelope.user_id,
                    entity="用户",
                    attribute="疾病",
                    value=value,
                    content=f"用户患有{value}",
                    source="用户自述",
                    observed_at=observed_at,
                    evidence=[diabetes_match.group(0)],
                    dedupe_key=_build_dedupe_key(MemoryLayer.FACTUAL, "用户", "疾病", value),
                    tags=["clinical", "disease"],
                )
            )

        for allergy in re.findall(r"(?:对|有)([^，。；,;]{1,12})过敏", joined_text):
            clean_allergy = allergy.strip()
            result.factual_memories.append(
                FactualMemory(
                    user_id=envelope.user_id,
                    entity="用户",
                    attribute="过敏原",
                    value=clean_allergy,
                    content=f"用户对{clean_allergy}过敏",
                    source="用户自述",
                    observed_at=observed_at,
                    evidence=[f"对{clean_allergy}过敏"],
                    dedupe_key=_build_dedupe_key(MemoryLayer.FACTUAL, "用户", "过敏原", clean_allergy),
                    tags=["clinical", "allergy"],
                )
            )

        for medicine in re.findall(r"(?:正在吃|在吃|服用)([^。；\n]+)", joined_text):
            clean_medicine = medicine.strip()
            result.factual_memories.append(
                FactualMemory(
                    user_id=envelope.user_id,
                    entity="用户",
                    attribute="用药",
                    value=clean_medicine,
                    content=f"用户正在服用{clean_medicine}",
                    source="用户自述",
                    observed_at=observed_at,
                    evidence=[clean_medicine],
                    dedupe_key=_build_dedupe_key(MemoryLayer.FACTUAL, "用户", "用药", clean_medicine),
                    tags=["clinical", "medication"],
                )
            )

        for advice in re.findall(r"(?:医生说|医生建议|医嘱|医生让我)([^。；\n]+)", joined_text):
            clean_advice = advice.strip()
            result.factual_memories.append(
                FactualMemory(
                    user_id=envelope.user_id,
                    entity="用户",
                    attribute="医生建议",
                    value=clean_advice,
                    content=f"医生建议用户{clean_advice}",
                    source="医生建议",
                    observed_at=observed_at,
                    evidence=[clean_advice],
                    dedupe_key=_build_dedupe_key(MemoryLayer.FACTUAL, "用户", "医生建议", clean_advice),
                    tags=["clinical", "doctor-advice"],
                )
            )

        meal_match = re.search(r"(早餐|午餐|晚餐|加餐).{0,30}", joined_text)
        if meal_match:
            meal_text = meal_match.group(0).strip()
            importance = 0.72 if any(term in joined_text for term in ["血糖", "高", "风险", "不舒服"]) else 0.56
            result.episodic_memories.append(
                EpisodicMemory(
                    user_id=envelope.user_id,
                    entity="用户",
                    attribute="饮食事件",
                    value=meal_text,
                    content=f"发生了一次饮食相关事件：{meal_text}",
                    source="对话事件",
                    observed_at=observed_at,
                    importance=importance,
                    weight=importance,
                    evidence=[joined_text[:240]],
                    dedupe_key=_build_dedupe_key(MemoryLayer.EPISODIC, "用户", "饮食事件", meal_text),
                    tags=["clinical", "meal-event"],
                )
            )

        if retrieval_hints:
            hint = retrieval_hints[0]
            result.semantic_memories.append(
                SemanticMemory(
                    user_id=envelope.user_id,
                    entity="用户",
                    attribute="语义线索",
                    value=hint[:80],
                    content=hint,
                    source="PowerMem检索提示",
                    observed_at=observed_at,
                    importance=0.78,
                    weight=0.78,
                    evidence=retrieval_hints[:3],
                    dedupe_key=_build_dedupe_key(MemoryLayer.SEMANTIC, "用户", "语义线索", hint[:80]),
                    tags=["semantic", "retrieval-hint"],
                )
            )

        if not result.factual_memories and not result.episodic_memories and not result.semantic_memories:
            result.is_noise = True
            result.noise_reason = "规则抽取未发现可写入长期记忆的内容"
        return result


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _parse_datetime(raw_value: Any, fallback: datetime) -> datetime:
    if not raw_value:
        return fallback
    try:
        return datetime.fromisoformat(str(raw_value).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return fallback


def _build_dedupe_key(layer: MemoryLayer, entity: str, attribute: str, value: str) -> str:
    normalized = re.sub(r"\s+", "", f"{entity}:{attribute}:{value}".lower())
    return f"{layer.value}:{normalized}"
