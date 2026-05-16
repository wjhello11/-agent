from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Any

from .models import (
    EpisodicMemory,
    ExtractionResult,
    FactualMemory,
    MemoryLayer,
    SemanticMemory,
    StructuredMemory,
)
from .prompts import SEMANTIC_SUMMARY_SYSTEM_PROMPT, SEMANTIC_SUMMARY_USER_TEMPLATE


class MemoryLifecycleManager:
    def __init__(self, config: dict[str, Any], llm, logger):
        self.config = config
        self.llm = llm
        self.logger = logger
        self.episodic_half_life_days = float(config.get("episodic_half_life_days", 14))
        self.semantic_half_life_days = float(config.get("semantic_half_life_days", 90))
        self.min_weight = float(config.get("min_weight", 0.08))
        self.promotion_min_count = int(config.get("promotion_min_count", 3))
        self.promotion_window_days = int(config.get("promotion_window_days", 30))

    def prepare_for_store(
        self,
        extraction: ExtractionResult,
        embedder,
    ) -> list[StructuredMemory]:
        prepared: list[StructuredMemory] = []

        for memory in extraction.factual_memories:
            memory.locked = True
            memory.importance = 1.0
            memory.weight = 1.0
            memory.embedding = embedder(f"{memory.attribute} {memory.value} {memory.content}")
            prepared.append(memory)

        for memory in extraction.episodic_memories:
            memory.weight = max(memory.importance, 0.15)
            memory.locked = False
            memory.embedding = embedder(f"{memory.attribute} {memory.value} {memory.content}")
            prepared.append(memory)

        for memory in extraction.semantic_memories:
            memory.weight = max(memory.importance, 0.2)
            memory.locked = False
            memory.embedding = embedder(f"{memory.attribute} {memory.value} {memory.content}")
            prepared.append(memory)

        return prepared

    async def apply_decay(self, store, user_id: str) -> None:
        await store.apply_forgetting_curve(
            user_id=user_id,
            episodic_half_life_days=self.episodic_half_life_days,
            semantic_half_life_days=self.semantic_half_life_days,
            min_weight=self.min_weight,
        )

    async def synthesize_semantic_memories(self, store, user_id: str) -> list[SemanticMemory]:
        recent_episodes = await store.list_recent_memories(
            user_id=user_id,
            layer=MemoryLayer.EPISODIC,
            limit=40,
        )
        if len(recent_episodes) < self.promotion_min_count:
            return []

        cutoff = datetime.utcnow() - timedelta(days=self.promotion_window_days)
        recent_episodes = [item for item in recent_episodes if item.observed_at >= cutoff]
        if len(recent_episodes) < self.promotion_min_count:
            return []

        if self.llm is not None:
            try:
                summarized = await self._summarize_with_llm(user_id, recent_episodes)
                if summarized:
                    return summarized
            except Exception as exc:
                self.logger.bind(tag=__name__).warning(f"语义记忆总结失败，回退规则聚合: {exc}")

        return self._fallback_semantic_summary(user_id, recent_episodes)

    async def _summarize_with_llm(
        self,
        user_id: str,
        episodes: list[StructuredMemory],
    ) -> list[SemanticMemory]:
        episode_lines = []
        for item in episodes[:20]:
            episode_lines.append(
                f"- [{item.observed_at.isoformat()}] {item.attribute}: {item.value} | {item.content}"
            )
        user_prompt = SEMANTIC_SUMMARY_USER_TEMPLATE.format(
            user_id=user_id,
            now=datetime.utcnow().isoformat(),
            episodes="\n".join(episode_lines),
        )
        raw = await asyncio.to_thread(
            self.llm.response_no_stream,
            SEMANTIC_SUMMARY_SYSTEM_PROMPT,
            user_prompt,
            max_tokens=self.config.get("summary_max_tokens", 800),
            temperature=self.config.get("summary_temperature", 0.2),
        )
        items = _extract_json_array(raw)
        results: list[SemanticMemory] = []
        for item in items:
            value = str(item.get("value", "")).strip()
            if not value:
                continue
            attribute = str(item.get("attribute", "长期规律")).strip() or "长期规律"
            observed_at = _parse_datetime(item.get("observed_at"), datetime.utcnow())
            importance = max(0.2, min(1.0, float(item.get("importance", 0.84))))
            results.append(
                SemanticMemory(
                    user_id=user_id,
                    entity=str(item.get("entity", "用户")),
                    attribute=attribute,
                    value=value,
                    content=str(item.get("content", "")).strip() or value,
                    source=str(item.get("source", "多轮总结")),
                    observed_at=observed_at,
                    importance=importance,
                    weight=importance,
                    evidence=[episode.content for episode in episodes[:3]],
                    dedupe_key=_build_dedupe_key(attribute, value),
                    tags=["semantic", "summary"],
                )
            )
        return results

    def _fallback_semantic_summary(
        self,
        user_id: str,
        episodes: list[StructuredMemory],
    ) -> list[SemanticMemory]:
        grouped: dict[str, list[StructuredMemory]] = {}
        for item in episodes:
            key = item.attribute.strip() or "长期规律"
            grouped.setdefault(key, []).append(item)

        semantic_memories: list[SemanticMemory] = []
        for attribute, items in grouped.items():
            if len(items) < self.promotion_min_count:
                continue
            common_values = [item.value for item in items[:5] if item.value]
            value_preview = "；".join(common_values[:3]) or f"{attribute}重复出现"
            content = f"最近 {len(items)} 次记录显示：{attribute}反复出现，代表用户可能存在稳定行为模式。典型表现：{value_preview}"
            semantic_memories.append(
                SemanticMemory(
                    user_id=user_id,
                    entity="用户",
                    attribute=attribute,
                    value=value_preview[:120],
                    content=content,
                    source="规则聚合总结",
                    observed_at=datetime.utcnow(),
                    importance=min(0.92, 0.65 + 0.05 * len(items)),
                    weight=min(0.92, 0.65 + 0.05 * len(items)),
                    evidence=[item.content for item in items[:4]],
                    dedupe_key=_build_dedupe_key(attribute, value_preview[:120]),
                    tags=["semantic", "aggregated"],
                )
            )
        return semantic_memories


def _extract_json_array(raw_text: str) -> list[dict[str, Any]]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []


def _parse_datetime(raw_value: Any, fallback: datetime) -> datetime:
    if not raw_value:
        return fallback
    try:
        return datetime.fromisoformat(str(raw_value).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return fallback


def _build_dedupe_key(attribute: str, value: str) -> str:
    normalized = re.sub(r"\s+", "", f"{attribute}:{value}".lower())
    return f"{MemoryLayer.SEMANTIC.value}:{normalized}"
