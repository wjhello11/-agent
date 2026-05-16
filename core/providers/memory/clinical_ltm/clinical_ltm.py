from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..base import MemoryProviderBase, logger
from .extractor import ClinicalCognitiveExtractor
from .health_profile import (
    HealthProfileStore,
    extract_health_profile_update,
    merge_profile_updates,
)
from .interceptor import MemoryRetrievalInterceptor
from .lifecycle import MemoryLifecycleManager
from .models import ExtractionEnvelope, WorkingTurn
from .powermem_index import PowerMemOfficialIndex
from .prompts import SHORT_TERM_SUMMARY_SYSTEM_PROMPT, SHORT_TERM_SUMMARY_USER_TEMPLATE
from .store import PowerMemSQLiteStore

TAG = __name__


class MemoryProvider(MemoryProviderBase):
    """
    面向临床营养师 Agent 的多维长效记忆引擎。

    设计目标：
    1. 使用 PowerMem 官方 AsyncMemory 负责长期记忆检索，并为记忆抽取阶段提供相关历史 hints。
    2. 使用 SQLite 保存四层结构与生命周期元数据。
    3. 本地 SQLite 负责四层结构、冲突管理、遗忘曲线、短期记忆和健康档案。
    4. 在生成回答前，自动把高权重历史记忆拼接到 system prompt 的 <memory> 区块。
    """

    def __init__(self, config, summary_memory=None):
        super().__init__(config)
        self.summary_memory = summary_memory or ""
        self.working_turns = int(config.get("working_memory_turns", 12))
        self.short_term_summary_enabled = bool(config.get("short_term_summary_enabled", True))
        self.short_term_summary_max_chars = int(config.get("short_term_summary_max_chars", 2000))
        self.short_term_recent_messages = int(config.get("short_term_recent_messages", 8))
        self.short_term_compact_trigger_messages = int(
            config.get("short_term_compact_trigger_messages", 18)
        )
        prompts = config.get("prompts") if isinstance(config.get("prompts"), dict) else {}
        self.short_term_summary_system_prompt = (
            str(prompts.get("short_term_summary_system_prompt") or "").strip()
            or SHORT_TERM_SUMMARY_SYSTEM_PROMPT
        )
        self.short_term_summary_user_template = (
            str(prompts.get("short_term_summary_user_template") or "").strip()
            or SHORT_TERM_SUMMARY_USER_TEMPLATE
        )
        self.enabled = True

        db_path = config.get("sqlite_path")
        if not db_path:
            db_path = str(Path("data") / "clinical_ltm.db")
        self.health_profile_enabled = bool(config.get("health_profile_enabled", True))

        try:
            self.store = PowerMemSQLiteStore(
                db_path=db_path,
                embedding_dimensions=int(config.get("embedding_dimensions", 256)),
            )
            self.health_profile_store = None
            if self.health_profile_enabled:
                profile_db_path = config.get("health_profile_sqlite_path")
                if not profile_db_path:
                    profile_db_path = str(Path(db_path).with_name("clinical_health_profile.db"))
                self.health_profile_store = HealthProfileStore(profile_db_path)
            self.powermem_index = PowerMemOfficialIndex(config, logger)
            self.extractor = ClinicalCognitiveExtractor(
                config, None, logger, powermem_index=self.powermem_index
            )
            self.lifecycle = MemoryLifecycleManager(config, None, logger)
            self.interceptor = MemoryRetrievalInterceptor(
                config,
                self.store,
                self.powermem_index,
                logger,
            )
            logger.bind(tag=TAG).info(
                f"多维长效记忆引擎初始化成功: sqlite={db_path}, "
                f"working_turns={self.working_turns}"
            )
            if self.health_profile_store:
                logger.bind(tag=TAG).info(
                    f"结构化健康档案初始化成功: sqlite={self.health_profile_store.db_path}"
                )
        except Exception as exc:
            logger.bind(tag=TAG).error(f"多维长效记忆引擎初始化失败: {exc}")
            self.enabled = False

    def init_memory(self, role_id, llm, **kwargs):
        super().init_memory(role_id, llm, **kwargs)
        self.set_llm(llm)

    def set_llm(self, llm):
        self.llm = llm
        if hasattr(self, "extractor"):
            self.extractor.llm = llm
        if hasattr(self, "lifecycle"):
            self.lifecycle.llm = llm

    async def save_memory(self, msgs, session_id=None):
        if not self.enabled or not getattr(self, "role_id", None):
            return None

        normalized_messages = self._normalize_messages(msgs)
        if len(normalized_messages) < 2:
            return None

        working_turns = [
            WorkingTurn(
                user_id=self.role_id,
                session_id=session_id or "unknown-session",
                role=item["role"],
                content=item["content"],
                created_at=datetime.utcnow(),
            )
            for item in normalized_messages[-self.working_turns:]
        ]

        await self.store.save_working_memory(
            user_id=self.role_id,
            session_id=session_id or "unknown-session",
            turns=working_turns,
            keep_last=self.working_turns,
        )

        await self._update_health_profile_from_messages(normalized_messages)
        await self.update_short_term_summary_from_messages(
            normalized_messages,
            session_id=session_id or "unknown-session",
            reason="session_end",
        )

        envelope = ExtractionEnvelope(
            user_id=self.role_id,
            session_id=session_id or "unknown-session",
            messages=normalized_messages,
            query_hint=normalized_messages[-1]["content"],
        )

        extraction_task = asyncio.create_task(self.extractor.extract(envelope))
        decay_task = asyncio.create_task(self.lifecycle.apply_decay(self.store, self.role_id))
        extraction_result, _ = await asyncio.gather(extraction_task, decay_task)

        if extraction_result.is_noise:
            logger.bind(tag=TAG).debug(
                f"对话被噪音过滤器丢弃: user_id={self.role_id}, reason={extraction_result.noise_reason}"
            )
            return {
                "status": "ignored",
                "reason": extraction_result.noise_reason,
            }

        prepared_memories = self.lifecycle.prepare_for_store(
            extraction_result,
            self.store.embed_text,
        )
        persisted_memories = await self.store.upsert_memories(prepared_memories)
        await self._sync_powermem_index(persisted_memories)

        semantic_memories = await self.lifecycle.synthesize_semantic_memories(self.store, self.role_id)
        if semantic_memories:
            for memory in semantic_memories:
                memory.embedding = self.store.embed_text(
                    f"{memory.attribute} {memory.value} {memory.content}"
                )
            persisted_semantic = await self.store.upsert_memories(semantic_memories)
            await self._sync_powermem_index(persisted_semantic)

        return {
            "status": "ok",
            "factual_count": len(extraction_result.factual_memories),
            "episodic_count": len(extraction_result.episodic_memories),
            "semantic_count": len(semantic_memories) + len(extraction_result.semantic_memories),
        }

    async def query_memory(self, query: str) -> str:
        memory_context = await self.interceptor.build_prompt_context(
            user_id=self.role_id,
            session_id=None,
            query=query,
            dialogue_messages=None,
        )
        memory_context = await self._prepend_short_term_summary_context(memory_context)
        return await self._prepend_health_profile_context(memory_context)

    async def build_memory_context(self, query: str, dialogue_messages=None, session_id=None) -> str:
        memory_context = await self.interceptor.build_prompt_context(
            user_id=self.role_id,
            session_id=session_id,
            query=query,
            dialogue_messages=dialogue_messages,
        )
        memory_context = await self._prepend_short_term_summary_context(memory_context)
        return await self._prepend_health_profile_context(memory_context)

    async def clear_user_memory(self) -> None:
        if not getattr(self, "role_id", None):
            return
        await self.store.clear_all_user_data(self.role_id)
        if getattr(self, "health_profile_store", None):
            await self.health_profile_store.clear_profile(self.role_id)
        if hasattr(self, "powermem_index"):
            await self.powermem_index.clear_user_memory(self.role_id)

    def should_compact_dialogue(self, dialogue_messages) -> bool:
        if not self.short_term_summary_enabled:
            return False
        if self.short_term_compact_trigger_messages <= 0:
            return False
        normalized = self._normalize_messages(dialogue_messages)
        return len(normalized) > self.short_term_compact_trigger_messages

    def get_short_term_recent_message_count(self) -> int:
        return max(2, self.short_term_recent_messages)

    async def update_short_term_summary_from_messages(
        self,
        messages,
        session_id: str | None = None,
        reason: str = "manual",
    ) -> dict | None:
        if (
            not self.short_term_summary_enabled
            or not getattr(self, "store", None)
            or not getattr(self, "role_id", None)
        ):
            return None

        normalized_messages = self._normalize_messages(messages)
        if len(normalized_messages) < 2:
            return await self.store.get_short_term_summary(self.role_id)

        previous = await self.store.get_short_term_summary(self.role_id)
        previous_summary = (previous or {}).get("summary", "")
        summary = await self._summarize_short_term(
            previous_summary=previous_summary,
            messages=normalized_messages,
            session_id=session_id or "unknown-session",
        )
        if not summary:
            return previous

        return await self.store.upsert_short_term_summary(
            user_id=self.role_id,
            summary=summary,
            source_session_id=session_id or "unknown-session",
            source_turn_count=len(normalized_messages),
            max_chars=self.short_term_summary_max_chars,
            metadata={"reason": reason},
        )

    async def update_health_profile_from_messages(
        self,
        messages,
        reason: str = "manual",
    ) -> dict | None:
        if not self.enabled or not getattr(self, "role_id", None):
            return None
        normalized_messages = self._normalize_messages(messages)
        if len(normalized_messages) < 1:
            return None
        return await self._update_health_profile_from_messages(
            normalized_messages,
            reason=reason,
        )

    async def _summarize_short_term(
        self,
        *,
        previous_summary: str,
        messages: list[dict[str, str]],
        session_id: str,
    ) -> str:
        dialogue = self._format_messages_for_summary(messages)
        if getattr(self, "llm", None) is not None:
            try:
                system_prompt = self.short_term_summary_system_prompt.replace(
                    "{max_chars}",
                    str(self.short_term_summary_max_chars),
                )
                user_prompt = self.short_term_summary_user_template.format(
                    user_id=self.role_id,
                    session_id=session_id,
                    now=datetime.utcnow().isoformat(),
                    max_chars=self.short_term_summary_max_chars,
                    previous_summary=previous_summary or "暂无",
                    dialogue=dialogue,
                )
                raw = await asyncio.to_thread(
                    self.llm.response_no_stream,
                    system_prompt,
                    user_prompt,
                    max_tokens=self.config.get("short_term_summary_max_tokens", 1200),
                    temperature=self.config.get("short_term_summary_temperature", 0.2),
                )
                cleaned = self._clean_short_term_summary(raw)
                if cleaned:
                    return cleaned
            except Exception as exc:
                logger.bind(tag=TAG).warning(f"短期记忆摘要生成失败，回退规则压缩: {exc}")

        return self._fallback_short_term_summary(previous_summary, messages)

    def _format_messages_for_summary(self, messages: list[dict[str, str]]) -> str:
        lines = []
        for item in messages[-80:]:
            role = item.get("role", "user")
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            if len(content) > 800:
                content = content[:800] + "..."
            lines.append(f"{role}: {content}")
        text = "\n".join(lines)
        return text[-12000:]

    def _clean_short_term_summary(self, text: str) -> str:
        cleaned = str(text or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("text"):
                cleaned = cleaned[4:].strip()
        prefixes = ("当前记忆：", "当前记忆:", "短期记忆：", "短期记忆:", "摘要：", "摘要:")
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        return cleaned[: self.short_term_summary_max_chars].strip()

    def _fallback_short_term_summary(
        self,
        previous_summary: str,
        messages: list[dict[str, str]],
    ) -> str:
        valuable = []
        keywords = (
            "糖尿病",
            "血糖",
            "过敏",
            "用药",
            "药",
            "早餐",
            "午餐",
            "晚餐",
            "奶茶",
            "饮料",
            "能不能喝",
            "能不能吃",
            "热量",
            "蛋白质",
            "碳水",
            "肾",
            "痛风",
            "尿酸",
        )
        for item in messages[-24:]:
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            if any(token in content for token in keywords) or item.get("role") == "user":
                valuable.append(f"{item.get('role', 'user')}: {content[:180]}")
        merged = "\n".join(part for part in [previous_summary.strip(), *valuable[-12:]] if part)
        return merged[: self.short_term_summary_max_chars].strip()

    async def _prepend_short_term_summary_context(self, memory_context: str) -> str:
        if not self.short_term_summary_enabled or not getattr(self, "store", None):
            return memory_context
        try:
            payload = await self.store.get_short_term_summary(self.role_id)
        except Exception as exc:
            logger.bind(tag=TAG).warning(f"短期记忆摘要读取失败: {exc}")
            return memory_context
        summary = (payload or {}).get("summary", "").strip()
        if not summary:
            return memory_context
        summary_context = (
            "【Short-Term Summary Memory｜压缩短期记忆】\n"
            f"- 更新时间: {(payload or {}).get('updated_at', '')}\n"
            f"- 当前记忆: {summary}\n"
            "- 使用规则: 用它理解当前会话脉络和追问指代；如果与结构化健康档案冲突，以健康档案为准。"
        )
        if memory_context:
            return f"{summary_context}\n\n{memory_context}"
        return summary_context

    async def _update_health_profile_from_messages(
        self,
        normalized_messages: list[dict[str, str]],
        reason: str = "session_end",
    ) -> dict | None:
        if not getattr(self, "health_profile_store", None) or not getattr(self, "role_id", None):
            return None
        user_texts = [
            item["content"]
            for item in normalized_messages
            if item.get("role") == "user" and item.get("content")
        ]
        if not user_texts:
            return None
        update = merge_profile_updates(
            [
                extract_health_profile_update(text, source="dialogue_user")
                for text in user_texts[-6:]
            ]
        )
        if update.is_empty():
            return None
        try:
            stats = await self.health_profile_store.apply_update(self.role_id, update)
            logger.bind(tag=TAG).info(
                "结构化健康档案已更新: "
                f"user_id={self.role_id}, scalar_count={stats['scalar_count']}, "
                f"item_count={stats['item_count']}, reason={reason}"
            )
            return stats
        except Exception as exc:
            logger.bind(tag=TAG).warning(f"结构化健康档案更新失败: {exc}")
            return None

    async def _prepend_health_profile_context(self, memory_context: str) -> str:
        if not getattr(self, "health_profile_store", None) or not getattr(self, "role_id", None):
            return memory_context
        try:
            profile_context = await self.health_profile_store.build_prompt_context(self.role_id)
        except Exception as exc:
            logger.bind(tag=TAG).warning(f"结构化健康档案检索失败: {exc}")
            return memory_context
        if not profile_context:
            return memory_context
        if memory_context:
            return f"{profile_context}\n\n{memory_context}"
        return profile_context

    async def _sync_powermem_index(self, memories) -> None:
        if not memories or not getattr(self, "powermem_index", None):
            return
        if not getattr(self.powermem_index, "enabled", False):
            return

        for memory in memories:
            try:
                index_id = await self.powermem_index.upsert_memory(memory)
                if index_id is not None:
                    await self.store.attach_powermem_index_id(memory.memory_id, index_id)
            except Exception as exc:
                logger.bind(tag=TAG).warning(
                    f"同步 PowerMem 官方检索索引失败: memory_id={memory.memory_id}, error={exc}"
                )

    @staticmethod
    def _normalize_messages(msgs) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for message in msgs:
            if isinstance(message, dict):
                role = message.get("role")
                content = message.get("content", "") or ""
            else:
                role = getattr(message, "role", None)
                content = getattr(message, "content", "") or ""
            if role == "system":
                continue

            if not content:
                continue

            text = str(content)
            try:
                if text.strip().startswith("{") and text.strip().endswith("}"):
                    data = json.loads(text)
                    if isinstance(data, dict) and "content" in data:
                        text = str(data.get("content", ""))
            except Exception:
                pass

            text = text.strip()
            if not text:
                continue
            normalized.append({"role": role or "user", "content": text})
        return normalized
