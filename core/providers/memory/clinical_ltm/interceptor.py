from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .models import MemoryLayer, RetrievalBundle, RetrievedMemory, WorkingTurn


class MemoryRetrievalInterceptor:
    def __init__(self, config: dict[str, Any], store, powermem_index, logger):
        self.config = config
        self.store = store
        self.powermem_index = powermem_index
        self.logger = logger
        self.retrieval_top_k = int(config.get("retrieval_top_k", 8))
        self.min_weight = float(config.get("retrieval_min_weight", 0.08))
        self.working_turns = int(config.get("working_memory_turns", 12))

    async def build_prompt_context(
        self,
        *,
        user_id: str,
        session_id: str | None,
        query: str,
        dialogue_messages: list[Any] | None = None,
    ) -> str:
        query_text = self._normalize_query(query)
        retrieved = await self._retrieve_long_term_memories(user_id=user_id, query_text=query_text)
        working_memory = self._build_working_turns(
            user_id=user_id,
            session_id=session_id or "active-session",
            dialogue_messages=dialogue_messages or [],
        )
        if not working_memory:
            working_memory = await self.store.get_working_memory(
                user_id=user_id,
                session_id=session_id,
                limit=self.working_turns,
            )

        bundle = RetrievalBundle(
            working_memory=working_memory,
            factual_memories=[item for item in retrieved if item.layer == MemoryLayer.FACTUAL],
            episodic_memories=[item for item in retrieved if item.layer == MemoryLayer.EPISODIC],
            semantic_memories=[item for item in retrieved if item.layer == MemoryLayer.SEMANTIC],
        )
        return self._format_bundle(bundle)

    async def _retrieve_long_term_memories(
        self,
        *,
        user_id: str,
        query_text: str,
    ) -> list[RetrievedMemory]:
        if self.powermem_index and getattr(self.powermem_index, "enabled", False):
            try:
                search_result = await self.powermem_index.search(
                    query=query_text,
                    user_id=user_id,
                    limit=max(self.retrieval_top_k * 3, 12),
                )
                retrieved: list[RetrievedMemory] = []
                for item in search_result.get("results", []):
                    metadata = item.get("metadata") or {}
                    ltm_memory_id = metadata.get("ltm_memory_id")
                    if not ltm_memory_id:
                        continue
                    stored = await self.store.get_memory_by_id(str(ltm_memory_id))
                    if stored is None or stored.weight < self.min_weight:
                        continue
                    retrieved.append(
                        RetrievedMemory(
                            memory_id=stored.memory_id,
                            user_id=stored.user_id,
                            layer=stored.layer,
                            content=stored.content,
                            source=stored.source,
                            observed_at=stored.observed_at,
                            weight=stored.weight,
                            importance=stored.importance,
                            locked=stored.locked,
                            score=float(item.get("score", 0.0)),
                            metadata=stored.metadata,
                        )
                    )
                if retrieved:
                    retrieved.sort(key=lambda row: (row.score, row.weight), reverse=True)
                    return retrieved[: self.retrieval_top_k]
            except Exception as exc:
                self.logger.bind(tag=__name__).warning(f"PowerMem 官方检索失败，回退本地检索: {exc}")

        query_embedding = self.store.embed_text(query_text)
        return await self.store.search_memories(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=self.retrieval_top_k,
            min_weight=self.min_weight,
        )

    def _build_working_turns(
        self,
        *,
        user_id: str,
        session_id: str,
        dialogue_messages: list[Any],
    ) -> list[WorkingTurn]:
        turns: list[WorkingTurn] = []
        for message in dialogue_messages[-self.working_turns:]:
            if getattr(message, "role", None) == "system":
                continue
            content = getattr(message, "content", None)
            if not content:
                continue
            turns.append(
                WorkingTurn(
                    user_id=user_id,
                    session_id=session_id,
                    role=getattr(message, "role", "user"),
                    content=str(content),
                    created_at=datetime.utcnow(),
                )
            )
        return turns

    def _format_bundle(self, bundle: RetrievalBundle) -> str:
        parts: list[str] = []

        if bundle.working_memory:
            parts.append("【Working Memory｜当前多轮上下文】")
            for turn in bundle.working_memory[-self.working_turns:]:
                parts.append(f"- {turn.role}: {turn.content}")

        if bundle.factual_memories:
            parts.append("【Factual Memory｜永久锁定事实】")
            for item in bundle.factual_memories[:6]:
                parts.append(self._format_memory_line(item))

        if bundle.semantic_memories:
            parts.append("【Semantic Memory｜行为规律总结】")
            for item in bundle.semantic_memories[:4]:
                parts.append(self._format_memory_line(item))

        if bundle.episodic_memories:
            parts.append("【Episodic Memory｜高关联历史事件】")
            for item in bundle.episodic_memories[:4]:
                parts.append(self._format_memory_line(item))

        if not parts:
            return "暂无可用长期记忆。"
        return "\n".join(parts)

    @staticmethod
    def _format_memory_line(item: RetrievedMemory) -> str:
        lock_tag = " | locked" if item.locked else ""
        return (
            f"- [{item.observed_at.strftime('%Y-%m-%d %H:%M')}] "
            f"{item.content} (weight={item.weight:.2f}, score={item.score:.2f}{lock_tag})"
        )

    @staticmethod
    def _normalize_query(query: str) -> str:
        try:
            if query.strip().startswith("{") and query.strip().endswith("}"):
                data = json.loads(query)
                return str(data.get("content", query))
        except Exception:
            return query
        return query
