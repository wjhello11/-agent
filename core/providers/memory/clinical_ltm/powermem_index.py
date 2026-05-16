from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from powermem import AsyncMemory


def _sanitize_provider_config(component_cfg: dict[str, Any]) -> dict[str, Any]:
    sanitized = deepcopy(component_cfg or {})
    provider = str(sanitized.get("provider", "")).lower()
    config = sanitized.get("config")
    if not isinstance(config, dict):
        return sanitized

    cleaned = deepcopy(config)
    if provider == "qwen":
        cleaned.pop("openai_base_url", None)
        cleaned.pop("base_url", None)
        cleaned.pop("api_base", None)
        cleaned.pop("openai_api_base", None)
    elif provider == "openai":
        cleaned.pop("dashscope_base_url", None)

    sanitized["config"] = cleaned
    return sanitized


class PowerMemOfficialIndex:
    """
    使用 PowerMem 官方 AsyncMemory 作为长期记忆检索层。

    我们仍然在本地 SQLite 中维护四层结构与生命周期元数据，
    但真正的语义/混合检索交给 PowerMem 官方 SDK。
    """

    def __init__(self, config: dict[str, Any], logger):
        self.config = config
        self.logger = logger
        self.enabled = False
        self.client: AsyncMemory | None = None
        self.runtime_config: dict[str, Any] = self._build_runtime_config()

        try:
            self.client = AsyncMemory(config=self.runtime_config)
            self.enabled = True
            vector_cfg = self.runtime_config.get("vector_store", {})
            embedder_cfg = self.runtime_config.get("embedder", {})
            self.logger.bind(tag=__name__).info(
                "PowerMem 官方检索层初始化成功: "
                f"vector_store={vector_cfg.get('provider')}, "
                f"embedder={embedder_cfg.get('provider')}"
            )
        except Exception as exc:
            self.logger.bind(tag=__name__).error(f"PowerMem 官方检索层初始化失败: {exc}")
            self.enabled = False

    async def upsert_memory(self, memory) -> int | None:
        if not self.enabled or self.client is None:
            return None

        metadata = deepcopy(memory.metadata or {})
        metadata.update(
            {
                "ltm_memory_id": memory.memory_id,
                "layer": memory.layer.value,
                "attribute": memory.attribute,
                "value": memory.value,
                "weight": memory.weight,
                "importance": memory.importance,
                "locked": memory.locked,
                "dedupe_key": memory.dedupe_key,
            }
        )

        existing_index_id = metadata.get("powermem_memory_id")
        if existing_index_id:
            await self.client.update(
                memory_id=int(existing_index_id),
                content=memory.content,
                user_id=memory.user_id,
                metadata=metadata,
            )
            return int(existing_index_id)

        result = await self.client.add(
            messages=memory.content,
            user_id=memory.user_id,
            metadata=metadata,
            infer=False,
        )
        first = (result.get("results") or [{}])[0]
        index_id = first.get("id")
        return int(index_id) if index_id is not None else None

    async def search(self, query: str, user_id: str, limit: int = 12) -> dict[str, Any]:
        if not self.enabled or self.client is None:
            return {"results": []}
        return await self.client.search(query=query, user_id=user_id, limit=limit)

    async def clear_user_memory(self, user_id: str) -> None:
        if not self.enabled or self.client is None:
            return
        await self.client.delete_all(user_id=user_id)

    def _build_runtime_config(self) -> dict[str, Any]:
        powermem_cfg = deepcopy(self.config.get("powermem", {}))
        embedding_dimensions = int(self.config.get("embedding_dimensions", 256))

        llm_cfg = powermem_cfg.get("llm") or {
            "provider": "openai",
            "config": {
                "api_key": "replace-with-real-key",
                "model": "replace-with-real-model",
                "openai_base_url": "https://api.openai.com/v1",
            },
        }
        llm_cfg = _sanitize_provider_config(llm_cfg)

        embedder_cfg = powermem_cfg.get("embedder") or {
            "provider": "mock",
            "config": {
                "embedding_dims": embedding_dimensions,
            },
        }
        embedder_cfg = _sanitize_provider_config(embedder_cfg)

        vector_store_cfg = powermem_cfg.get("vector_store") or {
            "provider": "sqlite",
            "config": {
                "database_path": str(Path(self.config.get("sqlite_path", "data/clinical_ltm.db")).with_name("clinical_ltm_powermem.db")),
                "collection_name": "clinical_ltm_memories",
                "embedding_model_dims": embedding_dimensions,
            },
        }
        vector_store_cfg = deepcopy(vector_store_cfg)
        vector_store_cfg.setdefault("config", {})
        vector_store_cfg["config"].setdefault("embedding_model_dims", embedding_dimensions)
        vector_store_cfg["config"].setdefault(
            "database_path",
            str(Path(self.config.get("sqlite_path", "data/clinical_ltm.db")).with_name("clinical_ltm_powermem.db")),
        )
        vector_store_cfg["config"].setdefault("collection_name", "clinical_ltm_memories")

        return {
            "llm": llm_cfg,
            "embedder": embedder_cfg,
            "vector_store": vector_store_cfg,
        }
