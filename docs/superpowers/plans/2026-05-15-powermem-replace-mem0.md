# PowerMem 替代 mem0 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 移除 mem0 依赖，让 PowerMem 替代 mem0 在记忆抽取阶段的语义提示（hints）功能。

**Architecture:** 将 `Mem0CognitiveExtractor` 重命名为 `ClinicalCognitiveExtractor`，删除所有 mem0 相关代码，新增 `_get_powermem_hints()` 方法调用已有的 `PowerMemOfficialIndex.search()` 获取检索提示。调整初始化顺序，先创建 `powermem_index` 再传给 extractor。

**Tech Stack:** Python 3.11, SQLite, PowerMem (powermem SDK), Pydantic

**Base Path:** `xiaozhi-esp32-server-main/main/xiaozhi-server/`

---

### Task 1: 修改 prompts.py — 更新提取提示词模板

**Files:**
- Modify: `core/providers/memory/clinical_ltm/prompts.py:55-65`
- Modify: `core/providers/memory/clinical_ltm/prompts.py:88-95`

- [ ] **Step 1: 更新 EXTRACTION_USER_TEMPLATE**

将 `可选的 mem0 语义提示:` 改为 `PowerMem 检索到的相关历史记忆:`，将 `{mem0_hints}` 改为 `{retrieval_hints}`。

```python
# 修改前（第55-65行）：
EXTRACTION_USER_TEMPLATE = """
当前用户ID: {user_id}
当前会话ID: {session_id}
当前时间: {now}

可选的 mem0 语义提示:
{mem0_hints}

最近对话:
{dialogue}
"""

# 修改后：
EXTRACTION_USER_TEMPLATE = """
当前用户ID: {user_id}
当前会话ID: {session_id}
当前时间: {now}

PowerMem 检索到的相关历史记忆:
{retrieval_hints}

最近对话:
{dialogue}
"""
```

- [ ] **Step 2: 删除 MEM0_EXTRACTION_PROMPT**

删除第88-95行的 `MEM0_EXTRACTION_PROMPT` 常量（这是给 mem0 的 `infer=True` 模式用的，PowerMem 不需要）。

```python
# 删除以下内容：
MEM0_EXTRACTION_PROMPT = """
You are a medical nutrition memory pre-processor.

Extract only medically relevant, longitudinally useful facts from the messages.
Discard weather, greetings, jokes, generic fillers, and chit-chat.
Prefer disease history, allergies, dietary restrictions, clinician instructions, medication, glucose response, repeated meal patterns.
Return concise memories only.
"""
```

- [ ] **Step 3: 验证 prompts.py 语法**

Run: `python -m py_compile core/providers/memory/clinical_ltm/prompts.py`
Expected: 无输出（编译成功）

---

### Task 2: 修改 extractor.py — 重构抽取器

**Files:**
- Modify: `core/providers/memory/clinical_ltm/extractor.py`

- [ ] **Step 1: 删除 mem0 导入，更新 prompts 导入**

```python
# 修改前（第10行）：
from mem0 import AsyncMemory, AsyncMemoryClient

# 删除该行

# 修改前（第20-24行）：
from .prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_TEMPLATE,
    MEM0_EXTRACTION_PROMPT,
)

# 修改后：
from .prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_TEMPLATE,
)
```

- [ ] **Step 2: 重命名类并重构构造函数**

```python
# 修改前（第53-68行）：
class Mem0CognitiveExtractor:
    def __init__(self, config: dict[str, Any], llm, logger):
        self.config = config
        self.llm = llm
        self.logger = logger
        self.mem0_config = config.get("mem0", {})
        prompts = config.get("prompts") if isinstance(config.get("prompts"), dict) else {}
        self.extraction_system_prompt = (
            str(prompts.get("long_term_extraction_system_prompt") or "").strip()
            or EXTRACTION_SYSTEM_PROMPT
        )
        self.extraction_user_template = (
            str(prompts.get("long_term_extraction_user_template") or "").strip()
            or EXTRACTION_USER_TEMPLATE
        )
        self._mem0_engine = None

# 修改后：
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
```

- [ ] **Step 3: 修改 extract 方法 — 变量名 mem0_hints → retrieval_hints**

```python
# 修改前（第70-81行）：
    async def extract(self, envelope: ExtractionEnvelope) -> ExtractionResult:
        mem0_hints = await self._get_mem0_hints(envelope.messages, envelope.user_id)
        if self.llm is not None:
            try:
                raw = await self._extract_with_llm(envelope, mem0_hints)
                parsed = self._parse_llm_result(raw, envelope)
                if parsed is not None:
                    return parsed
            except Exception as exc:
                self.logger.bind(tag=__name__).warning(f"LLM 结构化抽取失败，回退规则抽取: {exc}")

        return self._fallback_extract(envelope, mem0_hints)

# 修改后：
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
```

- [ ] **Step 4: 修改 _extract_with_llm 方法 — 变量名**

```python
# 修改前（第83-109行）：
    async def _extract_with_llm(
        self,
        envelope: ExtractionEnvelope,
        mem0_hints: list[str],
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
            mem0_hints="\n".join(f"- {item}" for item in mem0_hints) or "- 无",
            dialogue="\n".join(dialogue_lines),
        )
        return await asyncio.to_thread(
            self.llm.response_no_stream,
            self.extraction_system_prompt,
            user_prompt,
            max_tokens=self.config.get("extract_max_tokens", 1200),
            temperature=self.config.get("extract_temperature", 0.1),
        )

# 修改后：
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
```

- [ ] **Step 5: 替换 _get_mem0_hints 为 _get_powermem_hints**

删除原来的 `_get_mem0_hints`、`_get_mem0_engine`、`_build_default_mem0_local_config` 三个方法（第111-201行），替换为：

```python
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
```

- [ ] **Step 6: 修改 _fallback_extract 方法**

```python
# 修改前（第293-444行）：
    def _fallback_extract(
        self,
        envelope: ExtractionEnvelope,
        mem0_hints: list[str],
    ) -> ExtractionResult:
        joined_text = "\n".join(item.get("content", "") for item in envelope.messages)
        observed_at = envelope.generated_at
        result = ExtractionResult(raw_response="\n".join(mem0_hints))
        # ... 中间代码不变 ...
        if mem0_hints:
            hint = mem0_hints[0]
            result.semantic_memories.append(
                SemanticMemory(
                    user_id=envelope.user_id,
                    entity="用户",
                    attribute="语义线索",
                    value=hint[:80],
                    content=hint,
                    source="mem0语义提示",
                    observed_at=observed_at,
                    importance=0.78,
                    weight=0.78,
                    evidence=mem0_hints[:3],
                    dedupe_key=_build_dedupe_key(MemoryLayer.SEMANTIC, "用户", "语义线索", hint[:80]),
                    tags=["semantic", "mem0-hint"],
                )
            )

# 修改后：
    def _fallback_extract(
        self,
        envelope: ExtractionEnvelope,
        retrieval_hints: list[str],
    ) -> ExtractionResult:
        joined_text = "\n".join(item.get("content", "") for item in envelope.messages)
        observed_at = envelope.generated_at
        result = ExtractionResult(raw_response="\n".join(retrieval_hints))
        # ... 中间代码不变 ...
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
```

注意：`_fallback_extract` 方法中间的正则抽取代码（第302-421行）保持不变，只改参数名和最后的 hints 相关部分。

- [ ] **Step 7: 删除不再需要的 _sanitize_provider_config 函数**

删除第27-50行的 `_sanitize_provider_config` 函数（这个函数只被 `_build_default_mem0_local_config` 使用）。

- [ ] **Step 8: 验证 extractor.py 语法**

Run: `python -m py_compile core/providers/memory/clinical_ltm/extractor.py`
Expected: 无输出（编译成功）

---

### Task 3: 修改 clinical_ltm.py — 更新导入和初始化

**Files:**
- Modify: `core/providers/memory/clinical_ltm/clinical_ltm.py:10`
- Modify: `core/providers/memory/clinical_ltm/clinical_ltm.py:26-35`
- Modify: `core/providers/memory/clinical_ltm/clinical_ltm.py:74-76`

- [ ] **Step 1: 更新导入**

```python
# 修改前（第10行）：
from .extractor import Mem0CognitiveExtractor

# 修改后：
from .extractor import ClinicalCognitiveExtractor
```

- [ ] **Step 2: 更新类文档字符串**

```python
# 修改前（第28-35行）：
    """
    面向临床营养师 Agent 的多维长效记忆引擎。

    设计目标：
    1. 使用 mem0 进行认知分类与语义预提取。
    2. 使用 SQLite 保存四层结构与生命周期元数据。
    3. 使用 PowerMem 官方 AsyncMemory 负责长期记忆检索。
    4. 在生成回答前，自动把高权重历史记忆拼接到 system prompt 的 <memory> 区块。
    """

# 修改后：
    """
    面向临床营养师 Agent 的多维长效记忆引擎。

    设计目标：
    1. 使用 PowerMem 官方 AsyncMemory 负责长期记忆检索，并为记忆抽取阶段提供相关历史 hints。
    2. 使用 SQLite 保存四层结构与生命周期元数据。
    3. 本地 SQLite 负责四层结构、冲突管理、遗忘曲线、短期记忆和健康档案。
    4. 在生成回答前，自动把高权重历史记忆拼接到 system prompt 的 <memory> 区块。
    """
```

- [ ] **Step 3: 调整初始化顺序 — 先 powermem_index 再 extractor**

```python
# 修改前（第74-76行）：
            self.extractor = Mem0CognitiveExtractor(config, None, logger)
            self.lifecycle = MemoryLifecycleManager(config, None, logger)
            self.powermem_index = PowerMemOfficialIndex(config, logger)

# 修改后：
            self.powermem_index = PowerMemOfficialIndex(config, logger)
            self.extractor = ClinicalCognitiveExtractor(
                config, None, logger, powermem_index=self.powermem_index
            )
            self.lifecycle = MemoryLifecycleManager(config, None, logger)
```

- [ ] **Step 4: 验证 clinical_ltm.py 语法**

Run: `python -m py_compile core/providers/memory/clinical_ltm/clinical_ltm.py`
Expected: 无输出（编译成功）

---

### Task 4: 修改 config.yaml — 更新配置

**Files:**
- Modify: `config.yaml:382-450`

- [ ] **Step 1: 更新 clinical_ltm 注释**

```yaml
# 修改前（第383-388行）：
    # 面向临床营养师 Agent 的多维长效记忆引擎
    # 特点：
    # 1. 四层认知结构：working / factual / episodic / semantic
    # 2. mem0 local OSS 负责语义预提取与认知分类提示
    # 3. PowerMem 官方 AsyncMemory 负责长期记忆检索
    # 4. 本地 SQLite 负责四层结构、冲突管理、遗忘曲线与 working memory

# 修改后：
    # 面向临床营养师 Agent 的多维长效记忆引擎
    # 特点：
    # 1. 四层认知结构：working / factual / episodic / semantic
    # 2. PowerMem 官方 AsyncMemory 负责长期记忆检索，并为记忆抽取阶段提供相关历史 hints
    # 3. 本地 SQLite 负责四层结构、冲突管理、遗忘曲线、短期记忆和健康档案
```

- [ ] **Step 2: 禁用 mem0 配置段**

```yaml
# 修改前（第412-430行）：
    mem0:
      enabled: true
      # 推荐：local，使用 mem0 open-source 版本
      # cloud: 使用 mem0 平台托管服务
      # local: 使用本地 AsyncMemory.from_config
      # disabled: 关闭 mem0，只保留本地抽取链
      mode: local
      api_key: ""
      host: ""
      # 如果留空，会自动复用下面 powermem 的 llm / embedder 配置，
      # 并默认使用本地 qdrant 路径存储 mem0 索引
      config: {}
      # 可选：显式覆盖 mem0 local 的向量存储
      # vector_store:
      #   provider: qdrant
      #   config:
      #     path: data/mem0_qdrant
      #     collection_name: clinical_ltm_mem0
      #     embedding_model_dims: 1024

# 修改后：
    # mem0 已废弃：PowerMem 接管语义检索 hints
    mem0:
      enabled: false
      mode: disabled
```

- [ ] **Step 3: 在 powermem 段新增 extraction_hints 配置**

```yaml
# 修改前（第431-450行）：
    powermem:
      # PowerMem 官方检索层
      # 检索与索引都走官方 AsyncMemory，当前建议使用 sqlite 作为本地调试存储
      llm:
        # ... 保持不变 ...

# 修改后（在 powermem: 下新增三个字段，其余保持不变）：
    powermem:
      # PowerMem 官方检索层
      # 检索与索引都走官方 AsyncMemory，当前建议使用 sqlite 作为本地调试存储
      # 同时为记忆抽取阶段提供相关历史 hints（替代原 mem0 语义预提取）
      extraction_hints_enabled: true
      extraction_hints_limit: 6
      extraction_hints_max_chars: 1200
      llm:
        # ... 保持不变 ...
```

- [ ] **Step 4: 验证 config.yaml 格式**

Run: `python -c "import yaml; yaml.safe_load(open('config.yaml'))"`
Expected: 无输出（YAML 合法）

---

### Task 5: 全局验证 — 确认 mem0 引用已清除

**Files:**
- 验证范围: `core/providers/memory/clinical_ltm/`

- [ ] **Step 1: 编译所有修改过的文件**

Run: `python -m py_compile core/providers/memory/clinical_ltm/clinical_ltm.py core/providers/memory/clinical_ltm/extractor.py core/providers/memory/clinical_ltm/powermem_index.py`
Expected: 无输出（全部编译成功）

- [ ] **Step 2: 搜索残留的 mem0 运行时引用**

Run: `rg "from mem0|AsyncMemoryClient|_get_mem0|Mem0CognitiveExtractor" core/providers/memory/clinical_ltm/`
Expected: 无匹配结果

- [ ] **Step 3: 确认 prompts.py 不再导出 MEM0_EXTRACTION_PROMPT**

Run: `rg "MEM0_EXTRACTION_PROMPT" core/providers/memory/clinical_ltm/`
Expected: 无匹配结果

---

### Task 6: 可选 — 创建历史数据同步脚本

**Files:**
- Create: `scripts/resync_clinical_ltm_powermem.py`

- [ ] **Step 1: 创建同步脚本**

```python
"""
将 clinical_ltm.db 中尚未同步到 PowerMem 的记忆记录同步到 PowerMem 检索索引。

用法:
    python scripts/resync_clinical_ltm_powermem.py

可选参数:
    --db-path       SQLite 数据库路径（默认 data/clinical_ltm.db）
    --dry-run       只统计不同步
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.providers.memory.clinical_ltm.powermem_index import PowerMemOfficialIndex
from core.providers.memory.clinical_ltm.models import MemoryLayer, StructuredMemory
from datetime import datetime


def load_config() -> dict:
    import yaml
    config_path = ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_unsynced_memories(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT * FROM ltm_memory_items
        WHERE metadata_json NOT LIKE '%"powermem_memory_id"%'
           OR metadata_json IS NULL
        ORDER BY created_at DESC
        """
    ).fetchall()
    conn.close()
    results = []
    for row in rows:
        metadata = json.loads(row["metadata_json"] or "{}")
        if "powermem_memory_id" in metadata:
            continue
        results.append(
            {
                "memory_id": row["memory_id"],
                "user_id": row["user_id"],
                "layer": row["layer"],
                "entity": row["entity"],
                "attribute": row["attribute"],
                "value": row["value"],
                "content": row["content"],
                "source": row["source"],
                "observed_at": row["observed_at"],
                "importance": float(row["importance"]),
                "weight": float(row["weight"]),
                "locked": bool(row["locked"]),
                "dedupe_key": row["dedupe_key"],
                "evidence": json.loads(row["evidence_json"] or "[]"),
                "tags": json.loads(row["tags_json"] or "[]"),
                "metadata": metadata,
                "embedding": json.loads(row["embedding_json"] or "[]"),
            }
        )
    return results


def row_to_memory(row: dict) -> StructuredMemory:
    return StructuredMemory(
        memory_id=row["memory_id"],
        user_id=row["user_id"],
        layer=MemoryLayer(row["layer"]),
        entity=row["entity"],
        attribute=row["attribute"],
        value=row["value"],
        content=row["content"],
        source=row["source"],
        observed_at=datetime.fromisoformat(row["observed_at"]),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        importance=row["importance"],
        weight=row["weight"],
        locked=row["locked"],
        dedupe_key=row["dedupe_key"],
        evidence=row["evidence"],
        tags=row["tags"],
        metadata=row["metadata"],
        embedding=row["embedding"],
    )


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="同步 clinical_ltm.db 到 PowerMem")
    parser.add_argument("--db-path", default="data/clinical_ltm.db", help="SQLite 数据库路径")
    parser.add_argument("--dry-run", action="store_true", help="只统计不同步")
    args = parser.parse_args()

    db_path = str(ROOT / args.db_path)
    if not Path(db_path).exists():
        print(f"数据库不存在: {db_path}")
        return

    config = load_config()
    ltm_config = config.get("Memory", {}).get("clinical_ltm", {})

    print(f"正在扫描未同步记忆: {db_path}")
    unsynced = get_unsynced_memories(db_path)
    print(f"找到 {len(unsynced)} 条未同步记忆")

    if not unsynced:
        print("无需同步")
        return

    if args.dry_run:
        print("[dry-run] 以下记忆将被同步:")
        for item in unsynced[:10]:
            print(f"  - [{item['layer']}] {item['attribute']}: {item['value'][:60]}")
        if len(unsynced) > 10:
            print(f"  ... 还有 {len(unsynced) - 10} 条")
        return

    powermem_index = PowerMemOfficialIndex(ltm_config, __import__("loguru").logger)
    if not powermem_index.enabled:
        print("PowerMem 未启用，无法同步")
        return

    success_count = 0
    fail_count = 0
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    for item in unsynced:
        memory = row_to_memory(item)
        try:
            index_id = await powermem_index.upsert_memory(memory)
            if index_id is not None:
                metadata = item["metadata"]
                metadata["powermem_memory_id"] = int(index_id)
                conn.execute(
                    "UPDATE ltm_memory_items SET metadata_json = ? WHERE memory_id = ?",
                    (json.dumps(metadata, ensure_ascii=False), item["memory_id"]),
                )
                success_count += 1
            else:
                fail_count += 1
        except Exception as exc:
            print(f"  同步失败: {item['memory_id'][:16]}... error={exc}")
            fail_count += 1

    conn.commit()
    conn.close()

    print(f"同步完成: 成功 {success_count}, 失败 {fail_count}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 验收检查清单

完成所有任务后，运行以下验证：

- [ ] `python -m py_compile core/providers/memory/clinical_ltm/clinical_ltm.py` — 通过
- [ ] `python -m py_compile core/providers/memory/clinical_ltm/extractor.py` — 通过
- [ ] `python -m py_compile core/providers/memory/clinical_ltm/powermem_index.py` — 通过
- [ ] `rg "from mem0|AsyncMemoryClient|_get_mem0|Mem0CognitiveExtractor" core/providers/memory/clinical_ltm/` — 无匹配
- [ ] `rg "MEM0_EXTRACTION_PROMPT" core/providers/memory/clinical_ltm/` — 无匹配
- [ ] `python scripts/validate_clinical_knowledge_base.py` — 通过（如有）
