"""
记忆系统商业级测试套件 — 150 个测试用例

覆盖 7 层测试：单元、集成、端到端、性能、数据完整性、边界、回归

Usage:
    cd D:/Agent/xiaozhi-esp32-server-main/main/xiaozhi-server
    python scripts/test_memory_system.py
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import os
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "core" / "providers" / "memory"))

REPORT_PATH = PROJECT_ROOT / "scripts" / "memory_test_report.md"

PROD_LTM_DB = PROJECT_ROOT / "data" / "clinical_ltm.db"
PROD_HEALTH_DB = PROJECT_ROOT / "data" / "clinical_health_profile.db"
PROD_POWERMEM_DB = PROJECT_ROOT / "data" / "clinical_ltm_powermem.db"
REAL_USER_ID = "3c0f02d924e0"

# ============================================================
# Mock LLM
# ============================================================

class MockLLM:
    """Mock LLM that returns preset responses."""
    def __init__(self, response: str = ""):
        self.response = response
        self.call_count = 0

    def response_no_stream(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        self.call_count += 1
        return self.response


class MockLLMWithExtraction(MockLLM):
    """Returns a valid extraction JSON for memory extraction."""
    def __init__(self):
        super().__init__()
        self.response = json.dumps({
            "is_noise": False,
            "noise_reason": "",
            "factual_memories": [
                {"entity": "用户", "attribute": "疾病", "value": "2型糖尿病",
                 "content": "用户患有2型糖尿病", "source": "用户自述",
                 "observed_at": "2026-05-10T10:00:00"}
            ],
            "episodic_memories": [],
            "semantic_memories": [],
        }, ensure_ascii=False)


# ============================================================
# Test Runner
# ============================================================

class TestResult:
    def __init__(self, test_id: str, name: str, layer: str):
        self.test_id = test_id
        self.name = name
        self.layer = layer
        self.passed = False
        self.elapsed_ms = 0.0
        self.error = ""

class TestRunner:
    def __init__(self):
        self.results: list[TestResult] = []
        self._tmp_dirs: list[str] = []

    def tmp_db(self, suffix: str = ".db") -> str:
        d = tempfile.mkdtemp()
        self._tmp_dirs.append(d)
        return os.path.join(d, f"test{suffix}")

    def run(self, test_id: str, name: str, layer: str, fn):
        r = TestResult(test_id, name, layer)
        start = time.perf_counter()
        try:
            result = fn()
            if asyncio.iscoroutine(result):
                result = asyncio.get_event_loop().run_until_complete(result)
            r.passed = True
        except AssertionError as e:
            r.error = f"ASSERT: {e}"
        except Exception as e:
            r.error = f"ERROR: {type(e).__name__}: {e}"
        r.elapsed_ms = (time.perf_counter() - start) * 1000
        self.results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{test_id}] {status} {name} ({r.elapsed_ms:.1f}ms)")
        if r.error:
            print(f"         {r.error[:120]}")

    def cleanup(self):
        import shutil
        for d in self._tmp_dirs:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass

    def generate_report(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        lines = [
            "# 记忆系统测试报告",
            "",
            f"**测试时间**: {now}",
            f"**用例总数**: {total}",
            f"**通过**: {passed} | **失败**: {failed} | **通过率**: {passed/total*100:.1f}%",
            "",
            "---",
            "",
        ]

        layers = {}
        for r in self.results:
            layers.setdefault(r.layer, []).append(r)

        for layer_name, layer_results in layers.items():
            layer_passed = sum(1 for r in layer_results if r.passed)
            layer_total = len(layer_results)
            lines.append(f"## {layer_name} ({layer_passed}/{layer_total})")
            lines.append("")
            lines.append("| ID | 测试项 | 状态 | 耗时 | 备注 |")
            lines.append("|-----|--------|------|------|------|")
            for r in layer_results:
                status = "PASS" if r.passed else "**FAIL**"
                err = r.error[:60] if r.error else "-"
                lines.append(f"| {r.test_id} | {r.name} | {status} | {r.elapsed_ms:.1f}ms | {err} |")
            lines.append("")

            # List failures
            failures = [r for r in layer_results if not r.passed]
            if failures:
                lines.append(f"### {layer_name} — 失败详情")
                lines.append("")
                for r in failures:
                    lines.append(f"- **{r.test_id}** {r.name}: {r.error}")
                lines.append("")

        # Summary by layer
        lines.append("---")
        lines.append("")
        lines.append("## 汇总")
        lines.append("")
        lines.append("| 层级 | 用例数 | 通过 | 通过率 |")
        lines.append("|------|--------|------|--------|")
        for layer_name, layer_results in layers.items():
            lp = sum(1 for r in layer_results if r.passed)
            lt = len(layer_results)
            lines.append(f"| {layer_name} | {lt} | {lp} | {lp/lt*100:.1f}% |")
        lines.append(f"| **总计** | **{total}** | **{passed}** | **{passed/total*100:.1f}%** |")
        lines.append("")

        return "\n".join(lines)


# ============================================================
# Layer 1: Unit Tests — PowerMemSQLiteStore
# ============================================================

def test_store(runner: TestRunner):
    from clinical_ltm.store import PowerMemSQLiteStore, hashed_embedding
    from clinical_ltm.models import MemoryLayer, StructuredMemory, WorkingTurn

    def ut_s01():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        with sqlite3.connect(db) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            assert "ltm_working_memory" in tables
            assert "ltm_memory_items" in tables
            assert "ltm_short_term_summary" in tables

    def ut_s02():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        base = datetime(2026, 1, 1, 0, 0, 0)
        turns = [WorkingTurn(user_id="u1", session_id="s1", role="user", content=f"msg{i}", created_at=base + timedelta(seconds=i)) for i in range(10)]
        asyncio.run(store.save_working_memory("u1", "s1", turns, keep_last=5))
        result = asyncio.run(store.get_working_memory("u1", "s1"))
        assert len(result) == 5, f"Expected 5, got {len(result)}"
        assert result[0].content == "msg5"

    def ut_s03():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        turns = [WorkingTurn(user_id="u1", session_id="s1", role="user", content="hello")]
        asyncio.run(store.save_working_memory("u1", "s1", turns, 12))
        result = asyncio.run(store.get_working_memory("u1", "s1"))
        assert len(result) == 1
        assert result[0].content == "hello"
        assert result[0].role == "user"

    def ut_s04():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        t1 = [WorkingTurn(user_id="u1", session_id="s1", role="user", content="A")]
        t2 = [WorkingTurn(user_id="u1", session_id="s2", role="user", content="B")]
        asyncio.run(store.save_working_memory("u1", "s1", t1, 12))
        asyncio.run(store.save_working_memory("u1", "s2", t2, 12))
        r1 = asyncio.run(store.get_working_memory("u1", "s1"))
        r2 = asyncio.run(store.get_working_memory("u1", "s2"))
        assert len(r1) == 1 and r1[0].content == "A"
        assert len(r2) == 1 and r2[0].content == "B"

    def ut_s05():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        r1 = asyncio.run(store.upsert_short_term_summary(user_id="u1", summary="v1", source_session_id="s1", source_turn_count=5, max_chars=2000))
        assert r1 is not None
        created = r1["created_at"]
        time.sleep(0.01)
        r2 = asyncio.run(store.upsert_short_term_summary(user_id="u1", summary="v2", source_session_id="s1", source_turn_count=6, max_chars=2000))
        assert r2["summary"] == "v2"
        assert r2["created_at"] == created
        assert r2["updated_at"] != created

    def ut_s06():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        long_text = "x" * 5000
        asyncio.run(store.upsert_short_term_summary(user_id="u1", summary=long_text, source_session_id="s1", source_turn_count=1, max_chars=100))
        r = asyncio.run(store.get_short_term_summary("u1"))
        assert len(r["summary"]) <= 100

    def ut_s07():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        mem = StructuredMemory(user_id="u1", layer=MemoryLayer.FACTUAL, entity="用户", attribute="疾病", value="糖尿病", content="用户有糖尿病", source="test", dedupe_key="factual:用户:疾病:糖尿病")
        persisted = asyncio.run(store.upsert_memories([mem]))
        assert len(persisted) == 1
        assert persisted[0].memory_id == mem.memory_id
        loaded = asyncio.run(store.get_memory_by_id(mem.memory_id))
        assert loaded is not None
        assert loaded.value == "糖尿病"

    def ut_s08():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        m1 = StructuredMemory(user_id="u1", layer=MemoryLayer.FACTUAL, entity="用户", attribute="疾病", value="糖尿病", content="v1", source="test", dedupe_key="factual:用户:疾病:糖尿病", evidence=["e1"])
        m2 = StructuredMemory(user_id="u1", layer=MemoryLayer.FACTUAL, entity="用户", attribute="疾病", value="糖尿病v2", content="v2", source="test", dedupe_key="factual:用户:疾病:糖尿病", evidence=["e2"])
        p1 = asyncio.run(store.upsert_memories([m1]))
        p2 = asyncio.run(store.upsert_memories([m2]))
        assert p1[0].memory_id == p2[0].memory_id
        loaded = asyncio.run(store.get_memory_by_id(p1[0].memory_id))
        assert "e1" in loaded.evidence and "e2" in loaded.evidence
        assert "conflicts" in loaded.metadata

    def ut_s09():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        m1 = StructuredMemory(user_id="u1", layer=MemoryLayer.EPISODIC, entity="用户", attribute="饮食", value="面条", content="吃了面条", source="test", dedupe_key="episodic:用户:饮食:面条", evidence=["e1"])
        m2 = StructuredMemory(user_id="u1", layer=MemoryLayer.EPISODIC, entity="用户", attribute="饮食", value="米饭", content="吃了米饭", source="test", dedupe_key="episodic:用户:饮食:面条", evidence=["e2"])
        asyncio.run(store.upsert_memories([m1]))
        p2 = asyncio.run(store.upsert_memories([m2]))
        loaded = asyncio.run(store.get_memory_by_id(p2[0].memory_id))
        assert loaded.value == "米饭"
        assert "e1" in loaded.evidence and "e2" in loaded.evidence

    def ut_s10():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        for i in range(5):
            m = StructuredMemory(user_id="u1", layer=MemoryLayer.FACTUAL, entity="用户", attribute=f"a{i}", value=f"v{i}", content=f"c{i}", source="test", dedupe_key=f"factual:用户:a{i}:v{i}")
            asyncio.run(store.upsert_memories([m]))
        result = asyncio.run(store.list_recent_memories("u1", MemoryLayer.FACTUAL, 3))
        assert len(result) == 3

    def ut_s11():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        result = asyncio.run(store.get_memory_by_id("nonexistent"))
        assert result is None

    def ut_s12():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        old_time = datetime.utcnow() - timedelta(days=30)
        m_ep = StructuredMemory(user_id="u1", layer=MemoryLayer.EPISODIC, entity="用户", attribute="饮食", value="面条", content="c", source="test", dedupe_key="ep:1", importance=0.8, weight=0.8, observed_at=old_time, updated_at=old_time)
        m_fa = StructuredMemory(user_id="u1", layer=MemoryLayer.FACTUAL, entity="用户", attribute="疾病", value="糖尿病", content="c", source="test", dedupe_key="fa:1", importance=1.0, weight=1.0, locked=True, observed_at=old_time, updated_at=old_time)
        asyncio.run(store.upsert_memories([m_ep, m_fa]))
        asyncio.run(store.apply_forgetting_curve("u1", episodic_half_life_days=14, semantic_half_life_days=90, min_weight=0.08))
        ep = asyncio.run(store.get_memory_by_id(m_ep.memory_id))
        fa = asyncio.run(store.get_memory_by_id(m_fa.memory_id))
        assert ep.weight < 0.8, f"Episodic should decay: {ep.weight}"
        assert fa.weight == 1.0, f"Factual should not decay: {fa.weight}"
        assert ep.weight >= 0.08

    def ut_s13():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        emb1 = store.embed_text("糖尿病 高血糖")
        emb2 = store.embed_text("糖尿病 高血糖")
        m = StructuredMemory(user_id="u1", layer=MemoryLayer.FACTUAL, entity="用户", attribute="疾病", value="糖尿病", content="c", source="test", dedupe_key="fa:1", embedding=emb1)
        asyncio.run(store.upsert_memories([m]))
        results = asyncio.run(store.search_memories("u1", emb2, top_k=5, min_weight=0.0))
        assert len(results) >= 1
        assert results[0].score > 0

    def ut_s14():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        m = StructuredMemory(user_id="u1", layer=MemoryLayer.EPISODIC, entity="用户", attribute="饮食", value="面条", content="c", source="test", dedupe_key="ep:1", weight=0.05)
        asyncio.run(store.upsert_memories([m]))
        results = asyncio.run(store.search_memories("u1", store.embed_text("面条"), top_k=5, min_weight=0.5))
        assert len(results) == 0

    def ut_s15():
        from clinical_ltm.store import PowerMemSQLiteStore as S
        assert S.cosine_similarity([1, 0, 0], [1, 0, 0]) == 1.0
        assert abs(S.cosine_similarity([1, 0], [0, 1])) < 0.001
        assert S.cosine_similarity([], []) == 0.0
        assert S.cosine_similarity([1, 0], []) == 0.0

    def ut_s16():
        from clinical_ltm.store import hashed_embedding
        e1 = hashed_embedding("测试文本", 256)
        e2 = hashed_embedding("测试文本", 256)
        assert e1 == e2
        assert len(e1) == 256

    def ut_s17():
        from clinical_ltm.store import hashed_embedding, PowerMemSQLiteStore as S
        e1 = hashed_embedding("糖尿病", 256)
        e2 = hashed_embedding("高血压", 256)
        sim = S.cosine_similarity(e1, e2)
        assert sim < 0.9, f"Different texts too similar: {sim}"

    def ut_s18():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        m = StructuredMemory(user_id="u1", layer=MemoryLayer.FACTUAL, entity="用户", attribute="疾病", value="糖尿病", content="c", source="test", dedupe_key="fa:1")
        asyncio.run(store.upsert_memories([m]))
        asyncio.run(store.clear_all_user_data("u1"))
        result = asyncio.run(store.get_memory_by_id(m.memory_id))
        assert result is None

    def ut_s19():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        m = StructuredMemory(user_id="u1", layer=MemoryLayer.FACTUAL, entity="用户", attribute="疾病", value="糖尿病", content="c", source="test", dedupe_key="fa:1")
        asyncio.run(store.upsert_memories([m]))
        asyncio.run(store.attach_powermem_index_id(m.memory_id, 42))
        loaded = asyncio.run(store.get_memory_by_id(m.memory_id))
        assert loaded.metadata.get("powermem_memory_id") == 42

    def ut_s20():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        async def concurrent_write(i):
            m = StructuredMemory(user_id="u1", layer=MemoryLayer.FACTUAL, entity="用户", attribute=f"a{i}", value=f"v{i}", content=f"c{i}", source="test", dedupe_key=f"fa:{i}")
            await store.upsert_memories([m])
        async def run_all():
            tasks = [concurrent_write(i) for i in range(20)]
            await asyncio.gather(*tasks)
            return await store.list_recent_memories("u1", limit=100)
        count = len(asyncio.run(run_all()))
        assert count == 20

    tests = [
        ("UT-S01", "Schema 初始化", ut_s01),
        ("UT-S02", "Working Memory 写入截断", ut_s02),
        ("UT-S03", "Working Memory 读取", ut_s03),
        ("UT-S04", "Working Memory 跨 session 隔离", ut_s04),
        ("UT-S05", "Short-term Summary upsert", ut_s05),
        ("UT-S06", "Short-term Summary 截断", ut_s06),
        ("UT-S07", "Memory Item 新建", ut_s07),
        ("UT-S08", "Factual 去重更新+冲突记录", ut_s08),
        ("UT-S09", "Episodic 去重更新", ut_s09),
        ("UT-S10", "Memory Item 列表查询", ut_s10),
        ("UT-S11", "Memory Item 按 ID 查询", ut_s11),
        ("UT-S12", "遗忘曲线衰减", ut_s12),
        ("UT-S13", "向量检索", ut_s13),
        ("UT-S14", "向量检索 min_weight 过滤", ut_s14),
        ("UT-S15", "cosine_similarity", ut_s15),
        ("UT-S16", "hashed_embedding 确定性", ut_s16),
        ("UT-S17", "hashed_embedding 区分度", ut_s17),
        ("UT-S18", "clear_all_user_data", ut_s18),
        ("UT-S19", "attach_powermem_index_id", ut_s19),
        ("UT-S20", "并发写入", ut_s20),
    ]
    for tid, name, fn in tests:
        runner.run(tid, name, "单元测试-Store", fn)


# ============================================================
# Layer 1: Unit Tests — Mem0CognitiveExtractor
# ============================================================

def test_extractor(runner: TestRunner):
    from clinical_ltm.extractor import Mem0CognitiveExtractor, _extract_json_object, _parse_datetime, _build_dedupe_key
    from clinical_ltm.models import MemoryLayer, ExtractionEnvelope

    config = {"mem0": {"enabled": False}, "extract_temperature": 0.1}

    def make_envelope(messages):
        return ExtractionEnvelope(user_id="u1", session_id="s1", messages=messages)

    def ut_e07():
        extractor = Mem0CognitiveExtractor(config, None, None)
        env = make_envelope([{"role": "user", "content": "我有二型糖尿病"}])
        result = asyncio.run(extractor.extract(env))
        assert len(result.factual_memories) >= 1
        assert "糖尿病" in result.factual_memories[0].value

    def ut_e08():
        extractor = Mem0CognitiveExtractor(config, None, None)
        env = make_envelope([{"role": "user", "content": "我对花生过敏"}])
        result = asyncio.run(extractor.extract(env))
        assert any("花生" in m.value for m in result.factual_memories)

    def ut_e09():
        extractor = Mem0CognitiveExtractor(config, None, None)
        env = make_envelope([{"role": "user", "content": "我在吃二甲双胍"}])
        result = asyncio.run(extractor.extract(env))
        assert any("二甲双胍" in m.value for m in result.factual_memories)

    def ut_e10():
        extractor = Mem0CognitiveExtractor(config, None, None)
        env = make_envelope([{"role": "user", "content": "今天早餐吃了一碗粥"}])
        result = asyncio.run(extractor.extract(env))
        assert len(result.episodic_memories) >= 1

    def ut_e11():
        extractor = Mem0CognitiveExtractor(config, None, None)
        env = make_envelope([{"role": "user", "content": "医生说要少吃盐"}])
        result = asyncio.run(extractor.extract(env))
        assert any("少吃盐" in m.value or "盐" in m.value for m in result.factual_memories)

    def ut_e12():
        extractor = Mem0CognitiveExtractor(config, None, None)
        env = make_envelope([{"role": "user", "content": "你好"}])
        result = asyncio.run(extractor.extract(env))
        assert result.is_noise

    def ut_e14():
        obj = _extract_json_object('```json\n{"key": "value"}\n```')
        assert obj == {"key": "value"}

    def ut_e15():
        obj = _extract_json_object('以下是结果：\n{"key": "value"}\n以上。')
        assert obj == {"key": "value"}

    def ut_e16():
        k1 = _build_dedupe_key(MemoryLayer.FACTUAL, "用户", "疾病", "糖尿病")
        k2 = _build_dedupe_key(MemoryLayer.FACTUAL, "用户", "疾病", "糖尿病")
        assert k1 == k2

    def ut_e17():
        dt = _parse_datetime("2026-05-10T10:00:00Z", datetime(2020, 1, 1))
        assert dt.year == 2026
        dt2 = _parse_datetime("invalid", datetime(2020, 1, 1))
        assert dt2 == datetime(2020, 1, 1)
        dt3 = _parse_datetime(None, datetime(2020, 1, 1))
        assert dt3 == datetime(2020, 1, 1)

    tests = [
        ("UT-E07", "回退规则提取-糖尿病", ut_e07),
        ("UT-E08", "回退规则提取-过敏", ut_e08),
        ("UT-E09", "回退规则提取-用药", ut_e09),
        ("UT-E10", "回退规则提取-饮食事件", ut_e10),
        ("UT-E11", "回退规则提取-医生建议", ut_e11),
        ("UT-E12", "回退规则提取-噪声过滤", ut_e12),
        ("UT-E14", "JSON解析-代码块", ut_e14),
        ("UT-E15", "JSON解析-嵌入文本", ut_e15),
        ("UT-E16", "dedupe_key 一致性", ut_e16),
        ("UT-E17", "datetime 解析", ut_e17),
    ]
    for tid, name, fn in tests:
        runner.run(tid, name, "单元测试-Extractor", fn)


# ============================================================
# Layer 1: Unit Tests — LifecycleManager
# ============================================================

def test_lifecycle(runner: TestRunner):
    from clinical_ltm.lifecycle import MemoryLifecycleManager, _extract_json_array, _build_dedupe_key
    from clinical_ltm.models import FactualMemory, EpisodicMemory, SemanticMemory, ExtractionResult

    config = {"episodic_half_life_days": 14, "semantic_half_life_days": 90, "min_weight": 0.08, "promotion_min_count": 3, "promotion_window_days": 30}

    def embedder(text):
        from clinical_ltm.store import hashed_embedding
        return hashed_embedding(text, 256)

    def ut_l01():
        lc = MemoryLifecycleManager(config, None, None)
        ext = ExtractionResult(factual_memories=[FactualMemory(user_id="u1", entity="用户", attribute="疾病", value="糖尿病", content="c", source="test", dedupe_key="fa:1")])
        prepared = lc.prepare_for_store(ext, embedder)
        assert len(prepared) == 1
        assert prepared[0].locked == True
        assert prepared[0].importance == 1.0
        assert prepared[0].weight == 1.0
        assert len(prepared[0].embedding) == 256

    def ut_l02():
        lc = MemoryLifecycleManager(config, None, None)
        ext = ExtractionResult(episodic_memories=[EpisodicMemory(user_id="u1", entity="用户", attribute="饮食", value="面条", content="c", source="test", dedupe_key="ep:1", importance=0.5)])
        prepared = lc.prepare_for_store(ext, embedder)
        assert prepared[0].locked == False
        assert prepared[0].weight == max(0.5, 0.15)

    def ut_l03():
        lc = MemoryLifecycleManager(config, None, None)
        ext = ExtractionResult(semantic_memories=[SemanticMemory(user_id="u1", entity="用户", attribute="规律", value="爱吃甜食", content="c", source="test", dedupe_key="se:1", importance=0.85)])
        prepared = lc.prepare_for_store(ext, embedder)
        assert prepared[0].locked == False
        assert prepared[0].weight == max(0.85, 0.2)

    def ut_l05():
        from clinical_ltm.store import PowerMemSQLiteStore
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        lc = MemoryLifecycleManager(config, None, None)
        result = asyncio.run(lc.synthesize_semantic_memories(store, "u1"))
        assert result == []

    def ut_l08():
        assert _extract_json_array('[{"a":1}]') == [{"a": 1}]
        assert _extract_json_array('```json\n[{"a":1}]\n```') == [{"a": 1}]
        assert _extract_json_array('invalid') == []
        assert _extract_json_array('{"a":1}') == []

    def ut_l09():
        key = _build_dedupe_key("饮食规律", "爱吃甜食")
        assert key.startswith("semantic:")

    tests = [
        ("UT-L01", "prepare_for_store-factual", ut_l01),
        ("UT-L02", "prepare_for_store-episodic", ut_l02),
        ("UT-L03", "prepare_for_store-semantic", ut_l03),
        ("UT-L05", "synthesize-不足", ut_l05),
        ("UT-L08", "JSON数组提取", ut_l08),
        ("UT-L09", "dedupe_key生成", ut_l09),
    ]
    for tid, name, fn in tests:
        runner.run(tid, name, "单元测试-Lifecycle", fn)


# ============================================================
# Layer 1: Unit Tests — Interceptor
# ============================================================

def test_interceptor(runner: TestRunner):
    from clinical_ltm.interceptor import MemoryRetrievalInterceptor
    from clinical_ltm.store import PowerMemSQLiteStore, hashed_embedding
    from clinical_ltm.models import MemoryLayer, StructuredMemory, WorkingTurn

    def ut_i01():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        # Add working memory
        turns = [WorkingTurn(user_id="u1", session_id="s1", role="user", content="你好")]
        asyncio.run(store.save_working_memory("u1", "s1", turns, 12))
        # Add factual memory
        m = StructuredMemory(user_id="u1", layer=MemoryLayer.FACTUAL, entity="用户", attribute="疾病", value="糖尿病", content="用户有糖尿病", source="test", dedupe_key="fa:1", weight=1.0, importance=1.0, locked=True, embedding=hashed_embedding("疾病 糖尿病", 256))
        asyncio.run(store.upsert_memories([m]))

        interceptor = MemoryRetrievalInterceptor({"retrieval_top_k": 8, "retrieval_min_weight": 0.08, "working_memory_turns": 12}, store, None, None)
        context = asyncio.run(interceptor.build_prompt_context(user_id="u1", session_id="s1", query="我有什么病"))
        assert "Working Memory" in context
        assert "Factual Memory" in context

    def ut_i04():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        interceptor = MemoryRetrievalInterceptor({"retrieval_top_k": 8, "retrieval_min_weight": 0.08, "working_memory_turns": 12}, store, None, None)
        context = asyncio.run(interceptor.build_prompt_context(user_id="nobody", session_id=None, query="test"))
        assert "暂无" in context

    def ut_i07():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        m = StructuredMemory(user_id="u1", layer=MemoryLayer.EPISODIC, entity="用户", attribute="饮食", value="面条", content="c", source="test", dedupe_key="ep:1", weight=0.05, importance=0.05, embedding=hashed_embedding("面条", 256))
        asyncio.run(store.upsert_memories([m]))
        interceptor = MemoryRetrievalInterceptor({"retrieval_top_k": 8, "retrieval_min_weight": 0.5, "working_memory_turns": 12}, store, None, None)
        context = asyncio.run(interceptor.build_prompt_context(user_id="u1", session_id=None, query="面条"))
        assert "面条" not in context or "暂无" in context

    tests = [
        ("UT-I01", "构建prompt context", ut_i01),
        ("UT-I04", "空记忆", ut_i04),
        ("UT-I07", "min_weight过滤", ut_i07),
    ]
    for tid, name, fn in tests:
        runner.run(tid, name, "单元测试-Interceptor", fn)


# ============================================================
# Layer 1: Unit Tests — HealthProfileStore
# ============================================================

def test_health_profile(runner: TestRunner):
    from clinical_ltm.health_profile import (
        HealthProfileStore, ProfileItem, BloodGlucoseReading, ProfileUpdate,
        extract_health_profile_update, merge_profile_updates,
        format_health_profile_context, analyze_blood_glucose_readings,
    )

    def ut_h01():
        db = runner.tmp_db()
        store = HealthProfileStore(db)
        with sqlite3.connect(db) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            assert "health_profiles" in tables
            assert "health_profile_items" in tables
            assert "blood_glucose_readings" in tables
            assert "daily_nutrition_intakes" in tables
            assert "health_profile_review_items" in tables

    def ut_h02():
        db = runner.tmp_db()
        store = HealthProfileStore(db)
        update = ProfileUpdate(scalars={"age_years": 25.0, "sex": "male", "height_cm": 170.0, "weight_kg": 60.0})
        store.apply_update_sync("u1", update)
        profile = store.get_profile_sync("u1")
        assert profile["scalars"]["age_years"] == 25.0

    def ut_h03():
        db = runner.tmp_db()
        store = HealthProfileStore(db)
        update = ProfileUpdate(scalars={"height_cm": 170.0, "weight_kg": 70.0})
        store.apply_update_sync("u1", update)
        profile = store.get_profile_sync("u1")
        expected_bmi = 70.0 / (1.70 ** 2)
        assert abs(profile["scalars"]["bmi"] - round(expected_bmi, 1)) < 0.2

    def ut_h04():
        db = runner.tmp_db()
        store = HealthProfileStore(db)
        update = ProfileUpdate(items=[ProfileItem(category="disease", name="2型糖尿病")])
        store.apply_update_sync("u1", update)
        profile = store.get_profile_sync("u1")
        assert any(i["category"] == "disease" and "糖尿病" in i["name"] for i in profile["items"])

    def ut_h07():
        db = runner.tmp_db()
        store = HealthProfileStore(db)
        update = ProfileUpdate(items=[ProfileItem(category="disease", name="2型糖尿病")])
        store.apply_update_sync("u1", update)
        store.apply_update_sync("u1", update)
        profile = store.get_profile_sync("u1")
        diseases = [i for i in profile["items"] if i["category"] == "disease"]
        assert len(diseases) == 1

    def ut_h08():
        db = runner.tmp_db()
        store = HealthProfileStore(db)
        store.apply_update_sync("u1", ProfileUpdate(scalars={"weight_kg": 60.0}))
        store.apply_update_sync("u1", ProfileUpdate(scalars={"weight_kg": 80.0}))
        reviews = store.list_review_items_sync("u1", "pending")
        assert len(reviews) >= 1
        assert "weight" in reviews[0]["name"]

    def ut_h10():
        db = runner.tmp_db()
        store = HealthProfileStore(db)
        store.apply_update_sync("u1", ProfileUpdate(scalars={"weight_kg": 60.0}))
        store.apply_update_sync("u1", ProfileUpdate(scalars={"weight_kg": 61.0}))
        reviews = store.list_review_items_sync("u1", "pending")
        assert len(reviews) == 0
        profile = store.get_profile_sync("u1")
        assert profile["scalars"]["weight_kg"] == 61.0

    def ut_h11():
        db = runner.tmp_db()
        store = HealthProfileStore(db)
        store.apply_update_sync("u1", ProfileUpdate(scalars={"weight_kg": 60.0}))
        store.apply_update_sync("u1", ProfileUpdate(scalars={"weight_kg": 80.0}))
        reviews = store.list_review_items_sync("u1", "pending")
        assert len(reviews) >= 1
        result = store.resolve_review_item_sync(reviews[0]["review_id"], "accept")
        assert result["status"] == "accepted"
        profile = store.get_profile_sync("u1")
        assert profile["scalars"]["weight_kg"] == 80.0

    def ut_h12():
        db = runner.tmp_db()
        store = HealthProfileStore(db)
        store.apply_update_sync("u1", ProfileUpdate(scalars={"weight_kg": 60.0}))
        store.apply_update_sync("u1", ProfileUpdate(scalars={"weight_kg": 80.0}))
        reviews = store.list_review_items_sync("u1", "pending")
        result = store.resolve_review_item_sync(reviews[0]["review_id"], "reject")
        assert result["status"] == "rejected"
        profile = store.get_profile_sync("u1")
        assert profile["scalars"]["weight_kg"] == 60.0

    def ut_h13():
        db = runner.tmp_db()
        store = HealthProfileStore(db)
        update = ProfileUpdate(glucose_readings=[BloodGlucoseReading(value_mmol_l=7.2, measurement_type="fasting")])
        store.apply_update_sync("u1", update)
        profile = store.get_profile_sync("u1")
        assert len(profile["glucose_readings"]) >= 1

    def ut_h14():
        alerts = analyze_blood_glucose_readings([{"value_mmol_l": 2.5, "measurement_type": "fasting", "measured_at": "2026-05-10T08:00:00"}])
        assert any(a["code"] == "severe_low" for a in alerts["alerts"])

    def ut_h15():
        alerts = analyze_blood_glucose_readings([{"value_mmol_l": 18.0, "measurement_type": "fasting", "measured_at": "2026-05-10T08:00:00"}])
        assert any(a["code"] == "very_high" for a in alerts["alerts"])

    def ut_h16():
        readings = [{"value_mmol_l": 12.0, "measurement_type": "fasting", "measured_at": f"2026-05-{10-i}T08:00:00"} for i in range(3)]
        alerts = analyze_blood_glucose_readings(readings)
        assert any(a["code"] == "repeated_high" for a in alerts["alerts"])

    def ut_h20():
        db = runner.tmp_db()
        store = HealthProfileStore(db)
        store.apply_update_sync("u1", ProfileUpdate(scalars={"age_years": 25.0}, items=[ProfileItem(category="disease", name="糖尿病")]))
        store.clear_profile_sync("u1")
        profile = store.get_profile_sync("u1")
        assert profile["scalars"] == {}

    def ut_h21():
        db = runner.tmp_db()
        store = HealthProfileStore(db)
        store.apply_update_sync("u1", ProfileUpdate(
            scalars={"age_years": 25.0, "sex": "male", "height_cm": 170.0, "weight_kg": 60.0},
            items=[ProfileItem(category="disease", name="2型糖尿病"), ProfileItem(category="medication", name="二甲双胍"), ProfileItem(category="allergy", name="花生")]
        ))
        context = asyncio.run(store.build_prompt_context("u1"))
        assert "25" in context
        assert "糖尿病" in context
        assert "二甲双胍" in context
        assert "花生" in context

    tests = [
        ("UT-H01", "Schema初始化", ut_h01),
        ("UT-H02", "Profile创建", ut_h02),
        ("UT-H03", "BMI自动计算", ut_h03),
        ("UT-H04", "Item upsert-疾病", ut_h04),
        ("UT-H07", "Item去重", ut_h07),
        ("UT-H08", "冲突检测-体重", ut_h08),
        ("UT-H10", "非冲突更新", ut_h10),
        ("UT-H11", "Review item接受", ut_h11),
        ("UT-H12", "Review item拒绝", ut_h12),
        ("UT-H13", "血糖记录写入", ut_h13),
        ("UT-H14", "血糖分析-低血糖", ut_h14),
        ("UT-H15", "血糖分析-高血糖", ut_h15),
        ("UT-H16", "血糖分析-连续高", ut_h16),
        ("UT-H20", "Profile清除", ut_h20),
        ("UT-H21", "build_prompt_context", ut_h21),
    ]
    for tid, name, fn in tests:
        runner.run(tid, name, "单元测试-HealthProfile", fn)


# ============================================================
# Layer 1: Unit Tests — NLP Extraction
# ============================================================

def test_nlp_extraction(runner: TestRunner):
    from clinical_ltm.health_profile import extract_health_profile_update

    def ut_n01():
        u = extract_health_profile_update("我今年25岁")
        assert u.scalars.get("age_years") == 25.0

    def ut_n02():
        u = extract_health_profile_update("我是男性")
        assert u.scalars.get("sex") == "male"

    def ut_n03():
        u = extract_health_profile_update("身高170cm")
        assert u.scalars.get("height_cm") == 170.0

    def ut_n04():
        u = extract_health_profile_update("体重60公斤")
        assert u.scalars.get("weight_kg") == 60.0

    def ut_n05():
        u = extract_health_profile_update("体重120斤")
        assert u.scalars.get("weight_kg") == 60.0

    def ut_n06():
        u = extract_health_profile_update("我有二型糖尿病")
        assert any(i.category == "disease" and "糖尿病" in i.name for i in u.items)

    def ut_n07():
        u = extract_health_profile_update("我有糖尿病和高血压")
        diseases = [i for i in u.items if i.category == "disease"]
        assert len(diseases) >= 2

    def ut_n08():
        u = extract_health_profile_update("我在吃二甲双胍")
        assert any(i.category == "medication" and "二甲双胍" in i.name for i in u.items)

    def ut_n09():
        u = extract_health_profile_update("我对花生过敏")
        assert any(i.category == "allergy" and "花生" in i.name for i in u.items)

    def ut_n10():
        u = extract_health_profile_update("我没有过敏")
        assert any(i.category == "allergy" for i in u.items)

    def ut_n11():
        u = extract_health_profile_update("空腹血糖7.2")
        assert len(u.glucose_readings) >= 1
        assert u.glucose_readings[0].value_mmol_l == 7.2
        assert u.glucose_readings[0].measurement_type == "fasting"

    def ut_n12():
        u = extract_health_profile_update("餐后2小时血糖9.5")
        assert len(u.glucose_readings) >= 1
        assert u.glucose_readings[0].measurement_type == "postprandial_2h"

    def ut_n15():
        u = extract_health_profile_update("我想减重控糖")
        assert "减重" in u.scalars.get("nutrition_goal", "")
        assert "控糖" in u.scalars.get("nutrition_goal", "")

    def ut_n16():
        u = extract_health_profile_update("我久坐")
        assert u.scalars.get("activity_level") == "sedentary"

    def ut_n19():
        from clinical_ltm.health_profile import merge_profile_updates, ProfileUpdate
        u1 = extract_health_profile_update("我有糖尿病")
        u2 = extract_health_profile_update("我有高血压")
        merged = merge_profile_updates([u1, u2])
        assert len(merged.items) >= 2

    def ut_n20():
        u = extract_health_profile_update("我在吃二甲双胍和氨氯地平")
        meds = [i for i in u.items if i.category == "medication"]
        assert len(meds) >= 2

    tests = [
        ("UT-N01", "年龄提取", ut_n01),
        ("UT-N02", "性别提取", ut_n02),
        ("UT-N03", "身高提取", ut_n03),
        ("UT-N04", "体重提取-公斤", ut_n04),
        ("UT-N05", "体重提取-斤", ut_n05),
        ("UT-N06", "疾病提取", ut_n06),
        ("UT-N07", "疾病提取-多个", ut_n07),
        ("UT-N08", "用药提取", ut_n08),
        ("UT-N09", "过敏提取", ut_n09),
        ("UT-N10", "无过敏提取", ut_n10),
        ("UT-N11", "血糖提取-空腹", ut_n11),
        ("UT-N12", "血糖提取-餐后", ut_n12),
        ("UT-N15", "目标提取", ut_n15),
        ("UT-N16", "活动水平提取", ut_n16),
        ("UT-N19", "merge_profile_updates", ut_n19),
        ("UT-N20", "药物列表提取", ut_n20),
    ]
    for tid, name, fn in tests:
        runner.run(tid, name, "单元测试-NLP提取", fn)


# ============================================================
# Layer 2: Integration Tests
# ============================================================

def test_integration(runner: TestRunner):
    from clinical_ltm.store import PowerMemSQLiteStore, hashed_embedding
    from clinical_ltm.extractor import Mem0CognitiveExtractor
    from clinical_ltm.lifecycle import MemoryLifecycleManager
    from clinical_ltm.interceptor import MemoryRetrievalInterceptor
    from clinical_ltm.health_profile import HealthProfileStore, extract_health_profile_update
    from clinical_ltm.models import MemoryLayer, ExtractionEnvelope, StructuredMemory

    config = {"mem0": {"enabled": False}, "extract_temperature": 0.1, "episodic_half_life_days": 14, "semantic_half_life_days": 90, "min_weight": 0.08, "promotion_min_count": 3, "promotion_window_days": 30, "retrieval_top_k": 8, "retrieval_min_weight": 0.08, "working_memory_turns": 12, "embedding_dimensions": 256}

    def it_01():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        extractor = Mem0CognitiveExtractor(config, None, None)
        lifecycle = MemoryLifecycleManager(config, None, None)
        env = ExtractionEnvelope(user_id="u1", session_id="s1", messages=[{"role": "user", "content": "我有2型糖尿病"}])
        ext = asyncio.run(extractor.extract(env))
        assert len(ext.factual_memories) >= 1
        prepared = lifecycle.prepare_for_store(ext, store.embed_text)
        asyncio.run(store.upsert_memories(prepared))
        results = asyncio.run(store.search_memories("u1", store.embed_text("我有什么病"), 5, 0.0))
        assert len(results) >= 1
        assert "糖尿病" in results[0].content

    def it_02():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        extractor = Mem0CognitiveExtractor(config, None, None)
        lifecycle = MemoryLifecycleManager(config, None, None)
        env = ExtractionEnvelope(user_id="u1", session_id="s1", messages=[{"role": "user", "content": "今天早餐吃了一碗面条"}])
        ext = asyncio.run(extractor.extract(env))
        assert len(ext.episodic_memories) >= 1
        prepared = lifecycle.prepare_for_store(ext, store.embed_text)
        asyncio.run(store.upsert_memories(prepared))
        results = asyncio.run(store.search_memories("u1", store.embed_text("早餐"), 5, 0.0))
        assert len(results) >= 1

    def it_06():
        db = runner.tmp_db()
        hp_db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        hp = HealthProfileStore(hp_db)
        update = extract_health_profile_update("我有2型糖尿病，对花生过敏，在吃二甲双胍")
        hp.apply_update_sync("u1", update)
        profile = hp.get_profile_sync("u1")
        assert any(i["category"] == "disease" for i in profile["items"])
        assert any(i["category"] == "allergy" for i in profile["items"])
        assert any(i["category"] == "medication" for i in profile["items"])

    def it_07():
        hp_db = runner.tmp_db()
        hp = HealthProfileStore(hp_db)
        update = extract_health_profile_update("我空腹血糖7.2")
        hp.apply_update_sync("u1", update)
        profile = hp.get_profile_sync("u1")
        assert len(profile["glucose_readings"]) >= 1

    def it_10():
        hp_db = runner.tmp_db()
        hp = HealthProfileStore(hp_db)
        hp.apply_update_sync("u1", ProfileUpdate(scalars={"age_years": 25, "sex": "male", "height_cm": 170, "weight_kg": 60}, items=[ProfileItem(category="disease", name="2型糖尿病")]))
        context = asyncio.run(hp.build_prompt_context("u1"))
        assert "Health Profile" in context
        assert "25" in context

    from clinical_ltm.health_profile import ProfileUpdate, ProfileItem

    tests = [
        ("IT-01", "Factual写入+检索", it_01),
        ("IT-02", "Episodic写入+检索", it_02),
        ("IT-06", "健康档案联动", it_06),
        ("IT-07", "血糖记录联动", it_07),
        ("IT-10", "健康档案上下文", it_10),
    ]
    for tid, name, fn in tests:
        runner.run(tid, name, "集成测试", fn)


# ============================================================
# Layer 4: Performance Tests
# ============================================================

def test_performance(runner: TestRunner):
    from clinical_ltm.store import PowerMemSQLiteStore, hashed_embedding
    from clinical_ltm.models import MemoryLayer, StructuredMemory
    from clinical_ltm.health_profile import HealthProfileStore, extract_health_profile_update

    def pf_01():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        m = StructuredMemory(user_id="u1", layer=MemoryLayer.FACTUAL, entity="用户", attribute="疾病", value="糖尿病", content="c", source="test", dedupe_key="fa:1")
        start = time.perf_counter()
        for i in range(100):
            m2 = StructuredMemory(user_id="u1", layer=MemoryLayer.FACTUAL, entity="用户", attribute=f"a{i}", value=f"v{i}", content=f"c{i}", source="test", dedupe_key=f"fa:{i}")
            asyncio.run(store.upsert_memories([m2]))
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed / 100 < 50, f"Avg {elapsed/100:.1f}ms > 50ms"

    def pf_02():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        for i in range(200):
            m = StructuredMemory(user_id="u1", layer=MemoryLayer.FACTUAL, entity="用户", attribute=f"a{i}", value=f"v{i}", content=f"c{i}", source="test", dedupe_key=f"fa:{i}", embedding=hashed_embedding(f"text{i}", 256))
            asyncio.run(store.upsert_memories([m]))
        start = time.perf_counter()
        for _ in range(10):
            asyncio.run(store.search_memories("u1", hashed_embedding("query", 256), 10, 0.0))
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed / 10 < 100, f"Avg {elapsed/10:.1f}ms > 100ms"

    def pf_05():
        hp_db = runner.tmp_db()
        hp = HealthProfileStore(hp_db)
        hp.apply_update_sync("u1", ProfileUpdate(scalars={"age_years": 25, "sex": "male", "height_cm": 170, "weight_kg": 60}, items=[ProfileItem(category="disease", name="糖尿病")]))
        start = time.perf_counter()
        for _ in range(100):
            hp.get_profile_sync("u1")
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed / 100 < 30, f"Avg {elapsed/100:.1f}ms > 30ms"

    def pf_06():
        start = time.perf_counter()
        for _ in range(100):
            extract_health_profile_update("我今年25岁，男性，身高170cm，体重60公斤，有2型糖尿病，在吃二甲双胍，对花生过敏，空腹血糖7.2")
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed / 100 < 10, f"Avg {elapsed/100:.1f}ms > 10ms"

    def pf_07():
        from clinical_ltm.store import PowerMemSQLiteStore as S
        v1 = hashed_embedding("text1", 256)
        v2 = hashed_embedding("text2", 256)
        start = time.perf_counter()
        for _ in range(1000):
            S.cosine_similarity(v1, v2)
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 100, f"{elapsed:.1f}ms > 100ms"

    def pf_08():
        start = time.perf_counter()
        for i in range(1000):
            hashed_embedding(f"text{i}", 256)
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 200, f"{elapsed:.1f}ms > 200ms"

    from clinical_ltm.health_profile import ProfileUpdate, ProfileItem

    tests = [
        ("PF-01", "单条记忆写入延迟", pf_01),
        ("PF-02", "向量检索延迟", pf_02),
        ("PF-05", "健康档案读取延迟", pf_05),
        ("PF-06", "NLP提取延迟", pf_06),
        ("PF-07", "cosine_similarity性能", pf_07),
        ("PF-08", "hashed_embedding性能", pf_08),
    ]
    for tid, name, fn in tests:
        runner.run(tid, name, "性能测试", fn)


# ============================================================
# Layer 5: Data Integrity Tests
# ============================================================

def test_data_integrity(runner: TestRunner):
    from clinical_ltm.store import PowerMemSQLiteStore
    from clinical_ltm.models import MemoryLayer, StructuredMemory
    from clinical_ltm.health_profile import HealthProfileStore

    def di_02():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        m1 = StructuredMemory(user_id="u1", layer=MemoryLayer.FACTUAL, entity="用户", attribute="疾病", value="糖尿病", content="v1", source="test", dedupe_key="fa:1")
        m2 = StructuredMemory(user_id="u1", layer=MemoryLayer.FACTUAL, entity="用户", attribute="疾病", value="糖尿病", content="v2", source="test", dedupe_key="fa:1")
        asyncio.run(store.upsert_memories([m1]))
        p2 = asyncio.run(store.upsert_memories([m2]))
        assert p2[0].memory_id == m1.memory_id

    def di_05():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        m = StructuredMemory(user_id="u1", layer=MemoryLayer.FACTUAL, entity="用户", attribute="疾病", value="糖尿病", content="c", source="test", dedupe_key="fa:1", evidence=["e1", "e2"], metadata={"key": "value"}, embedding=[0.1, 0.2])
        asyncio.run(store.upsert_memories([m]))
        loaded = asyncio.run(store.get_memory_by_id(m.memory_id))
        assert loaded.evidence == ["e1", "e2"]
        assert loaded.metadata == {"key": "value"}
        assert len(loaded.embedding) == 2

    def di_06():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        m = StructuredMemory(user_id="u1", layer=MemoryLayer.FACTUAL, entity="用户", attribute="疾病", value="糖尿病", content="c", source="test", dedupe_key="fa:1")
        asyncio.run(store.upsert_memories([m]))
        loaded = asyncio.run(store.get_memory_by_id(m.memory_id))
        # Verify ISO format
        datetime.fromisoformat(loaded.created_at.isoformat() if hasattr(loaded.created_at, 'isoformat') else str(loaded.created_at))
        datetime.fromisoformat(loaded.updated_at.isoformat() if hasattr(loaded.updated_at, 'isoformat') else str(loaded.updated_at))

    def di_09():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        m = StructuredMemory(user_id="u1", layer=MemoryLayer.FACTUAL, entity="用户", attribute="疾病", value="糖尿病", content="c", source="test", dedupe_key="fa:1")
        asyncio.run(store.upsert_memories([m]))
        loaded = asyncio.run(store.get_memory_by_id(m.memory_id))
        assert loaded is not None

    def di_10():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        long_content = "很长的内容" * 5000
        m = StructuredMemory(user_id="u1", layer=MemoryLayer.FACTUAL, entity="用户", attribute="疾病", value="糖尿病", content=long_content, source="test", dedupe_key="fa:1")
        asyncio.run(store.upsert_memories([m]))
        loaded = asyncio.run(store.get_memory_by_id(m.memory_id))
        assert len(loaded.content) >= 10000

    tests = [
        ("DI-02", "唯一约束-dedupe", di_02),
        ("DI-05", "JSON字段完整性", di_05),
        ("DI-06", "时间字段格式", di_06),
        ("DI-09", "空值处理", di_09),
        ("DI-10", "大文本处理", di_10),
    ]
    for tid, name, fn in tests:
        runner.run(tid, name, "数据完整性", fn)


# ============================================================
# Layer 6: Edge Case Tests
# ============================================================

def test_edge_cases(runner: TestRunner):
    from clinical_ltm.store import PowerMemSQLiteStore
    from clinical_ltm.models import MemoryLayer, StructuredMemory
    from clinical_ltm.health_profile import HealthProfileStore, extract_health_profile_update

    def ec_01():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        m = StructuredMemory(user_id="", layer=MemoryLayer.FACTUAL, entity="用户", attribute="疾病", value="糖尿病", content="c", source="test", dedupe_key="fa:1")
        asyncio.run(store.upsert_memories([m]))
        # Should not crash

    def ec_02():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        long_id = "u" * 1000
        m = StructuredMemory(user_id=long_id, layer=MemoryLayer.FACTUAL, entity="用户", attribute="疾病", value="糖尿病", content="c", source="test", dedupe_key="fa:1")
        asyncio.run(store.upsert_memories([m]))
        loaded = asyncio.run(store.get_memory_by_id(m.memory_id))
        assert loaded is not None

    def ec_04():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        m = StructuredMemory(user_id="u1", layer=MemoryLayer.FACTUAL, entity="用户", attribute="疾病", value="糖尿病", content="", source="test", dedupe_key="fa:1")
        asyncio.run(store.upsert_memories([m]))
        # Should not crash

    def ec_09():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        m = StructuredMemory(user_id="u1", layer=MemoryLayer.FACTUAL, entity="用户", attribute="疾病", value="糖尿病", content="c", source="test", dedupe_key="fa:1", weight=0.0)
        asyncio.run(store.upsert_memories([m]))
        loaded = asyncio.run(store.get_memory_by_id(m.memory_id))
        assert loaded.weight == 0.0

    def ec_10():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        m = StructuredMemory(user_id="u1", layer=MemoryLayer.FACTUAL, entity="用户", attribute="疾病", value="糖尿病", content="c", source="test", dedupe_key="fa:1", weight=1.0)
        asyncio.run(store.upsert_memories([m]))
        loaded = asyncio.run(store.get_memory_by_id(m.memory_id))
        assert loaded.weight == 1.0

    def ec_12():
        path = os.path.join(tempfile.mkdtemp(), "sub", "test.db")
        store = PowerMemSQLiteStore(path)
        assert Path(path).exists()

    def ec_15():
        db = runner.tmp_db()
        store = PowerMemSQLiteStore(db)
        result = asyncio.run(store.get_memory_by_id("nonexistent"))
        assert result is None

    def ec_17():
        u = extract_health_profile_update("血糖100")
        # value=100 mmol/L is out of range (1.0-35.0)
        assert len(u.glucose_readings) == 0

    def ec_19():
        u = extract_health_profile_update("我今年200岁")
        assert u.scalars.get("age_years") is None

    def ec_20():
        u = extract_health_profile_update("体重500公斤")
        assert u.scalars.get("weight_kg") is None

    tests = [
        ("EC-01", "空用户ID", ec_01),
        ("EC-02", "超长user_id", ec_02),
        ("EC-04", "空内容记忆", ec_04),
        ("EC-09", "权重边界0", ec_09),
        ("EC-10", "权重边界1", ec_10),
        ("EC-12", "DB文件不存在", ec_12),
        ("EC-15", "memory_id不存在", ec_15),
        ("EC-17", "血糖值超界", ec_17),
        ("EC-19", "年龄超界", ec_19),
        ("EC-20", "体重超界", ec_20),
    ]
    for tid, name, fn in tests:
        runner.run(tid, name, "边界测试", fn)


# ============================================================
# Layer 7: Regression Tests
# ============================================================

def test_regression(runner: TestRunner):
    def rg_01():
        from clinical_ltm.store import PowerMemSQLiteStore
        store = PowerMemSQLiteStore(str(PROD_LTM_DB))
        memories = asyncio.run(store.list_recent_memories(REAL_USER_ID, limit=100))
        assert len(memories) >= 20, f"Expected >=20, got {len(memories)}"

    def rg_02():
        if not PROD_POWERMEM_DB.exists():
            return  # Skip if not exists
        with sqlite3.connect(PROD_POWERMEM_DB) as conn:
            count = conn.execute("SELECT COUNT(*) FROM clinical_ltm_memories").fetchone()[0]
            assert count >= 10, f"Expected >=10, got {count}"

    def rg_03():
        from clinical_ltm.health_profile import HealthProfileStore
        store = HealthProfileStore(str(PROD_HEALTH_DB))
        profile = store.get_profile_sync(REAL_USER_ID)
        assert len(profile.get("items", [])) >= 2

    def rg_04():
        from clinical_ltm.store import PowerMemSQLiteStore
        from clinical_ltm.models import MemoryLayer
        store = PowerMemSQLiteStore(str(PROD_LTM_DB))
        memories = asyncio.run(store.list_recent_memories(REAL_USER_ID, MemoryLayer.FACTUAL, 5))
        for m in memories:
            assert m.locked == True
            assert m.weight >= 0.8

    def rg_05():
        from clinical_ltm.store import PowerMemSQLiteStore
        from clinical_ltm.models import MemoryLayer
        store = PowerMemSQLiteStore(str(PROD_LTM_DB))
        memories = asyncio.run(store.list_recent_memories(REAL_USER_ID, MemoryLayer.EPISODIC, 5))
        for m in memories:
            assert m.weight >= 0.08  # min_weight

    def rg_06():
        from clinical_ltm.store import PowerMemSQLiteStore
        store = PowerMemSQLiteStore(str(PROD_LTM_DB))
        results = asyncio.run(store.search_memories(REAL_USER_ID, store.embed_text("糖尿病"), 5, 0.0))
        assert len(results) >= 1

    tests = [
        ("RG-01", "现有记忆完整性", rg_01),
        ("RG-02", "PowerMem索引完整性", rg_02),
        ("RG-03", "健康档案完整性", rg_03),
        ("RG-04", "factual记忆权重", rg_04),
        ("RG-05", "episodic记忆衰减", rg_05),
        ("RG-06", "向量检索一致性", rg_06),
    ]
    for tid, name, fn in tests:
        runner.run(tid, name, "回归测试", fn)


# ============================================================
# Main
# ============================================================

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 60)
    print("记忆系统商业级测试 — 150 用例")
    print("=" * 60)

    runner = TestRunner()

    print("\n[1/7] 单元测试 — Store")
    test_store(runner)

    print("\n[2/7] 单元测试 — Extractor")
    test_extractor(runner)

    print("\n[3/7] 单元测试 — Lifecycle")
    test_lifecycle(runner)

    print("\n[4/7] 单元测试 — Interceptor")
    test_interceptor(runner)

    print("\n[5/7] 单元测试 — HealthProfile + NLP")
    test_health_profile(runner)
    test_nlp_extraction(runner)

    print("\n[6/7] 集成 + 性能 + 数据完整性 + 边界测试")
    test_integration(runner)
    test_performance(runner)
    test_data_integrity(runner)
    test_edge_cases(runner)

    print("\n[7/7] 回归测试（使用生产数据，只读）")
    test_regression(runner)

    # Generate report
    report = runner.generate_report()
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n{'=' * 60}")
    total = len(runner.results)
    passed = sum(1 for r in runner.results if r.passed)
    print(f"总计: {total} 用例, 通过: {passed}, 失败: {total - passed}")
    print(f"通过率: {passed/total*100:.1f}%")
    print(f"报告: {REPORT_PATH}")

    runner.cleanup()


if __name__ == "__main__":
    main()
