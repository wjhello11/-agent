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
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.providers.memory.clinical_ltm.models import MemoryLayer, StructuredMemory
from core.providers.memory.clinical_ltm.powermem_index import PowerMemOfficialIndex


def load_config() -> dict:
    import yaml
    config_path = ROOT / "config.yaml"
    data_config_path = ROOT / "data" / ".config.yaml"
    for path in [data_config_path, config_path]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}


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

    try:
        from loguru import logger
    except ImportError:
        import logging
        logger = logging.getLogger("resync")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    powermem_index = PowerMemOfficialIndex(ltm_config, logger)
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
