from pathlib import Path
from typing import TYPE_CHECKING

from config.config_loader import get_project_dir
from config.logger import setup_logging
from core.clinical_nutrition.structured_knowledge import search_structured_knowledge
from plugins_func.register import Action, ActionResponse, ToolType, register_function

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

SEARCH_CLINICAL_STRUCTURED_KNOWLEDGE_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "search_clinical_structured_knowledge",
        "description": (
            "查询本地临床营养结构化知识库。适合查指南表格、BMI/腰围判定、"
            "身体活动 MET、食物交换份、减重食谱、食养方、菜谱食材克数和来源页码。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "用户提出的结构化临床营养问题。",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量，默认 6。",
                },
            },
            "required": ["question"],
        },
    },
}


@register_function(
    "search_clinical_structured_knowledge",
    SEARCH_CLINICAL_STRUCTURED_KNOWLEDGE_FUNCTION_DESC,
    ToolType.SYSTEM_CTL,
)
def search_clinical_structured_knowledge(
    conn: "ConnectionHandler",
    question=None,
    limit=6,
):
    question = str(question or "").strip()
    if not question:
        return ActionResponse(Action.RESPONSE, None, "请告诉我需要查询的结构化营养问题。")

    try:
        limit = max(1, min(int(limit or 6), 12))
    except (TypeError, ValueError):
        limit = 6

    db_path = _resolve_db_path(conn)
    if not db_path.exists():
        return ActionResponse(
            Action.RESPONSE,
            None,
            f"临床结构化知识库还没有初始化：{db_path}",
        )

    try:
        rows = search_structured_knowledge(db_path, question, limit=limit)
    except Exception as exc:
        logger.bind(tag=TAG).error(f"Search clinical structured knowledge failed: {exc}")
        return ActionResponse(Action.RESPONSE, None, "查询临床结构化知识库失败，请稍后再试。")

    if not rows:
        return ActionResponse(
            Action.RESPONSE,
            None,
            f"临床结构化知识库暂时没有查到“{question}”。",
        )

    return ActionResponse(Action.REQLLM, _format_context(question, rows), None)


def _resolve_db_path(conn: "ConnectionHandler") -> Path:
    clinical_knowledge = conn.config.get("clinical_knowledge", {})
    plugin_config = conn.config.get("plugins", {}).get("search_clinical_structured_knowledge", {})
    configured = clinical_knowledge.get("db_path") or plugin_config.get("db_path") or "data/clinical_knowledge.db"
    db_path = Path(str(configured))
    if not db_path.is_absolute():
        db_path = Path(get_project_dir()) / db_path
    return db_path


def _format_context(question: str, rows: list[dict]) -> str:
    lines = [
        f"# 临床结构化知识库查询结果：{question}",
        (
            "以下内容来自本地结构化知识库，适合回答表格、菜谱、食养方、BMI/腰围标准、MET、交换份等问题。"
            "回答时请优先给可执行结论，并保留来源页码。"
        ),
    ]
    for index, item in enumerate(rows, start=1):
        lines.append(f"\n## 结果 {index}: {item.get('title') or item.get('type')}")
        lines.append(f"- 类型: {item.get('type', '')}")
        lines.append(f"- 内容: {item.get('content', '')}")
        if item.get("citation"):
            lines.append(f"- 来源: {item['citation']}")
    return "\n".join(lines)
