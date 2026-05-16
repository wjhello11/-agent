from pathlib import Path
from typing import TYPE_CHECKING

from config.config_loader import get_project_dir
from config.logger import setup_logging
from core.clinical_nutrition.clinical_rag import ClinicalRAGService
from plugins_func.register import Action, ActionResponse, ToolType, register_function

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

SEARCH_CLINICAL_RAG_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "search_clinical_rag",
        "description": (
            "检索本地 Clinical RAG 原文证据库。适合需要查看上传文档原文片段、指南证据、"
            "共识依据、菜谱原文、表格片段、疾病饮食原则、糖尿病、肾病、体重管理等问题。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "用户提出的临床营养或上传文档相关问题。",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的证据片段数量，默认 6。",
                },
            },
            "required": ["question"],
        },
    },
}


@register_function(
    "search_clinical_rag", SEARCH_CLINICAL_RAG_FUNCTION_DESC, ToolType.SYSTEM_CTL
)
def search_clinical_rag(conn: "ConnectionHandler", question=None, top_k=None):
    question = question if isinstance(question, str) else str(question or "")
    question = question.strip()
    if not question:
        return ActionResponse(
            Action.RESPONSE,
            None,
            "请告诉我需要检索的临床营养问题。",
        )

    try:
        limit = int(top_k or 0) or None
    except (TypeError, ValueError):
        limit = None

    try:
        service = ClinicalRAGService(
            project_root=Path(get_project_dir()),
            config=conn.config,
            logger=logger,
        )
        results = service.search(question, top_k=limit)
    except Exception as exc:
        logger.bind(tag=TAG).error(f"Clinical RAG search failed: {exc}")
        return ActionResponse(
            Action.RESPONSE,
            None,
            "本地 Clinical RAG 检索失败，请稍后再试。",
        )

    if not results:
        return ActionResponse(
            Action.RESPONSE,
            None,
            "本地 Clinical RAG 暂时没有检索到足够相关的资料。",
        )

    context_lines = [
        f"# 本地 Clinical RAG 检索结果：{question}",
        (
            "请优先依据以下原文证据回答。涉及指南、共识或上传资料内容时，要说明依据；"
            "如果证据不足，请直接说资料里没有找到明确依据。结构化营养数字仍以营养库工具为准，"
            "用户个人疾病、用药、过敏、体重等事实仍以健康档案为准，安全红线规则优先级最高。"
        ),
    ]
    for index, item in enumerate(results, start=1):
        score = item.get("score", 0.0)
        context_lines.extend(
            [
                "",
                f"## 证据 {index}",
                f"引用: {item.get('citation', '')}",
                f"标题: {item.get('title', '')}",
                f"章节: {item.get('section_title', '') or '未标注'}",
                f"相关度: {score:.3f}",
                str(item.get("text") or "").strip(),
            ]
        )

    return ActionResponse(Action.REQLLM, "\n".join(context_lines), None)
