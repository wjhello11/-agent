import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

from config.config_loader import get_project_dir
from config.logger import setup_logging
from plugins_func.register import Action, ActionResponse, ToolType, register_function

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()
VERSIONED_WIKI_DIR_RE = re.compile(r"^(?P<base>.+)-v(?P<ts>\d{14})(?:-\d+)?$")
DEFAULT_EXCLUDED_WIKI_DIRS = {"archive", "archived", "raw", "templates"}

# 中文关键词 → 英文 metadata 扩展映射
QUERY_EXPANSIONS: dict[str, list[str]] = {
    "低血糖": ["hypoglycemia", "低血糖", "血糖低"],
    "高血糖": ["hyperglycemia", "高血糖", "血糖高"],
    "过敏": ["allergy", "过敏"],
    "花生": ["peanut", "花生"],
    "肾病": ["kidney_disease", "肾脏", "肾功能", "肾"],
    "肾脏": ["kidney", "肾脏", "肾功能", "肾"],
    "蛋白质": ["protein", "蛋白质", "蛋白"],
    "碳水": ["carbohydrate", "碳水化合物", "碳水"],
    "碳水化合物": ["carbohydrate", "碳水化合物", "碳水"],
    "GI": ["gi", "glycemic_index", "升糖指数"],
    "GL": ["gl", "glycemic_load", "血糖负荷"],
    "早餐": ["breakfast", "早餐"],
    "主食": ["staple", "主食", "碳水", "谷薯"],
    "膳食纤维": ["fiber", "膳食纤维", "纤维"],
    "禁忌": ["contraindication", "禁忌", "不宜"],
    "药物": ["drug", "药物", "用药", "降糖药"],
    "降糖药": ["drug", "降糖药", "药物", "用药"],
    "肥胖": ["obesity", "overweight", "肥胖", "超重"],
    "糖尿病": ["diabetes", "糖尿病"],
    "高血压": ["hypertension", "高血压"],
    "脂肪": ["fat", "脂肪", "油脂"],
    "能量": ["energy", "能量", "热量"],
    "胰岛素": ["insulin", "胰岛素"],
    "运动": ["exercise", "运动", "锻炼"],
}

SEARCH_FROM_LLMWIKI_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "search_from_llmwiki",
        "description": (
            "检索本地 LLMWiki 临床营养知识库。适合回答已经整理成知识页的指南原则、"
            "糖尿病营养、肥胖管理、食物选择、过敏、早餐搭配、饮食原则和安全边界等问题。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "用户提出的临床营养相关问题。",
                }
            },
            "required": ["question"],
        },
    },
}


@register_function(
    "search_from_llmwiki", SEARCH_FROM_LLMWIKI_FUNCTION_DESC, ToolType.SYSTEM_CTL
)
def search_from_llmwiki(conn: "ConnectionHandler", question=None):
    question = question if isinstance(question, str) else str(question or "")
    question = question.strip()
    if not question:
        return ActionResponse(Action.RESPONSE, None, "知识库查询问题不能为空。")

    wiki_config = conn.config.get("plugins", {}).get("search_from_llmwiki", {})
    project_dir = Path(get_project_dir())
    wiki_root = Path(
        wiki_config.get("wiki_root", "knowledge_base/llmwiki/clinical-nutrition")
    )
    if not wiki_root.is_absolute():
        wiki_root = project_dir / wiki_root
    index_file_name = str(wiki_config.get("index_file", "_index.md"))
    top_k = int(wiki_config.get("top_k", 4))
    snippet_chars = int(wiki_config.get("snippet_chars", 480))
    excluded_dirs = {
        str(item).strip().lower()
        for item in wiki_config.get("excluded_dirs", ["raw"])
    }

    if not wiki_root.exists():
        logger.bind(tag=TAG).error(f"LLMWiki root not found: {wiki_root}")
        return ActionResponse(
            Action.RESPONSE,
            None,
            f"本地 LLMWiki 目录不存在：{wiki_root}",
        )

    documents = _load_markdown_documents(wiki_root, excluded_dirs)
    if not documents:
        return ActionResponse(
            Action.RESPONSE,
            None,
            "本地 LLMWiki 中还没有可检索的内容。",
        )

    ranked = _rank_documents(
        question,
        documents,
        top_k=top_k,
        snippet_chars=snippet_chars,
    )
    if not ranked:
        fallback = next(
            (doc for doc in documents if doc["path"].name == index_file_name),
            None,
        )
        if fallback is None:
            return ActionResponse(
                Action.RESPONSE,
                None,
                "LLMWiki 里暂时没有找到相关内容。",
            )
        ranked = [
            {
                "title": fallback["title"],
                "relative_path": fallback["relative_path"],
                "snippet": fallback["snippet"],
                "score": 0.0,
            }
        ]

    context_lines = [f"# 本地 LLMWiki 检索结果：{question}"]
    context_lines.append(
        "请优先依据以下已整理的 LLMWiki 知识页回答。"
        "如果资料不足，再结合通用医学营养知识，并明确区分“知识库内容”和“通用推断”。"
    )
    for idx, item in enumerate(ranked, start=1):
        context_lines.append(f"\n## 资料 {idx}: {item['title']}")
        context_lines.append(f"来源文件: {item['relative_path']}")
        context_lines.append(item["snippet"])

    return ActionResponse(Action.REQLLM, "\n".join(context_lines), None)


def _load_markdown_documents(wiki_root: Path, excluded_dirs: set[str]) -> list[dict]:
    excluded_dirs = {str(item).strip().lower() for item in excluded_dirs}
    excluded_dirs.update(DEFAULT_EXCLUDED_WIKI_DIRS)
    latest_version_dirs = _latest_version_dirs(wiki_root, excluded_dirs)
    documents = []
    for path in wiki_root.rglob("*.md"):
        if _should_skip_wiki_markdown(path, wiki_root, excluded_dirs, latest_version_dirs):
            continue
        # 排除噪音文件
        if path.name.lower() in ("readme.md",):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        metadata, body = _split_frontmatter(text)
        cleaned = body.strip()
        if not cleaned:
            continue
        title = _extract_title(path, cleaned, metadata)
        relative_path = path.relative_to(wiki_root).as_posix()

        # 将 frontmatter 元数据（clinical_domain, conditions）也纳入 tokens
        meta_tokens_source = " ".join([
            metadata.get("clinical_domain", ""),
            metadata.get("conditions", ""),
            metadata.get("kb_layer", ""),
        ]).replace("[", "").replace("]", "").replace(",", " ").replace('"', "")

        all_text = f"{title}\n{meta_tokens_source}\n{cleaned}"
        documents.append(
            {
                "path": path,
                "relative_path": relative_path,
                "title": title,
                "metadata": metadata,
                "text": cleaned,
                "snippet": _normalize_whitespace(cleaned)[:600],
                "tokens": _tokenize(all_text),
                "meta_token_set": set(_tokenize(meta_tokens_source)),
            }
        )
    return documents


def _latest_version_dirs(wiki_root: Path, excluded_dirs: set[str]) -> dict[tuple[str, str], tuple[str, Path]]:
    latest: dict[tuple[str, str], tuple[str, Path]] = {}
    for directory in wiki_root.rglob("*"):
        if not directory.is_dir():
            continue
        try:
            relative_parts = directory.relative_to(wiki_root).parts
        except ValueError:
            relative_parts = directory.parts
        if any(part.lower() in excluded_dirs for part in relative_parts):
            continue
        match = VERSIONED_WIKI_DIR_RE.match(directory.name)
        if not match:
            continue
        parent = directory.parent.relative_to(wiki_root).as_posix()
        base = _normalize_version_base(match.group("base"))
        key = (parent, base)
        timestamp = match.group("ts")
        previous = latest.get(key)
        if previous is None or timestamp > previous[0]:
            latest[key] = (timestamp, directory.resolve())
    return latest


def _should_skip_wiki_markdown(
    path: Path,
    wiki_root: Path,
    excluded_dirs: set[str],
    latest_version_dirs: dict[tuple[str, str], tuple[str, Path]],
) -> bool:
    try:
        relative_parts = path.relative_to(wiki_root).parts
    except ValueError:
        relative_parts = path.parts
    if any(part.lower() in excluded_dirs for part in relative_parts):
        return True

    version_info = _version_dir_info(path, wiki_root)
    if version_info:
        key, timestamp, directory = version_info
        latest = latest_version_dirs.get(key)
        return latest is not None and (
            timestamp != latest[0] or directory.resolve() != latest[1]
        )

    return _is_legacy_dir_shadowed_by_version(path, wiki_root, latest_version_dirs)


def _version_dir_info(path: Path, wiki_root: Path) -> tuple[tuple[str, str], str, Path] | None:
    for directory in [path.parent, *path.parents]:
        if directory == wiki_root or wiki_root not in directory.parents:
            continue
        match = VERSIONED_WIKI_DIR_RE.match(directory.name)
        if not match:
            continue
        parent = directory.parent.relative_to(wiki_root).as_posix()
        base = _normalize_version_base(match.group("base"))
        return (parent, base), match.group("ts"), directory
    return None


def _is_legacy_dir_shadowed_by_version(
    path: Path,
    wiki_root: Path,
    latest_version_dirs: dict[tuple[str, str], tuple[str, Path]],
) -> bool:
    for directory in [path.parent, *path.parents]:
        if directory == wiki_root or wiki_root not in directory.parents:
            continue
        if VERSIONED_WIKI_DIR_RE.match(directory.name):
            continue
        parent = directory.parent.relative_to(wiki_root).as_posix()
        key = (parent, _normalize_version_base(directory.name))
        if key in latest_version_dirs:
            return True
    return False


def _normalize_version_base(name: str) -> str:
    value = re.sub(r"-v\d{14}(?:-\d+)?$", "", str(name).strip())
    previous = None
    while previous != value:
        previous = value
        value = re.sub(
            r"[-_\s]*wiki[-_\s]*(?:\u603b\u7d22\u5f15|index)?$",
            "",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(r"[-_\s]+$", "", value)
    return value.lower()


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            metadata = _parse_frontmatter_lines(lines[1:idx])
            body = "\n".join(lines[idx + 1 :])
            return metadata, body
    return {}, text


def _parse_frontmatter_lines(lines: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata


def _extract_title(path: Path, text: str, metadata: dict[str, str] | None = None) -> str:
    metadata = metadata or {}
    if metadata.get("title"):
        return metadata["title"]
    for line in text.splitlines():
        striped = line.strip()
        if striped.startswith("#"):
            return striped.lstrip("#").strip()
    return path.stem.replace("-", " ").replace("_", " ").strip() or path.name


def _rank_documents(
    question: str,
    documents: list[dict],
    top_k: int,
    snippet_chars: int,
) -> list[dict]:
    query_tokens = _expand_query_tokens(question)
    if not query_tokens:
        return []

    n_docs = len(documents)
    # 计算 IDF: idf(token) = log(N / df(token))
    doc_freq: dict[str, int] = {}
    for doc in documents:
        for token in doc["tokens"]:
            doc_freq[token] = doc_freq.get(token, 0) + 1

    def _idf(token: str) -> float:
        df = doc_freq.get(token, 1)
        return math.log((n_docs + 1) / (df + 1)) + 1.0  # smoothed IDF, always > 0

    ranked = []
    for doc in documents:
        title_tokens = set(_tokenize(doc["title"]))
        body_token_set = doc["tokens"]  # already a list, used as set-like
        body_token_set_set = set(body_token_set)
        meta_token_set = doc.get("meta_token_set", set())

        # Body score: IDF-weighted overlap
        body_score = sum(_idf(t) for t in query_tokens if t in body_token_set_set)
        # Title score: IDF-weighted, boosted by 3x
        title_score = sum(_idf(t) for t in query_tokens if t in title_tokens) * 3
        # Metadata score: query tokens matching frontmatter domain/conditions
        meta_score = sum(_idf(t) * 1.5 for t in query_tokens if t in meta_token_set)

        total_score = body_score + title_score + meta_score
        if total_score <= 0:
            continue

        snippet = _build_snippet(doc["text"], query_tokens, snippet_chars)
        ranked.append(
            {
                "title": doc["title"],
                "relative_path": doc["relative_path"],
                "snippet": snippet,
                "score": float(total_score),
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def _expand_query_tokens(question: str) -> list[str]:
    """对问题做 query expansion，将中文关键词映射到英文 metadata 标签。"""
    base_tokens = _tokenize(question)
    expanded = list(base_tokens)
    question_lower = question.lower()

    for cn_keyword, expansions in QUERY_EXPANSIONS.items():
        if cn_keyword.lower() in question_lower or any(t == cn_keyword.lower() for t in base_tokens):
            for exp in expansions:
                exp_tokens = _tokenize(exp)
                for t in exp_tokens:
                    if t not in expanded:
                        expanded.append(t)

    return expanded


def _build_snippet(text: str, query_tokens: list[str], snippet_chars: int) -> str:
    normalized = _normalize_whitespace(text)
    if not normalized:
        return ""

    hit_index = -1
    for token in query_tokens:
        hit_index = normalized.lower().find(token.lower())
        if hit_index >= 0:
            break

    if hit_index < 0:
        return normalized[:snippet_chars]

    half = max(snippet_chars // 2, 80)
    start = max(hit_index - half, 0)
    end = min(start + snippet_chars, len(normalized))
    snippet = normalized[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(normalized):
        snippet = snippet + "..."
    return snippet


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    words = re.findall(r"[a-z0-9_+-]+", lowered)
    cjk_chars = [ch for ch in lowered if "\u4e00" <= ch <= "\u9fff"]
    bigrams = [
        "".join(cjk_chars[i : i + 2])
        for i in range(max(len(cjk_chars) - 1, 0))
    ]
    seen = set()
    tokens = []
    for token in words + cjk_chars + bigrams:
        if len(token.strip()) < 1:
            continue
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens
