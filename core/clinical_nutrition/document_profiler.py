"""
Document Profiler — AI-driven document structure analysis and structured extraction.

Replaces hardcoded page-matching rules with:
1. LLM analyzes document → IngestionPlan (blocks with types and routing)
2. For each structured block, LLM extracts data → Pydantic schema validation
3. Failed validation → needs_review table

Usage:
    profiler = DocumentProfiler(llm_caller=your_llm_json_function)
    plan = profiler.generate_ingestion_plan(pages, source_name="肥胖指南")
    results = profiler.extract_structured_blocks(plan, pages)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Any, Callable

from core.clinical_nutrition.ingestion_schemas import (
    BLOCK_TYPE_SCHEMA_MAP,
    ActivityMET,
    DiagnosticThreshold,
    DocumentProfile,
    ExchangePortion,
    GenericTableRow,
    GuideTable,
    GuideTableRow,
    IngestionBlock,
    IngestionPlan,
    NeedsReviewItem,
    NutritionTarget,
    RecipeDish,
    RecipeIngredient,
    RecipeMeal,
    RecipePlan,
    SafetyRuleCandidate,
    TherapeuticRecipe,
    get_schema_for_block_type,
)

logger = logging.getLogger(__name__)

# Type alias for the LLM caller function
# Signature: (prompt: str, max_tokens: int) -> dict[str, Any]
LlmCaller = Callable[[str, int], dict[str, Any]]


# ============================================================
# Block type descriptions for LLM prompts
# ============================================================

BLOCK_TYPE_DESCRIPTIONS = """
可用的 block_type 类型：
- narrative_guideline: 叙述性指南建议（段落形式的推荐意见）
- recommendation: 明确的推荐/建议条目
- diagnostic_threshold: 诊断阈值/标准（如 BMI≥28, 血糖≥7.0）
- nutrition_target: 营养素目标/推荐量（如碳水45%-60%, 蛋白质15%-20%）
- food_exchange_portion: 食物交换份/食物等值份（每份食物的营养素含量）
- recipe_plan: 食谱方案（一日三餐的具体搭配）
- therapeutic_recipe: 食养方/药膳（有药材的食疗方）
- activity_met: 体力活动代谢当量（MET值表）
- contraindication: 禁忌/注意事项
- safety_rule_candidate: 安全红线候选（需要警示的情况）
- generic_table: 通用表格（不属于以上特定类型的表格）
- cover: 封面
- toc: 目录
- reference: 参考文献
- appendix: 附录

should_store_in 路由说明：
- wiki: 适合整理为知识页的叙述性内容
- rag: 适合向量检索的段落（指南建议、说明文字）
- structured: 有明确结构的数据（表格、食谱、交换份、MET值等）
- skip: 不需要入库的内容（封面、目录、空白页、参考文献）
"""


# ============================================================
# DocumentProfiler
# ============================================================

class DocumentProfiler:
    """Analyzes PDF pages and extracts structured knowledge via LLM + schema validation."""

    def __init__(
        self,
        *,
        llm_caller: LlmCaller | None = None,
        source_name: str = "",
        document_id: str = "",
    ):
        self.llm_caller = llm_caller
        self.source_name = source_name
        self.document_id = document_id

    def profile_document(
        self,
        pages: list[dict[str, Any]],
        *,
        source_name: str = "",
        quality_report: dict[str, Any] | None = None,
    ) -> DocumentProfile:
        """Return a compact document profile before detailed block planning."""
        source_name = source_name or self.source_name
        quality_report = quality_report or {}
        if not self.llm_caller:
            return self._fallback_document_profile(pages, source_name, quality_report)

        prompt = self._build_document_profile_prompt(pages, source_name, quality_report)
        try:
            raw = self.llm_caller(prompt, 2048)
            content = self._extract_content(raw)
            data = self._try_parse_json_object(content) or {}
            profile = DocumentProfile.model_validate(data)
            profile.source_document = source_name
            profile.page_count = len(pages)
            profile.quality_status = str(quality_report.get("quality_status") or profile.quality_status or "ok")
            if not profile.suggested_status:
                profile.suggested_status = "profiled"
            return profile
        except Exception as exc:
            logger.warning(f"Document profile generation failed: {exc}")
            return self._fallback_document_profile(pages, source_name, quality_report)

    # ----------------------------------------------------------
    # Step 1: Generate ingestion plan
    # ----------------------------------------------------------

    def generate_ingestion_plan(
        self,
        pages: list[dict[str, Any]],
        *,
        source_name: str = "",
        batch_size: int = 20,
    ) -> IngestionPlan:
        """Analyze all pages and produce an IngestionPlan with typed blocks.

        For large documents, pages are split into batches to avoid LLM timeout.
        """
        source_name = source_name or self.source_name
        total_pages = len(pages)

        if not self.llm_caller:
            logger.warning("No LLM caller provided, generating fallback plan")
            return self._fallback_plan(pages, source_name)

        # Split pages into batches
        all_blocks: list[IngestionBlock] = []
        all_knowledge_types: set[str] = set()
        document_type = "other"
        block_counter = 0

        for batch_start in range(0, total_pages, batch_size):
            batch_pages = pages[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (total_pages + batch_size - 1) // batch_size

            page_summary = self._build_page_summary(batch_pages)
            prompt = self._build_profiling_prompt(
                page_summary, total_pages, source_name,
                batch_offset=batch_start,
            )

            try:
                raw = self.llm_caller(prompt, 4096)
                batch_plan = self._parse_ingestion_plan(raw, batch_pages, source_name)

                if batch_plan.document_type != "other":
                    document_type = batch_plan.document_type
                all_knowledge_types.update(batch_plan.knowledge_types)

                for block in batch_plan.blocks:
                    block_counter += 1
                    block.block_id = f"b{block_counter:03d}"
                    all_blocks.append(block)

                logger.info(
                    f"Profiled batch {batch_num}/{total_batches}: "
                    f"{len(batch_plan.blocks)} blocks from {len(batch_pages)} pages"
                )
            except Exception as exc:
                logger.warning(f"Batch {batch_num} profiling failed: {exc}")
                # Add fallback blocks for this batch
                for page in batch_pages:
                    block_counter += 1
                    pn = page.get("page_number", 0)
                    text = (page.get("text") or "").strip()
                    if not text or len(text) < 50:
                        all_blocks.append(IngestionBlock(
                            block_id=f"b{block_counter:03d}",
                            block_type="cover",
                            page_start=pn,
                            page_end=pn,
                            confidence=0.3,
                            should_store_in="skip",
                            skip_reason="LLM profiling failed, fallback",
                        ))
                    else:
                        all_blocks.append(IngestionBlock(
                            block_id=f"b{block_counter:03d}",
                            block_type="narrative_guideline",
                            page_start=pn,
                            page_end=pn,
                            confidence=0.3,
                            should_store_in="rag",
                            skip_reason="",
                        ))

        plan = IngestionPlan(
            document_type=document_type,
            knowledge_types=sorted(all_knowledge_types),
            blocks=all_blocks,
            total_pages=total_pages,
            source_document=source_name,
        )
        plan = self._ensure_page_coverage(plan, pages)
        self._enrich_block_hashes(plan, pages)
        return plan

    # ----------------------------------------------------------
    # Step 2: Extract structured blocks
    # ----------------------------------------------------------

    def extract_structured_blocks(
        self,
        plan: IngestionPlan,
        pages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """For each structured block, call LLM to extract data and validate against schemas.

        Returns:
            {
                "extracted": {block_type: [validated_items]},
                "needs_review": [NeedsReviewItem],
                "stats": {block_type: count},
            }
        """
        extracted: dict[str, list[dict[str, Any]]] = {}
        needs_review: list[NeedsReviewItem] = []
        stats: dict[str, int] = {}

        page_map = {p["page_number"]: p.get("text", "") for p in pages}

        for block in plan.blocks:
            if block.should_store_in != "structured":
                continue
            if block.block_type in ("cover", "toc", "reference", "appendix"):
                continue

            # Get raw text for this block's page range
            raw_text = self._get_block_text(page_map, block.page_start, block.page_end)
            if not raw_text.strip():
                continue

            # Determine schema
            schema_cls = get_schema_for_block_type(block.block_type)
            if schema_cls is None:
                # Try generic_table as fallback
                schema_cls = GuideTable

            # Call LLM to extract structured data
            items, schema_errors, llm_output = self._extract_block(block, raw_text, schema_cls)

            if items:
                block_type = block.block_type
                if block_type not in extracted:
                    extracted[block_type] = []
                extracted[block_type].extend(items)
                stats[block_type] = stats.get(block_type, 0) + len(items)
                block.extraction_status = "done"
            else:
                # Extraction failed or returned nothing
                block.extraction_status = "needs_review"
                needs_review.append(NeedsReviewItem(
                    document_id=self.document_id,
                    block_id=block.block_id,
                    block_type=block.block_type,
                    section_path=block.section_path,
                    page_start=block.page_start,
                    page_end=block.page_end,
                    raw_text=raw_text[:2000],
                    llm_output=llm_output[:4000],
                    schema_errors=schema_errors,
                    confidence=block.confidence,
                ))

        return {
            "extracted": extracted,
            "needs_review": needs_review,
            "stats": stats,
        }

    # ----------------------------------------------------------
    # Internal: LLM extraction for a single block
    # ----------------------------------------------------------

    def _extract_block(
        self,
        block: IngestionBlock,
        raw_text: str,
        schema_cls: type,
    ) -> tuple[list[dict[str, Any]], str, str]:
        """Call LLM to extract structured data from a block, validate against schema."""
        if not self.llm_caller:
            return [], "no LLM caller configured", ""

        prompt = self._build_extraction_prompt(block, raw_text, schema_cls)

        try:
            raw_output = self.llm_caller(prompt, 3072)
            items, errors, content = self._parse_extraction_output(raw_output, schema_cls)
            return items, errors, content
        except Exception as exc:
            logger.warning(f"Block {block.block_id} extraction failed: {exc}")
            return [], str(exc), ""

    def _build_extraction_prompt(
        self,
        block: IngestionBlock,
        raw_text: str,
        schema_cls: type,
    ) -> str:
        """Build LLM prompt for structured extraction from a single block."""
        schema_json = json.dumps(schema_cls.model_json_schema(), ensure_ascii=False, indent=2)

        return f"""你是临床营养结构化数据提取器。从以下文本中提取 **{block.block_type}** 类型的结构化数据。

输出格式：严格 JSON 数组，每个元素符合以下 schema：
{schema_json}

要求：
1. 如果文本中有多个同类数据项，全部提取
2. 如果无法确定字段值，使用 null
3. 不要猜测或编造数据
4. source_document 填 "{self.source_name}"
5. source_pages 填 [{block.page_start}, {block.page_end}]

原文（p.{block.page_start}-p.{block.page_end}，章节：{block.section_path}）：
---
{raw_text[:3000]}
---

输出严格 JSON 数组（不要包含其他文字）："""

    def _parse_extraction_output(
        self,
        raw_output: dict[str, Any],
        schema_cls: type,
    ) -> tuple[list[dict[str, Any]], str, str]:
        """Parse LLM output and validate against schema. Returns list of validated dicts."""
        # Extract JSON from LLM response
        content = self._extract_content(raw_output)
        if not content:
            return [], "empty LLM response", ""

        # Try to parse as JSON array
        items_raw = self._try_parse_json_array(content)
        if not items_raw:
            # Maybe it's a single object
            single = self._try_parse_json_object(content)
            if single:
                items_raw = [single]
            else:
                return [], "LLM response did not contain a JSON array or object", content

        validated = []
        errors = []
        for item in items_raw:
            if not isinstance(item, dict):
                errors.append("non-object item in JSON array")
                continue
            try:
                obj = schema_cls.model_validate(item)
                validated.append(obj.model_dump())
            except Exception as exc:
                logger.debug(f"Schema validation failed for item: {exc}")
                errors.append(str(exc))

        return validated, "; ".join(errors), content

    # ----------------------------------------------------------
    # Internal: Profiling prompt construction
    # ----------------------------------------------------------

    def _build_page_summary(
        self,
        pages: list[dict[str, Any]],
        *,
        max_chars_per_page: int = 300,
    ) -> str:
        """Build a condensed summary of all pages for the profiling LLM call."""
        lines = []
        for page in pages:
            pn = page.get("page_number", 0)
            text = (page.get("text") or "").strip()
            if not text:
                lines.append(f"--- 第 {pn} 页 [空白] ---")
                continue
            # Truncate
            snippet = re.sub(r"\s+", " ", text)[:max_chars_per_page]
            lines.append(f"--- 第 {pn} 页 ---\n{snippet}")
        return "\n".join(lines)

    def _build_document_profile_prompt(
        self,
        pages: list[dict[str, Any]],
        source_name: str,
        quality_report: dict[str, Any],
    ) -> str:
        page_summary = self._build_page_summary(pages, max_chars_per_page=220)
        return f"""你是临床营养文档画像分析器。请先判断文档类型和知识类型，只输出严格 JSON。

文档名：{source_name}
质量报告：{json.dumps(quality_report, ensure_ascii=False)}

允许的 document_type：
clinical_guideline, dietary_guideline, food_composition_table, recipe_manual, drug_food_interaction_reference, patient_education_material, other

允许的 knowledge_types：
narrative_guideline, recommendation, diagnostic_threshold, nutrition_target, food_exchange_portion, recipe_plan, therapeutic_recipe, activity_met, contraindication, safety_rule_candidate, generic_table

输出 schema：
{{
  "document_type": "dietary_guideline",
  "knowledge_types": ["narrative_guideline", "recipe_plan"],
  "source_document": "{source_name}",
  "page_count": {len(pages)},
  "quality_status": "{quality_report.get('quality_status', 'ok')}",
  "suggested_status": "profiled",
  "confidence": 0.8,
  "summary": "一句话概括这份文档的内容和入库重点"
}}

页面摘要：
{page_summary}"""

    def _build_profiling_prompt(
        self,
        page_summary: str,
        total_pages: int,
        source_name: str,
        *,
        batch_offset: int = 0,
    ) -> str:
        return f"""你是临床营养文档结构分析器。分析以下文档页面，输出严格 JSON。

文档名：{source_name}
总页数：{total_pages}（当前分析第 {batch_offset + 1}-{batch_offset + page_summary.count('第 ')} 页）

{BLOCK_TYPE_DESCRIPTIONS}

输出格式（严格 JSON，不要包含其他文字）：
{{
  "document_type": "clinical_guideline | dietary_guideline | food_composition_table | recipe_manual | drug_food_interaction_reference | patient_education_material | other",
  "knowledge_types": ["诊断阈值", "营养目标", "食物交换份", "食谱", "食养方", "运动MET", "安全规则"],
  "chapter_outline": [
    {{"title": "章节标题", "page_start": 1, "page_end": 5, "section_path": "第一章 概述"}}
  ],
  "blocks": [
    {{
      "block_id": "b001",
      "block_type": "narrative_guideline",
      "section_path": "第三章 营养治疗",
      "page_start": 10,
      "page_end": 12,
      "raw_text_hash": "",
      "confidence": 0.85,
      "should_store_in": "wiki",
      "extraction_status": "pending",
      "skip_reason": ""
    }}
  ]
}}

要求：
1. 每页至少属于一个 block，不能遗漏任何页面
2. 连续的同类内容可以合并为一个 block
3. 表格内容 should_store_in 设为 "structured"
4. 叙述性指南建议设为 "wiki" 或 "rag"
5. 封面、目录、参考文献设为 "skip"
6. 对不确定的类型，confidence 设低一些（0.5-0.6）

以下是文档页面内容：

{page_summary}"""

    # ----------------------------------------------------------
    # Internal: Parsing helpers
    # ----------------------------------------------------------

    def _parse_ingestion_plan(
        self,
        raw: dict[str, Any],
        pages: list[dict[str, Any]],
        source_name: str,
    ) -> IngestionPlan:
        """Parse LLM output into IngestionPlan."""
        content = self._extract_content(raw)
        if not content:
            return self._fallback_plan(pages, source_name)

        data = self._try_parse_json_object(content)
        if not data:
            return self._fallback_plan(pages, source_name)

        try:
            plan = IngestionPlan.model_validate(data)
            plan.source_document = source_name
            plan.total_pages = len(pages)
            self._enrich_block_hashes(plan, pages)
            return plan
        except Exception as exc:
            logger.warning(f"IngestionPlan validation failed: {exc}")
            return self._fallback_plan(pages, source_name)

    def _fallback_document_profile(
        self,
        pages: list[dict[str, Any]],
        source_name: str,
        quality_report: dict[str, Any],
    ) -> DocumentProfile:
        text = "\n".join(str(page.get("text") or "") for page in pages[:8])
        document_type = "dietary_guideline" if any(k in source_name + text for k in ("膳食", "食养", "营养")) else "clinical_guideline"
        knowledge_types = ["narrative_guideline"]
        if any(k in text for k in ("表", "阈值", "标准")):
            knowledge_types.append("generic_table")
        if any(k in text for k in ("食谱", "早餐", "午餐", "晚餐")):
            knowledge_types.append("recipe_plan")
        if any(k in text for k in ("食养方", "主要材料", "制作方法")):
            knowledge_types.append("therapeutic_recipe")
        return DocumentProfile(
            document_type=document_type,
            knowledge_types=sorted(set(knowledge_types)),
            source_document=source_name,
            page_count=len(pages),
            quality_status=str(quality_report.get("quality_status") or "ok"),
            suggested_status="profiled",
            confidence=0.45,
            summary="由系统回退生成的文档画像，建议人工复核。",
        )

    def _ensure_page_coverage(
        self,
        plan: IngestionPlan,
        pages: list[dict[str, Any]],
    ) -> IngestionPlan:
        """Ensure every page is covered by at least one block. Add missing pages as rag blocks."""
        covered_pages: set[int] = set()
        for block in plan.blocks:
            for p in range(block.page_start, block.page_end + 1):
                covered_pages.add(p)

        all_pages = {p["page_number"] for p in pages}
        missing = sorted(all_pages - covered_pages)

        if not missing:
            self._enrich_block_hashes(plan, pages)
            return plan

        # Group consecutive missing pages
        groups = self._group_consecutive(missing)
        for group in groups:
            plan.blocks.append(IngestionBlock(
                block_id=f"b_auto_{group[0]}",
                block_type="narrative_guideline",
                section_path="（自动补充）",
                page_start=group[0],
                page_end=group[-1],
                confidence=0.5,
                should_store_in="rag",
                skip_reason="",
            ))
            logger.info(f"Auto-added block for uncovered pages: {group[0]}-{group[-1]}")

        self._enrich_block_hashes(plan, pages)
        return plan

    def _enrich_block_hashes(self, plan: IngestionPlan, pages: list[dict[str, Any]]) -> None:
        page_map = {int(p.get("page_number") or idx): str(p.get("text") or "") for idx, p in enumerate(pages, start=1)}
        for block in plan.blocks:
            raw_text = self._get_block_text(page_map, block.page_start, block.page_end)
            block.raw_text_hash = hashlib.sha1(raw_text.encode("utf-8")).hexdigest()[:16] if raw_text else ""
            if block.should_store_in == "skip" and not block.skip_reason:
                block.skip_reason = "封面/目录/参考文献或无有效正文"

    def _fallback_plan(
        self,
        pages: list[dict[str, Any]],
        source_name: str,
    ) -> IngestionPlan:
        """Generate a simple fallback plan when LLM is not available."""
        blocks = []
        for page in pages:
            pn = page.get("page_number", 0)
            text = (page.get("text") or "").strip()
            if not text:
                block_type = "cover"
                should_store = "skip"
                skip_reason = "空白页"
            elif len(text) < 50:
                block_type = "cover"
                should_store = "skip"
                skip_reason = "内容过短"
            else:
                block_type = "narrative_guideline"
                should_store = "rag"
                skip_reason = ""

            blocks.append(IngestionBlock(
                block_id=f"b_fallback_{pn}",
                block_type=block_type,
                section_path="",
                page_start=pn,
                page_end=pn,
                confidence=0.3,
                should_store_in=should_store,
                skip_reason=skip_reason,
            ))

        return IngestionPlan(
            document_type="other",
            blocks=blocks,
            total_pages=len(pages),
            source_document=source_name,
        )

    def _get_block_text(
        self,
        page_map: dict[int, str],
        page_start: int,
        page_end: int,
    ) -> str:
        """Get concatenated text for a page range."""
        parts = []
        for pn in range(page_start, page_end + 1):
            text = page_map.get(pn, "")
            if text.strip():
                parts.append(f"[第{pn}页] {text}")
        return "\n\n".join(parts)

    @staticmethod
    def _extract_content(raw: dict[str, Any]) -> str:
        """Extract text content from LLM response."""
        if isinstance(raw, dict):
            # OpenAI-compatible format
            choices = raw.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                return str(msg.get("content") or "")
            # Direct content
            return str(raw.get("content") or raw.get("text") or "")
        return str(raw)

    @staticmethod
    def _try_parse_json_object(text: str) -> dict[str, Any] | None:
        """Try to parse a JSON object from text, stripping markdown fences."""
        text = text.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        # Try to find JSON object in text
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _try_parse_json_array(text: str) -> list[Any] | None:
        """Try to parse a JSON array from text."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                return arr
        except json.JSONDecodeError:
            pass
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                arr = json.loads(match.group(0))
                if isinstance(arr, list):
                    return arr
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _group_consecutive(nums: list[int]) -> list[list[int]]:
        """Group consecutive integers: [1,2,3,5,6] -> [[1,2,3],[5,6]]"""
        if not nums:
            return []
        groups = [[nums[0]]]
        for n in nums[1:]:
            if n == groups[-1][-1] + 1:
                groups[-1].append(n)
            else:
                groups.append([n])
        return groups
