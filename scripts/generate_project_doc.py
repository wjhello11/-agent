#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目技术文档 PDF 生成脚本
基于 reportlab 生成 xiaozhi-esp32-server 临床营养师 Agent 的完整技术文档。
"""

import os
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Preformatted,
    KeepTogether,
    HRFlowable,
)

# ---------------------------------------------------------------------------
# Font registration
# ---------------------------------------------------------------------------
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
FONT_PATH_BOLD = "C:/Windows/Fonts/msyhbd.ttc"

try:
    pdfmetrics.registerFont(TTFont("MSYaHei", FONT_PATH))
    pdfmetrics.registerFont(TTFont("MSYaHeiBold", FONT_PATH_BOLD))
    CN_FONT = "MSYaHei"
    CN_FONT_BOLD = "MSYaHeiBold"
except Exception:
    # Fallback: try SimSun or other common Chinese fonts
    for fallback in ["C:/Windows/Fonts/simsun.ttc", "C:/Windows/Fonts/simhei.ttf"]:
        if os.path.exists(fallback):
            pdfmetrics.registerFont(TTFont("CNFallback", fallback))
            CN_FONT = "CNFallback"
            CN_FONT_BOLD = "CNFallback"
            break
    else:
        print("WARNING: No Chinese font found, PDF may not display Chinese correctly.")
        CN_FONT = "Helvetica"
        CN_FONT_BOLD = "Helvetica-Bold"

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = A4


def _build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="CoverTitle",
        fontName=CN_FONT_BOLD,
        fontSize=28,
        leading=36,
        alignment=TA_CENTER,
        spaceAfter=12,
        textColor=colors.HexColor("#1a1a2e"),
    ))
    styles.add(ParagraphStyle(
        name="CoverSubtitle",
        fontName=CN_FONT,
        fontSize=14,
        leading=20,
        alignment=TA_CENTER,
        spaceAfter=8,
        textColor=colors.HexColor("#4a4a6a"),
    ))
    styles.add(ParagraphStyle(
        name="ChapterTitle",
        fontName=CN_FONT_BOLD,
        fontSize=20,
        leading=28,
        spaceBefore=24,
        spaceAfter=12,
        textColor=colors.HexColor("#1a1a2e"),
        borderPadding=(0, 0, 4, 0),
    ))
    styles.add(ParagraphStyle(
        name="SectionTitle",
        fontName=CN_FONT_BOLD,
        fontSize=14,
        leading=20,
        spaceBefore=16,
        spaceAfter=8,
        textColor=colors.HexColor("#2d3748"),
    ))
    styles.add(ParagraphStyle(
        name="SubSectionTitle",
        fontName=CN_FONT_BOLD,
        fontSize=12,
        leading=16,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor("#4a5568"),
    ))
    styles.add(ParagraphStyle(
        name="BodyCN",
        fontName=CN_FONT,
        fontSize=10,
        leading=16,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        textColor=colors.HexColor("#2d3748"),
    ))
    styles.add(ParagraphStyle(
        name="BulletCN",
        fontName=CN_FONT,
        fontSize=10,
        leading=16,
        leftIndent=20,
        spaceAfter=4,
        textColor=colors.HexColor("#2d3748"),
        bulletIndent=8,
        bulletFontName=CN_FONT,
    ))
    styles.add(ParagraphStyle(
        name="CodeBlock",
        fontName="Courier",
        fontSize=8,
        leading=11,
        leftIndent=12,
        rightIndent=12,
        spaceBefore=6,
        spaceAfter=6,
        backColor=colors.HexColor("#f7fafc"),
        borderColor=colors.HexColor("#e2e8f0"),
        borderWidth=0.5,
        borderPadding=6,
    ))
    styles.add(ParagraphStyle(
        name="Caption",
        fontName=CN_FONT,
        fontSize=9,
        leading=13,
        alignment=TA_CENTER,
        spaceAfter=10,
        textColor=colors.HexColor("#718096"),
    ))
    styles.add(ParagraphStyle(
        name="TableHeader",
        fontName=CN_FONT_BOLD,
        fontSize=9,
        leading=13,
        textColor=colors.white,
    ))
    styles.add(ParagraphStyle(
        name="TableCell",
        fontName=CN_FONT,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2d3748"),
    ))
    styles.add(ParagraphStyle(
        name="TableCellCode",
        fontName="Courier",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#2d3748"),
    ))
    return styles


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------
def hr():
    return HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceAfter=8, spaceBefore=8)


def code_block(text, styles):
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Preformatted(safe, styles["CodeBlock"])


def make_table(headers, rows, col_widths=None):
    """Create a styled table."""
    header_paras = [Paragraph(h, _styles["TableHeader"]) for h in headers]
    data = [header_paras]
    for row in rows:
        data.append([Paragraph(str(c), _styles["TableCell"]) for c in row])

    avail = PAGE_W - 2 * 2 * cm
    if col_widths is None:
        n = len(headers)
        col_widths = [avail / n] * n

    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3748")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, -1), CN_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
        ("TOPPADDING", (0, 1), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
    ]))
    return t


# Global styles reference (set in build_pdf)
_styles = None


# ---------------------------------------------------------------------------
# Document content builders
# ---------------------------------------------------------------------------

def build_cover(story, styles):
    story.append(Spacer(1, 6 * cm))
    story.append(Paragraph("xiaozhi-esp32-server", styles["CoverTitle"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("临床营养师 AI Agent 系统", styles["CoverTitle"]))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph("项目技术文档", styles["CoverSubtitle"]))
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("基于 ESP32 语音交互的慢病管理 AI 营养师平台", styles["CoverSubtitle"]))
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("版本: 1.0  |  日期: 2026-05-11", styles["CoverSubtitle"]))
    story.append(PageBreak())


def build_toc(story, styles):
    story.append(Paragraph("目录", styles["ChapterTitle"]))
    story.append(hr())
    toc_items = [
        "第 1 章  项目概述",
        "第 2 章  系统架构",
        "第 3 章  知识检索系统",
        "第 4 章  知识摄入管线",
        "第 5 章  记忆系统",
        "第 6 章  健康档案系统",
        "第 7 章  临床安全拦截",
        "第 8 章  插件函数系统",
        "第 9 章  数据库架构",
        "第 10 章  配置与部署",
    ]
    for item in toc_items:
        story.append(Paragraph(item, styles["BodyCN"]))
    story.append(PageBreak())


# ---- Chapter 1 ----
def build_ch1(story, styles):
    story.append(Paragraph("第 1 章  项目概述", styles["ChapterTitle"]))
    story.append(hr())

    story.append(Paragraph("1.1 项目定位", styles["SectionTitle"]))
    story.append(Paragraph(
        "xiaozhi-esp32-server 是一个面向慢病管理的 AI 临床营养师系统。"
        "系统通过 ESP32 硬件设备进行语音交互，结合多层知识检索、长期记忆、"
        "健康档案和临床安全拦截，为用户提供个性化的饮食与营养指导。",
        styles["BodyCN"],
    ))

    story.append(Paragraph("1.2 技术栈", styles["SectionTitle"]))
    avail = PAGE_W - 4 * cm
    story.append(make_table(
        ["层级", "技术", "说明"],
        [
            ["硬件", "ESP32-S3", "语音采集与播放终端"],
            ["通信", "WebSocket", "全双工实时音频传输"],
            ["ASR", "FunASR", "语音识别（Paraformer）"],
            ["LLM", "ChatGLM / Qwen", "大语言模型推理"],
            ["TTS", "EdgeTTS", "文本转语音"],
            ["VAD", "SileroVAD", "语音活动检测"],
            ["知识库", "SQLite + FTS5", "Wiki / RAG / 结构化知识"],
            ["记忆", "mem0 + PowerMem", "四层认知记忆系统"],
            ["安全", "规则引擎", "药物-食物禁忌拦截"],
            ["语言", "Python 3.10+", "aiohttp / asyncio 异步架构"],
        ],
        [avail * 0.15, avail * 0.25, avail * 0.6],
    ))

    story.append(Paragraph("1.3 核心能力", styles["SectionTitle"]))
    capabilities = [
        "语音对话：实时语音识别 + LLM 推理 + 语音合成，支持多轮上下文",
        "多知识库检索：Wiki（TF-IDF）、RAG（BM25+向量）、结构化知识（SQLite）三路并行",
        "长期记忆：四层认知架构（working / factual / episodic / semantic），跨会话持久化",
        "健康档案：结构化存储用户疾病、用药、过敏、血糖等健康数据",
        "临床安全拦截：确定性规则引擎，药物-食物禁忌检查，防止有害建议",
        "知识摄入：PDF 指南自动解析 → Wiki 页面 + RAG 分块 + 结构化提取",
    ]
    for cap in capabilities:
        story.append(Paragraph(f"•  {cap}", styles["BulletCN"]))

    story.append(PageBreak())


# ---- Chapter 2 ----
def build_ch2(story, styles):
    story.append(Paragraph("第 2 章  系统架构", styles["ChapterTitle"]))
    story.append(hr())

    story.append(Paragraph("2.1 整体架构", styles["SectionTitle"]))
    arch = """\
+-------------------------------------------------------------+
|                      ESP32 设备端                            |
|   [麦克风] --> [VAD] --> [ASR] --> [WebSocket] -->          |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                    服务端 (Python asyncio)                    |
|                                                              |
|  app.py                                                      |
|    |                                                         |
|    +--> WebSocketServer                                      |
|    |       |                                                 |
|    |       +--> ConnectionHandler                            |
|    |              |                                          |
|    |              +--> IntentDispatcher (function_call)       |
|    |              |       |                                  |
|    |              |       +--> PluginManager                 |
|    |              |              |                           |
|    |              |              +--> search_from_llmwiki    |
|    |              |              +--> search_clinical_rag    |
|    |              |              +--> search_food_nutrition  |
|    |              |              +--> ... (7 plugins)        |
|    |              |                                          |
|    |              +--> MemoryProvider (clinical_ltm)         |
|    |              |       |                                  |
|    |              |       +--> HealthProfileStore            |
|    |              |       +--> Mem0CognitiveExtractor        |
|    |              |       +--> PowerMemSQLiteStore           |
|    |              |                                          |
|    |              +--> ClinicalSafetyInterceptor             |
|    |                      |                                  |
|    |                      +--> SafetyRuleEngine              |
|    |                                                          |
|    +--> SimpleHttpServer (OTA / Vision API)                   |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                     知识层 (SQLite + Files)                   |
|                                                              |
|  knowledge_base/llmwiki/    (Wiki Markdown 文件)             |
|  knowledge_base/structured/ (食物营养数据库)                  |
|  data/clinical_knowledge.db (RAG + 结构化知识)                |
|  data/clinical_ltm.db       (长期记忆)                        |
+-------------------------------------------------------------+"""
    story.append(code_block(arch, styles))
    story.append(Paragraph("图 2-1: 系统整体架构图", styles["Caption"]))

    story.append(Paragraph("2.2 请求处理流程", styles["SectionTitle"]))
    story.append(Paragraph(
        "当 ESP32 设备发送语音数据时，处理流程如下：",
        styles["BodyCN"],
    ))
    steps = [
        "1. VAD 检测语音活动，判断用户是否在说话",
        "2. ASR（FunASR）将语音转为文本",
        "3. IntentDispatcher 判断是否触发插件函数（function_call 模式）",
        "4. 若触发插件：调用对应工具函数，获取知识上下文",
        "5. ClinicalSafetyInterceptor 检查安全规则（药物禁忌、过敏原）",
        "6. MemoryProvider 注入记忆上下文（working + long-term）",
        "7. LLM 生成回答（结合工具结果 + 记忆 + 健康档案）",
        "8. TTS 将文本转为语音，通过 WebSocket 返回设备",
    ]
    for step in steps:
        story.append(Paragraph(step, styles["BulletCN"]))

    story.append(Paragraph("2.3 模块选择", styles["SectionTitle"]))
    story.append(Paragraph(
        "各模块通过 config.yaml 的 selected_module 字段配置，支持热插拔。"
        "当前默认配置：VAD=SileroVAD, ASR=FunASR, LLM=ChatGLMLLM, "
        "TTS=EdgeTTS, Memory=clinical_ltm, Intent=function_call。",
        styles["BodyCN"],
    ))

    story.append(PageBreak())


# ---- Chapter 3 ----
def build_ch3(story, styles):
    avail = PAGE_W - 4 * cm
    story.append(Paragraph("第 3 章  知识检索系统", styles["ChapterTitle"]))
    story.append(hr())

    story.append(Paragraph(
        "系统采用三路并行检索架构，分别针对不同类型的临床营养知识：",
        styles["BodyCN"],
    ))

    # 3.1 Wiki
    story.append(Paragraph("3.1 Wiki 知识库（TF-IDF 关键词检索）", styles["SectionTitle"]))
    story.append(Paragraph(
        "Wiki 知识库以 Markdown 文件形式存储在 knowledge_base/llmwiki/clinical-nutrition/ 目录下。"
        "每个 Wiki 页面包含 frontmatter 元数据（clinical_domain, conditions, kb_layer）和正文内容。",
        styles["BodyCN"],
    ))
    story.append(Paragraph("检索流程：", styles["SubSectionTitle"]))
    wiki_flow = """\
用户问题
  |
  v
_query_expansion()  -- 中文关键词 -> 英文映射
  |                    例: "糖尿病" -> ["diabetes","糖尿病","血糖"]
  v
_tokenize()  -- 中文字符 + bigram + 英文分词
  |
  v
_rank_documents()  -- IDF 加权评分
  |   body_score   = sum(idf(t) for t in query if t in body)
  |   title_score  = sum(idf(t) for t in query if t in title) * 3
  |   meta_score   = sum(idf(t)*1.5 for t in query if t in meta)
  |   total = body + title + meta
  v
_build_snippet()  -- 480 字符窗口，首次出现位置截取
  |
  v
Action.REQLLM  -- 格式化为 LLM 上下文"""
    story.append(code_block(wiki_flow, styles))

    story.append(Paragraph("关键实现文件：plugins_func/functions/search_from_llmwiki.py", styles["SubSectionTitle"]))
    story.append(make_table(
        ["参数", "默认值", "说明"],
        [
            ["top_k", "4", "返回最多 4 条结果"],
            ["snippet_chars", "480", "每条结果截取的字符数"],
            ["title 权重", "3x", "标题匹配得分为正文的 3 倍"],
            ["meta 权重", "1.5x", "frontmatter 元数据匹配加权"],
        ],
        [avail * 0.25, avail * 0.25, avail * 0.5],
    ))

    # 3.2 RAG
    story.append(Paragraph("3.2 RAG 混合检索（BM25 + 向量）", styles["SectionTitle"]))
    story.append(Paragraph(
        "RAG 检索由 ClinicalRAGService 实现，采用 BM25 词法检索与向量语义检索的混合策略。"
        "向量使用 DashScope text-embedding-v4 模型生成（256 维），存储为 SQLite BLOB，"
        "查询时全表扫描 + Python cosine similarity 计算。",
        styles["BodyCN"],
    ))
    story.append(Paragraph("混合评分公式：", styles["SubSectionTitle"]))
    rag_score = """\
total_score = 0.55 * bm25_score + 0.45 * cosine_similarity

bm25_score    -- FTS5 全文检索（BM25 算法）
cosine_score  -- 向量余弦相似度（全表扫描）
mmr_select()  -- Maximal Marginal Relevance 多样性过滤"""
    story.append(code_block(rag_score, styles))

    story.append(Paragraph("分块策略：", styles["SubSectionTitle"]))
    story.append(make_table(
        ["参数", "值", "说明"],
        [
            ["chunk_chars", "850", "每块最大字符数"],
            ["chunk_overlap", "150", "相邻块重叠字符数"],
            ["embedding_model", "text-embedding-v4", "DashScope 文本向量化模型"],
            ["dimensions", "256", "向量维度"],
        ],
        [avail * 0.25, avail * 0.3, avail * 0.45],
    ))

    # 3.3 Structured
    story.append(Paragraph("3.3 结构化知识库（SQLite 关系表）", styles["SectionTitle"]))
    story.append(Paragraph(
        "结构化知识由 StructuredKnowledgeStore 管理，存储在 clinical_knowledge.db 中。"
        "包含 15+ 张表，涵盖食物营养、食谱方案、食养方、MET 系数、诊断阈值等。",
        styles["BodyCN"],
    ))
    story.append(Paragraph("核心表：", styles["SubSectionTitle"]))
    story.append(make_table(
        ["表名", "内容", "查询函数"],
        [
            ["guide_tables / rows", "指南表格数据", "search_guide_tables()"],
            ["food_exchange_portions", "食物交换份", "search_exchange_portions()"],
            ["recipe_plans / meals / dishes", "食谱方案", "search_recipe_plans()"],
            ["therapeutic_recipes", "中医食养方", "search_therapeutic_recipes()"],
            ["activity_mets", "运动 MET 系数", "search_activity_mets()"],
            ["diagnostic_thresholds", "诊断阈值", "search_diagnostic_thresholds()"],
            ["nutrition_targets", "营养目标", "search_nutrition_targets()"],
        ],
        [avail * 0.35, avail * 0.3, avail * 0.35],
    ))

    # 3.4 Food Nutrition
    story.append(Paragraph("3.4 食物营养数据库", styles["SectionTitle"]))
    story.append(Paragraph(
        "食物营养数据存储在 knowledge_base/structured/clinical_nutrition.db 中，"
        "包含中国食物成分表、USDA 数据、血糖生成指数等。",
        styles["BodyCN"],
    ))
    story.append(make_table(
        ["表名", "字段", "说明"],
        [
            ["food_items", "food_id, canonical_name, food_category", "食物基础信息"],
            ["food_aliases", "food_id, alias, language", "食物别名（中英文）"],
            ["food_nutrients_per_100g", "energy_kcal, protein_g, fat_g, ...", "每 100g 营养成分"],
            ["glycemic_values", "gi_value, gl_value, serving_g", "血糖生成指数"],
        ],
        [avail * 0.3, avail * 0.4, avail * 0.3],
    ))

    story.append(PageBreak())


# ---- Chapter 4 ----
def build_ch4(story, styles):
    avail = PAGE_W - 4 * cm
    story.append(Paragraph("第 4 章  知识摄入管线", styles["ChapterTitle"]))
    story.append(hr())

    story.append(Paragraph(
        "知识摄入管线（Knowledge Ingestion Pipeline）负责将 PDF 指南文档自动解析为 "
        "Wiki 页面、RAG 分块和结构化知识，是知识库的生产系统。",
        styles["BodyCN"],
    ))

    story.append(Paragraph("4.1 处理流程", styles["SectionTitle"]))
    pipeline = """\
PDF 文件
  |
  v
Step 1: 文本提取 (PyMuPDF / pdfplumber)
  |       提取每页文本，保留页面编号
  v
Step 2: DocumentProfiler (LLM 分析)
  |       分析文档结构：识别 14 种块类型
  |       路由决策：每页 -> wiki / rag / structured / skip
  v
Step 3: Wiki Compiler v2 (LLM 处理)
  |       按语义分块 (~4200 chars/块)
  |       生成：核心结论、行动建议、安全边界
  |       标记：哪些内容进入结构化库
  v
Step 4: LLM Review (质量审查)
  |       检查覆盖率、一致性、安全性
  |       输出：overall_status, confidence, issues
  v
Step 5: Human approve_draft (人工审批)
  |       人类确认后写入知识库
  v
输出:
  +-- Wiki: knowledge_base/llmwiki/ (Markdown 文件)
  +-- RAG:  clinical_knowledge.db (rag_chunks + rag_embeddings)
  +-- Structured: clinical_knowledge.db (guide_tables, recipes, etc.)"""
    story.append(code_block(pipeline, styles))
    story.append(Paragraph("图 4-1: 知识摄入管线流程", styles["Caption"]))

    story.append(Paragraph("4.2 文档分析器（DocumentProfiler）", styles["SectionTitle"]))
    story.append(Paragraph(
        "DocumentProfiler 使用 LLM 分析 PDF 文档结构，识别 14 种内容块类型，"
        "并决定每页内容的存储路由。",
        styles["BodyCN"],
    ))
    story.append(make_table(
        ["块类型", "说明", "存储路由"],
        [
            ["narrative_guideline", "叙述性指南内容", "wiki + rag"],
            ["diagnostic_threshold", "诊断阈值数据", "structured"],
            ["nutrition_target", "营养目标数据", "structured"],
            ["food_exchange_table", "食物交换份表", "structured"],
            ["recipe_plan", "食谱方案", "structured"],
            ["therapeutic_recipe", "中医食养方", "structured"],
            ["met_coefficient", "运动 MET 系数", "structured"],
            ["safety_contraindication", "安全禁忌", "structured + wiki"],
            ["table_data", "表格数据", "structured"],
            ["figure_legend", "图表说明", "rag"],
            ["reference", "参考文献", "skip"],
            ["toc", "目录页", "skip"],
        ],
        [avail * 0.35, avail * 0.35, avail * 0.3],
    ))

    story.append(Paragraph("4.3 Wiki 编译器（Wiki Compiler v2）", styles["SectionTitle"]))
    story.append(Paragraph(
        "Wiki Compiler 将文档按语义分块（约 4200 字符/块），每块由 LLM 生成："
        "核心结论（key_conclusions）、行动建议（clinical_recommendations）、"
        "安全边界（safety_boundaries）、结构化候选（structured_candidates）。",
        styles["BodyCN"],
    ))

    story.append(Paragraph("4.4 实际案例：成人肥胖指南", styles["SectionTitle"]))
    story.append(Paragraph(
        "以《成人肥胖食养指南（2024年版）》（70 页）为例：",
        styles["BodyCN"],
    ))
    story.append(make_table(
        ["指标", "数值"],
        [
            ["源文件页数", "70 页"],
            ["Wiki 页面数", "7 个（overview, principles, energy-control, food-selection, safe-weight-loss, regional-recipes, tcm-diet-therapy）"],
            ["RAG 分块数", "21 块"],
            ["结构化提取", "49 页内容"],
            ["跳过页面", "1 页（目录页）"],
            ["Wiki 覆盖率", "69/70 页"],
            ["RAG 覆盖率", "70/70 页"],
            ["LLM 审查状态", "passed (confidence: 0.95)"],
        ],
        [avail * 0.3, avail * 0.7],
    ))

    story.append(Paragraph("关键文件：", styles["SubSectionTitle"]))
    story.append(make_table(
        ["文件", "作用"],
        [
            ["core/clinical_nutrition/knowledge_ingestion.py", "管线编排：create_draft() / approve_draft()"],
            ["core/clinical_nutrition/document_profiler.py", "LLM 文档结构分析"],
            ["core/clinical_nutrition/wiki_compiler.py", "Wiki 页面编译"],
            ["data/knowledge_ingestion/drafts/", "草稿输出目录"],
        ],
        [avail * 0.5, avail * 0.5],
    ))

    story.append(PageBreak())


# ---- Chapter 5 ----
def build_ch5(story, styles):
    avail = PAGE_W - 4 * cm
    story.append(Paragraph("第 5 章  记忆系统（Clinical LTM）", styles["ChapterTitle"]))
    story.append(hr())

    story.append(Paragraph(
        "记忆系统采用四层认知架构，模拟人类记忆的形成、巩固和衰减过程，"
        "为 Agent 提供跨会话的用户理解能力。",
        styles["BodyCN"],
    ))

    story.append(Paragraph("5.1 四层认知架构", styles["SectionTitle"]))
    mem_arch = """\
+---------------------------------------------------+
|                记忆层级架构                         |
|                                                    |
|  [Working Memory]  -- 当前会话上下文               |
|       |  (每轮对话后提取)                           |
|       v                                            |
|  [Factual Memory]  -- 锁定事实                    |
|       |  用户确诊 2 型糖尿病                        |
|       |  用户对虾过敏                               |
|       |  用户服用二甲双胍                           |
|       v                                            |
|  [Episodic Memory] -- 事件记忆                     |
|       |  早餐吃了包子+粥，餐后血糖偏高              |
|       |  重要度 0.72，半衰期 14 天                  |
|       v                                            |
|  [Semantic Memory] -- 抽象规律                     |
|       |  用户早餐碳水偏好高，需控制份量             |
|       |  重要度 0.86，半衰期 90 天                  |
+---------------------------------------------------+"""
    story.append(code_block(mem_arch, styles))
    story.append(Paragraph("图 5-1: 四层认知记忆架构", styles["Caption"]))

    story.append(Paragraph("5.2 MemoryProvider 编排流程", styles["SectionTitle"]))
    mem_flow = """\
save_memory(messages, user_id, session_id):
  1. WorkingMemory  -- 解析当前轮对话为 WorkingTurn
  2. HealthProfile  -- 提取健康信息更新档案
  3. ShortTermSummary -- 滚动窗口摘要 (超过 working_memory_turns 轮时触发)
  4. Extraction     -- mem0 hints + LLM 结构化抽取
  |   +-- factual: 疾病、过敏、用药、医嘱
  |   +-- episodic: 饮食事件、血糖记录
  |   +-- semantic: 行为规律、偏好
  5. Decay          -- 按半衰期衰减 importance/weight
  6. Upsert         -- 去重写入 SQLite (dedupe_key)
  7. PowerMem Sync  -- 同步到 PowerMem 向量存储
  8. Semantic Synthesis -- factual+episodic -> semantic 规律"""
    story.append(code_block(mem_flow, styles))

    story.append(Paragraph("5.3 mem0 集成", styles["SectionTitle"]))
    story.append(Paragraph(
        "mem0 作为语义预提取层，在 LLM 抽取之前提供 hints。"
        "支持两种模式：cloud（mem0 Cloud API）和 local（mem0 本地 Qdrant 向量库）。"
        "当前配置为 local 模式，使用 Qdrant 内存向量库 + DashScope embedding。",
        styles["BodyCN"],
    ))

    story.append(Paragraph("5.4 记忆衰减机制", styles["SectionTitle"]))
    story.append(make_table(
        ["记忆类型", "半衰期", "衰减公式", "说明"],
        [
            ["episodic", "14 天", "w(t) = w0 * 0.5^(t/14)", "事件记忆逐渐淡化"],
            ["semantic", "90 天", "w(t) = w0 * 0.5^(t/90)", "规律记忆长期保留"],
            ["factual", "不衰减", "weight 固定", "锁定事实永远有效"],
        ],
        [avail * 0.2, avail * 0.15, avail * 0.35, avail * 0.3],
    ))

    story.append(Paragraph("5.5 语义合成", styles["SectionTitle"]))
    story.append(Paragraph(
        "当同一实体的 factual + episodic 记忆积累到一定数量时，"
        "系统自动触发语义合成：LLM 将多条具体事件归纳为抽象规律。"
        "例如：多条「早餐吃高碳水食物导致血糖偏高」的 episodic 记忆，"
        "被合成为「用户早餐碳水偏好高，需控制份量」的 semantic 记忆。",
        styles["BodyCN"],
    ))

    story.append(Paragraph("5.6 关键文件", styles["SectionTitle"]))
    story.append(make_table(
        ["文件", "作用"],
        [
            ["core/providers/memory/clinical_ltm/clinical_ltm.py", "MemoryProvider 主编排"],
            ["core/providers/memory/clinical_ltm/store.py", "PowerMemSQLiteStore"],
            ["core/providers/memory/clinical_ltm/extractor.py", "Mem0CognitiveExtractor"],
            ["core/providers/memory/clinical_ltm/lifecycle.py", "MemoryLifecycleManager"],
            ["core/providers/memory/clinical_ltm/interceptor.py", "MemoryRetrievalInterceptor"],
            ["core/providers/memory/clinical_ltm/models.py", "Pydantic 数据模型"],
            ["core/providers/memory/clinical_ltm/prompts.py", "LLM 提取提示词"],
        ],
        [avail * 0.55, avail * 0.45],
    ))

    story.append(PageBreak())


# ---- Chapter 6 ----
def build_ch6(story, styles):
    avail = PAGE_W - 4 * cm
    story.append(Paragraph("第 6 章  健康档案系统", styles["ChapterTitle"]))
    story.append(hr())

    story.append(Paragraph(
        "HealthProfileStore 为每个用户维护结构化的健康数据，"
        "支持从对话中自动提取健康信息，并提供冲突检测和血糖追踪功能。",
        styles["BodyCN"],
    ))

    story.append(Paragraph("6.1 数据结构", styles["SectionTitle"]))
    story.append(make_table(
        ["数据类别", "字段示例", "存储表"],
        [
            ["基本信息", "age, sex, height_cm, weight_kg, bmi", "health_profiles"],
            ["疾病", "type-2-diabetes, hypertension", "health_profile_items"],
            ["用药", "metformin, insulin", "health_profile_items"],
            ["过敏原", "shrimp, peanut", "health_profile_items"],
            ["血糖记录", "fasting=7.2, postprandial=11.5", "blood_glucose_readings"],
            ["回顾项", "待确认的健康信息", "health_profile_items (pending)"],
        ],
        [avail * 0.2, avail * 0.4, avail * 0.4],
    ))

    story.append(Paragraph("6.2 自动提取", styles["SectionTitle"]))
    story.append(Paragraph(
        "每轮对话后，系统调用 extract_health_profile_update() 从用户消息中提取健康信息。"
        "提取逻辑包括：身高体重计算 BMI、血糖数值记录、疾病/用药/过敏原识别。"
        "新增信息会与已有档案进行冲突检测：若用户之前说「对虾过敏」，"
        "后来又说「昨天吃了虾」，系统会标记为待回顾项。",
        styles["BodyCN"],
    ))

    story.append(Paragraph("6.3 血糖追踪", styles["SectionTitle"]))
    story.append(Paragraph(
        "blood_glucose_readings 表记录用户的血糖测量值，包括：测量时间、"
        "血糖值（mmol/L）、测量类型（空腹/餐后/随机）。"
        "系统可自动分析血糖趋势，并在 Agent 回答中引用最近的血糖数据。",
        styles["BodyCN"],
    ))

    story.append(PageBreak())


# ---- Chapter 7 ----
def build_ch7(story, styles):
    avail = PAGE_W - 4 * cm
    story.append(Paragraph("第 7 章  临床安全拦截", styles["ChapterTitle"]))
    story.append(hr())

    story.append(Paragraph(
        "ClinicalSafetyInterceptor 是一个确定性规则引擎，在 LLM 生成回答之前"
        "检查潜在的临床安全风险。它不依赖 LLM，而是基于预定义的规则进行判定。",
        styles["BodyCN"],
    ))

    story.append(Paragraph("7.1 检查流程", styles["SectionTitle"]))
    safety_flow = """\
用户问题 + 工具检索结果
  |
  v
ClinicalSafetyInterceptor.check(message, tool_context, health_profile)
  |
  +--> SafetyRuleEngine.evaluate()
  |      |
  |      +--> 过敏原检查: 用户过敏原 x 工具结果中的食物
  |      +--> 药物禁忌检查: 用户用药 x 食物成分
  |      +--> 疾病不适宜: 用户疾病 x 食物 GI/嘌呤/钠含量
  |      +--> 安全边界: 是否涉及自行调药/停药
  |
  v
结果:
  +-- PASS: 无安全问题，正常进入 LLM
  +-- WARN: 存在风险，注入安全提示到 LLM prompt
  +-- BLOCK: 严重风险，直接返回安全警告"""
    story.append(code_block(safety_flow, styles))
    story.append(Paragraph("图 7-1: 临床安全拦截流程", styles["Caption"]))

    story.append(Paragraph("7.2 规则类型", styles["SectionTitle"]))
    story.append(make_table(
        ["规则类型", "示例", "处理方式"],
        [
            ["过敏原检查", "用户对虾过敏 + 推荐了虾仁", "BLOCK"],
            ["药物-食物禁忌", "服用华法林 + 推荐菠菜", "BLOCK"],
            ["疾病不适宜食物", "痛风 + 推荐高嘌呤海鲜", "WARN"],
            ["自行调药风险", "用户想减量二甲双胍", "WARN + 安全提示"],
        ],
        [avail * 0.25, avail * 0.4, avail * 0.35],
    ))

    story.append(Paragraph("7.3 安全边界约束", styles["SectionTitle"]))
    story.append(Paragraph(
        "Agent 在回答中必须遵守以下安全边界（通过系统提示词强制约束）：",
        styles["BodyCN"],
    ))
    safety_rules = [
        "绝不建议用户自行调整药物剂量",
        "绝不建议用户停药",
        "食物建议必须注明「在医生指导下」",
        "检测到严重症状时建议立即就医",
        "不提供具体药物剂量建议",
        "区分「知识库内容」和「通用推断」",
    ]
    for rule in safety_rules:
        story.append(Paragraph(f"•  {rule}", styles["BulletCN"]))

    story.append(PageBreak())


# ---- Chapter 8 ----
def build_ch8(story, styles):
    avail = PAGE_W - 4 * cm
    story.append(Paragraph("第 8 章  插件函数系统", styles["ChapterTitle"]))
    story.append(hr())

    story.append(Paragraph(
        "插件函数系统通过 @register_function 装饰器注册，由 PluginManager 统一管理。"
        "LLM 通过 function_call 机制调用插件，插件返回工具上下文供 LLM 生成回答。",
        styles["BodyCN"],
    ))

    story.append(Paragraph("8.1 注册机制", styles["SectionTitle"]))
    story.append(Paragraph(
        "每个插件函数使用 @register_function(name, description, tool_type) 装饰器注册。"
        "函数签名统一为 func(conn: ConnectionHandler, **params) -> ActionResponse。"
        "ActionResponse 包含三个字段：action（REQLLM/RESPONSE）、content、error。",
        styles["BodyCN"],
    ))

    story.append(Paragraph("8.2 已注册插件", styles["SectionTitle"]))
    story.append(make_table(
        ["插件名称", "功能", "输入参数", "返回类型"],
        [
            ["search_from_llmwiki", "Wiki 知识库检索", "question: str", "Action.REQLLM"],
            ["search_clinical_rag", "RAG 混合检索", "question: str", "Action.REQLLM"],
            ["search_clinical_structured_knowledge", "结构化知识检索", "question: str", "Action.REQLLM"],
            ["search_food_nutrition", "食物营养查询", "food_name: str", "Action.REQLLM"],
            ["analyze_meal_nutrition", "膳食营养分析", "meal_description: str", "Action.REQLLM"],
            ["get_health_profile", "读取健康档案", "user_id: str", "Action.RESPONSE"],
            ["update_health_profile", "更新健康档案", "user_id, profile_data", "Action.RESPONSE"],
        ],
        [avail * 0.3, avail * 0.25, avail * 0.25, avail * 0.2],
    ))

    story.append(Paragraph("8.3 调用流程", styles["SectionTitle"]))
    plugin_flow = """\
用户消息
  |
  v
IntentDispatcher 判断是否触发 function_call
  |
  v
LLM 生成 function_call (选择插件 + 参数)
  |
  v
PluginManager.execute(name, args)
  |
  v
插件函数执行 (Wiki/RAG/结构化/食物营养...)
  |
  v
ActionResponse(action=REQLLM, content=context)
  |
  v
LLM 基于工具上下文生成最终回答"""
    story.append(code_block(plugin_flow, styles))

    story.append(PageBreak())


# ---- Chapter 9 ----
def build_ch9(story, styles):
    avail = PAGE_W - 4 * cm
    story.append(Paragraph("第 9 章  数据库架构", styles["ChapterTitle"]))
    story.append(hr())

    story.append(Paragraph("9.1 数据库总览", styles["SectionTitle"]))
    story.append(make_table(
        ["数据库文件", "用途", "主要表"],
        [
            ["clinical_knowledge.db", "结构化知识 + RAG 分块 + 记忆", "rag_documents, rag_chunks, rag_embeddings, guide_tables, recipe_plans, ..."],
            ["clinical_ltm.db", "长期记忆 + 工作记忆", "ltm_working_memory, ltm_short_term_summary, ltm_memory_items"],
            ["mem0_qdrant/", "mem0 向量存储", "Qdrant 内存向量库"],
            ["clinical_nutrition.db", "食物营养基础数据", "food_items, food_nutrients_per_100g, glycemic_values, food_aliases"],
        ],
        [avail * 0.3, avail * 0.3, avail * 0.4],
    ))

    story.append(Paragraph("9.2 RAG 存储结构", styles["SectionTitle"]))
    rag_schema = """\
rag_documents (文档注册)
  +-- document_id, title, source_path, status, created_at

rag_pages (页面/章节)
  +-- page_id, document_id, page_number, content, block_type

rag_chunks (分块)
  +-- chunk_id, page_id, chunk_index, content, char_count

rag_embeddings (向量)
  +-- chunk_id, embedding (BLOB), model_name, dimensions

rag_chunks_fts (FTS5 全文索引)
  +-- content (BM25 检索)"""
    story.append(code_block(rag_schema, styles))

    story.append(Paragraph("9.3 记忆存储结构", styles["SectionTitle"]))
    mem_schema = """\
ltm_working_memory (工作记忆)
  +-- user_id, session_id, turn_index, role, content, created_at

ltm_short_term_summary (短期摘要)
  +-- user_id, session_id, summary, turn_range, created_at

ltm_memory_items (长期记忆)
  +-- memory_id, user_id, layer (factual/episodic/semantic)
  +-- entity, attribute, value, content, source
  +-- importance, weight, dedupe_key
  +-- observed_at, last_accessed, access_count
  +-- tags, evidence (JSON), created_at"""
    story.append(code_block(mem_schema, styles))

    story.append(Paragraph("9.4 结构化知识存储", styles["SectionTitle"]))
    struct_schema = """\
source_documents (数据来源)
guide_tables / guide_table_rows (指南表格)
food_exchange_portions (食物交换份)
recipe_plans / recipe_meals / recipe_dishes / recipe_ingredients (食谱)
therapeutic_recipes (中医食养方)
activity_mets (运动 MET 系数)
diagnostic_thresholds (诊断阈值)
nutrition_targets (营养目标)
safety_rule_candidates (安全规则候选)
needs_review (待审核项)"""
    story.append(code_block(struct_schema, styles))

    story.append(PageBreak())


# ---- Chapter 10 ----
def build_ch10(story, styles):
    avail = PAGE_W - 4 * cm
    story.append(Paragraph("第 10 章  配置与部署", styles["ChapterTitle"]))
    story.append(hr())

    story.append(Paragraph("10.1 配置文件结构", styles["SectionTitle"]))
    story.append(Paragraph(
        "系统使用 config.yaml 作为主配置文件。支持通过 data/.config.yaml 覆盖部分配置，"
        "保护密钥安全。",
        styles["BodyCN"],
    ))
    story.append(make_table(
        ["配置块", "说明", "关键字段"],
        [
            ["server", "服务器基础配置", "ip, port, websocket, http_port"],
            ["selected_module", "模块选择", "VAD, ASR, LLM, TTS, Memory, Intent"],
            ["plugins", "插件配置", "各插件的 API key、参数"],
            ["clinical_rag", "RAG 配置", "chunk_chars, embedding, dimensions"],
            ["clinical_ltm", "记忆配置", "working_memory_turns, mem0, powermem"],
            ["llm", "LLM 配置", "provider, model, api_key, temperature"],
            ["asr", "ASR 配置", "provider, model"],
            ["tts", "TTS 配置", "provider, voice"],
        ],
        [avail * 0.2, avail * 0.3, avail * 0.5],
    ))

    story.append(Paragraph("10.2 LLM 配置", styles["SectionTitle"]))
    llm_config = """\
llm:
  type: openai  # or dashscope, zhipuai, etc.
  # DashScope (Qwen)
  model: qwen-plus
  api_key: "your-api-key"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  # 温度控制 (0.0-1.0)
  temperature: 0.7
  # 上下文窗口
  max_tokens: 2000"""
    story.append(code_block(llm_config, styles))

    story.append(Paragraph("10.3 RAG 配置", styles["SectionTitle"]))
    rag_config = """\
plugins:
  search_clinical_rag:
    chunk_chars: 850
    chunk_overlap: 150
    embedding:
      provider: dashscope
      model: text-embedding-v4
      dimensions: 256
    hybrid_weights:
      lexical: 0.55   # BM25 权重
      vector: 0.45    # 向量权重
    top_k: 5
    mmr_lambda: 0.7   # MMR 多样性参数"""
    story.append(code_block(rag_config, styles))

    story.append(Paragraph("10.4 记忆系统配置", styles["SectionTitle"]))
    ltm_config = """\
clinical_ltm:
  working_memory_turns: 12
  short_term_summary:
    enabled: true
    max_tokens: 500
  extraction:
    max_tokens: 1200
    temperature: 0.1
  lifecycle:
    episodic_half_life_days: 14
    semantic_half_life_days: 90
    synthesis_threshold: 5
  mem0:
    enabled: true
    mode: local  # or cloud
  powermem:
    enabled: true
    vector_store:
      provider: qdrant
      path: data/mem0_qdrant"""
    story.append(code_block(ltm_config, styles))

    story.append(Paragraph("10.5 部署步骤", styles["SectionTitle"]))
    deploy_steps = [
        "1. 安装依赖：pip install -r requirements.txt",
        "2. 创建数据目录：mkdir -p data/knowledge_base/llmwiki",
        "3. 复制配置模板：cp config.yaml data/.config.yaml",
        "4. 编辑 data/.config.yaml 填入 API Key",
        "5. 初始化数据库：python -c \"from core.clinical_nutrition.structured_knowledge import ...\"",
        "6. 导入食物营养数据：python scripts/import_china_food_composition_excel.py",
        "7. 摄入指南 PDF：python scripts/ingest_hypertension_guide.py",
        "8. 启动服务：python app.py",
    ]
    for step in deploy_steps:
        story.append(Paragraph(step, styles["BulletCN"]))

    story.append(Paragraph("10.6 环境依赖", styles["SectionTitle"]))
    story.append(make_table(
        ["依赖", "版本", "用途"],
        [
            ["Python", "3.10+", "运行时"],
            ["aiohttp", "3.9+", "WebSocket / HTTP 服务"],
            ["aiosqlite", "0.19+", "异步 SQLite"],
            ["reportlab", "4.4+", "PDF 文档生成"],
            ["pydantic", "2.0+", "数据模型"],
            ["loguru", "0.7+", "日志"],
            ["funasr", "1.0+", "语音识别"],
            ["edge-tts", "6.0+", "文本转语音"],
            ["mem0", "0.1+", "语义记忆"],
            ["PyMuPDF", "1.23+", "PDF 文本提取"],
        ],
        [avail * 0.25, avail * 0.2, avail * 0.55],
    ))


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------
def build_pdf(output_path: str):
    global _styles
    _styles = _build_styles()
    styles = _styles

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="xiaozhi-esp32-server 临床营养师 AI Agent 技术文档",
        author="Clinical Nutrition Team",
    )

    story = []
    build_cover(story, styles)
    build_toc(story, styles)
    build_ch1(story, styles)
    build_ch2(story, styles)
    build_ch3(story, styles)
    build_ch4(story, styles)
    build_ch5(story, styles)
    build_ch6(story, styles)
    build_ch7(story, styles)
    build_ch8(story, styles)
    build_ch9(story, styles)
    build_ch10(story, styles)

    doc.build(story)
    print(f"PDF generated: {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    output_dir = project_dir / "docs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "project_documentation.pdf"
    build_pdf(str(output_file))
