# 临床营养 Agent 检索层改进记录

日期：2026-05-09

## 一、Clinical RAG 改进

文件：`core/clinical_nutrition/clinical_rag.py`

### 1.1 `_adjust_rag_candidate_score` 重构

**问题**：原加分系统硬编码最高 +1.35 分，且不检查 chunk 来自哪个文档。导致"痛风患者可以喝酒"这种问题给肥胖指南的 chunk 加了大量分。

**改动**：
- 加分前先检查 `source_name` 是否匹配问题主题（如"肥胖"问题只给肥胖指南 chunk 加分）
- 单项加分从最高 1.35 降到 0.5
- 新增 `section_title` 匹配加分（章节标题匹配比正文匹配更有意义）
- 新增 `_source_matches()` 和 `_section_boost()` 辅助函数

### 1.2 `_expand_rag_question` 扩展

**改动**：新增 10+ 个主题扩展：
- 食谱/做法类（铁皮石斛、药膳、烹饪）
- 禁忌/不能吃类
- 蛋白质/脂肪/碳水具体数字
- 能量/热量类
- 肾功能/并发症类
- 早餐/晚餐/加餐类

### 1.3 BM25 权重调整

**问题**：原 BM25 权重 0.45，向量 0.55。但 BM25 分数 `1/(rank+1) * 0.45` 最高仅 0.225，被向量分数淹没。

**改动**：三处权重调整：
- `_lexical_search`：`0.45 → 0.55`
- `_vector_search`：`0.55 → 0.45`
- `_merge_candidates`：`lexical * 0.45 + vector * 0.55 → lexical * 0.55 + vector * 0.45`

### 1.4 效果

| 指标 | 改进前 (17题) | 改进后 (21题) | 全新题目 (19题) |
|------|-------------|-------------|---------------|
| Hit Rate | 82.4% | 100.0% | 100.0% |
| Recall@3 | 64.7% | 95.2% | 89.5% |
| MRR | 0.623 | 0.700 | 0.632 |

---

## 二、LLMWiki 搜索改进

文件：`plugins_func/functions/search_from_llmwiki.py`

### 2.1 IDF 加权

**问题**：原评分 `score = body_overlap + title_overlap * 2`，纯计数无 IDF，常见词（"糖尿""食物"）权重过高。

**改动**：新增 `_idf()` 函数，使用 smoothed IDF：
```
idf(token) = log((N+1) / (df+1)) + 1.0
```
body 匹配、title 匹配、metadata 匹配均使用 IDF 加权。

### 2.2 Frontmatter 元数据匹配

**改动**：
- `_load_markdown_documents` 中将 `clinical_domain`、`conditions`、`kb_layer` 字段加入 tokens
- 新增 `meta_token_set` 字段
- `_rank_documents` 中新增 metadata 匹配加分（权重 1.5x）

### 2.3 Query Expansion（中英映射）

**改动**：新增 `QUERY_EXPANSIONS` 字典（30+ 条映射）和 `_expand_query_tokens()` 函数：
- "低血糖" → hypoglycemia
- "过敏" → allergy
- "蛋白质" → protein
- "GI" → gi, glycemic_index
- 等等

### 2.4 Title 匹配权重提升

**改动**：`title_overlap * 2 → title_overlap * 3`

### 2.5 噪音页面排除

**改动**：`_load_markdown_documents` 中排除 `README.md` 和 `templates/` 目录。

### 2.6 效果

| 指标 | 改进前 (第一套) | 改进后 (第二套) |
|------|---------------|---------------|
| Top-1 Accuracy | 60.0% | 63.2% |
| Hit@4 | 95.0% | 100.0% |
| MRR | 0.750 | 0.807 |
| 安全边界 Hit@4 | 75% | 100% |
| 安全边界 MRR | 0.46 | 0.88 |

---

## 三、新增评估脚本

### 3.1 `scripts/eval_clinical_rag.py`

RAG 评估脚本，19 个测试用例，覆盖：
- 肥胖（口语化问法）
- 糖尿病（患者视角）
- 痛风/高尿酸（常见困惑）
- 跨文档综合问题

指标：Hit Rate, Recall@3/6, Precision@3/6, MRR, Answer Coverage, Latency

### 3.2 `scripts/eval_llmwiki_search.py`

Wiki 搜索评估脚本，19 个测试用例，覆盖：
- 精确匹配（专业术语）
- 主题匹配（口语化表达）
- 场景匹配（日常饮食）
- 安全边界（红旗信号）
- 跨页（综合问题）

指标：Top-1 Accuracy, Hit@3/4, MRR, Precision@3

### 3.3 `scripts/ingest_hypertension_guide.py`

完整入库流程脚本：注册 → 索引（分块+向量化+FTS5）→ RAG 搜索验证 → LLMWiki 编译 → Wiki 搜索验证
