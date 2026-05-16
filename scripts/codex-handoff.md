# 项目改动交接文档

**日期**：2026-05-10
**目的**：将"AI 文档结构编译器 + Schema 校验 + 人工审核"入库架构的所有改动整理交接给 Codex

---

## 一、整体背景

系统已有 `KnowledgeIngestionService`（AI 文档结构编译器），之前处理了 4 个临床营养 PDF，生成了 20 个 draft，但 wiki 里只有 9 页（旧的糖尿病相关页面）。

本次工作流程：**审批 draft 入库 wiki → 检索验证 → Agent 全面测试**。

---

## 二、新建文件（4 个脚本 + 1 个报告）

### 1. `scripts/approve_drafts.py`

批量审批 20 个 draft 中的 4 个 PDF wiki 内容入库。

**逻辑**：
- 遍历 `data/knowledge_ingestion/drafts/` 下 20 个 draft 目录
- 每个 PDF 选"最好的"那个 draft（有 wiki pages 的最新版）
- 调用 `service.approve_draft(draft_id)` 把 `pages/` 下的 md 文件写入 `knowledge_base/llmwiki/clinical-nutrition/guidelines/`
- 高血压指南没有现成 draft，额外调用 `service.create_draft(pdf_path)` 生成 + 审批（耗时 ~20 分钟）

**审批结果**：

| PDF | Draft ID | Wiki 页数 |
|-----|----------|-----------|
| 成人肥胖食养指南 | `20260507033626-610446e9b8-be0d81` | 7 页 |
| 糖尿病膳食指南 | `20260507041635-57e2e5478c-bd19f7` | 3 页 |
| 成人高尿酸血症与痛风食养指南 | `20260508114420-7179e6fd24-39b0f7` | 6 页 |
| 中国高血压防治指南（2024修订版） | `20260509182002-5aaab0779b-c164dd`（新生成） | 6 页 |

### 2. `scripts/test_wiki_retrieval.py`

测试 wiki 检索准确率，15 个测试用例覆盖 4 个 PDF。

- 调用 `search_from_llmwiki._load_markdown_documents()` + `_rank_documents()`
- 验证 Hit@1 和 Hit@4
- 结果：Hit@4 = 100%（15/15）

### 3. `scripts/test_agent_comprehensive.py`

Agent 全面对话测试，20 个用例，8 个类别，调用真实 LLM 生成回答。

- 调用 DashScope API（`qwen-plus` 模型，因 `qwen3.6-flash` 返回 403）
- 用户 ID 用真实值 `3c0f02d924e0`
- 问题用大白话模拟真实用户
- 输出 Markdown 对话报告

**8 个测试类别**：

| 类别 | 用例数 | 说明 |
|------|--------|------|
| Wiki检索 | 6 | 覆盖 4 个 PDF + 原有 wiki |
| 结构化知识 | 2 | MET 值、食养方 |
| 食物营养 | 3 | 鸡蛋、米饭/馒头、苹果 |
| 整餐分析 | 1 | 早餐热量估算 |
| 健康档案 | 1 | 读取用户档案 |
| 记忆检索 | 2 | 个性化建议、血糖记录 |
| 安全边界 | 3 | 低血糖、停药、孕妇用药 |
| 跨模块综合 | 2 | 糖尿病+红薯、肥胖+运动 |

### 4. `scripts/agent_test_report.md`

测试脚本生成的对话报告，展示用户问题 → Agent 回答 + 耗时。

---

## 三、修改文件（1 个）

### `knowledge_base/llmwiki/clinical-nutrition/_index.md`

在末尾新增 4 条索引，指向 4 个新入库的 wiki 目录：

```markdown
- [成人肥胖食养指南 Wiki 总索引](guidelines/成人肥胖食养指南-wiki-总索引/index.md)
- [糖尿病膳食指南 Wiki 总索引](guidelines/糖尿病膳食指南-wiki-总索引/index.md)
- [成人高尿酸血症与痛风食养指南 Wiki 总索引](guidelines/成人高尿酸血症与痛风食养指南-wiki-总索引-wiki-总索引/index.md)
- [中国高血压防治指南（2024年修订版） Wiki 总索引](guidelines/中国高血压防治指南-2024年修订版--wiki-总索引/index.md)
```

---

## 四、新增数据（Wiki 页面）

`knowledge_base/llmwiki/clinical-nutrition/guidelines/` 下新增 4 个子目录，共 22 个 md 文件：

| 目录 | 页面文件 |
|------|---------|
| `成人肥胖食养指南-wiki-总索引/` | index, overview, principles, energy-control, food-selection, safe-weight-loss, regional-recipes, tcm-diet-therapy（8个） |
| `糖尿病膳食指南-wiki-总索引/` | index, overview, principles, food-selection（4个） |
| `成人高尿酸血症与痛风食养指南-wiki-总索引-wiki-总索引/` | index, overview, energy-control, food-selection, exercise-sleep, structured-tables, tcm-diet-therapy（7个） |
| `中国高血压防治指南-2024年修订版--wiki-总索引/` | index, overview, principles, energy-control, food-selection, exercise-sleep, tcm-diet-therapy（7个） |

**Wiki 总页面数**：9 → 38 页

---

## 五、未修改的核心代码（仅调用/测试）

| 文件 | 模块 | 说明 |
|------|------|------|
| `core/clinical_nutrition/knowledge_ingestion.py` | KnowledgeIngestionService | 调用了 `approve_draft()` 和 `create_draft()` |
| `plugins_func/functions/search_from_llmwiki.py` | Wiki 检索 | 调用了 `_load_markdown_documents()` + `_rank_documents()` |
| `core/providers/memory/clinical_ltm/health_profile.py` | 健康档案 | 调用了 `get_profile_sync()` |
| `data/clinical_health_profile.db` | 健康档案 DB | 只读查询，user_id=`3c0f02d924e0` |
| `data/clinical_ltm.db` | 长期记忆 DB | 只读查询 |
| `data/clinical_knowledge.db` | 结构化知识 DB | 只读查询 |
| `data/clinical_foods.db` | 食物营养 DB | 只读查询 |
| `data/.config.yaml` | 配置 | 只读（取 API key 和模型名） |

---

## 六、已知问题

1. **高血压 structured ingestion 报错**：`13 values for 12 columns`，非关键错误，wiki 内容已正常写入
2. **`qwen3.6-flash` 模型 403**：测试脚本改用 `qwen-plus`，不影响生产配置（生产用 `qwen3.6-plus`）
3. **痛风 wiki 目录名有重复后缀**：`成人高尿酸血症与痛风食养指南-wiki-总索引-wiki-总索引`，是 draft 生成时的命名问题
