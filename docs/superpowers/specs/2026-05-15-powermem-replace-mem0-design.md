# 用 PowerMem 替代 mem0 的语义提示功能

## 背景

当前长期记忆系统使用三层架构：
- **mem0**：语义预提取与认知分类提示（hints）
- **PowerMem**：长期记忆语义检索
- **本地 SQLite**：四层结构化记忆、短期记忆、健康档案、冲突管理

目标：**移除 mem0 依赖，让 PowerMem 替代 mem0 的语义提示部分**，其余保持不变。

## 设计原则

1. 不删除本地 SQLite 记忆主库
2. 不删除结构化健康档案
3. 不删除短期记忆摘要
4. 不把所有记忆交给 PowerMem 黑盒管理
5. 只把"mem0 提供语义 hints"替换为"PowerMem 检索相关历史记忆作为 hints"
6. 优先级：当前对话事实 > 结构化健康档案 > 本地 SQLite 长期记忆 > PowerMem 检索 hints

## 目标架构

```
用户对话
  ↓
本地规则/LLM 抽取候选记忆
  ↓
PowerMem 检索相关历史记忆，作为 extraction hints
  ↓
LLM 根据当前对话 + PowerMem hints 抽取 factual/episodic/semantic memories
  ↓
写入本地 SQLite 主记忆库
  ↓
同步到 PowerMem 检索索引
  ↓
对话前从 PowerMem + SQLite 检索记忆并拼进 prompt
```

最终结构：
```
ClinicalLTM
├── Short-Term Summary Memory
├── Health Profile Store
├── SQLite Structured Memory Store
├── PowerMem Retrieval Index
└── Clinical Cognitive Extractor
    └── 使用 PowerMem hints，不再使用 mem0
```

## 改造范围

### 需要修改的文件

| 文件 | 改动类型 |
|------|----------|
| `core/providers/memory/clinical_ltm/extractor.py` | 重构 |
| `core/providers/memory/clinical_ltm/clinical_ltm.py` | 小改 |
| `core/providers/memory/clinical_ltm/prompts.py` | 小改 |
| `config.yaml` | 配置更新 |

### 可选新增

| 文件 | 用途 |
|------|------|
| `scripts/resync_clinical_ltm_powermem.py` | 历史数据同步脚本 |

### 不修改的文件

- `store.py` — SQLite 主库
- `health_profile.py` — 健康档案
- `lifecycle.py` — 生命周期管理
- `interceptor.py` — 检索拦截器
- `models.py` — 数据模型
- `powermem_index.py` — PowerMem 索引层

## 详细改动

### 1. extractor.py — 重构抽取器

**类重命名**：`Mem0CognitiveExtractor` → `ClinicalCognitiveExtractor`

**删除内容**：
- `from mem0 import AsyncMemory, AsyncMemoryClient`
- `_get_mem0_hints()` 方法
- `_get_mem0_engine()` 方法
- `_build_default_mem0_local_config()` 方法
- `self._mem0_engine` 属性
- `self.mem0_config` 属性

**新增内容**：
- 构造函数增加 `powermem_index` 参数
- `_get_powermem_hints(messages, user_id)` 方法

**`_get_powermem_hints` 逻辑**：
```python
async def _get_powermem_hints(self, messages, user_id):
    if not self.powermem_index or not getattr(self.powermem_index, "enabled", False):
        return []

    query = self._build_hint_query(messages)
    if not query:
        return []

    limit = int(self.config.get("extraction_hints_limit", 6))

    try:
        result = await self.powermem_index.search(
            query=query, user_id=user_id, limit=limit
        )
    except Exception as exc:
        self.logger.warning(f"PowerMem hints 检索失败: {exc}")
        return []

    hints = []
    for item in result.get("results") or []:
        text = item.get("memory") or item.get("content") or item.get("text") or ""
        if text:
            hints.append(str(text).strip())

    return hints[:limit]
```

**`_build_hint_query` 逻辑**：
- 取最近几轮用户消息拼成 query
- 截断到合理长度

**变量重命名**：所有 `mem0_hints` → `retrieval_hints`

**`_fallback_extract` 改动**：
- `source="mem0语义提示"` → `source="PowerMem检索提示"`
- `tags=["semantic", "mem0-hint"]` → `tags=["semantic", "retrieval-hint"]`

**`_extract_with_llm` 改动**：
- `mem0_hints` 参数名 → `retrieval_hints`
- 传入 prompt 的变量名同步更新

### 2. clinical_ltm.py — 调整初始化顺序

**导入改动**：
```python
# 之前
from .extractor import Mem0CognitiveExtractor
# 之后
from .extractor import ClinicalCognitiveExtractor
```

**初始化顺序**：
```python
# 之前
self.extractor = Mem0CognitiveExtractor(config, None, logger)
self.powermem_index = PowerMemOfficialIndex(config, logger)

# 之后（先 powermem_index，再传给 extractor）
self.powermem_index = PowerMemOfficialIndex(config, logger)
self.extractor = ClinicalCognitiveExtractor(
    config, None, logger, powermem_index=self.powermem_index
)
```

**文档字符串更新**：删除 "使用 mem0 进行认知分类与语义预提取" 相关描述。

### 3. prompts.py — 更新提示词

**`EXTRACTION_USER_TEMPLATE` 改动**：
```
# 之前
可选的 mem0 语义提示:
{mem0_hints}

# 之后
PowerMem 检索到的相关历史记忆:
{retrieval_hints}
```

**删除 `MEM0_EXTRACTION_PROMPT`**：这个 prompt 是给 mem0 的 `infer=True` 模式用的，PowerMem 检索不需要。

### 4. config.yaml — 配置更新

**禁用 mem0**：
```yaml
clinical_ltm:
  # mem0 已废弃：PowerMem 接管语义检索 hints
  mem0:
    enabled: false
    mode: disabled
```

**PowerMem 新增配置**：
```yaml
powermem:
  extraction_hints_enabled: true
  extraction_hints_limit: 6
  extraction_hints_max_chars: 1200
  # ... 原有 llm/embedder/vector_store 配置不变
```

**注释更新**：
```
# 之前
# 2. mem0 local OSS 负责语义预提取与认知分类提示
# 3. PowerMem 官方 AsyncMemory 负责长期记忆检索

# 之后
# 2. PowerMem 官方 AsyncMemory 负责长期记忆检索，并为记忆抽取阶段提供相关历史 hints
# 3. 本地 SQLite 负责四层结构、冲突管理、遗忘曲线、短期记忆和健康档案
```

### 5. 可选：resync_clinical_ltm_powermem.py

同步脚本逻辑：
- 读取 `ltm_memory_items` 表
- 筛选没有 `powermem_memory_id` 的记录
- 调用 `PowerMemOfficialIndex.upsert_memory` 写入 PowerMem
- 写回 `powermem_memory_id` 到 metadata

## 验收标准

### 代码层

```powershell
python -m py_compile core\providers\memory\clinical_ltm\clinical_ltm.py core\providers\memory\clinical_ltm\extractor.py core\providers\memory\clinical_ltm\powermem_index.py
```

```powershell
rg "from mem0|AsyncMemoryClient|_get_mem0|Mem0CognitiveExtractor" core\providers\memory\clinical_ltm
```
要求运行时代码里不再出现这些内容（文档注释里的 "mem0 已废弃" 说明除外）。

### 启动层

启动后日志应出现：
```
PowerMem 官方检索层初始化成功
```

不应出现：
```
mem0 初始化
mem0 语义预提取
mem0 qdrant
```

### 功能层

1. 用户说"我有二型糖尿病，体重60公斤" → 健康档案更新 + 长期记忆保存 + 不依赖 mem0
2. 下一轮问"我能喝奶茶吗？" → 能结合糖尿病背景
3. 用户说"我刚才说我的体重是多少？" → 能从记忆中回答

## 风险提醒

**禁止做的事**：
1. 不删除健康档案系统
2. 不删除短期记忆摘要
3. 不把疾病、过敏、用药等只放进 PowerMem
4. 不在 extractor 阶段直接把原始对话写进 PowerMem
5. 不让 PowerMem hints 覆盖当前用户新输入
