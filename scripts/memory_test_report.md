# 记忆系统测试报告

**测试时间**: 2026-05-10 19:11:07
**用例总数**: 102
**通过**: 102 | **失败**: 0 | **通过率**: 100.0%

---

## 单元测试-Store (20/20)

| ID | 测试项 | 状态 | 耗时 | 备注 |
|-----|--------|------|------|------|
| UT-S01 | Schema 初始化 | PASS | 40.3ms | - |
| UT-S02 | Working Memory 写入截断 | PASS | 60.2ms | - |
| UT-S03 | Working Memory 读取 | PASS | 51.5ms | - |
| UT-S04 | Working Memory 跨 session 隔离 | PASS | 68.9ms | - |
| UT-S05 | Short-term Summary upsert | PASS | 72.0ms | - |
| UT-S06 | Short-term Summary 截断 | PASS | 53.6ms | - |
| UT-S07 | Memory Item 新建 | PASS | 52.7ms | - |
| UT-S08 | Factual 去重更新+冲突记录 | PASS | 68.0ms | - |
| UT-S09 | Episodic 去重更新 | PASS | 66.1ms | - |
| UT-S10 | Memory Item 列表查询 | PASS | 119.8ms | - |
| UT-S11 | Memory Item 按 ID 查询 | PASS | 41.0ms | - |
| UT-S12 | 遗忘曲线衰减 | PASS | 67.1ms | - |
| UT-S13 | 向量检索 | PASS | 54.6ms | - |
| UT-S14 | 向量检索 min_weight 过滤 | PASS | 54.8ms | - |
| UT-S15 | cosine_similarity | PASS | 0.0ms | - |
| UT-S16 | hashed_embedding 确定性 | PASS | 0.1ms | - |
| UT-S17 | hashed_embedding 区分度 | PASS | 0.1ms | - |
| UT-S18 | clear_all_user_data | PASS | 62.8ms | - |
| UT-S19 | attach_powermem_index_id | PASS | 65.2ms | - |
| UT-S20 | 并发写入 | PASS | 344.1ms | - |

## 单元测试-Extractor (10/10)

| ID | 测试项 | 状态 | 耗时 | 备注 |
|-----|--------|------|------|------|
| UT-E07 | 回退规则提取-糖尿病 | PASS | 2.6ms | - |
| UT-E08 | 回退规则提取-过敏 | PASS | 1.3ms | - |
| UT-E09 | 回退规则提取-用药 | PASS | 1.3ms | - |
| UT-E10 | 回退规则提取-饮食事件 | PASS | 1.8ms | - |
| UT-E11 | 回退规则提取-医生建议 | PASS | 1.6ms | - |
| UT-E12 | 回退规则提取-噪声过滤 | PASS | 1.3ms | - |
| UT-E14 | JSON解析-代码块 | PASS | 0.1ms | - |
| UT-E15 | JSON解析-嵌入文本 | PASS | 0.1ms | - |
| UT-E16 | dedupe_key 一致性 | PASS | 0.0ms | - |
| UT-E17 | datetime 解析 | PASS | 0.0ms | - |

## 单元测试-Lifecycle (6/6)

| ID | 测试项 | 状态 | 耗时 | 备注 |
|-----|--------|------|------|------|
| UT-L01 | prepare_for_store-factual | PASS | 0.2ms | - |
| UT-L02 | prepare_for_store-episodic | PASS | 0.1ms | - |
| UT-L03 | prepare_for_store-semantic | PASS | 0.1ms | - |
| UT-L05 | synthesize-不足 | PASS | 39.9ms | - |
| UT-L08 | JSON数组提取 | PASS | 0.1ms | - |
| UT-L09 | dedupe_key生成 | PASS | 0.0ms | - |

## 单元测试-Interceptor (3/3)

| ID | 测试项 | 状态 | 耗时 | 备注 |
|-----|--------|------|------|------|
| UT-I01 | 构建prompt context | PASS | 65.5ms | - |
| UT-I04 | 空记忆 | PASS | 40.9ms | - |
| UT-I07 | min_weight过滤 | PASS | 54.5ms | - |

## 单元测试-HealthProfile (15/15)

| ID | 测试项 | 状态 | 耗时 | 备注 |
|-----|--------|------|------|------|
| UT-H01 | Schema初始化 | PASS | 63.2ms | - |
| UT-H02 | Profile创建 | PASS | 77.7ms | - |
| UT-H03 | BMI自动计算 | PASS | 74.2ms | - |
| UT-H04 | Item upsert-疾病 | PASS | 75.2ms | - |
| UT-H07 | Item去重 | PASS | 82.1ms | - |
| UT-H08 | 冲突检测-体重 | PASS | 79.6ms | - |
| UT-H10 | 非冲突更新 | PASS | 82.5ms | - |
| UT-H11 | Review item接受 | PASS | 98.6ms | - |
| UT-H12 | Review item拒绝 | PASS | 95.5ms | - |
| UT-H13 | 血糖记录写入 | PASS | 73.5ms | - |
| UT-H14 | 血糖分析-低血糖 | PASS | 0.0ms | - |
| UT-H15 | 血糖分析-高血糖 | PASS | 0.0ms | - |
| UT-H16 | 血糖分析-连续高 | PASS | 0.1ms | - |
| UT-H20 | Profile清除 | PASS | 80.9ms | - |
| UT-H21 | build_prompt_context | PASS | 78.9ms | - |

## 单元测试-NLP提取 (16/16)

| ID | 测试项 | 状态 | 耗时 | 备注 |
|-----|--------|------|------|------|
| UT-N01 | 年龄提取 | PASS | 2.9ms | - |
| UT-N02 | 性别提取 | PASS | 0.1ms | - |
| UT-N03 | 身高提取 | PASS | 0.1ms | - |
| UT-N04 | 体重提取-公斤 | PASS | 0.0ms | - |
| UT-N05 | 体重提取-斤 | PASS | 0.0ms | - |
| UT-N06 | 疾病提取 | PASS | 0.1ms | - |
| UT-N07 | 疾病提取-多个 | PASS | 0.1ms | - |
| UT-N08 | 用药提取 | PASS | 0.4ms | - |
| UT-N09 | 过敏提取 | PASS | 0.1ms | - |
| UT-N10 | 无过敏提取 | PASS | 0.1ms | - |
| UT-N11 | 血糖提取-空腹 | PASS | 4.5ms | - |
| UT-N12 | 血糖提取-餐后 | PASS | 0.2ms | - |
| UT-N15 | 目标提取 | PASS | 0.1ms | - |
| UT-N16 | 活动水平提取 | PASS | 0.0ms | - |
| UT-N19 | merge_profile_updates | PASS | 0.1ms | - |
| UT-N20 | 药物列表提取 | PASS | 0.1ms | - |

## 集成测试 (5/5)

| ID | 测试项 | 状态 | 耗时 | 备注 |
|-----|--------|------|------|------|
| IT-01 | Factual写入+检索 | PASS | 56.6ms | - |
| IT-02 | Episodic写入+检索 | PASS | 55.4ms | - |
| IT-06 | 健康档案联动 | PASS | 110.2ms | - |
| IT-07 | 血糖记录联动 | PASS | 74.5ms | - |
| IT-10 | 健康档案上下文 | PASS | 79.5ms | - |

## 性能测试 (6/6)

| ID | 测试项 | 状态 | 耗时 | 备注 |
|-----|--------|------|------|------|
| PF-01 | 单条记忆写入延迟 | PASS | 1289.0ms | - |
| PF-02 | 向量检索延迟 | PASS | 3061.4ms | - |
| PF-05 | 健康档案读取延迟 | PASS | 395.4ms | - |
| PF-06 | NLP提取延迟 | PASS | 15.4ms | - |
| PF-07 | cosine_similarity性能 | PASS | 39.6ms | - |
| PF-08 | hashed_embedding性能 | PASS | 29.3ms | - |

## 数据完整性 (5/5)

| ID | 测试项 | 状态 | 耗时 | 备注 |
|-----|--------|------|------|------|
| DI-02 | 唯一约束-dedupe | PASS | 60.7ms | - |
| DI-05 | JSON字段完整性 | PASS | 53.7ms | - |
| DI-06 | 时间字段格式 | PASS | 58.1ms | - |
| DI-09 | 空值处理 | PASS | 53.9ms | - |
| DI-10 | 大文本处理 | PASS | 74.0ms | - |

## 边界测试 (10/10)

| ID | 测试项 | 状态 | 耗时 | 备注 |
|-----|--------|------|------|------|
| EC-01 | 空用户ID | PASS | 48.6ms | - |
| EC-02 | 超长user_id | PASS | 52.7ms | - |
| EC-04 | 空内容记忆 | PASS | 47.3ms | - |
| EC-09 | 权重边界0 | PASS | 53.6ms | - |
| EC-10 | 权重边界1 | PASS | 53.3ms | - |
| EC-12 | DB文件不存在 | PASS | 36.0ms | - |
| EC-15 | memory_id不存在 | PASS | 40.2ms | - |
| EC-17 | 血糖值超界 | PASS | 0.1ms | - |
| EC-19 | 年龄超界 | PASS | 0.1ms | - |
| EC-20 | 体重超界 | PASS | 0.0ms | - |

## 回归测试 (6/6)

| ID | 测试项 | 状态 | 耗时 | 备注 |
|-----|--------|------|------|------|
| RG-01 | 现有记忆完整性 | PASS | 8.0ms | - |
| RG-02 | PowerMem索引完整性 | PASS | 1.0ms | - |
| RG-03 | 健康档案完整性 | PASS | 3.5ms | - |
| RG-04 | factual记忆权重 | PASS | 5.9ms | - |
| RG-05 | episodic记忆衰减 | PASS | 6.3ms | - |
| RG-06 | 向量检索一致性 | PASS | 7.7ms | - |

---

## 汇总

| 层级 | 用例数 | 通过 | 通过率 |
|------|--------|------|--------|
| 单元测试-Store | 20 | 20 | 100.0% |
| 单元测试-Extractor | 10 | 10 | 100.0% |
| 单元测试-Lifecycle | 6 | 6 | 100.0% |
| 单元测试-Interceptor | 3 | 3 | 100.0% |
| 单元测试-HealthProfile | 15 | 15 | 100.0% |
| 单元测试-NLP提取 | 16 | 16 | 100.0% |
| 集成测试 | 5 | 5 | 100.0% |
| 性能测试 | 6 | 6 | 100.0% |
| 数据完整性 | 5 | 5 | 100.0% |
| 边界测试 | 10 | 10 | 100.0% |
| 回归测试 | 6 | 6 | 100.0% |
| **总计** | **102** | **102** | **100.0%** |
