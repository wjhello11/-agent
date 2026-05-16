"""
Clinical RAG 评估脚本

评估指标：
- Recall@K: 前 K 个结果中是否包含正确答案片段
- Precision@K: 前 K 个结果中有多少是相关的
- MRR: 第一个正确结果排在第几位的倒数
- Hit Rate: 至少命中一个正确文档的比例
- Answer Coverage: 检索到的文本是否包含标准答案的关键信息

用法：
    cd D:/Agent/xiaozhi-esp32-server-main/main/xiaozhi-server
    python scripts/eval_clinical_rag.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import load_config
from core.clinical_nutrition.clinical_rag import ClinicalRAGService
from config.logger import setup_logging


# ============================================================
# 评估数据集：问题 + 标准答案 + 来源页码 + 关键词
# ============================================================

@dataclass
class EvalCase:
    question: str
    expected_answer: str  # 标准答案摘要
    expected_pages: list[int]  # 答案应出现的页码
    expected_keywords: list[str]  # 答案中应包含的关键词
    document_hint: str = ""  # 期望来自哪个文档（模糊匹配文件名）


EVAL_DATASET: list[EvalCase] = [
    # ================================================================
    # 全新题目集：不同问法、不同角度，验证泛化能力
    # ================================================================

    # ---- 肥胖：口语化/场景化问法 ----
    EvalCase(
        question="我身高170体重90公斤算胖吗？",
        expected_answer="BMI计算可知属于肥胖范围，BMI≥28为肥胖",
        expected_pages=[1, 2, 3, 4],
        expected_keywords=["BMI", "肥胖"],
        document_hint="肥胖",
    ),
    EvalCase(
        question="减肥的时候一个月瘦多少斤比较健康？",
        expected_answer="每月减2-4kg，每周约0.5kg",
        expected_pages=[11, 12, 13],
        expected_keywords=["每月", "2", "4kg"],
        document_hint="肥胖",
    ),
    EvalCase(
        question="大肚子型肥胖怎么判断？",
        expected_answer="中心型肥胖，男性腰围≥90cm，女性≥85cm",
        expected_pages=[3, 4, 5, 6],
        expected_keywords=["腰围", "中心型"],
        document_hint="肥胖",
    ),
    EvalCase(
        question="每天少吃多少才能瘦下来？",
        expected_answer="每日减少500-1000kcal",
        expected_pages=[7, 8, 9, 10, 11],
        expected_keywords=["500", "1000", "kcal"],
        document_hint="肥胖",
    ),
    EvalCase(
        question="不想跑步，走路能减肥吗？",
        expected_answer="每周150-300分钟中等强度有氧运动，快走属于中等强度",
        expected_pages=[10, 11, 12],
        expected_keywords=["150", "300", "中等强度"],
        document_hint="肥胖",
    ),

    # ---- 糖尿病：患者视角/日常问法 ----
    EvalCase(
        question="得了糖尿病是不是就不能吃米饭了？",
        expected_answer="可以吃但要控制量，选择低GI主食，粗细搭配",
        expected_pages=[4, 5, 6, 7],
        expected_keywords=["GI"],
        document_hint="糖尿病",
    ),
    EvalCase(
        question="糖尿病一天最多能吃多少碳水？",
        expected_answer="碳水化合物占总能量50%-60%",
        expected_pages=[3, 4, 5],
        expected_keywords=["50%", "60%", "碳水"],
        document_hint="糖尿病",
    ),
    EvalCase(
        question="糖尿病人为什么要多吃粗粮？",
        expected_answer="全谷物GI较低，膳食纤维有助于血糖控制，建议占主食1/3",
        expected_pages=[4, 5, 6],
        expected_keywords=["全谷物", "GI", "膳食纤维"],
        document_hint="糖尿病",
    ),
    EvalCase(
        question="糖尿病患者每天要吃多少克膳食纤维？",
        expected_answer="每日25-30g膳食纤维",
        expected_pages=[5, 6, 7],
        expected_keywords=["25", "30", "膳食纤维"],
        document_hint="糖尿病",
    ),
    EvalCase(
        question="妊娠糖尿病和普通糖尿病饮食一样吗？",
        expected_answer="本标准不适用于妊娠糖尿病等特殊类型",
        expected_pages=[1, 2, 3],
        expected_keywords=["妊娠糖尿病"],
        document_hint="糖尿病",
    ),

    # ---- 痛风/高尿酸：患者常见困惑 ----
    EvalCase(
        question="体检发现尿酸高但没有症状需要管吗？",
        expected_answer="高尿酸血症即使无症状也需要管理，非同日2次>420μmol/L即确诊",
        expected_pages=[1, 2, 3, 4, 5],
        expected_keywords=["420", "高尿酸血症", "非同日"],
        document_hint="hyperuricemia",
    ),
    EvalCase(
        question="痛风的人能不能吃海鲜？",
        expected_answer="应避免高嘌呤食物，海鲜属于高嘌呤",
        expected_pages=[12, 13, 14, 15, 16],
        expected_keywords=["海鲜", "嘌呤", "避免"],
        document_hint="hyperuricemia",
    ),
    EvalCase(
        question="尿酸高喝啤酒好还是白酒好？",
        expected_answer="都应限制，啤酒风险最高，白酒黄酒也需限制",
        expected_pages=[6, 7, 8, 9],
        expected_keywords=["啤酒", "限制饮酒"],
        document_hint="hyperuricemia",
    ),
    EvalCase(
        question="痛风发作的时候饮食怎么调整？",
        expected_answer="急性期严格限制嘌呤，多饮水，避免酒精",
        expected_pages=[10, 11, 12, 13, 14],
        expected_keywords=["嘌呤", "饮水", "限制"],
        document_hint="hyperuricemia",
    ),
    EvalCase(
        question="果糖和尿酸有什么关系？",
        expected_answer="果糖会升高尿酸，应限制含糖饮料和高果糖食物",
        expected_pages=[13, 14, 15, 16],
        expected_keywords=["果糖", "尿酸"],
        document_hint="hyperuricemia",
    ),
    EvalCase(
        question="痛风病人一天要喝够多少水？",
        expected_answer="每日2000-3000mL，保证尿量>2000mL",
        expected_pages=[10, 11, 12, 13, 14, 15],
        expected_keywords=["2000", "3000"],
        document_hint="hyperuricemia",
    ),

    # ---- 跨文档 / 综合问题 ----
    EvalCase(
        question="我又有糖尿病又有痛风，吃饭该怎么搭配？",
        expected_answer="低GI饮食控制血糖，同时限制嘌呤，多饮水",
        expected_pages=[1, 2, 3, 4, 5],
        expected_keywords=["GI", "嘌呤"],
        document_hint="",
    ),
    EvalCase(
        question="肥胖加高尿酸，饮食上有什么要注意的？",
        expected_answer="控制能量减重，限制高嘌呤食物，避免果糖",
        expected_pages=[6, 7, 8, 9, 10, 20],
        expected_keywords=["嘌呤", "果糖"],
        document_hint="",
    ),
    EvalCase(
        question="糖尿病肾病患者的蛋白质怎么控制？",
        expected_answer="肾功能不全时限制蛋白质摄入",
        expected_pages=[6, 7, 8],
        expected_keywords=["蛋白质", "肾"],
        document_hint="",
    ),
]


# ============================================================
# 评估逻辑
# ============================================================

@dataclass
class EvalResult:
    question: str
    recall_at_3: float
    recall_at_6: float
    precision_at_3: float
    precision_at_6: float
    mrr: float
    hit: bool
    answer_covered: bool
    top_pages: list[int] = field(default_factory=list)
    top_scores: list[float] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


def evaluate_case(
    service: ClinicalRAGService,
    case: EvalCase,
    top_k: int = 6,
) -> EvalResult:
    start = time.perf_counter()
    results = service.search(case.question, top_k=top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # 收集检索到的页码
    retrieved_pages: list[int] = []
    for item in results:
        p_start = item.get("page_start", 0)
        p_end = item.get("page_end", p_start)
        for p in range(p_start, p_end + 1):
            retrieved_pages.append(p)

    # 判断哪些 chunk 是"相关的"（页码与期望页码有交集）
    expected_page_set = set(case.expected_pages)
    relevant_indices: list[int] = []
    for i, item in enumerate(results):
        p_start = item.get("page_start", 0)
        p_end = item.get("page_end", p_start)
        chunk_pages = set(range(p_start, p_end + 1))
        if chunk_pages & expected_page_set:
            relevant_indices.append(i)

    # Recall@K: 是否至少有一个相关结果
    recall_at_3 = 1.0 if any(i < 3 for i in relevant_indices) else 0.0
    recall_at_6 = 1.0 if any(i < 6 for i in relevant_indices) else 0.0

    # Precision@K
    precision_at_3 = sum(1 for i in relevant_indices if i < 3) / 3.0
    precision_at_6 = sum(1 for i in relevant_indices if i < 6) / 6.0

    # MRR: 第一个相关结果的排名倒数
    mrr = 0.0
    for rank, i in enumerate(relevant_indices):
        if i < top_k:
            mrr = 1.0 / (i + 1)
            break

    # Hit: 至少有一个相关结果
    hit = len(relevant_indices) > 0

    # Answer Coverage: 检索文本中是否包含标准答案的关键词
    all_text = " ".join(str(item.get("text", "")) for item in results)
    matched_keywords = [kw for kw in case.expected_keywords if kw in all_text]
    answer_covered = len(matched_keywords) >= max(1, len(case.expected_keywords) // 2)

    return EvalResult(
        question=case.question,
        recall_at_3=recall_at_3,
        recall_at_6=recall_at_6,
        precision_at_3=precision_at_3,
        precision_at_6=precision_at_6,
        mrr=mrr,
        hit=hit,
        answer_covered=answer_covered,
        top_pages=retrieved_pages[:12],
        top_scores=[item.get("score", 0.0) for item in results[:6]],
        matched_keywords=matched_keywords,
        elapsed_ms=elapsed_ms,
    )


def run_evaluation() -> None:
    print("=" * 70)
    print("Clinical RAG Evaluation")
    print("=" * 70)

    config = load_config()
    logger = setup_logging()
    service = ClinicalRAGService(
        project_root=PROJECT_ROOT,
        config=config,
        logger=logger,
    )

    results: list[EvalResult] = []
    for i, case in enumerate(EVAL_DATASET, 1):
        print(f"\n[{i}/{len(EVAL_DATASET)}] {case.question}")
        result = evaluate_case(service, case)
        results.append(result)

        status = "HIT" if result.hit else "MISS"
        coverage = "OK" if result.answer_covered else "WEAK"
        print(f"  {status} | Recall@3={result.recall_at_3:.0f} | MRR={result.mrr:.2f} | "
              f"Coverage={coverage} | {result.elapsed_ms:.0f}ms")
        if result.matched_keywords:
            print(f"  Keywords: {', '.join(result.matched_keywords)}")
        if result.top_scores:
            scores_str = ", ".join(f"{s:.3f}" for s in result.top_scores)
            print(f"  Top scores: {scores_str}")

    # 汇总
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    n = len(results)
    avg_recall_3 = sum(r.recall_at_3 for r in results) / n
    avg_recall_6 = sum(r.recall_at_6 for r in results) / n
    avg_precision_3 = sum(r.precision_at_3 for r in results) / n
    avg_precision_6 = sum(r.precision_at_6 for r in results) / n
    avg_mrr = sum(r.mrr for r in results) / n
    hit_rate = sum(1 for r in results if r.hit) / n
    coverage_rate = sum(1 for r in results if r.answer_covered) / n
    avg_latency = sum(r.elapsed_ms for r in results) / n

    print(f"  Test cases:        {n}")
    print(f"  Hit Rate:          {hit_rate:.1%}")
    print(f"  Recall@3:          {avg_recall_3:.1%}")
    print(f"  Recall@6:          {avg_recall_6:.1%}")
    print(f"  Precision@3:       {avg_precision_3:.1%}")
    print(f"  Precision@6:       {avg_precision_6:.1%}")
    print(f"  MRR:               {avg_mrr:.3f}")
    print(f"  Answer Coverage:   {coverage_rate:.1%}")
    print(f"  Avg Latency:       {avg_latency:.0f}ms")

    # 失败案例
    misses = [r for r in results if not r.hit]
    if misses:
        print(f"\n  Missed cases ({len(misses)}):")
        for r in misses:
            print(f"    - {r.question}")

    weak = [r for r in results if r.hit and not r.answer_covered]
    if weak:
        print(f"\n  Weak coverage ({len(weak)}):")
        for r in weak:
            print(f"    - {r.question} (matched: {', '.join(r.matched_keywords)})")

    # 输出 JSON 报告
    report = {
        "summary": {
            "test_cases": n,
            "hit_rate": round(hit_rate, 4),
            "recall_at_3": round(avg_recall_3, 4),
            "recall_at_6": round(avg_recall_6, 4),
            "precision_at_3": round(avg_precision_3, 4),
            "precision_at_6": round(avg_precision_6, 4),
            "mrr": round(avg_mrr, 4),
            "answer_coverage": round(coverage_rate, 4),
            "avg_latency_ms": round(avg_latency, 1),
        },
        "details": [
            {
                "question": r.question,
                "recall_at_3": r.recall_at_3,
                "recall_at_6": r.recall_at_6,
                "precision_at_3": r.precision_at_3,
                "precision_at_6": r.precision_at_6,
                "mrr": r.mrr,
                "hit": r.hit,
                "answer_covered": r.answer_covered,
                "matched_keywords": r.matched_keywords,
                "top_scores": [round(s, 4) for s in r.top_scores],
                "elapsed_ms": round(r.elapsed_ms, 1),
            }
            for r in results
        ],
    }

    report_path = PROJECT_ROOT / "scripts" / "rag_eval_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Report saved to: {report_path}")


if __name__ == "__main__":
    run_evaluation()
