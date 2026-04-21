import asyncio
import collections
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Import cả V1 và V2 từ file main_agent của bạn
from agent.main_agent import MainAgent, MainAgentV2 
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner
try:
    from ragas.metrics.collections import faithfulness, answer_relevancy
except Exception:
    from ragas.metrics import faithfulness, answer_relevancy
try:
    from ragas.dataset_schema import SingleTurnSample
except Exception:
    SingleTurnSample = None

load_dotenv()

class ExpertEvaluator:
    def __init__(self, top_k: int = 5):
        self.retrieval_eval = RetrievalEvaluator(top_k=top_k)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower())

    def _jaccard_similarity(self, left: str, right: str) -> float:
        left_tokens = set(self._tokenize(left))
        right_tokens = set(self._tokenize(right))
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    def _heuristic_faithfulness(self, answer: str, contexts: List[str]) -> float:
        return round(self._jaccard_similarity(answer, " ".join(contexts or [])), 4)

    def _heuristic_relevancy(self, answer: str, expected_answer: str, question: str) -> float:
        answer_to_expected = self._jaccard_similarity(answer, expected_answer)
        answer_to_question = self._jaccard_similarity(answer, question)
        return round((0.8 * answer_to_expected) + (0.2 * answer_to_question), 4)

    async def _compute_ragas_metric(
        self,
        metric,
        row: Dict,
        fallback_value: float,
    ) -> float:
        sample = SingleTurnSample(**row) if SingleTurnSample is not None else row

        async_methods = (
            ("single_turn_ascore", sample),
            ("ascore", row),
        )
        sync_methods = (
            ("single_turn_score", sample),
            ("score", row),
        )

        for method_name, payload in async_methods:
            method = getattr(metric, method_name, None)
            if method is None:
                continue
            try:
                result = method(payload)
                if asyncio.iscoroutine(result):
                    result = await result
                return float(result)
            except Exception:
                continue

        for method_name, payload in sync_methods:
            method = getattr(metric, method_name, None)
            if method is None:
                continue
            try:
                result = await asyncio.to_thread(method, payload)
                return float(result)
            except Exception:
                continue

        return fallback_value
    
    async def score(self, case, resp): 
        """
        Score: Tính Hit Rate, MRR, Faithfulness, Relevancy.
        """
        # 1. Tính toán Retrieval Metrics (dựa trên ID)
        expected_ids = case.get("ground_truth_ids", [])
        retrieved_ids = resp.get("retrieved_ids", [])
        
        hit_rate = self.retrieval_eval.calculate_hit_rate(expected_ids, retrieved_ids)
        mrr = self.retrieval_eval.calculate_mrr(expected_ids, retrieved_ids)
        
        # 2. Chuẩn bị data cho Ragas (BẮT BUỘC phải là text)
        row = {
            "question": case.get("question", ""),
            "answer": resp.get("answer", ""),
            "contexts": resp.get("retrieved_context") or resp.get("contexts", [])
        }
        
        heuristic_faithfulness = self._heuristic_faithfulness(row["answer"], row["contexts"])
        heuristic_relevancy = self._heuristic_relevancy(
            row["answer"],
            case.get("expected_answer", ""),
            row["question"],
        )

        # 3. Tính toán Generation Metrics bằng Ragas với fallback an toàn theo version
        faithfulness_score = await self._compute_ragas_metric(
            faithfulness,
            row,
            heuristic_faithfulness,
        )
        relevancy_score = await self._compute_ragas_metric(
            answer_relevancy,
            row,
            heuristic_relevancy,
        )
        
        return {
            "faithfulness": faithfulness_score,
            "relevancy": relevancy_score,
            "retrieval": {
                "hit_rate": hit_rate, 
                "mrr": mrr
            }
        }


def normalize_test_case(raw_case: Dict, line_number: int) -> Optional[Dict]:
    normalized = dict(raw_case)

    question = normalized.get("question")
    if not question:
        messages = normalized.get("messages", [])
        user_messages = [
            message.get("content", "").strip()
            for message in messages
            if message.get("role") == "user" and message.get("content")
        ]
        if user_messages:
            question = user_messages[-1]

    expected_answer = (
        normalized.get("expected_answer")
        or normalized.get("answer")
        or normalized.get("ground_truth")
    )

    if not question or not expected_answer:
        print(f"Bo qua test case dong {line_number}: thieu question hoac expected_answer.")
        return None

    normalized["question"] = question
    normalized["expected_answer"] = expected_answer
    normalized["ground_truth_ids"] = normalized.get("ground_truth_ids", []) or normalized.get("expected_retrieval_ids", [])
    normalized.setdefault("metadata", {})
    return normalized


def load_dataset(path: str = "data/golden_set.jsonl") -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Thiếu {path}. Hãy chạy 'python data/synthetic_gen.py' trước.")

    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            raw_case = json.loads(line)
            normalized_case = normalize_test_case(raw_case, line_number)
            if normalized_case is not None:
                dataset.append(normalized_case)

    if not dataset:
        raise ValueError(f"File {path} rỗng. Hãy tạo ít nhất 1 test case.")

    return dataset


def calculate_cohens_kappa(results: List[Dict]) -> Tuple[float, Dict[str, Optional[str]]]:
    score_maps = [result.get("judge", {}).get("individual_scores", {}) for result in results]
    total_pairs = len(results)

    model_names = sorted(
        {
            model
            for score_map in score_maps
            for model, score in score_map.items()
            if isinstance(score, (int, float))
        }
    )

    best_pair: Tuple[Optional[str], Optional[str]] = (None, None)
    best_overlap = 0

    for idx, rater_a in enumerate(model_names):
        for rater_b in model_names[idx + 1:]:
            overlap = sum(
                1
                for score_map in score_maps
                if isinstance(score_map.get(rater_a), (int, float))
                and isinstance(score_map.get(rater_b), (int, float))
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_pair = (rater_a, rater_b)

    rater_a, rater_b = best_pair
    paired_scores = []
    if rater_a and rater_b:
        for score_map in score_maps:
            score_a = score_map.get(rater_a)
            score_b = score_map.get(rater_b)
            if isinstance(score_a, (int, float)) and isinstance(score_b, (int, float)):
                paired_scores.append((int(score_a), int(score_b)))

    if len(paired_scores) < 2:
        return 0.0, {
            "rater_a": rater_a,
            "rater_b": rater_b,
            "valid_pairs": len(paired_scores),
            "total_pairs": total_pairs,
            "coverage_rate": round(len(paired_scores) / total_pairs, 4) if total_pairs else 0.0,
            "status": "insufficient_overlap",
        }

    observed_agreement = sum(1 for score_a, score_b in paired_scores if score_a == score_b) / len(paired_scores)
    categories = sorted({score for pair in paired_scores for score in pair}, key=str)
    counts_a = collections.Counter(score_a for score_a, _ in paired_scores)
    counts_b = collections.Counter(score_b for _, score_b in paired_scores)
    total_valid_pairs = len(paired_scores)
    expected_agreement = sum(
        (counts_a[category] / total_valid_pairs) * (counts_b[category] / total_valid_pairs)
        for category in categories
    )

    if expected_agreement >= 1.0:
        kappa = 1.0
    else:
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)

    return round(kappa, 4), {
        "rater_a": rater_a,
        "rater_b": rater_b,
        "valid_pairs": len(paired_scores),
        "total_pairs": total_pairs,
        "coverage_rate": round(len(paired_scores) / total_pairs, 4) if total_pairs else 0.0,
        "status": "ok",
    }


def build_summary(results: List[Dict], agent_version: str) -> Dict:
    total = len(results)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    if total == 0:
        return {
            "metadata": {"version": agent_version, "total": 0, "timestamp": timestamp},
            "metrics": {
                "avg_score": 0.0,
                "avg_faithfulness": 0.0,
                "avg_relevancy": 0.0,
                "hit_rate": 0.0,
                "avg_mrr": 0.0,
                "agreement_rate": 0.0,
                "conflict_rate": 0.0,
                "pass_rate": 0.0,
            },
        }

    avg_score = sum(r["judge"]["final_score"] for r in results) / total
    avg_faithfulness = sum(r["ragas"]["faithfulness"] for r in results) / total
    avg_relevancy = sum(r["ragas"]["relevancy"] for r in results) / total
    avg_hit_rate = sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total
    avg_mrr = sum(r["ragas"]["retrieval"]["mrr"] for r in results) / total
    agreement_rate, agreement_meta = calculate_cohens_kappa(results)
    conflict_rate = sum(1 for r in results if not r["judge"].get("consensus_reached", False)) / total
    pass_rate = sum(1 for r in results if r["status"] == "pass") / total

    return {
        "metadata": {
            "version": agent_version,
            "total": total,
            "timestamp": timestamp,
            "agreement_method": "cohen_kappa",
            "agreement_raters": agreement_meta,
        },
        "metrics": {
            "avg_score": round(avg_score, 4),
            "avg_faithfulness": round(avg_faithfulness, 4),
            "avg_relevancy": round(avg_relevancy, 4),
            "hit_rate": round(avg_hit_rate, 4),
            "avg_mrr": round(avg_mrr, 4),
            "agreement_rate": agreement_rate,
            "judge_coverage_rate": agreement_meta["coverage_rate"],
            "conflict_rate": round(conflict_rate, 4),
            "pass_rate": round(pass_rate, 4),
        },
    }


def add_regression_section(
    candidate_summary: Dict,
    baseline_summary: Optional[Dict],
    agent_version: str,
    baseline_version: str,
) -> Dict:
    summary = dict(candidate_summary)

    if baseline_summary is None:
        summary["metadata"]["baseline_version"] = None
        summary["regression"] = {
            "candidate_version": agent_version,
            "baseline_version": baseline_version,
            "delta_avg_score": None,
            "decision": "NO_BASELINE",
        }
        return summary

    delta = candidate_summary["metrics"]["avg_score"] - baseline_summary["metrics"]["avg_score"]
    summary["metadata"]["baseline_version"] = baseline_version
    summary["regression"] = {
        "candidate_version": agent_version,
        "baseline_version": baseline_version,
        "delta_avg_score": round(delta, 4),
        # Nếu V2 tốt hơn hoặc bằng V1 thì cho phép Release
        "decision": "APPROVE" if delta >= 0 else "BLOCK_RELEASE",
    }
    return summary

# 🚀 ĐÃ SỬA: Nhận tham số `agent_instance` để biết đang chạy V1 hay V2
async def run_benchmark_with_results(
    agent_instance, 
    agent_version: str,
    dataset: Optional[List[Dict]] = None,
    batch_size: Optional[int] = None,
) -> Tuple[List[Dict], Dict]:
    print(f"⏳ Bắt đầu thread Benchmark cho: {agent_version}")
    cases = dataset or load_dataset()
    effective_batch_size = batch_size or int(os.getenv("BATCH_SIZE", "5"))

    runner = BenchmarkRunner(
        agent_instance, # Truyền instance vào đây (MainAgent hoặc MainAgentV2)
        ExpertEvaluator(top_k=int(os.getenv("TOP_K", "5"))),
        LLMJudge(),
        agreement_threshold=float(os.getenv("JUDGE_AGREEMENT_THRESHOLD", "0.5")),
    )
    results = await runner.run_all(cases, batch_size=effective_batch_size)
    summary = build_summary(results, agent_version)
    print(f"✅ Đã hoàn thành Benchmark cho: {agent_version}")
    return results, summary


async def main():
    agent_version = os.getenv("AGENT_VERSION", "SupportAgent-v2")
    baseline_version = os.getenv("BASELINE_VERSION", "SupportAgent-v1")

    try:
        dataset = load_dataset()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Khong the chay benchmark: {exc}")
        return

    batch_size = int(os.getenv("BATCH_SIZE", "5"))
    print("\n⚡ ĐANG CHẠY BENCHMARK SONG SONG CHO CẢ V1 VÀ V2 ⚡\n")

    # 🚀 TỐI ƯU HÓA ASYNC: Khởi tạo 2 Agent và gom vào asyncio.gather để chạy cùng lúc
    agent_v1 = MainAgent()
    agent_v2 = MainAgentV2()

    task_v1 = run_benchmark_with_results(
        agent_instance=agent_v1,
        agent_version=baseline_version,
        dataset=dataset,
        batch_size=batch_size
    )
    
    task_v2 = run_benchmark_with_results(
        agent_instance=agent_v2,
        agent_version=agent_version,
        dataset=dataset,
        batch_size=batch_size
    )

    # Chờ cả 2 pipeline hoàn thành đồng thời
    (baseline_results, baseline_summary), (candidate_results, candidate_summary) = await asyncio.gather(task_v1, task_v2)

    # Tổng hợp báo cáo Release Gate
    final_summary = add_regression_section(
        candidate_summary=candidate_summary,
        baseline_summary=baseline_summary,
        agent_version=agent_version,
        baseline_version=baseline_version,
    )

    os.makedirs("reports", exist_ok=True)
    # Lưu file report
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results_v2.json", "w", encoding="utf-8") as f:
        json.dump(candidate_results, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results_v1.json", "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, ensure_ascii=False, indent=2)

    # In kết quả so sánh ra Terminal
    print("\n" + "="*50)
    print("📊 KẾT QUẢ SO SÁNH REGRESSION (V2 vs V1)")
    print("="*50)
    
    total_cases = final_summary["metadata"]["total"]
    metrics_v2 = final_summary["metrics"]
    metrics_v1 = baseline_summary["metrics"]

    print(f"📌 Tổng số Test Cases: {total_cases}")
    print("\n1️⃣ ĐÁNH GIÁ TỪ LLM JUDGE (Điểm tổng thể)")
    print(f"   - Score V2: {metrics_v2['avg_score']:.2f}")
    print(f"   - Score V1: {metrics_v1['avg_score']:.2f}")
    print(f"   - Mức độ đồng thuận của Judge: {metrics_v2['agreement_rate']:.2f} (Kappa)")
    
    print("\n2️⃣ RAGAS METRICS (Chất lượng nội dung)")
    print(f"   - Faithfulness V2: {metrics_v2['avg_faithfulness']:.2f}  |  V1: {metrics_v1['avg_faithfulness']:.2f}")
    print(f"   - Relevancy V2:    {metrics_v2['avg_relevancy']:.2f}  |  V1: {metrics_v1['avg_relevancy']:.2f}")
    
    print("\n3️⃣ RETRIEVAL METRICS (Chất lượng Vector DB)")
    print(f"   - Hit Rate V2:     {metrics_v2['hit_rate']:.4f}  |  V1: {metrics_v1['hit_rate']:.4f}")
    print(f"   - MRR V2:          {metrics_v2['avg_mrr']:.4f}  |  V1: {metrics_v1['avg_mrr']:.4f}")

    print("\n" + "-"*50)
    regression = final_summary.get("regression", {})
    decision = regression.get('decision', 'N/A')
    delta = regression.get('delta_avg_score', 0)
    
    if decision == "APPROVE":
        print(f"✅ QUYẾT ĐỊNH HỆ THỐNG: {decision}")
        print(f"🎉 Agent V2 hoạt động tốt hơn (hoặc bằng) V1 (+{delta:.2f} điểm). Cho phép Release lên Production!")
    else:
        print(f"❌ QUYẾT ĐỊNH HỆ THỐNG: {decision}")
        print(f"⚠️ Agent V2 gây lỗi hồi quy ({delta:.2f} điểm). Chặn Release, giữ lại V1!")
    print("-"*50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
