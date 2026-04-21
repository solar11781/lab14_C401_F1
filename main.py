import asyncio
import collections
import json
import os
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from agent.main_agent import MainAgent
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner

load_dotenv()


class ExpertEvaluator:
    def __init__(self, top_k: int = 5):
        self.retrieval_eval = RetrievalEvaluator(top_k=top_k)

    async def score(self, case: Dict, resp: Dict) -> Dict:
        expected_ids = case.get("ground_truth_ids", []) or case.get("expected_retrieval_ids", [])
        retrieved_ids = resp.get("retrieved_ids", [])
        hit_rate = self.retrieval_eval.calculate_hit_rate(expected_ids, retrieved_ids)
        mrr = self.retrieval_eval.calculate_mrr(expected_ids, retrieved_ids)

        return {
            "faithfulness": None,
            "relevancy": None,
            "retrieval": {
                "hit_rate": hit_rate,
                "mrr": mrr,
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids,
            },
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
                "hit_rate": 0.0,
                "avg_mrr": 0.0,
                "agreement_rate": 0.0,
                "conflict_rate": 0.0,
                "pass_rate": 0.0,
            },
        }

    avg_score = sum(r["judge"]["final_score"] for r in results) / total
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
        "decision": "APPROVE" if delta >= 0 else "BLOCK_RELEASE",
    }
    return summary


async def run_benchmark_with_results(
    agent_version: str,
    dataset: Optional[List[Dict]] = None,
    batch_size: Optional[int] = None,
) -> Tuple[List[Dict], Dict]:
    print(f"Khoi dong Benchmark cho {agent_version}...")
    cases = dataset or load_dataset()
    effective_batch_size = batch_size or int(os.getenv("BATCH_SIZE", "5"))

    runner = BenchmarkRunner(
        MainAgent(),
        ExpertEvaluator(top_k=int(os.getenv("TOP_K", "5"))),
        LLMJudge(),
        agreement_threshold=float(os.getenv("JUDGE_AGREEMENT_THRESHOLD", "0.5")),
    )
    results = await runner.run_all(cases, batch_size=effective_batch_size)
    summary = build_summary(results, agent_version)
    return results, summary


async def main():
    agent_version = os.getenv("AGENT_VERSION", "SupportAgent-v1")
    baseline_version = os.getenv("BASELINE_VERSION", "SupportAgent-v1")
    run_baseline = os.getenv("RUN_BASELINE_COMPARISON", "false").lower() == "true"

    try:
        dataset = load_dataset()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Khong the chay benchmark: {exc}")
        return

    baseline_summary = None
    if run_baseline:
        _, baseline_summary = await run_benchmark_with_results(
            baseline_version,
            dataset=dataset,
            batch_size=int(os.getenv("BATCH_SIZE", "5")),
        )

    candidate_results, candidate_summary = await run_benchmark_with_results(
        agent_version,
        dataset=dataset,
        batch_size=int(os.getenv("BATCH_SIZE", "5")),
    )

    final_summary = add_regression_section(
        candidate_summary=candidate_summary,
        baseline_summary=baseline_summary,
        agent_version=agent_version,
        baseline_version=baseline_version,
    )

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(candidate_results, f, ensure_ascii=False, indent=2)

    print("\nKet qua benchmark")
    print(f"Version: {final_summary['metadata']['version']}")
    print(f"So cases: {final_summary['metadata']['total']}")
    print(f"Avg score: {final_summary['metrics']['avg_score']}")
    print(f"Hit rate: {final_summary['metrics']['hit_rate']}")
    print(f"MRR: {final_summary['metrics']['avg_mrr']}")
    print(f"Agreement rate: {final_summary['metrics']['agreement_rate']}")
    print(f"Conflict rate: {final_summary['metrics']['conflict_rate']}")

    regression = final_summary.get("regression", {})
    if regression.get("decision"):
        print(f"Regression decision: {regression['decision']}")


if __name__ == "__main__":
    asyncio.run(main())
