import asyncio
import collections
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

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
        expected_ids = case.get("ground_truth_ids", [])
        retrieved_ids = resp.get("retrieved_ids", [])
        
        hit_rate = self.retrieval_eval.calculate_hit_rate(expected_ids, retrieved_ids)
        mrr = self.retrieval_eval.calculate_mrr(expected_ids, retrieved_ids)
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
                "mrr": mrr,
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids[: self.retrieval_eval.top_k],
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


def _metric_mean(results: List[Dict], getter) -> float:
    values = []
    for result in results:
        try:
            values.append(float(getter(result)))
        except Exception:
            continue
    return sum(values) / len(values) if values else 0.0


def build_summary(
    results: List[Dict],
    agent_version: str,
    total_runtime_seconds: Optional[float] = None,
) -> Dict:
    total = len(results)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    if total == 0:
        return {
            "metadata": {
                "version": agent_version,
                "total": 0,
                "timestamp": timestamp,
                "agreement_method": "cohen_kappa",
                "agreement_raters": {
                    "rater_a": None,
                    "rater_b": None,
                    "valid_pairs": 0,
                    "total_pairs": 0,
                    "coverage_rate": 0.0,
                    "status": "insufficient_overlap",
                },
            },
            "metrics": {
                "avg_score": 0.0,
                "avg_faithfulness": 0.0,
                "avg_relevancy": 0.0,
                "hit_rate": 0.0,
                "avg_mrr": 0.0,
                "agreement_rate": 0.0,
                "judge_coverage_rate": 0.0,
                "conflict_rate": 0.0,
                "pass_rate": 0.0,
                "avg_latency": 0.0,
                "p95_latency": 0.0,
                "total_runtime_seconds": round(total_runtime_seconds or 0.0, 4),
                "throughput_cases_per_min": 0.0,
                "total_tokens": 0,
                "avg_tokens_per_case": 0.0,
            },
        }

    avg_score = _metric_mean(results, lambda r: r["judge"]["final_score"])
    avg_faithfulness = _metric_mean(results, lambda r: r["ragas"]["faithfulness"])
    avg_relevancy = _metric_mean(results, lambda r: r["ragas"]["relevancy"])
    avg_hit_rate = _metric_mean(results, lambda r: r["ragas"]["retrieval"]["hit_rate"])
    avg_mrr = _metric_mean(results, lambda r: r["ragas"]["retrieval"]["mrr"])

    agreement_rate, agreement_meta = calculate_cohens_kappa(results)
    conflict_rate = sum(1 for r in results if not r["judge"].get("consensus_reached", False)) / total
    pass_rate = sum(1 for r in results if r["status"] == "pass") / total

    latencies = [float(r.get("latency", 0.0) or 0.0) for r in results]
    avg_latency = sum(latencies) / total if latencies else 0.0

    sorted_latencies = sorted(latencies)
    p95_index = int((len(sorted_latencies) - 1) * 0.95) if sorted_latencies else 0
    p95_latency = sorted_latencies[p95_index] if sorted_latencies else 0.0

    total_tokens = sum(
        int((r.get("agent_metadata") or {}).get("tokens_used", 0) or 0)
        for r in results
    )
    avg_tokens_per_case = total_tokens / total if total > 0 else 0.0
    
    runtime_seconds = float(total_runtime_seconds or 0.0)
    throughput_cases_per_min = (total / runtime_seconds) * 60 if runtime_seconds > 0 else 0.0

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
            "agreement_rate": round(agreement_rate, 4),
            "judge_coverage_rate": round(agreement_meta["coverage_rate"], 4),
            "conflict_rate": round(conflict_rate, 4),
            "pass_rate": round(pass_rate, 4),
            "avg_latency": round(avg_latency, 4),
            "p95_latency": round(p95_latency, 4),
            "total_runtime_seconds": round(runtime_seconds, 4),
            "throughput_cases_per_min": round(throughput_cases_per_min, 4),
            "total_tokens": total_tokens,
            "avg_tokens_per_case": round(avg_tokens_per_case, 4),
        },
    }


def derive_default_gate_thresholds(baseline_summary: Dict) -> Dict[str, float]:
    metrics = baseline_summary["metrics"]
    total_cases = max(int(baseline_summary["metadata"].get("total", 0) or 0), 1)
    one_case_step = 1 / total_cases

    return {
        # QUALITY / RELIABILITY
        "min_avg_score": round(max(0.0, metrics["avg_score"] - 0.05), 4),
        "min_delta_avg_score": -0.05,
        "min_avg_faithfulness": round(max(0.10, metrics["avg_faithfulness"] - 0.02), 4),
        "min_delta_avg_faithfulness": -0.02,
        "min_avg_relevancy": round(max(0.15, metrics["avg_relevancy"] - 0.05), 4),
        "min_delta_avg_relevancy": -0.05,
        "min_pass_rate": round(max(0.0, metrics["pass_rate"] - (2 * one_case_step)), 4),
        "min_delta_pass_rate": round(-(2 * one_case_step), 4),
        "min_agreement_rate": round(max(0.50, metrics["agreement_rate"] - 0.10), 4),
        "min_delta_agreement_rate": -0.10,
        "max_conflict_rate": round(min(1.0, metrics["conflict_rate"] + one_case_step), 4),
        "max_delta_conflict_rate": round(one_case_step, 4),

        # RETRIEVAL
        "min_hit_rate": round(max(0.0, metrics["hit_rate"] - one_case_step), 4),
        "min_delta_hit_rate": round(-one_case_step, 4),
        "min_avg_mrr": round(max(0.0, metrics["avg_mrr"] - 0.01), 4),
        "min_delta_avg_mrr": -0.01,

        # PERFORMANCE / COST
        "max_avg_latency": round(metrics["avg_latency"] * 1.10, 4),
        "max_p95_latency": round(metrics["p95_latency"] * 1.15, 4),
        "max_total_runtime_seconds": round(min(120.0, max(metrics.get("total_runtime_seconds", 0.0) * 1.10, 120.0)), 4),
        "min_throughput_cases_per_min": round(max(metrics.get("throughput_cases_per_min", 0.0) * 0.90, 0.0), 4),
        "max_total_tokens": round(metrics["total_tokens"] * 1.10, 4),
        "max_avg_tokens_per_case": round(metrics["avg_tokens_per_case"] * 1.10, 4),
    }


def load_gate_thresholds(baseline_summary: Optional[Dict]) -> Dict[str, Optional[float]]:
    defaults: Dict[str, Optional[float]] = {}
    if baseline_summary is not None:
        defaults = derive_default_gate_thresholds(baseline_summary)

    def env_or_default(name: str, default: Optional[float]) -> Optional[float]:
        raw = os.getenv(name)
        if raw is None or raw == "":
            return default
        return float(raw)

    keys = [
        "min_avg_score",
        "min_delta_avg_score",
        "min_avg_faithfulness",
        "min_delta_avg_faithfulness",
        "min_avg_relevancy",
        "min_delta_avg_relevancy",
        "min_hit_rate",
        "min_delta_hit_rate",
        "min_avg_mrr",
        "min_delta_avg_mrr",
        "min_pass_rate",
        "min_delta_pass_rate",
        "min_agreement_rate",
        "min_delta_agreement_rate",
        "max_conflict_rate",
        "max_delta_conflict_rate",
        "max_avg_latency",
        "max_p95_latency",
        "max_total_runtime_seconds",
        "min_throughput_cases_per_min",
        "max_total_tokens",
        "max_avg_tokens_per_case",
    ]

    env_map = {
        "min_avg_score": "GATE_MIN_AVG_SCORE",
        "min_delta_avg_score": "GATE_MIN_DELTA_AVG_SCORE",
        "min_avg_faithfulness": "GATE_MIN_AVG_FAITHFULNESS",
        "min_delta_avg_faithfulness": "GATE_MIN_DELTA_AVG_FAITHFULNESS",
        "min_avg_relevancy": "GATE_MIN_AVG_RELEVANCY",
        "min_delta_avg_relevancy": "GATE_MIN_DELTA_AVG_RELEVANCY",
        "min_hit_rate": "GATE_MIN_HIT_RATE",
        "min_delta_hit_rate": "GATE_MIN_DELTA_HIT_RATE",
        "min_avg_mrr": "GATE_MIN_AVG_MRR",
        "min_delta_avg_mrr": "GATE_MIN_DELTA_AVG_MRR",
        "min_pass_rate": "GATE_MIN_PASS_RATE",
        "min_delta_pass_rate": "GATE_MIN_DELTA_PASS_RATE",
        "min_agreement_rate": "GATE_MIN_AGREEMENT_RATE",
        "min_delta_agreement_rate": "GATE_MIN_DELTA_AGREEMENT_RATE",
        "max_conflict_rate": "GATE_MAX_CONFLICT_RATE",
        "max_delta_conflict_rate": "GATE_MAX_DELTA_CONFLICT_RATE",
        "max_avg_latency": "GATE_MAX_AVG_LATENCY",
        "max_p95_latency": "GATE_MAX_P95_LATENCY",
        "max_total_runtime_seconds": "GATE_MAX_TOTAL_RUNTIME_SECONDS",
        "min_throughput_cases_per_min": "GATE_MIN_THROUGHPUT_CASES_PER_MIN",
        "max_total_tokens": "GATE_MAX_TOTAL_TOKENS",
        "max_avg_tokens_per_case": "GATE_MAX_AVG_TOKENS_PER_CASE",
    }

    return {
        key: env_or_default(env_map[key], defaults.get(key))
        for key in keys
    }


def add_regression_section(
    candidate_summary: Dict,
    baseline_summary: Optional[Dict],
    agent_version: str,
    baseline_version: str,
) -> Dict:
    summary = dict(candidate_summary)
    summary["metadata"]["baseline_version"] = baseline_version if baseline_summary else None

    if baseline_summary is None:
        summary["regression"] = {
            "candidate_version": agent_version,
            "baseline_version": baseline_version,
            "decision": "NO_BASELINE",
            "reason": "Khong tim thay baseline_summary hoac baseline run.",
            "thresholds": load_gate_thresholds(None),
            "policy": {
                "name": "baseline_relative_release_gate_v1",
                "dataset_case_step": None,
                "notes": [
                    "Can baseline summary de tinh threshold mac dinh.",
                    "Co the override bang bien moi truong GATE_*.",
                ],
            },
            "checks": [],
            "check_summary": {"passed": 0, "failed": 0},
            "deltas": {},
            "delta_avg_score": None,
            "tradeoff_summary": "missing_baseline",
        }
        return summary

    candidate_metrics = candidate_summary["metrics"]
    baseline_metrics = baseline_summary["metrics"]
    total_cases = max(int(candidate_summary["metadata"].get("total", 0) or 0), 1)
    one_case_step = round(1 / total_cases, 4)

    thresholds = load_gate_thresholds(baseline_summary)

    deltas = {
        "avg_score": round(candidate_metrics["avg_score"] - baseline_metrics["avg_score"], 4),
        "avg_faithfulness": round(candidate_metrics["avg_faithfulness"] - baseline_metrics["avg_faithfulness"], 4),
        "avg_relevancy": round(candidate_metrics["avg_relevancy"] - baseline_metrics["avg_relevancy"], 4),
        "hit_rate": round(candidate_metrics["hit_rate"] - baseline_metrics["hit_rate"], 4),
        "avg_mrr": round(candidate_metrics["avg_mrr"] - baseline_metrics["avg_mrr"], 4),
        "agreement_rate": round(candidate_metrics["agreement_rate"] - baseline_metrics["agreement_rate"], 4),
        "conflict_rate": round(candidate_metrics["conflict_rate"] - baseline_metrics["conflict_rate"], 4),
        "pass_rate": round(candidate_metrics["pass_rate"] - baseline_metrics["pass_rate"], 4),
        "avg_latency": round(candidate_metrics["avg_latency"] - baseline_metrics["avg_latency"], 4),
        "p95_latency": round(candidate_metrics["p95_latency"] - baseline_metrics["p95_latency"], 4),
        "total_runtime_seconds": round(candidate_metrics["total_runtime_seconds"] - baseline_metrics["total_runtime_seconds"], 4),
        "throughput_cases_per_min": round(candidate_metrics["throughput_cases_per_min"] - baseline_metrics["throughput_cases_per_min"], 4),
        "total_tokens": round(candidate_metrics["total_tokens"] - baseline_metrics["total_tokens"], 4),
        "avg_tokens_per_case": round(candidate_metrics["avg_tokens_per_case"] - baseline_metrics["avg_tokens_per_case"], 4),
    }

    checks = []

    def add_min_check(name: str, actual: float, threshold: Optional[float], category: str):
        if threshold is None:
            return
        checks.append({
            "name": name,
            "category": category,
            "operator": ">=",
            "actual": round(actual, 4),
            "threshold": round(float(threshold), 4),
            "passed": actual >= threshold,
        })

    def add_max_check(name: str, actual: float, threshold: Optional[float], category: str):
        if threshold is None:
            return
        checks.append({
            "name": name,
            "category": category,
            "operator": "<=",
            "actual": round(actual, 4),
            "threshold": round(float(threshold), 4),
            "passed": actual <= threshold,
        })

    # QUALITY
    add_min_check("avg_score", candidate_metrics["avg_score"], thresholds["min_avg_score"], "quality")
    add_min_check("delta_avg_score", deltas["avg_score"], thresholds["min_delta_avg_score"], "quality")
    add_min_check("avg_faithfulness", candidate_metrics["avg_faithfulness"], thresholds["min_avg_faithfulness"], "quality")
    add_min_check("delta_avg_faithfulness", deltas["avg_faithfulness"], thresholds["min_delta_avg_faithfulness"], "quality")
    add_min_check("avg_relevancy", candidate_metrics["avg_relevancy"], thresholds["min_avg_relevancy"], "quality")
    add_min_check("delta_avg_relevancy", deltas["avg_relevancy"], thresholds["min_delta_avg_relevancy"], "quality")
    add_min_check("pass_rate", candidate_metrics["pass_rate"], thresholds["min_pass_rate"], "quality")
    add_min_check("delta_pass_rate", deltas["pass_rate"], thresholds["min_delta_pass_rate"], "quality")

    # RETRIEVAL
    add_min_check("hit_rate", candidate_metrics["hit_rate"], thresholds["min_hit_rate"], "retrieval")
    add_min_check("delta_hit_rate", deltas["hit_rate"], thresholds["min_delta_hit_rate"], "retrieval")
    add_min_check("avg_mrr", candidate_metrics["avg_mrr"], thresholds["min_avg_mrr"], "retrieval")
    add_min_check("delta_avg_mrr", deltas["avg_mrr"], thresholds["min_delta_avg_mrr"], "retrieval")

    # RELIABILITY
    add_min_check("agreement_rate", candidate_metrics["agreement_rate"], thresholds["min_agreement_rate"], "reliability")
    add_min_check("delta_agreement_rate", deltas["agreement_rate"], thresholds["min_delta_agreement_rate"], "reliability")
    add_max_check("conflict_rate", candidate_metrics["conflict_rate"], thresholds["max_conflict_rate"], "reliability")
    add_max_check("delta_conflict_rate", deltas["conflict_rate"], thresholds["max_delta_conflict_rate"], "reliability")

    # PERFORMANCE / COST
    add_max_check("avg_latency", candidate_metrics["avg_latency"], thresholds["max_avg_latency"], "performance")
    add_max_check("p95_latency", candidate_metrics["p95_latency"], thresholds["max_p95_latency"], "performance")
    add_max_check("total_runtime_seconds", candidate_metrics["total_runtime_seconds"], thresholds["max_total_runtime_seconds"], "performance")
    add_min_check("throughput_cases_per_min", candidate_metrics["throughput_cases_per_min"], thresholds["min_throughput_cases_per_min"], "performance")
    add_max_check("total_tokens", candidate_metrics["total_tokens"], thresholds["max_total_tokens"], "cost")
    add_max_check("avg_tokens_per_case", candidate_metrics["avg_tokens_per_case"], thresholds["max_avg_tokens_per_case"], "cost")

    decision = "APPROVE" if all(check["passed"] for check in checks) else "BLOCK_RELEASE"

    failed_checks = [check["name"] for check in checks if not check["passed"]]
    passed_checks = [check["name"] for check in checks if check["passed"]]

    quality_regressed = any(
        name in failed_checks
        for name in [
            "avg_score",
            "delta_avg_score",
            "avg_faithfulness",
            "delta_avg_faithfulness",
            "avg_relevancy",
            "delta_avg_relevancy",
            "pass_rate",
            "delta_pass_rate",
            "hit_rate",
            "delta_hit_rate",
            "avg_mrr",
            "delta_avg_mrr",
            "agreement_rate",
            "delta_agreement_rate",
            "conflict_rate",
            "delta_conflict_rate",
        ]
    )
    perf_or_cost_improved = (
        deltas["avg_latency"] < 0
        or deltas["p95_latency"] < 0
        or deltas["total_runtime_seconds"] < 0
        or deltas["total_tokens"] < 0
        or deltas["avg_tokens_per_case"] < 0
    )

    if quality_regressed and perf_or_cost_improved:
        tradeoff_summary = "candidate_faster_or_cheaper_but_quality_or_reliability_regressed"
    elif quality_regressed:
        tradeoff_summary = "candidate_regressed_on_quality_or_reliability"
    elif perf_or_cost_improved:
        tradeoff_summary = "candidate_improved_without_quality_regression"
    else:
        tradeoff_summary = "candidate_mixed_changes_without_clear_perf_gain"

    summary["regression"] = {
        "candidate_version": agent_version,
        "baseline_version": baseline_version,
        "decision": decision,
        "reason": "all_checks_passed" if decision == "APPROVE" else "one_or_more_checks_failed",
        "thresholds": thresholds,
        "policy": {
            "name": "baseline_relative_release_gate_v1",
            "dataset_case_step": one_case_step,
            "notes": [
                "Quality duoc bao ve bang nguong non-regression chat.",
                "Hit-rate/conflict-rate su dung tolerance theo buoc 1 case vi dataset hien tai co kich thuoc huu han.",
                "Performance va token duoc phep xau di trong bien an toan, neu khong anh huong chat luong.",
                "Tat ca threshold co the override bang bien moi truong GATE_*.",
            ],
        },
        "checks": checks,
        "check_summary": {
            "passed": len(passed_checks),
            "failed": len(failed_checks),
            "failed_checks": failed_checks,
        },
        "deltas": deltas,
        "delta_avg_score": deltas["avg_score"],
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "tradeoff_summary": tradeoff_summary,
    }
    return summary

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

    started_at = time.perf_counter()
    results = await runner.run_all(cases, batch_size=effective_batch_size)
    total_runtime_seconds = time.perf_counter() - started_at

    summary = build_summary(
        results,
        agent_version,
        total_runtime_seconds=total_runtime_seconds,
    )
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
        batch_size=batch_size,
    )
    
    task_v2 = run_benchmark_with_results(
        agent_instance=agent_v2,
        agent_version=agent_version,
        dataset=dataset,
        batch_size=batch_size,
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
    # 1) File chính để nộp
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(candidate_results, f, ensure_ascii=False, indent=2)

    # 2) File phụ để lưu riêng kết quả regression V1/V2
    with open("reports/benchmark_results_v2.json", "w", encoding="utf-8") as f:
        json.dump(candidate_results, f, ensure_ascii=False, indent=2)

    with open("reports/benchmark_results_v1.json", "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, ensure_ascii=False, indent=2)

    # In kết quả so sánh ra Terminal
    print("\n" + "=" * 60)
    print("📊 KẾT QUẢ SO SÁNH REGRESSION (V2 vs V1)")
    print("=" * 60)

    total_cases = final_summary["metadata"]["total"]
    metrics_v2 = final_summary["metrics"]
    metrics_v1 = baseline_summary["metrics"]
    regression = final_summary.get("regression", {})
    decision = regression.get("decision", "N/A")
    delta = regression.get("deltas", {}).get("avg_score", regression.get("delta_avg_score", 0))

    print(f"📌 Tổng số Test Cases: {total_cases}")
    print(f"📌 Trade-off Summary: {regression.get('tradeoff_summary', 'N/A')}")

    print("\n1️⃣ QUALITY")
    print(f"   - Avg Score V2:          {metrics_v2['avg_score']:.4f} | V1: {metrics_v1['avg_score']:.4f}")
    print(f"   - Faithfulness V2:       {metrics_v2['avg_faithfulness']:.4f} | V1: {metrics_v1['avg_faithfulness']:.4f}")
    print(f"   - Relevancy V2:          {metrics_v2['avg_relevancy']:.4f} | V1: {metrics_v1['avg_relevancy']:.4f}")
    print(f"   - Pass Rate V2:          {metrics_v2['pass_rate']:.4f} | V1: {metrics_v1['pass_rate']:.4f}")

    print("\n2️⃣ RETRIEVAL")
    print(f"   - Hit Rate V2:           {metrics_v2['hit_rate']:.4f} | V1: {metrics_v1['hit_rate']:.4f}")
    print(f"   - Avg MRR V2:            {metrics_v2['avg_mrr']:.4f} | V1: {metrics_v1['avg_mrr']:.4f}")

    print("\n3️⃣ RELIABILITY")
    print(f"   - Agreement (Kappa) V2:  {metrics_v2['agreement_rate']:.4f} | V1: {metrics_v1['agreement_rate']:.4f}")
    print(f"   - Conflict Rate V2:      {metrics_v2['conflict_rate']:.4f} | V1: {metrics_v1['conflict_rate']:.4f}")

    print("\n4️⃣ PERFORMANCE / COST")
    print(f"   - Avg Latency V2:        {metrics_v2['avg_latency']:.4f}s | V1: {metrics_v1['avg_latency']:.4f}s")
    print(f"   - P95 Latency V2:        {metrics_v2['p95_latency']:.4f}s | V1: {metrics_v1['p95_latency']:.4f}s")
    print(f"   - Total Runtime V2:      {metrics_v2['total_runtime_seconds']:.4f}s | V1: {metrics_v1['total_runtime_seconds']:.4f}s")
    print(f"   - Throughput V2:         {metrics_v2['throughput_cases_per_min']:.2f} cases/min | V1: {metrics_v1['throughput_cases_per_min']:.2f} cases/min")
    print(f"   - Total Tokens V2:       {metrics_v2['total_tokens']} | V1: {metrics_v1['total_tokens']}")
    print(f"   - Avg Tokens/Case V2:    {metrics_v2['avg_tokens_per_case']:.4f} | V1: {metrics_v1['avg_tokens_per_case']:.4f}")

    print("\n5️⃣ RELEASE GATE")
    print(f"   - Decision:              {decision}")
    print(f"   - Delta Avg Score:       {delta:+.4f}")
    check_summary = regression.get("check_summary", {})
    print(f"   - Checks Passed:         {check_summary.get('passed', 0)}")
    print(f"   - Checks Failed:         {check_summary.get('failed', 0)}")

    for check in regression.get("checks", []):
        status = "PASS" if check["passed"] else "FAIL"
        print(
            f"   - [{status}] ({check['category']}) {check['name']} "
            f"{check['operator']} {check['threshold']} (actual={check['actual']})"
        )

    print("-" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
