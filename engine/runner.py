import asyncio
import os
import time
from typing import List, Dict


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge, agreement_threshold: float = 0.5):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        self.agreement_threshold = agreement_threshold
        self.tie_breaker_model = os.getenv("JUDGE_TIEBREAKER_MODEL", "gpt-4o-mini")

    async def _evaluate_with_retry_and_tiebreak(
        self,
        test_case: Dict,
        response: Dict,
    ) -> Dict:
        retrieved_context = response.get("retrieved_context") or response.get("contexts", [])

        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case["expected_answer"],
            retrieved_context=retrieved_context,
        )

        judge_result.setdefault("notes", [])

        if judge_result.get("agreement_rate", 0.0) < self.agreement_threshold:
            judge_result["notes"].append(
                f"Low agreement detected (< {self.agreement_threshold}). Invoking tie-breaker judge."
            )
            tie_break_result = await self.judge.evaluate_multi_judge(
                test_case["question"],
                response["answer"],
                test_case["expected_answer"],
                retrieved_context=retrieved_context,
                extra_models=[self.tie_breaker_model],
            )
            tie_break_result.setdefault("notes", [])
            tie_break_result["notes"] = judge_result["notes"] + [
                f"Tie-breaker model used: {self.tie_breaker_model}"
            ]
            return tie_break_result

        return judge_result

    async def run_single_test(self, test_case: Dict) -> Dict:
        question = test_case.get("question")
        expected_answer = test_case.get("expected_answer", "")
        if not question:
            raise ValueError(f"Invalid test case format: missing 'question'. Keys={sorted(test_case.keys())}")

        start_time = time.perf_counter()
        
        # 1. Gọi Agent
        response = await self.agent.query(question)
        latency = time.perf_counter() - start_time
        
        # 2. Chạy RAGAS metrics
        ragas_scores = await self.evaluator.score(test_case, response)
        
        # 3. Chạy Multi-Judge
        judge_result = await self._evaluate_with_retry_and_tiebreak(test_case, response)
        
        return {
            "test_case": question,
            "agent_response": response["answer"],
            "retrieved_ids": response.get("retrieved_ids", []),
            "retrieved_context": response.get("retrieved_context") or response.get("contexts", []),
            "sources": response.get("sources", []),
            "agent_metadata": response.get("metadata", {}),
            "latency": latency,
            "ragas": ragas_scores,
            "judge": judge_result,
            "status": "fail" if judge_result["final_score"] < 3 else "pass"
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        """
        Chạy song song bằng asyncio.gather với giới hạn batch_size để không bị Rate Limit.
        """
        effective_batch_size = batch_size or int(os.getenv("BATCH_SIZE", "5"))
        results = []
        for i in range(0, len(dataset), effective_batch_size):
            batch = dataset[i:i + effective_batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results
