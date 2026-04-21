import asyncio
import json
import os
import time
from engine.runner import BenchmarkRunner
from engine.retrieval_eval import RetrievalEvaluator
from agent.main_agent import MainAgent
from ragas.metrics import faithfulness, answer_relevancy

# Giả lập các components Expert
class ExpertEvaluator:
    def __init__(self, top_k: int = 5):
        self.retrieval_eval = RetrievalEvaluator(top_k=top_k)
    
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
        # Giả định 'case' chứa question và 'resp' chứa answer + list các text context
        row = {
            "question": case.get("question", ""),
            "answer": resp.get("answer", ""),
            "contexts": resp.get("retrieved_texts", []) # Ragas cần nội dung text, không phải ID
        }
        
        # 3. Tính toán Generation Metrics bằng Ragas (gọi LLM)
        # Sử dụng ascore() để chạy async, tránh block event loop
        faithfulness_score = await faithfulness.ascore(row)
        relevancy_score = await answer_relevancy.ascore(row)
        
        return {
            "faithfulness": faithfulness_score,
            "relevancy": relevancy_score,
            "retrieval": {
                "hit_rate": hit_rate, 
                "mrr": mrr
            }
        }

class MultiModelJudge:
    async def evaluate_multi_judge(self, q, a, gt): 
        return {
            "final_score": 4.5, 
            "agreement_rate": 0.8,
            "reasoning": "Cả 2 model đồng ý đây là câu trả lời tốt."
        }

async def run_benchmark_with_results(agent_version: str):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    runner = BenchmarkRunner(MainAgent(), ExpertEvaluator(), MultiModelJudge())
    results = await runner.run_all(dataset)

    total = len(results)
    summary = {
        "metadata": {"version": agent_version, "total": total, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total
        }
    }
    return results, summary

async def run_benchmark(version):
    _, summary = await run_benchmark_with_results(version)
    return summary

async def main():
    v1_summary = await run_benchmark("Agent_V1_Base")
    
    # Giả lập V2 có cải tiến (để test logic)
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")
    
    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    if delta > 0:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")

if __name__ == "__main__":
    asyncio.run(main())
