from typing import List, Dict, Optional
import asyncio

class RetrievalEvaluator:
    """
    Đánh giá chất lượng Retrieval stage bằng Hit Rate & MRR.
    Hit Rate: Xem ít nhất 1 tài liệu liên quan có nằm trong top_k không.
    MRR: Mean Reciprocal Rank - trung bình thứ hạng của tài liệu liên quan.
    """
    
    def __init__(self, top_k: int = 3):
        """
        Args:
            top_k: Số lượng tài liệu top để tính Hit Rate (mặc định 3)
        """
        self.top_k = top_k

    def calculate_hit_rate(
        self, 
        expected_ids: List[str], 
        retrieved_ids: List[str], 
        top_k: Optional[int] = None
    ) -> float:
        """
        Tính Hit Rate: Kiểm tra ít nhất 1 expected_id có nằm trong top_k retrieved_ids không.
        
        Args:
            expected_ids: Danh sách ID tài liệu ground truth
            retrieved_ids: Danh sách ID tài liệu được retrieve (đã sắp xếp theo độ liên quan)
            top_k: Số lượng top documents. Nếu None, dùng self.top_k
            
        Returns:
            1.0 nếu có hit, 0.0 nếu không
            
        Example:
            >>> expected_ids = ["doc_1", "doc_5"]
            >>> retrieved_ids = ["doc_3", "doc_1", "doc_7", "doc_2"]
            >>> evaluator.calculate_hit_rate(expected_ids, retrieved_ids, top_k=2)
            1.0  # Vì doc_1 nằm ở vị trí 2, trong top_k=2
        """
        k = top_k if top_k is not None else self.top_k
        top_retrieved = retrieved_ids[:k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(
        self, 
        expected_ids: List[str], 
        retrieved_ids: List[str]
    ) -> float:
        """
        Tính Mean Reciprocal Rank: 1 / (vị trí đầu tiên của expected_id).
        
        Args:
            expected_ids: Danh sách ID tài liệu ground truth
            retrieved_ids: Danh sách ID tài liệu được retrieve (đã sắp xếp theo độ liên quan)
            
        Returns:
            MRR (0.0 - 1.0). Càng cao càng tốt. 
            - 1.0: Tài liệu liên quan ở vị trí 1 (hoàn hảo)
            - 0.5: Tài liệu liên quan ở vị trí 2
            - 0.33: Tài liệu liên quan ở vị trí 3
            - 0.0: Không tìm thấy tài liệu liên quan
            
        Example:
            >>> expected_ids = ["doc_1", "doc_5"]
            >>> retrieved_ids = ["doc_3", "doc_1", "doc_7", "doc_5"]
            >>> evaluator.calculate_mrr(expected_ids, retrieved_ids)
            0.5  # Vì doc_1 ở vị trí 2 (1-indexed), MRR = 1/2 = 0.5
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)  # i+1 vì vị trí 1-indexed
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict]) -> Dict:
        """
        Chạy eval cho toàn bộ bộ dữ liệu.
        
        Args:
            dataset: Danh sách các test case. Mỗi case cần có:
                - expected_retrieval_ids: List[str] - IDs của tài liệu ground truth
                - retrieved_ids: List[str] - IDs của tài liệu được retrieve (của Agent)
                - question (optional): Câu hỏi, để debug
                
        Returns:
            Dict chứa:
                - avg_hit_rate: Trung bình Hit Rate trên toàn bộ dataset
                - avg_mrr: Trung bình MRR trên toàn bộ dataset
                - total_cases: Tổng số cases đã eval
                - cases_with_hit: Số cases có ít nhất 1 hit
                - per_case_metrics: Chi tiết từng case (tuỳ chọn)
                - details: Phân tích chi tiết
        """
        if not dataset:
            return {
                "avg_hit_rate": 0.0,
                "avg_mrr": 0.0,
                "total_cases": 0,
                "cases_with_hit": 0,
                "details": "Dataset rỗng"
            }

        hit_rates = []
        mrrs = []
        cases_with_hit = 0
        per_case_metrics = []

        for idx, case in enumerate(dataset):
            # Validate required fields
            expected_ids = case.get("expected_retrieval_ids", [])
            retrieved_ids = case.get("retrieved_ids", [])
            
            if not expected_ids:
                # Nếu không có expected_ids, bỏ qua case này
                continue
            
            # Tính Hit Rate và MRR
            hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids)
            mrr = self.calculate_mrr(expected_ids, retrieved_ids)
            
            hit_rates.append(hit_rate)
            mrrs.append(mrr)
            
            if hit_rate > 0:
                cases_with_hit += 1
            
            # Lưu chi tiết từng case
            case_metric = {
                "case_idx": idx,
                "question": case.get("question", "N/A"),
                "hit_rate": hit_rate,
                "mrr": mrr,
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids[:self.top_k],  # Chỉ lưu top_k
                "status": "hit" if hit_rate > 0 else "miss"
            }
            per_case_metrics.append(case_metric)

        # Tính trung bình
        avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
        avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0

        total_cases = len(hit_rates)
        hit_rate_percentage = (cases_with_hit / total_cases * 100) if total_cases > 0 else 0.0

        return {
            "avg_hit_rate": round(avg_hit_rate, 4),
            "avg_mrr": round(avg_mrr, 4),
            "total_cases": total_cases,
            "cases_with_hit": cases_with_hit,
            "hit_rate_percentage": round(hit_rate_percentage, 2),
            "per_case_metrics": per_case_metrics,
            "details": {
                "top_k": self.top_k,
                "description": f"Đánh giá Retrieval trên {total_cases} cases. "
                               f"Hit Rate: {hit_rate_percentage:.1f}% (avg MRR: {avg_mrr:.4f})"
            }
        }
