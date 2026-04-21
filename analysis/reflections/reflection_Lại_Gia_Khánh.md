# Individual Report Lab Day 14

**Tên SV:** Lại Gia Khánh
**Vai trò:** Retrieval Eval (Nhóm Data)
**Ngày hoàn thành:** 21/04/2026

---

## 1. Engineering Contribution

### 1.1 Đóng góp cụ thể vào Project

#### **A. Tạo RetrievalEvaluator Class**
Tôi là người chịu trách nhiệm thiết kế và implement toàn bộ `engine/retrieval_eval.py` - module lõi để đánh giá chất lượng Retrieval stage trong AI Agent evaluation pipeline.

**Công việc cụ thể:**

1. **Implement Hit Rate Calculation (`calculate_hit_rate`)**
   - Xây dựng logic kiểm tra xem ít nhất 1 tài liệu ground truth có nằm trong top_k retrieved documents không
   - Support tham số `top_k` động để flexible với các kịch bản khác nhau
   - Xử lý edge cases: empty lists, duplicate IDs

2. **Implement Mean Reciprocal Rank (`calculate_mrr`)**
   - Tính toán vị trí (rank) của tài liệu liên quan đầu tiên
   - Công thức: MRR = 1 / position (1-indexed)
   - Return 0.0 khi không tìm thấy tài liệu liên quan

3. **Implement Batch Evaluation (`evaluate_batch`)**
   - Xử lý song song toàn bộ dataset (50+ test cases)
   - Aggregate metrics: avg_hit_rate, avg_mrr, hit_rate_percentage
   - Per-case tracking: lưu chi tiết từng case để debug
   - Validate dataset format: required fields handling, error resilience

---

### 1.2 Chứng minh qua Git Commits

```
- https://github.com/solar11781/lab14_C401_F1/commit/30757e25ac61a2b8d61bbfc2a6b2bd1b50c09702
- https://github.com/solar11781/lab14_C401_F1/commit/b0016cb2304bfa86dd3f1ce8dd9179f264bc9654 

```

### 1.3 Giải trình kỹ thuật

1. **Separate concerns**: `calculate_hit_rate()` và `calculate_mrr()` tách riêng để dễ unit test
2. **Aggregation pattern**: `evaluate_batch()` tính từng case rồi aggregate (chứ không tính trực tiếp trung bình)

---

## 2. Technical Depth

Các khái niệm cốt lõi về Retrieval Evaluation:

### 2.1 MRR (Mean Reciprocal Rank)

**Định nghĩa và ứng dụng:**
MRR đo lường thứ hạng (rank) của tài liệu liên quan đầu tiên trong danh sách retrieved documents. Công thức: `MRR = 1 / position` (với position 1-indexed).

**Tại sao quan trọng hơn Hit Rate:**
- Hit Rate chỉ cho biết "có hay không" (binary) mà không phân biệt tài liệu ở vị trí 1 hay vị trí 100
- MRR penalize khi tài liệu nằm xa top, vì LLM sẽ có xu hướng bỏ qua chunks ở cuối context window

---

### 2.2 Cohen's Kappa (Agreement Rate) cho Multi-Judge

**Định nghĩa:**
Khi sử dụng 2 LLM model (ví dụ GPT-4o + Claude) làm judge, Cohen's Kappa đo lường mức độ đồng thuận thực sự giữa chúng, loại trừ sự đồng ý do ngẫu nhiên.

**Công thức đơn giản (Agreement Rate):**
```
Agreement Rate = (Số case 2 judges cùng ý) / (Tổng cases)

```

**Tại sao cần tracking trong project:**
- Agreement Rate > 75% → Judges đáng tin cậy, merge scores bằng trung bình
- Agreement Rate < 60% → Judges không consistent, cần review rubric hoặc model

---

### 2.3 Position Bias trong Retrieval Ranking

**Định nghĩa:**
Position Bias xảy ra khi Vector DB ranking hoặc Judge evaluation bị ảnh hưởng bởi vị trí câu hỏi/response, thay vì chỉ dựa vào nội dung.

**Tại sao quan trọng:**
- Position bias → Kết quả retrieval không fair
- Một chunk liên quan vô tình bị rank thấp → Hallucination risk

---

### 2.4 Trade-off: Tốc độ Eval vs Chất lượng Retrieval

**Vấn đề:**
Retrieval Evaluation cần chạy nhanh nhưng cũng cần độ chính xác cao.

**Chi phí của evaluation:**

| Aspect | Cost | Impact |
|--------|------|--------|
| Top_k size | Lớn → Chậm | Kiểm tra nhiều docs → Chính xác hơn |
| Per-case analysis | Chi tiết → Chậm | Track từng chunk → Phát hiện bug dễ |
| Batching strategy | Song song → Nhanh | 5 cases/batch → 10x faster |

---

## 🔧 3. Problem Solving (10 điểm)

Trong quá trình tích hợp RetrievalEvaluator với ExpertEvaluator, tôi gặp một vấn đề thực tế: dù đã có cả retrieval metrics (Hit Rate, MRR) và generation metrics (faithfulness, relevancy), nhưng hệ thống vẫn chưa thể xác định rõ nguyên nhân của hallucination.

Cụ thể, trong implementation hiện tại, hàm score() đã thu thập đầy đủ tín hiệu từ cả hai phía:
- Retrieval quality: hit_rate, mrr, retrieved_ids
- Generation quality: faithfulness, relevancy

Tuy nhiên, các metric này đang được trả về dạng song song (parallel signals) mà chưa có bước tổng hợp để suy luận nguyên nhân. Tôi xác định root cause không nằm ở việc thiếu metric, mà là thiếu một abstraction layer để “diễn giải” các metric này thành insight.

Giải pháp tôi đề xuất là xây dựng một bước post-processing analysis ngay sau khi score() trả về kết quả, nhằm gom các tín hiệu thành các pattern có ý nghĩa. Điểm quan trọng trong cách tiếp cận này là: không cần thay đổi pipeline hiện tại, chỉ cần thêm một bước interpret layer. Điều này giúp tận dụng toàn bộ dữ liệu đã có từ retrieval_eval.py và ExpertEvaluator, đồng thời biến hệ thống evaluation từ “đo lường” sang “chẩn đoán”.

---
