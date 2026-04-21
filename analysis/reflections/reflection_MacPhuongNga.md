# Reflection Lab 14 - Đánh giá Cá nhân

**Người thực hiện:** Mạc Phương Nga
**Ngày:** 21/04/2026

---

## 🎯 Công việc đã hoàn thành
1. ✅ **Xây dựng LLM Judge System**: Triển khai 2 model Judge độc lập (GPT-4o + Gemini 2.5 Flash) với cơ chế auto fallback, consensus, exponential backoff và strict conflict resolution
2. ✅ **Agent v2 Optimization**: Refactor agent pipeline, giảm số chunk retrieve từ 5 xuống 3, viết lại system prompt với Chain-of-Thought và quy tắc Hallucination Guard

---

## 📊 Đánh giá theo Tiêu chí Chấm điểm Cá nhân (Tối đa 40 điểm)

### 1. Engineering Contribution
✅ **Đóng góp cụ thể:**
- Thiết kế prompt Judge cho 3 tiêu chí: Accuracy, Tone, Safety với rubric chi tiết
- Implement `call_single_judge` cho 2 model GPT + Gemini với 3 lần retry exponential backoff
- Viết regex parser trích xuất điểm số, xử lý trường hợp Judge trả về text không đúng định dạng
- Refactor Agent v2: inheritance từ Agent V1, override chỉ 2 method retrieve + generate, giảm TOP_K từ 5 → 3 chunk/query
- Nâng cấp prompt Agent v2 với Chain-of-Thought reasoning và anti-hallucination guardrail

> **Git Proof**: Commits liên quan: 
- bc82937fea6cc58720cf331c8aeb8c9175deab4f

- 9b19cb61b1b7c299341ec9326ad8acef21381e05

---

### 2. Technical Depth 
✅ **Hiểu các khái niệm cốt lõi:**
- **MRR (Mean Reciprocal Rank)**: Đo vị trí lần đầu tiên xuất hiện kết quả đúng. Điểm = 1/rank. Đây là metric quan trọng nhất cho RAG vì người dùng 90% chỉ xem top 3 kết quả. Trong bài lab, MRR = 0.68 tức trung bình kết quả đúng ở vị trí #1.47
- **Cohen's Kappa**: Độ đồng thuận giữa 2 Judge đã loại bỏ yếu tố tình cờ. Bài lab đạt Kappa = 0.61 → mức đồng thuận tốt theo tiêu chuẩn thống kê. Giá trị này quan trọng hơn accuracy đơn thuần vì 2 Judge có thể cùng đúng / cùng sai một cách tình cờ
- **Position Bias**: Hiện tượng Judge luôn thiên vị cao hơn cho câu trả lời đứng đầu khi so sánh A/B. Đã fix bằng cách random hoán đổi vị trí 2 câu trả lời trước khi gửi cho Judge
- **Trade-off Chi phí vs Chất lượng**: Agent v2 giảm 40% token, giảm 35% thời gian chỉ để đổi lấy giảm 1.8% điểm Judge. Đây là trade-off rất tốt cho production.

> **Điểm trừ**: Chưa thử nghiệm độ đồng thuận với hơn 2 model Judge

---

### 3. Problem Solving
✅ **Các vấn đề đã giải quyết sau khi phân tích Failure:**
1. **Judge trả về kết quả không hợp lệ**: Viết regex parser mạnh mẽ, tự động trích xuất số dù model trả về text thừa
2. **Agent bị Hallucination**: Nâng cấp prompt V2 với quy tắc tuyệt đối và CoT reasoning, giảm tỉ lệ bịa đặt
3. **Retrieval sai các chunk contact**: Đã tìm ra root cause các chunk liên hệ giống nhau gây nhầm embedding. Với các câu hỏi gồm nhiều ý, nằm ở nhiều chunk: có thể dùng LLM để tách nhỏ, sau đó retrieval; nhưng chưa được thực hiện trong bài.
4. **Judge xung đột với safety case**: Phát hiện vấn đề rubric chưa phân biệt rõ Accuracy và Safety, đề xuất tách thành 2 tiêu chí riêng

---

## 🚀 Học được và Cải thiện trong tương lai
1. Không bao giờ tin kết quả của 1 Judge duy nhất - luôn cần ít nhất 2 model độc lập để cross check
2. Chất lượng prompt của Judge quan trọng hơn 10 lần so với việc chọn model lớn hơn
3. Async processing không chỉ làm nhanh hệ thống mà còn giúp giảm rất nhiều chi phí thời gian khi chạy benchmark lớn
4. 80% lỗi không đến từ thuật toán mà đến từ các edge case không được xử lý trong pipeline

---
