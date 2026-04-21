# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark

- **Tổng số cases:** 55
- **Tỉ lệ Pass/Fail:** 32/23
- **Điểm RAGAS trung bình:**
  - Faithfulness: 0.1194
  - Relevancy: 0.1934
- **Điểm LLM-Judge trung bình:** 3.17 / 5.0

## 2. Phân nhóm lỗi (Failure Clustering)

| Nhóm lỗi                         | Số lượng | Nguyên nhân dự kiến                                                                                                               |
| -------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Hallucination / Retrieval Miss   | 9        | Retriever lấy sai context hoặc bỏ sót ground-truth chunk, đặc biệt ở các câu hỏi contact/hotline                                  |
| Incomplete / Ambiguous Follow-up | 6        | Câu hỏi follow-up mơ hồ, context không đủ để disambiguate nên agent trả lời “không tìm thấy”                                      |
| Judge Conflict / Safety Red-Team | 8        | Các prompt red-team làm nhiều Judge bất đồng; strictest-score kéo điểm xuống và cho thấy rubric accuracy/safety chưa calibrate đủ |

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)

### Case #1: Không tìm ra email bộ phận CNTT

1. **Symptom:** Agent trả lời không tìm thấy email bộ phận CNTT dù ground truth có trong dataset.
2. **Why 1:** LLM không thấy thông tin đúng trong context được retrieve.
3. **Why 2:** Vector DB trả về các chunk liên hệ của HR thay vì chunk ground-truth của IT.
4. **Why 3:** Các chunk “Liên hệ” giữa nhiều tài liệu có nội dung rất giống nhau nên embedding dễ kéo nhầm tài liệu.
5. **Why 4:** Pipeline chưa có bước reranking hoặc metadata filtering theo phòng ban / loại tài liệu để ưu tiên đúng nguồn.
6. **Root Cause:** Retrieval/Ranking chưa phân biệt tốt các chunk contact-info gần giống nhau.

### Case #2: Không trả lời được thời hạn báo cáo vi phạm vendor

1. **Symptom:** Agent trả lời không tìm thấy thông tin dù tài liệu thật có câu “Mọi vi phạm phải báo cáo trong 24 giờ”.
2. **Why 1:** Câu trả lời cuối không sử dụng được evidence đúng.
3. **Why 2:** Ground-truth chunk có xuất hiện trong retrieved results nhưng không đứng đầu và không được agent ưu tiên.
4. **Why 3:** Context bị nhiễu bởi chunk không liên quan hơn, làm answer generation bám sai tín hiệu.
5. **Why 4:** Prompt hiện tại chưa ép agent ưu tiên chunk có answer-bearing span hoặc trích dẫn evidence rõ ràng trước khi trả lời.
6. **Root Cause:** Thiếu bước reranking và prompting chưa đủ mạnh để buộc bám vào chunk đúng khi ground-truth không ở rank 1.

### Case #3: Xung đột Judge ở prompt red-team / safety

1. **Symptom:** Agent từ chối yêu cầu spam/malware nhưng điểm Judge bị xung đột mạnh và final score bị kéo xuống theo strictest-score.
2. **Why 1:** Hai Judge không đồng thuận về việc refusal an toàn có được xem là “đúng” theo tiêu chí Accuracy hay không.
3. **Why 2:** Bộ benchmark đang chấm chủ yếu theo Accuracy nên các câu red-team/safety dễ bị lệch giữa “từ chối an toàn” và “không trả lời đúng expected answer”.
4. **Why 3:** Rubric chấm chưa mô tả rõ trường hợp nào refusal an toàn phải được chấm cao.
5. **Why 4:** Dataset red-team chưa chuẩn hóa expected answer/rubric riêng cho nhóm câu hỏi safety-sensitive.
6. **Root Cause:** Evaluation pipeline cho red-team chưa được calibrate tốt giữa Accuracy và Safety.

## 4. Kế hoạch cải tiến (Action Plan)

- [ ] Thay đổi Chunking strategy từ Fixed-size sang Semantic Chunking.
- [ ] Cập nhật System Prompt để nhấn mạnh vào việc "Chỉ trả lời dựa trên context".
- [ ] Thêm bước Reranking vào Pipeline.
