# Individual Technical Report

## 1. Engineering Contribution

### 1.1. Async Benchmark Pipeline
- Tôi tham gia hoàn thiện pipeline benchmark bất đồng bộ trong [engine/runner.py](c:\Users\2tmy\Desktop\AI_Thuc_Chien\lab14_C401_F1\engine\runner.py).
- Tôi giữ cấu trúc `asyncio.gather(...)` để chạy nhiều test case theo batch, giúp benchmark xử lý được nhiều case nhanh hơn thay vì chạy tuần tự.
- Tôi bổ sung cơ chế `tie-breaker` khi mức đồng thuận giữa các judge quá thấp, nhằm tránh việc một case bị kết luận chỉ dựa trên một judge duy nhất.
- Tôi thêm các trường output phục vụ phân tích như:
  - `retrieved_ids`
  - `retrieved_context`
  - `sources`
  - `agent_metadata`

### 1.2. Multi-Judge Consensus Engine
- Tôi chỉnh sửa [engine/llm_judge.py](c:\Users\2tmy\Desktop\AI_Thuc_Chien\lab14_C401_F1\engine\llm_judge.py) để triển khai chấm điểm với nhiều judge model.
- Hệ thống sử dụng ít nhất 2 judge:
  - `gpt-4o`
  - `gemini-2.5-flash`
- Tôi bổ sung:
  - xử lý lỗi theo từng model
  - `error_details`
  - `resolution_strategy`
  - `consensus_reached`
  - fallback khi judge phụ bị lỗi quota
- Tôi thêm logic dùng judge thứ ba (`gpt-4o-mini`) làm trọng tài trong các case bất đồng mạnh.

### 1.3. Metrics và Evaluation Output
- Tôi sửa [main.py](c:\Users\2tmy\Desktop\AI_Thuc_Chien\lab14_C401_F1\main.py) và [main1.py](c:\Users\2tmy\Desktop\AI_Thuc_Chien\lab14_C401_F1\main1.py) để tổng hợp đầy đủ các chỉ số:
  - `avg_score`
  - `hit_rate`
  - `avg_mrr`
  - `agreement_rate`
  - `conflict_rate`
  - `pass_rate`
- Tôi bổ sung khả năng in output theo format phục vụ báo cáo:
  - tổng số case
  - pass/fail
  - faithfulness
  - relevancy
  - điểm LLM-Judge trung bình

### 1.4. Dataset và Schema Handling
- Tôi xử lý tình huống dataset có nhiều schema khác nhau trong [main.py](c:\Users\2tmy\Desktop\AI_Thuc_Chien\lab14_C401_F1\main.py):
  - case thường dùng `question`
  - case multi-turn dùng `messages`
- Tôi viết hàm `normalize_test_case(...)` để quy đổi dữ liệu về một format chung, tránh lỗi `KeyError: 'question'`.
- Tôi sửa [engine/retrieval_eval.py](c:\Users\2tmy\Desktop\AI_Thuc_Chien\lab14_C401_F1\engine\retrieval_eval.py) để hỗ trợ cả:
  - `ground_truth_ids`
  - `expected_retrieval_ids`

### 1.5. Chứng minh qua kỹ thuật và commit
- Các phần tôi trực tiếp tham gia chỉnh sửa nằm ở:
https://github.com/solar11781/lab14_C401_F1/commit/9776d04e32f4839036d9d0a10d7c79ab15eebf52
https://github.com/solar11781/lab14_C401_F1/commit/8dd70641eb7b4707d4ca2a1e5d8a1cebb503ef9e

## 2. Technical Depth

### 2.1. MRR
- MRR (Mean Reciprocal Rank) đo vị trí xuất hiện đầu tiên của tài liệu đúng trong danh sách retrieve.
- Công thức:

```text
MRR = 1 / rank
```

- Ví dụ:
  - nếu chunk đúng đứng vị trí 1 thì MRR = 1.0
  - nếu đứng vị trí 2 thì MRR = 0.5
  - nếu đứng vị trí 5 thì MRR = 0.2
- Trong bài lab, MRR giúp đánh giá không chỉ việc “có tìm thấy đúng tài liệu hay không”, mà còn đánh giá tài liệu đúng có được ưu tiên đủ cao hay không.

### 2.2. Cohen's Kappa
- Cohen’s Kappa đo mức đồng thuận giữa hai judge sau khi loại trừ phần đồng thuận có thể xảy ra ngẫu nhiên.
- Công thức:

```text
Kappa = (P_o - P_e) / (1 - P_e)
```

- Trong đó:
  - `P_o`: observed agreement
  - `P_e`: expected agreement by chance
- Tôi đã áp dụng Kappa ở mức toàn bộ benchmark thay vì từng case đơn lẻ.
- Tôi cũng xử lý bài toán thực tế là không phải lúc nào 2 judge chính cũng cùng hoạt động được. Vì vậy summary cần theo dõi thêm:
  - `valid_pairs`
  - `coverage_rate`
- Đây là điểm quan trọng vì nếu judge phụ bị lỗi quota thì Kappa thấp không đồng nghĩa với việc hai judge bất đồng, mà có thể do thiếu overlap.

### 2.3. Position Bias
- Position Bias là hiện tượng judge thiên vị đáp án đứng ở vị trí A hoặc B thay vì đánh giá hoàn toàn theo nội dung.
- Trong một hệ thống pairwise judge, cách kiểm tra cơ bản là:
  - chấm A trước B
  - sau đó đổi thứ tự B trước A
  - so sánh kết quả
- Nếu kết quả thay đổi nhiều theo vị trí trình bày, judge có thể đang bị bias vị trí.
- Trong code, phần này đã được định hướng bằng method kiểm tra position bias ở judge module, dù chưa phải phần mạnh nhất của pipeline hiện tại.

### 2.4. Trade-off giữa Chi phí và Chất lượng
- Dùng nhiều judge model giúp tăng độ tin cậy nhưng cũng làm tăng:
  - số request API
  - thời gian chạy
  - chi phí
- Ví dụ trong hệ thống này:
  - `gpt-4o` cho chất lượng judge tốt hơn nhưng tốn chi phí hơn
  - `gemini-2.5-flash` rẻ hơn nhưng bị giới hạn quota free tier
  - `gpt-4o-mini` phù hợp làm tie-breaker hoặc fallback
- Vì vậy tôi chọn hướng:
  - judge chính: `gpt-4o`
  - judge phụ: `gemini-2.5-flash`
  - fallback/tie-breaker: `gpt-4o-mini`
- Thiết kế này cân bằng giữa:
  - độ tin cậy
  - khả năng chạy thật
  - chi phí benchmark

## 3. Problem Solving

### 3.1. Xử lý lỗi schema dữ liệu
- Khi chạy benchmark, hệ thống bị lỗi `KeyError: 'question'`.
- Tôi điều tra và phát hiện một phần dataset multi-turn không có `question`, mà chỉ có `messages`.
- Cách giải quyết:
  - tạo `normalize_test_case(...)`
  - lấy câu hỏi cuối của user từ `messages`
  - chuẩn hóa toàn bộ dataset về schema benchmark chung

### 3.2. Xử lý lỗi agreement thấp bất thường
- Ban đầu `agreement_rate` rất thấp hoặc bằng 0, khiến kết quả trông không hợp lý.
- Tôi kiểm tra `benchmark_results.json` và phát hiện nguyên nhân thật là:
  - `gemini-2.5-flash` bị `429 RESOURCE_EXHAUSTED`
  - gần như toàn bộ case chỉ có `gpt-4o` chấm
- Cách giải quyết:
  - thêm `error_details`
  - thêm `judge_coverage_rate`
  - sửa cách tính Kappa để phản ánh đúng overlap giữa các judge
  - thêm fallback model khi judge phụ lỗi quota

### 3.3. Xử lý lỗi RAGAS API không tương thích
- Khi tích hợp RAGAS thật trong `main1.py`, tôi gặp lỗi:
  - `AttributeError: 'Faithfulness' object has no attribute 'ascore'`
- Điều này cho thấy version RAGAS trong môi trường đang dùng API khác với ví dụ ban đầu.
- Tôi xử lý bằng cách viết wrapper linh hoạt:
  - thử `single_turn_ascore`
  - thử `ascore`
  - thử `single_turn_score`
  - thử `score`
  - nếu không dùng được thì fallback heuristic


### 3.5. Xử lý tình huống thiếu collection ChromaDB
- Hệ thống từng bị lỗi do collection `internal_docs` chưa tồn tại.
- Tôi thêm fallback trong `MainAgent`:
  - nếu chưa có collection, trả retrieval rỗng thay vì crash toàn bộ benchmark
- Điều này giúp pipeline ổn định hơn trong lúc setup hoặc rebuild dữ liệu.

## 4. Kết luận cá nhân
- Phần việc tôi tập trung chủ yếu nằm ở các module phức tạp nhất của bài:
  - async evaluation runner
  - multi-judge consensus
  - agreement/Kappa metrics
  - ragas/metric integration
  - debug runtime issues
- Tôi không chỉ viết code mới mà còn xử lý nhiều lỗi tích hợp thực tế giữa:
  - dataset
  - vector DB
  - judge APIs
  - RAGAS
  - asyncio runtime
- Qua phần này, tôi hiểu rõ hơn cách xây dựng một evaluation pipeline thực tế: không chỉ đo metric, mà còn phải xử lý độ tin cậy, khả năng chạy ổn định và trade-off giữa chất lượng với chi phí.
