# Báo cáo cá nhân — Nhóm Regression Release Gate (DevOps)

- **Họ tên:** Bùi Trần Gia Bảo
- **Nhóm:** Regression Release Gate (DevOps/Analyst)
- **Vai trò phụ trách:** DevOps / Implementation owner
- **Git commit link:** https://github.com/solar11781/lab14_C401_F1/commit/678ce9952b84fff94361774bd96ade0eb392e7d9

## 1. Engineering Contribution

Phần việc chính của tôi là triển khai và hoàn thiện module Regression Release Gate trong `main.py`

- Reorganized pipeline benchmark để chạy song song V1 và V2 bằng `asyncio.gather`, giúp so sánh hai phiên bản trên cùng một dataset trong một lần chạy.
- Hoàn thiện phần `build_summary()` để tổng hợp đầy đủ các metrics cần cho release gate: `avg_score`, `avg_faithfulness`, `avg_relevancy`, `hit_rate`, `avg_mrr`, `agreement_rate`, `conflict_rate`, `pass_rate`, `avg_latency`, `p95_latency`, `total_runtime_seconds`, `throughput_cases_per_min`, `total_tokens`, `avg_tokens_per_case`.
- Hoàn thiện phần `add_regression_section()` để chuyển từ logic gate đơn giản theo `delta_avg_score` sang gate nhiều ngưỡng theo 4 nhóm: quality, retrieval, reliability, performance/cost.
- Chuẩn hóa output của hệ thống sang các file `reports/summary.json`, `reports/benchmark_results.json`, đồng thời lưu riêng `benchmark_results_v1.json` và `benchmark_results_v2.json` để phục vụ giải trình kỹ thuật.
- Hoàn thiện phần print kết quả ra terminal để nhìn trực tiếp được delta giữa V1 và V2 theo từng nhóm chỉ số.

## 2. Technical Depth

Qua phần implementation này, tôi hiểu rõ hơn cách thiết kế một release gate cho AI system theo hướng “evidence-based”, không dựa vào cảm tính. Tôi cũng đã hiểu thêm về các khái niệm sau:

- **MRR (Mean Reciprocal Rank):** đo vị trí xuất hiện đầu tiên của ground-truth chunk trong danh sách retrieve. Nếu tài liệu đúng đứng càng cao thì MRR càng lớn.
- **Cohen’s Kappa:** dùng để đo mức độ đồng thuận thực chất giữa các Judge model, tốt hơn việc chỉ nhìn raw agreement vì đã tính đến xác suất đồng thuận ngẫu nhiên.
- **Trade-off giữa Chi phí và Chất lượng:** summary của nhóm cho thấy V2 nhanh hơn và tốn ít token hơn V1, nhưng lại giảm ở `avg_score`, `faithfulness`, `relevancy`. Vì vậy hệ thống phải block release thay vì chỉ nhìn performance.

## 3. Problem Solving

Vấn đề lớn nhất tôi xử lý là làm sao để release gate không quá đơn giản. Ban đầu, logic cũ chỉ so sánh `avg_score` của V2 với V1. Cách đó dễ làm nhưng thiếu chiều sâu vì không phản ánh được trường hợp “nhanh hơn, rẻ hơn nhưng chất lượng kém hơn” hoặc “retrieval giảm nhưng answer chưa giảm ngay”. Để giải quyết vấn đề này, tôi chuyển sang cách tổng hợp và kiểm tra theo nhiều nhóm chỉ số:

- quality
- retrieval
- reliability
- performance
- cost
