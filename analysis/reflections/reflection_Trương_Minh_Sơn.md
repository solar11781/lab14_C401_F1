# Báo cáo cá nhân — Nhóm Regression Release Gate (Analyst)

- **Họ tên:** [Trương Minh Sơn]
- **Nhóm:** Regression Release Gate (DevOps/Analyst)
- **Vai trò phụ trách:** Analyst / Delta analysis & Failure analysis owner
- **Commit minh chứng:** [[Dán link commit `failure_analysis.md` ở đây](https://github.com/solar11781/lab14_C401_F1/blob/main/analysis/failure_analysis.md)]

## 1. Engineering Contribution

Phần việc chính của tôi là đọc và phân tích output benchmark để chuyển kết quả kỹ thuật thành báo cáo có thể chấm điểm.

Cụ thể, tôi phụ trách:

- Đọc `summary.json` để xác định quyết định cuối cùng của release gate.
- Phân tích các `failed_checks` trong phần regression, đặc biệt là các chỉ số quality bị giảm ở V2.
- Đối chiếu `benchmark_results.json` để chọn các case thất bại tiêu biểu phục vụ phần “5 Whys”.
- Hoàn thiện file `analysis/failure_analysis.md`, bao gồm benchmark overview, failure clustering, root-cause analysis và action plan.
- Chuyển kết quả kỹ thuật sang ngôn ngữ giải thích dễ hiểu cho báo cáo nhóm và báo cáo cá nhân.

Tôi không phụ trách viết code chính của `main.py`, nhưng phần đóng góp của tôi nằm ở việc biến các output benchmark thành phân tích có ý nghĩa, phục vụ trực tiếp cho quyết định `BLOCK_RELEASE`.

## 2. Technical Depth

Trong quá trình làm phần Analyst, tôi hiểu rõ hơn mối liên hệ giữa các nhóm metrics:

- **MRR** không chỉ là một con số retrieval, mà còn cho thấy tài liệu đúng có nằm đủ cao để model dễ bám vào hay không.
- **Cohen’s Kappa** cho biết mức độ ổn định của hệ thống Judge; nếu Kappa thấp thì kết luận benchmark cũng cần cẩn trọng hơn.
- **Trade-off giữa chi phí và chất lượng** là phần quan trọng nhất trong release gate: V2 của nhóm chạy nhanh hơn, ít token hơn, nhưng lại giảm `avg_score`, `faithfulness`, `relevancy`, nên không thể release chỉ vì “rẻ hơn”.

Từ output thực tế, tôi rút ra được thông điệp chính của bài lab: một agent không thể được release nếu chỉ tối ưu hiệu năng mà làm suy giảm chất lượng trả lời.

## 3. Problem Solving

Vấn đề chính tôi xử lý là phân biệt loại lỗi.

Khi đọc `benchmark_results.json`, tôi nhận ra không phải case fail nào cũng cùng nguyên nhân:

- có case fail vì **retrieval miss**,
- có case fail vì **ground-truth chunk đã xuất hiện nhưng answer vẫn không bám đúng evidence**,
- và có case fail vì **judge conflict** ở các prompt red-team / safety.

Ví dụ, có case vendor violation mà chunk đúng đã nằm trong retrieved results nhưng agent vẫn trả lời “không tìm thấy”, cho thấy lỗi không chỉ nằm ở retrieval mà còn ở bước grounding/generation. Ngoài ra, một số case red-team có disagreement giữa các Judge và bị chốt theo strictest-score, cho thấy bài toán calibration của evaluator cũng ảnh hưởng trực tiếp đến decision cuối cùng.

Nhờ cách phân loại này, tôi hoàn thiện được phần 5 Whys theo hướng chỉ ra lỗi hệ thống, thay vì chỉ mô tả triệu chứng bề mặt.

## 4. Tự đánh giá đóng góp

Tôi đánh giá phần đóng góp của mình nằm ở nhánh Analyst của nhóm Regression:

- đọc đúng ý nghĩa của các metrics,
- phân tích vì sao gate block release,
- và chuyển output kỹ thuật sang failure analysis có thể dùng trực tiếp khi nộp bài.

Điểm mạnh nhất trong phần của tôi là khả năng nối từ benchmark output sang root cause và action plan, giúp nhóm không chỉ “đo được” mà còn “biết phải cải thiện gì tiếp theo”.
