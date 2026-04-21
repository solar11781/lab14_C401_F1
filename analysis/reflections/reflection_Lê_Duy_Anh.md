# Báo Cáo Cá Nhân — Lab Day 14

**Họ và tên:** Lê Duy Anh
**Vai trò trong nhóm:** SDG (Nhóm Data)
**Ngày nộp:** 21/04/2026


## 1. Engineering Contribution
- Đóng góp trong bài lab: 
    Trong lab này, tôi đảm nhiệm vai trò thiết kế và phát triển module `synthetic_gen.py` để tạo tự động Golden Dataset. Các đóng góp kỹ thuật cốt lõi bao gồm:

    - Xây dựng từ nguyên bản (Building from scratch): Thay vì phụ thuộc vào các framework trừu tượng bậc cao, tôi đã trực tiếp lập trình luồng pipeline tạo dữ liệu bằng API thuần túy để kiểm soát hoàn toàn logic phía dưới, cấu trúc đầu ra và tối ưu hiệu năng.

    - Xử lý Bất đồng bộ (Async Pipeline): Triển khai thư viện `asyncio` và `openai.AsyncOpenAI` để gọi API song song, giúp tăng tốc độ sinh tổng cộng hơn 50 test cases. Việc tích hợp cơ chế van điều tiết `asyncio.Semaphore(5)` giúp quản lý tài nguyên hiệu quả, đảm bảo pipeline chạy cực nhanh (đáp ứng tiêu chí chạy dưới 2 phút) mà không gây quá tải hệ thống.

    - Thiết kế Prompt & Kỹ thuật SDG: Phân loại và xây dựng hệ thống prompt chi tiết để sinh ra 5 loại dữ liệu đánh giá toàn diện: Normal Q&A, Adversarial Prompts, Edge Cases, Multi-turn Complexity, và Technical Constraints.

    - Chuẩn hóa cấu trúc đánh giá: Ép LLM trả về định dạng JSON nghiêm ngặt `(response_format={"type": "json_object"})` và tự động mapping chính xác `ground_truth_ids` tới từng `chunk_id` cụ thể để phục vụ cho việc tính toán Hit Rate & MRR ở giai đoạn sau.

- Git commit:
    - commit 2f95b33a988122ef82e4043cd517ed866517b597
    Merge: 0de37f4 8dd7064
    Author: AnhLD2809 <leduyanh2k3@gmail.com>
    Date:   Tue Apr 21 17:19:16 2026 +0700

        Merge branch 'main' of https://github.com/solar11781/lab14_C401_F1

    - commit 0de37f4b31217ff57c8633f168dbf0e3a510c33e
    Author: AnhLD2809 <leduyanh2k3@gmail.com>
    Date:   Tue Apr 21 17:15:41 2026 +0700

        update faithfulness and relevancy score

    - commit 0d759b968707a1976a49846cc8ad97f12192a717
    Author: AnhLD2809 <leduyanh2k3@gmail.com>
    Date:   Tue Apr 21 16:43:45 2026 +0700

        synthetic_gen v2

    - commit 14637850519919bd17d55c41c48b7f174e2d4f52
    Author: AnhLD2809 <leduyanh2k3@gmail.com>
    Date:   Tue Apr 21 11:31:39 2026 +0700

        modify synthetic_gen

    - commit b02f627b6f04e1784ab593d5832a448f873cdd7f
    Author: AnhLD2809 <leduyanh2k3@gmail.com>
    Date:   Tue Apr 21 11:19:50 2026 +0700

        database and rag agent

    - commit 1522334020e3a2ee7ea249bcc74b42037608bc68
    Merge: cbaebaa 19cb566
    Author: AnhLD2809 <leduyanh2k3@gmail.com>
    Date:   Tue Apr 21 10:35:55 2026 +0700

        Merge branch 'main' of https://github.com/solar11781/lab14_C401_F1

    - commit cbaebaa2e6f2cbe22181c4ac38c694127d771794
    Author: AnhLD2809 <leduyanh2k3@gmail.com>
    Date:   Tue Apr 21 10:34:55 2026 +0700

        update docs data for RAG

## 2. Technical Depth
Các khái niệm cốt lõi trong đánh giá LLM:

- MRR (Mean Reciprocal Rank): Là chỉ số đánh giá chất lượng của Retrieval. Nó đo lường thứ hạng của chunk tài liệu đúng đầu tiên được truy xuất. Việc tôi thiết kế trường `ground_truth_ids` trong bộ Golden Dataset là điều kiện tiên quyết bắt buộc để hệ thống có thể đối chiếu và tính toán MRR tự động.

- Cohen's Kappa: Khi áp dụng Multi-Judge consensus (sử dụng cả GPT và một model khác làm giám khảo), Cohen's Kappa được dùng để đo lường mức độ đồng thuận thực sự giữa các giám khảo, giúp loại trừ các trường hợp chúng đồng tình với nhau chỉ do ngẫu nhiên.

- Position Bias: Hiện tượng các mô hình Judge (LLM-as-a-judge) có xu hướng thiên vị các câu trả lời nằm ở vị trí đầu hoặc cuối trong prompt. Hiểu rõ điều này giúp nhóm thiết kế cơ chế hoán đổi vị trí câu trả lời khi đưa vào luồng Multi-Judge.

- Trade-off giữa Chi phí và Chất lượng: Trong quá trình sinh dữ liệu, việc sử dụng mô hình gpt-4o đảm bảo dữ liệu đầu ra đạt độ khó và chất lượng tốt nhất. Tuy nhiên, để tối ưu chi phí (Cost vs. Quality), tôi đã thiết lập batching gom nhiều test cases vào một lần gọi API và tinh chỉnh temperature=0.7 để vừa giữ được tính đa dạng cho Adversarial/Edge cases, vừa đảm bảo tính chính xác chặt chẽ (Grounding) với ngữ cảnh.

## 3. Problem Solving
Trong quá trình phát triển module sinh dữ liệu tự động (SDG) từ đầu, tôi đã trực tiếp đối mặt và giải quyết một bài toán cốt lõi trong AI Engineering: Hiện tượng Hallucination (Sinh hoang tưởng) do ràng buộc quá chặt (Over-constrained Prompting).

- Bối cảnh vấn đề: Ban đầu, khi sinh 60 câu hỏi cho golden dataset, tôi chỉ tập trung vào 4 trường hợp Hard Case (mỗi trường hợp 15 câu) dẫn đến việc LLM tự sinh câu hỏi và tự trả lời cho câu hỏi đó mà không dựa trên chunk.
- Giải pháp đưa ra: Tăng số lượng các câu hỏi FAQ thông thường để giảm bớt áp lực kịch bản cho LLM đồng thời tôi đã can thiệp ở tầng gọi API bằng cách tinh chỉnh tham số `temperature=0.7`. Tham số này được tôi thử nghiệm để cắt giảm sự "phóng tác" quá đà của LLM, ép nó bám sát vào sự thật trong các chunk, nhưng vẫn duy trì đủ độ đa dạng cho các câu hỏi tấn công hệ thống.
