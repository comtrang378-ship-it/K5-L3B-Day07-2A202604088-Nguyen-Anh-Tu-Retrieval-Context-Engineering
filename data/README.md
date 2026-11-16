# Định dạng dữ liệu

Các file `example_*.json` chỉ minh họa định dạng, không đủ để nộp bài. Dữ liệu thật là JSON array UTF-8; nếu dùng định dạng khác, ghi cách chuyển trong README.

- Benchmark của bài này nằm tại [`benchmark.json`](benchmark.json): 15 câu dev có đáp án, 2 câu dev thiếu bằng chứng, 5 câu test có đáp án mới và 2 câu test thiếu bằng chứng. Năm câu test có đáp án được tạo riêng cho Day 07; không tái sử dụng các câu test Day 06 đã xem.
- Corpus là NIST AI RMF 1.0, trang PDF vật lý 6–47 (42 trang). Chunks và vector được tái tạo theo cấu hình B Day 06: chia theo đoạn/section, tối đa 700 ký tự; xác minh tái tạo cho kết quả 188 chunks với ID ổn định. Nguồn và quyền sử dụng xem [manifest Day 06](../../../Day06/K5-L3B-Day06-Nguyen-Anh-T---2A202604088-DataIngestionEngineering/starter/data/manifest.json).
- Không sao chép vector hay tạo embedding mới: dense retrieval phải truy vấn lại PostgreSQL/pgvector Day 06, bảng `day6_benchmark_chunks`, `strategy = 'B'`. Kết nối local lấy từ `DATABASE_URL` trong `.env` của Day 06; không đưa thông tin xác thực vào repo. Khi chuẩn bị benchmark, PostgreSQL chưa chạy nên cần khởi động và xác nhận kết nối trước khi benchmark.
- Embedding tương thích: EmbeddingGemma 2 local, 768 chiều, Sentence Transformers; prompt `Document` cho chunk, `SearchQuery` cho query và L2 normalization. Revision, runtime và cách kết nối được ghi trong [báo cáo](../report/individual/MSSV_HoTen.md).
- File benchmark chứa gold chỉ dành cho chấm điểm. Agent, prompt, retrieval và query phải chỉ nhận câu hỏi cùng filter được nêu trước; không truyền `gold`, `expected_answer`, hoặc `absence_reason`.

- **Chunk:** `id`, `text`, `source_id`, `source`, `page`, `metadata`. ID phải duy nhất; giữ phiên bản hoặc ngày nếu có.
- **Câu có đáp án:** `id`, `split` (`dev` hoặc `test`), `question`, `expected_answer`, `gold`: danh sách `{source_id, page, evidence}`. Với schema mẫu, mỗi evidence phải nằm trọn trong một chunk đúng nguồn và trang; các phần tử `gold` là evidence thay thế. Với câu multi-hop cần nhiều đoạn, bạn có thể mở rộng schema, ghi rõ nhóm evidence bắt buộc và cách chấm. Có thể thêm `filter` do người dùng cung cấp trước.
- **Câu thiếu evidence:** `id`, `split`, `question`, `answerable: false`, `expected_behavior: "search_and_abstain"`, `absence_reason`. Xem `agent_cases_example.json`.

Không đưa `gold` hoặc nhãn đáp án vào model và retrieval tool. Ghi nguồn dữ liệu và quyền sử dụng. Cách chuyển một số định dạng Day 6: [starter/README.md](../starter/README.md).
