# Báo cáo Day 7

## Chuẩn bị

- **Corpus:** NIST, *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*, NIST AI 100-1, DOI https://doi.org/10.6028/NIST.AI.100-1. Dùng trang PDF vật lý 6–47 (42 trang nội dung) từ cùng PDF Day 06; nguồn bao quát framing risk, governance, mapping, measurement, management, profiles và appendices. Manifest và lưu ý quyền Figure 2 được ghi trong [manifest Day 06](../../../../Day06/K5-L3B-Day06-Nguyen-Anh-T---2A202604088-DataIngestionEngineering/starter/data/manifest.json). PDF gốc không được sao chép vào repo.
- **Chunks:** Dùng cấu hình B Day 06 (chia theo đoạn/section, tối đa 700 ký tự), cùng ID SHA-256 32 ký tự, nguồn, trang và metadata. Tái tạo pipeline từ PDF cho kết quả 188 chunks; 5 evidence test mới đã được kiểm tra nằm trọn trong từng chunk B tương ứng. Không tạo vector mới.
- **Embedding:** EmbeddingGemma 2 local qua Sentence Transformers, revision `f9972ac22e696a105c482202646979e33e4b5a1e` (model commit ngày 2026-10-28), đầu ra 768 chiều; prompt `Document` cho chunk, `SearchQuery` cho query, L2 normalization, CPU. Runtime Day 06 ghi Sentence Transformers 6.3.0, Transformers 5.23.0 và PyTorch 2.15.0+cpu. Không đổi model; chi phí embedding Day 07 là **$0** vì tái sử dụng vectors.
- **Vector store và cách nối:** PostgreSQL 16.16 + pgvector 0.8.7; bảng `day6_benchmark_chunks`, lọc `strategy = 'B'`, tìm exact cosine; kết nối local qua `DATABASE_URL` trong `.env` Day 06 tại `127.0.0.1:55433/day6_lab`. Không ghi thông tin xác thực vào repo. Metadata filter `section` chỉ truyền khi câu hỏi nêu section trước retrieval.
- **Trạng thái kiểm tra:** Sau khi khởi động Docker Compose Day 06, đã xác minh kết nối PostgreSQL 16.16 + pgvector 0.8.7 và đủ 188 chunk B trong `day6_benchmark_chunks`. Không tạo vector mới; benchmark truy vấn index thật và dùng EmbeddingGemma local trên CPU.
- **Benchmark:** [`data/benchmark.json`](../../data/benchmark.json) có 15 câu có đáp án + 2 câu thiếu evidence ở dev; 5 câu có đáp án mới + 2 câu thiếu evidence ở test. Các test answerable mới lấy evidence từ trang 28, 31, 34, 36 và 44, không tái dùng test Day 06 đã được xem. Gold chỉ dùng để đánh giá, tuyệt đối không truyền vào agent/retriever.

## Phương pháp retrieval đã triển khai

| Method | Cấu hình đầu | Cách xếp hạng |
|---|---|---|
| `dense` | Day 6 strategy B, exact cosine, top-k 3 | Query EmbeddingGemma `SearchQuery`, vector 768 chiều |
| `tfidf` | Cùng 188 chunks, top-k 3 | TF-IDF cosine offline |
| `hybrid_rrf` | Dense + BM25 toàn corpus, RRF `k=60` | `1/(k+dense_rank) + 1/(k+bm25_rank)` |
| `dense_tfidf_rerank` | Dense candidate top-30, xuất top-3 | TF-IDF cosine trên candidate pool |
| `dense_keyword_rewrite` | Keyword-focused query, dense top-3 | Bỏ từ hỏi/chức năng; giữ thuật ngữ và phủ định |

Đây là cấu hình khởi đầu (top-k 3, context budget 800 ký tự; RRF `k=60`, rerank candidate top-30). Filter chỉ cho phép `section`, lấy từ câu hỏi nêu rõ section. Lệnh chạy, định dạng output và vị trí trace nằm trong [starter/README.md](../../starter/README.md). Mỗi method ghi candidates, rank/score, source, page, metadata, query/filter, context và latency; agent trace còn lưu tool call, answer, citation và error. Query rewrite hiện là quy tắc cục bộ, không phát sinh chi phí model.

Đã chạy benchmark retrieval thật trên dev (17 câu: 15 có đáp án, 2 thiếu evidence), không có lỗi truy vấn. So khớp evidence bỏ qua khác biệt khoảng trắng do xuống dòng khi trích xuất PDF; vẫn yêu cầu đúng nguồn và trang. Kết quả từng lượt và JSONL traces: `starter/results/retrieval/dev-20261116T041558320576Z-d6f0c8b4/`. Các thời gian là trên CPU local, không gồm thời gian khởi động model; chi phí dịch vụ là $0.

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk: NIST AI RMF 1.0, 42 trang nội dung, 188 chunk B; chi tiết corpus và manifest ở phần Chuẩn bị.
- Vector store, embedding model, phiên bản và số chiều: PostgreSQL 16.16 + pgvector 0.8.7; EmbeddingGemma 2 local, 768 chiều.
- Agent model, cách chạy và phân chia dev/test: Đã thử Qwen2.5-1.5B-Instruct local qua Transformers trên CPU với một câu smoke D01 thuộc dev; controller trả output không phải action JSON hợp lệ sau khoảng 141 giây. Vì vậy không có điểm answer/citation đáng tin; chưa chạy full dev hoặc test. Xem trace `starter/results/agent/smoke/dev-20261116T052727903716Z-6072dff7/traces.jsonl`.
- Cách xác định câu có/thiếu evidence và tiêu chí chấm: Gold source/page/evidence của benchmark; kiểm tra evidence sau retrieval, chuẩn hóa khoảng trắng do PDF xuống dòng nhưng vẫn yêu cầu đúng nguồn và trang. Retrieval metrics chỉ tính 15 câu answerable.

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| `dense` (baseline, top-3) | Strategy B, exact cosine; context 800 | Dense truy vấn vector Day 6 | dev | 53.3% (8/15) | 86.7% (13/15) | 53.3% (8/15) | median 448 ms; tổng 17.85 s; $0 |
| `tfidf` (top-3) | TF-IDF trên 188 chunks; context 800 | Lexical baseline offline | dev | 60.0% (9/15) | 66.7% (10/15) | 60.0% (9/15) | median 1.41 ms; tổng 0.10 s; $0 |
| `hybrid_rrf` (top-3) | Dense + BM25, RRF k=60; context 800 | Hợp nhất thứ hạng dense/lexical | dev | 53.3% (8/15) | 73.3% (11/15) | 53.3% (8/15) | median 939 ms; tổng 37.69 s; $0 |
| `dense_tfidf_rerank` (top-3) | Dense top-30, xếp lại TF-IDF; context 800 | Rerank candidate dense | dev | 60.0% (9/15) | 66.7% (10/15) | 60.0% (9/15) | median 1,137 ms; tổng 64.17 s; $0 |
| `dense_keyword_rewrite` (top-3) | Keyword rewrite + dense; context 800 | Biến đổi query bằng quy tắc | dev | 53.3% (8/15) | 80.0% (12/15) | 53.3% (8/15) | median 3,062 ms; tổng 64.72 s; $0 |
| Cấu hình chọn sau đánh giá agent | Chưa chốt | Thiếu grounded-answer/citation score để chọn theo tiêu chí | test | Chưa chạy | Chưa chạy | Chưa chạy | — |
| Baseline | Dense top-3, context 800 | Đối chứng holdout | test | Chưa chạy | Chưa chạy | Chưa chạy | — |

### Hai cấu hình cải thiện đã thử trên dev

| Thử nghiệm | Giả thuyết và thay đổi | Kết quả trên phương pháp bị tác động (cùng dev) | Thời gian / chi phí |
|---|---|---|---|
| A: tăng candidate/context | Dense top-k 3 → 5; context budget 800 → 1600 ký tự. Thêm candidates và chỗ chứa có thể tăng evidence trong context mà không đổi index/model. | Dense: Hit@1 8/15 (không đổi); Evidence Hit@3 13/15 (không đổi); Context Evidence 9/15, tăng từ 8/15. D02 và D04 vẫn không có gold trong top-5/context. | Median search 1,096 ms (baseline 448 ms); tổng 31.15 s; $0. |
| B: tăng trọng số rank cao trong hybrid | Giữ top-k 3/context 800; RRF `k=60` → `k=10`. Rank cao được tăng ảnh hưởng tương đối, có thể cải thiện top-3 hybrid. | Hybrid RRF: Hit@1 8/15 (không đổi); Evidence Hit@3 12/15, tăng từ 11/15; Context Evidence 8/15 (không đổi). D02 và D04 vẫn miss top-3/context. | Median search 751 ms (baseline 939 ms); tổng 15.88 s; $0. |

Cả hai lượt đều chạy đủ năm method trên cùng 17 câu dev; không có lỗi truy vấn. Kết quả đầy đủ và traces: `starter/results/retrieval/dev-20261116T042735822211Z-555d8e17/` (A), `starter/results/retrieval/dev-20261116T042919417386Z-1e0f6ba0/` (B). Mỗi method ghi 17 traces. Thời gian đo trên CPU local; dao động giữa lượt nên latency là chỉ báo, không phải bảo đảm cải thiện.
Tham số tái lập ngoài lệnh baseline trong README: A thêm `--top-k 5 --context-budget 1600`; B thêm `--rrf-k 10` (top-k 3/context 800 giữ mặc định).

### Đọc lỗi dev và quyết định trước test

- **D02 — “Why does the framework treat AI risk as more than a technical issue?”** Dense baseline xếp các đoạn Appendix B (trang 44), Appendix D (47), và 1.2.4 (13) vào top-3, trong khi evidence đúng ở trang 6. Các đoạn sai có chủ đề AI risk/frameworks nhưng không nêu phần evidence về societal/context factors. Câu hỏi dùng diễn đạt rộng (“more than a technical issue”), khiến embedding ưu tiên các đoạn gần chủ đề; đây là lỗi truy xuất/độ phân biệt, không phải thiếu evidence trong corpus. Sau A, gold vẫn không có trong dense top-5; sau B, hybrid vẫn không có trong top-3/context.
- **D04 — “Which four descriptors explain how the AI RMF is intended to be used?”** Dense baseline trả các đoạn nói về risk tolerance (trang 12) và profiles (38), không phải mô tả intended use ở trang 7. Từ “used/use” xuất hiện ở nhiều section nên semantic match bị kéo sang nội dung gần nghĩa nhưng sai điều kiện. Sau A, gold vẫn không có trong dense top-5; sau B, hybrid vẫn miss top-3/context.
- **Tiêu chí chọn đã công bố trước test:** ưu tiên Evidence Hit@3, tiếp đến Context Evidence Rate; hòa thì ưu tiên Hit@1, sau đó cần điểm grounded-answer/citation trên cùng dev, rồi mới dùng latency/chi phí để phân xử. Theo retrieval/context, A tăng context thêm 1/15 nhưng tốn thời gian hơn; B tăng hybrid Evidence Hit@3 thêm 1/15 nhưng không tăng context. Chưa thể chốt cấu hình toàn pipeline hoặc xác nhận chất lượng câu trả lời vì chưa có agent model thật và nhãn groundedness/citation. **Chưa chạy test**; cần hoàn tất đánh giá agent và chốt cấu hình trước.

## Agent và đánh giá

- Đã tích hợp controller Transformers local với model thật Qwen2.5-1.5B-Instruct, CPU, tối đa 96 token mỗi quyết định; agent gọi dense retrieval trên Day 6 strategy B, context budget 800 ký tự và giới hạn 1 lượt search trong smoke test. Trace lưu tool call, query, kết quả, context, câu trả lời/citation và lỗi; output parse lỗi được lưu tối đa 500 ký tự đầu để chẩn đoán.
- **Kết quả smoke D01 (dev, một câu, không phải benchmark):** search tool chạy thành công nhưng query do model tạo là “AI RMF definition”, không giữ được cụm đặc thù “AI system”; top-3 không có evidence trang 6 trong gold, context cũng không có evidence. Sau search, model không trả action JSON answer hợp lệ; trace kết thúc `error` sau 141.03 giây, không có câu trả lời hoặc citation để chấm. Thử dynamic INT8 riêng không đưa vào code sinh văn bản không liên quan thay vì JSON/action hợp lệ; chưa xác nhận chất lượng hoặc tốc độ của cấu hình này. Do chất lượng và thời gian không đạt, không chạy full agent dev/test, không chọn cấu hình agent và không dùng smoke để chốt test.
- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:
- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:
  - Trace smoke: `starter/results/agent/smoke/dev-20261116T052727903716Z-6072dff7/traces.jsonl`.
  - Không có điểm answer/citation; chỉ có retrieval trace smoke. Năm phương pháp retrieval và hai cải tiến được đánh giá riêng ở trên; chúng không thay thế đánh giá grounded answer của agent.

## Thử thách nâng cao (nếu thực hiện)

- Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:
- Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:
- Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | | | |
| Ảnh, sơ đồ | | | |
| Biểu đồ | | | |
| Multi-hop | | | |
| Text | | | |
| Thiếu bằng chứng | | | |
| **Toàn bộ** | | | |

- Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:
- Giới hạn, cách tái lập và đường dẫn benchmark:
