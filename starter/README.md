# Starter Day 7

| File | Công dụng |
|---|---|
| `agentic.py` | Khung agent ví dụ với số lượt search, top-k và context budget có thể cấu hình |
| `local_controller.py` | Controller Transformers local dùng model instruct để chọn search/answer |
| `agent_benchmark.py` | Chạy agent thật trên split và lưu traces, cases, review template |
| `baseline.py` | TF-IDF lexical baseline, chạy offline |
| `retrieval.py` | Dense Day 6, TF-IDF, BM25 + RRF, dense + TF-IDF rerank, keyword query rewrite |
| `retrieval_benchmark.py` | Chạy năm phương pháp trên cùng benchmark và lưu trace/summary |
| `tracing.py` | Ghi JSONL UTF-8 cho retrieval và agent |
| `convert_day6.py` | Chuyển một số trường JSON từ Day 6 |
| `agent_metrics.py` | Tổng hợp nhãn đánh giá do bạn tự chấm |

Các phương pháp dùng chung chunk và vector Day 6: dense exact cosine, TF-IDF, BM25 + dense RRF, dense candidate top-30 được xếp lại bằng TF-IDF, và dense retrieval với keyword-focused query rewrite. RRF và candidate count là cấu hình ban đầu, cần chọn trên dev. Rewrite bỏ từ hỏi/chức năng tiếng Anh, giữ thuật ngữ và phủ định; không gọi model ngoài. Dense không có fallback giả.

## 1. Chạy agent với model local

`EmbeddingGemma 2` tại `D:\Models\embeddinggemma-2` chỉ tạo query embedding, không sinh câu trả lời. `local_controller.py` dùng model instruct sinh văn bản; lượt chạy đã chuẩn bị dùng `Qwen2.5-1.5B-Instruct` tại `D:\Models\Qwen2.5-1.5B-Instruct` trên CPU. Inference chỉ đọc model local (`local_files_only=True`), không gửi câu hỏi hay tài liệu ra API. Model loading cần chạy bằng virtual environment Day 06 có `torch` và `transformers`.

`LocalTransformersController` trả JSON action `search` hoặc `answer`; text trong tài liệu được xem là dữ liệu không đáng tin, không phải chỉ dẫn. `agent_benchmark.py` nối controller với dense retriever Day 6 strategy B, không đưa gold/expected answer vào agent. Ví dụ chạy baseline dev từ thư mục Day 07:

```powershell
$python = "..\..\Day06\K5-L3B-Day06-Nguyen-Anh-T---2A202604088-DataIngestionEngineering\starter\.venv\Scripts\python.exe"
$day6 = "..\..\Day06\K5-L3B-Day06-Nguyen-Anh-T---2A202604088-DataIngestionEngineering\starter"
& $python "starter\agent_benchmark.py" --day6-starter $day6 `
  --benchmark "data\benchmark.json" --split dev `
  --model-path "D:\Models\Qwen2.5-1.5B-Instruct" `
  --output "starter\results\agent" --top-k 3 `
  --context-budget 800 --max-searches 2
```

Mỗi lượt tạo thư mục riêng có `summary.json`, `traces.jsonl`, `traces.json`, `cases.json` và `reviews_template.json`. Review template có nhãn trống cần người chấm điền. Dùng `--top-k 5 --context-budget 1600` để đo cấu hình context A trên dev. Chỉ chạy test sau khi review câu trả lời/citation trên dev và chốt cấu hình.
Model chạy CPU; mặc định tối đa 192 token mỗi quyết định và hai lượt search. Có thể dùng `--limit 1` để kiểm tra tích hợp, nhưng kết quả đó được đánh dấu partial và không thay cho benchmark đủ split.
So sánh smoke D01 trên Qwen local và Day 6 pgvector thật (188 chunks, dense strategy B, exact cosine): baseline top-k 3/budget 800/tối đa 1 search có context evidence `0/1`, thời gian 30,53 giây; cấu hình top-k 5/budget 1.600/tối đa 2 search có hit@1 `0/1`, hit@3 `0/1` nhưng context evidence `1/1`, thời gian 70,91 giây. Chunk đúng xếp hạng 4 nên chỉ xuất hiện trong context top-5. Cả hai lần đều dùng 1 search, model local cost `$0` và abstain vì Qwen không kèm citation; tăng context giúp đưa bằng chứng vào nhưng không sửa được lỗi citation. Các tham số được đổi cùng lúc và mỗi phép đo chỉ có một câu, nên đây là kết quả smoke định hướng, chưa tách được tác động của từng tham số hay đại diện chất lượng toàn dev/test. Ở cấu hình top-k 5, agent ghi câu trả lời thử và abstain (không phải lỗi). Đối chiếu thủ công: chunk `b01510dac0e9f0e2bc074865dd5596bc`, trang 6, hỗ trợ nội dung câu trả lời, nhưng ID nguồn không được model trích dẫn nên không phát hành câu trả lời. Baseline: `starter/results/agent/smoke-section3-fix/dev-20261116T102503992673Z-7a2ac4a6/`; cấu hình top-k 5: `starter/results/agent/real-model-smoke/dev-20261116T115442991447Z-eac5e1ab/`.
Full dev trên cùng 17 câu: baseline dense top-k 3/context 800/96 token ghi 12 abstain và 5 lỗi parse; cấu hình top-k 5/context 1.600/192 token cùng parser mới ghi 16 abstain và 1 lỗi parse. Context Evidence tăng 12/15→14/15 nhưng cả hai lần không phát hành answer nào; median latency 62,17→101,08 giây/câu, phí local `$0`. Kết quả không đủ để chốt agent cho test do answer/citation chưa đạt. Xem [`baseline summary`](results/agent/full-dev/dev-20261116T120603637985Z-029eb140/summary.json), [`context summary`](results/agent/full-dev-context/dev-20261116T122618082666Z-406a2bc6/summary.json) và báo cáo cá nhân.

Tham số `max_searches`, `top_k` và `budget` của `run_agent` có thể cấu hình. Nếu model bỏ thuật ngữ nội dung trong query, agent neo query tìm kiếm bằng câu hỏi gốc và trace lưu cả query do model đề xuất lẫn query thực tế. Context trích câu khớp thuật ngữ trong câu hỏi, xếp chunk theo retrieval score cao trước, giữ thứ tự khi hòa và deduplicate theo chunk ID; trace ghi cách chọn trên từng lượt search. Khi chạm giới hạn, trace cũng lưu query/model metadata của lượt được đề xuất nhưng không thực thi. Nếu model bỏ mảng `citations` nhưng câu trả lời có citation nội dòng `[chunk-id]`, controller trích đúng các ID đó; nếu không có citation nào, agent lưu câu trả lời như lần thử nhưng từ chối nó và trả `abstain`. Sentinel `INSUFFICIENT_EVIDENCE` cũng được coi là từ chối. Agent chỉ chấp nhận ID có trong context; ID hợp lệ chưa chứng minh nguồn hỗ trợ câu trả lời, nên cần đọc evidence. Báo lỗi model/vector store, không thay bằng kết quả giả.

## 2. Chạy TF-IDF baseline

Chạy từ thư mục gốc với Python 3.11 trở lên:

```bash
python3 -m unittest discover -s tests -v
python3 starter/baseline.py --chunks data/example_chunks.json --questions data/example_questions.json --split dev --output starter/results/example-dev
```

Kết quả có `summary.json` và `traces.json`. Dữ liệu mẫu chỉ có một câu, không đủ để nộp bài. TF-IDF là lexical retrieval, không dùng filter và không thay dense retrieval.

## 3. So sánh năm phương pháp

Chạy từ Day 07 root bằng virtual environment Day 06, vì nơi đó có `sentence-transformers`, `psycopg`, `pgvector` và `rank-bm25` đã dùng để tạo index:

```powershell
$python = "..\..\Day06\K5-L3B-Day06-Nguyen-Anh-T---2A202604088-DataIngestionEngineering\starter\.venv\Scripts\python.exe"
$day6 = "..\..\Day06\K5-L3B-Day06-Nguyen-Anh-T---2A202604088-DataIngestionEngineering\starter"
& $python "starter\retrieval_benchmark.py" --day6-starter $day6 `
  --benchmark "data\benchmark.json" --split dev `
  --output "starter\results\retrieval"
```

Sau khi chọn cấu hình chỉ bằng dev, chạy lại với `--split test`; không dùng test để chỉnh tham số. Runner đọc cấu hình `DATABASE_URL`/`EMBEDDING_MODEL` từ `.env` Day 06, chỉ đọc strategy B và mặc định yêu cầu đủ 188 chunks. Nó không tạo index, không tạo vector ngẫu nhiên và dừng rõ ràng nếu database hoặc model không sẵn sàng.

Các phương pháp, cùng top-k và filter section nếu câu hỏi nêu rõ trước retrieval:

1. `dense`: embedding query `SearchQuery` tương thích index và exact cosine trên pgvector B.
2. `tfidf`: TF-IDF cosine trên cùng 188 chunk.
3. `hybrid_rrf`: dense + BM25, kết hợp ranking bằng `1/(k+dense_rank) + 1/(k+bm25_rank)`; mặc định `k=60`.
4. `dense_tfidf_rerank`: lấy tối đa 30 dense candidates rồi xếp lại theo TF-IDF trên cùng corpus.
5. `dense_keyword_rewrite`: bỏ từ hỏi/chức năng trước khi embed; log cả query ban đầu và query đã chuyển đổi.

Mỗi lượt tạo thư mục riêng dưới `starter/results/retrieval/`, có `comparison.json`, summary theo phương pháp và `traces.jsonl`. Summary ghi thời gian build index, tổng thời gian từng cấu hình, median search latency, chi phí dịch vụ cục bộ và metric `n/N`. Trace lưu query, query đã rewrite, filter, cấu hình/candidate count, từng kết quả (ID, score/rank, text, source, page, metadata), context cuối, ID context, latency và lỗi. Retrieval metrics chỉ tính trên câu có đáp án; câu thiếu evidence vẫn được truy xuất và ghi trace nhưng không vào mẫu số. Gold chỉ được dùng sau retrieval để chấm, không truyền vào retriever.

Trong agent thật, truyền đường dẫn để lưu toàn bộ trace:

```python
result = run_agent(
    question, controller, retriever,
    filters=filters,
    trace_path="starter/results/agent/dev.jsonl",
)
```

Trace agent bổ sung từng tool call, kết quả, context, câu trả lời, citations, status/lỗi và thời gian. Nếu controller trả thêm `provider`, `model`, `usage` hoặc `cost_usd`, các trường đó cũng được ghi. Lỗi vẫn được log rồi ném tiếp; ghi trace thất bại cũng được báo rõ.

## 4. Dùng dữ liệu Day 6

Xuất chunk và câu hỏi thành hai JSON array. Starter Day 6 không quy định tên bảng hay cách xuất; bạn cần tự thực hiện. Nếu chunk có `chunk_id`, `document_id`, `source`, `page`, `text`, `metadata` và câu hỏi có `gold_source_id`, `gold_page`, `evidence_quote`, chạy:

```bash
python3 starter/convert_day6.py --chunks <day6_chunks.json> --questions <day6_questions.json> --output starter/data/converted
python3 starter/baseline.py --chunks starter/data/converted/chunks.json --questions starter/data/converted/questions.json --split dev --output starter/results/dev-tfidf
```

Thay phần trong `< >` bằng đường dẫn thật. Script chỉ đổi tên trường, không tự truy cập database hay đoán evidence. Nếu cấu trúc khác, tự viết adapter. Đối chiếu lại nguồn, trang và evidence.

`--split dev` chỉ chấm dev. Sau khi chốt cấu hình, dùng `--split test` và chọn thư mục kết quả khác. Nếu test Day 6 từng dùng để chỉnh hệ thống, tạo test mới theo [GUIDE.md](../GUIDE.md). Baseline mẫu dùng evidence nằm trong một chunk và context từ top-3; cần mở rộng hoặc tự đánh giá nếu dùng multi-hop, context từ nhiều lượt hay schema khác. Khớp nguyên văn bằng code không thay thế đánh giá thủ công.

## 5. Tổng hợp đánh giá agent

Sau khi chạy agent thật và tự chấm, chuẩn bị ba JSON array:

- `cases.json`: mỗi câu có `id`, `split`, `answerable` (`true` nếu có đủ evidence).
- `traces.json`: mỗi kết quả có `query_id`, `status` (`answered`, `abstain`, `error`), `events`, `citations`. `agent_benchmark.py` đã tạo file này cùng `traces.jsonl`; lỗi được ghi riêng với `status: "error"`. Script đếm tool use qua event `search`.
- `reviews.json`: mỗi nhãn có `query_id`, `citation_support` là danh sách đúng/sai ứng với từng citation. Câu có đáp án cần `answer_grade` 0–2 theo [BENCHMARK.md](../BENCHMARK.md); nếu bị từ chối, chấm 0. Câu thiếu evidence cần `abstention_correct` (`true`/`false`) do người chấm xác nhận. Lần chạy lỗi không cần nhãn.

```bash
python3 starter/agent_metrics.py --cases <cases.json> --traces <traces.json> --reviews <reviews.json> --split test --output starter/results/agent-test/summary.json
```

Thay đường dẫn trong `< >`. Script tổng hợp nhãn bạn đã chấm, không tự đánh giá nội dung. Mẫu số bằng 0 nghĩa là không áp dụng. Lỗi kết nối được báo riêng, không tính vào chất lượng; giữ trace và nhãn để đối chiếu.

## Lưu ý

TF-IDF dùng cách tách từ đơn giản. `baseline.py` dùng mặc định context budget 800 ký tự, gồm cả nguồn; chunk không vừa bị bỏ. Bạn có thể chọn cách khác. File kết quả cùng tên sẽ bị ghi đè. `starter/data/` và `starter/results/` không được Git theo dõi mặc định; nếu được phép chia sẻ, dùng `git add -f <đường-dẫn>`, hoặc ghi cách tái tạo. Không nộp dữ liệu mật và secret.

Khung agent ưu tiên giữ chunk từ lần search trước. Khi chunk đầu chiếm gần hết budget, lần search sau có thể không thêm được evidence; hãy kiểm tra trace và thử cách chọn context phù hợp trên dev. Citation ID hợp lệ hoặc verifier trả “đúng” không đảm bảo câu trả lời được hỗ trợ; cần chấm thủ công.
