# Báo cáo Day 7

## Dữ liệu và benchmark

- Nguồn: Luật Sở hữu trí tuệ, Văn bản hợp nhất số 155/VBHN-VPQH ngày 09-09-2025; tài liệu công khai, gồm 30 trang nội dung, được chia thành 98 chunk theo đoạn.
- Vector store: PostgreSQL/pgvector dùng lại từ Day 6, database `day6_lab`, bảng `public.ingestion_chunks`, strategy B. Embedding qua OpenRouter `openai/text-embedding-3-large`, 3072 chiều.
- Agent và benchmark: Controller dùng `openai/gpt-4o-mini` qua OpenRouter; tối đa 2 lượt search, top-3 kết quả và context budget 3000 ký tự. Benchmark có 20 câu đủ evidence (15 dev, 5 test) và 4 câu thiếu evidence (2 dev, 2 test); chọn cấu hình trên dev rồi giữ nguyên khi đánh giá test.
- Evidence và chấm điểm: Mỗi câu đủ evidence có đáp án chuẩn, nguồn, trang và câu trích dẫn; evidence được xác nhận bằng nội dung trong chunk đúng nguồn/trang (bỏ qua khác biệt khoảng trắng). Retrieval Hit@1/Hit@3 yêu cầu evidence hợp lệ ở ranking tương ứng; Context Evidence yêu cầu evidence có trong context cuối. Bốn câu thiếu evidence hỏi về mức phí, mã biểu mẫu, kênh nộp đơn trực tuyến và địa chỉ bưu chính mà corpus không cung cấp; chúng không tính vào retrieval metrics và agent cần tìm kiếm rồi từ chối. Câu trả lời agent được chấm groundedness 0–2; citation được kiểm tra thủ công xem có trong context và có hỗ trợ câu trả lời hay không.

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | TF-IDF, top-3, không filter, budget 3000 | Mốc so sánh lexical | dev | 8/15 (53,3%) | 10/15 (66,7%) | 10/15 (66,7%) | Median 0,21 ms; $0; context được đóng gói lại offline từ run budget 800 |
| Phương pháp 2 | Dense pgvector, strategy B, top-3, không filter, budget 3000 | Embedding tìm đoạn diễn đạt tương đương | dev | 6/15 (40%) | 10/15 (66,7%) | 10/15 (66,7%) | 2,66 s/15 câu ở run nguồn; 600 embedding tokens; $0 |
| Phương pháp 3 | BM25, k1=0,75, b=0,75, top-3, không filter, budget 3000 | BM25 với tokenizer Unicode word của TF-IDF; thử độ bão hòa tần suất từ | dev | 10/15 (66,7%) | 13/15 (86,7%) | 12/15 (80%) | Median 0,34 ms ở run nguồn; $0 |
| Cải thiện 1 | Hybrid RRF Dense + TF-IDF; top-3 mỗi nguồn, RRF k=60, lấy top-3, budget 3000 | Kết hợp thứ hạng hai nguồn; không cộng score khác thang đo | dev | 6/15 (40%) | 12/15 (80%) | 12/15 (80%) | Fusion 0,60 ms trên trace đã có; không gồm chi phí Dense nguồn |
| Cải thiện 2 | Hybrid min-max Dense + BM25; chuẩn hóa per-query, trọng số 0,5/0,5, top-3 mỗi nguồn và lấy top-3, budget 3000 | Chuẩn hóa score trước khi fusion; Context Evidence cao nhất trên dev | dev | 9/15 (60%) | 13/15 (86,7%) | 13/15 (86,7%) | Fusion 0,47 ms trên trace đã có; không gồm chi phí Dense nguồn |
| Cấu hình đã chọn | Hybrid min-max Dense + BM25; cùng cấu hình dev, top-3, budget 3000 | Chốt vì Context Evidence cao nhất trên dev; không đổi cấu hình theo test | test | 2/5 (40%) | 3/5 (60%) | 3/5 (60%) | Dense 19,04 s/5 câu (gồm embedding); fusion 0,29 ms; $0 |
| Baseline | TF-IDF, top-3, không filter, budget 3000 | Mốc so sánh test | test | 2/5 (40%) | 4/5 (80%) | 4/5 (80%) | Median 0,37 ms ở run nguồn; $0 |

Một cấu hình có thể đồng thời là phương pháp mới và cải thiện; Hybrid min-max cũng là phương pháp hybrid thứ hai. Ưu tiên chọn theo Context Evidence, sau đó Evidence Hit@3: Hybrid min-max đạt 13/15 cho cả hai chỉ số trên dev, cao hơn baseline; BM25 có Hit@1 cao hơn một câu (10/15 so với 9/15). Trên test, Hybrid không vượt baseline ở Hit@3 hoặc Context Evidence (3/5 so với 4/5); giữ nguyên kết quả này, không điều chỉnh cấu hình dựa trên test.

Hai lỗi dev của cấu hình đã chọn:
- **q01**: gold ở trang 2 không xuất hiện trong top-3 của Dense hoặc BM25; fusion trả các trang 3, 16 và 8. Đây là lỗi candidate recall, không phải thiếu budget.
- **q09**: gold ở trang 12 không xuất hiện trong candidate pool; top-3 hợp nhất gồm trang 13, 11 và 2. Đây cũng là lỗi truy hồi candidate.
## Agent và đánh giá

- Cấu hình dev: Controller `openai/gpt-4o-mini`; tool `search_evidence` dùng Hybrid min-max Dense + BM25, top-3, candidate-3 mỗi phương pháp, budget 3000 ký tự, tối đa 2 lượt search. Có 40 chat calls, 21 lượt search, 23 embedding calls; latency trung bình 6,76 giây/câu.
- Trace và nhãn: [`traces.json`](../../starter/results/agent-dev-hybrid-minmax-diagnostic2/traces.json), [`reviews.json`](../../starter/results/agent-dev-hybrid-minmax-diagnostic2/reviews.json), [`metrics.json`](../../starter/results/agent-dev-hybrid-minmax-diagnostic2/metrics.json).

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Hybrid min-max agent / dev | 17/17 (100%) | 40 chat calls; 21 search events | 4/1/10; trung bình 1,40/2 | 11/14 (78,6%) | 13/13 (100%) | 2/2 (100%) | 2/15 (13,3%) |

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q01 | Có đáp án | Search | Trả lời thiếu quyền liên quan; citations 2/2 hỗ trợ phần được nêu | 1; thiếu một nhóm quyền |
| q02 | Có đáp án | Search | Đúng; citation 1/1 | 2 |
| q03 | Có đáp án | Search | Đúng và đủ; citation 1/1 | 2 |
| q04 | Có đáp án | Search | Đúng và đủ; citation 1/1 | 2 |
| q05 | Có đáp án | Search | Nội dung đúng; citation 0/1 vì trỏ vào định nghĩa nhãn hiệu tập thể, không phải định nghĩa chung | 2 |
| q06 | Có đáp án | Search | Đúng và đủ; citation 1/1 | 2 |
| q07 | Có đáp án | Search | Bị từ chối sau citation ID không có trong context | 0; false abstention |
| q08 | Có đáp án | Search | Đúng; citation 1/1 | 2 |
| q09 | Có đáp án | Search | Trả lời khoản 1–2 thay vì danh sách ở khoản 3; citation 0/1 | 0 |
| q10 | Có đáp án | Search | Đúng, đủ 4 nhóm; citation 1/1 | 2 |
| q11 | Có đáp án | Search | Bị từ chối vì citation ID không có trong context; nội dung thử cũng chưa đủ các quyền | 0; false abstention |
| q12 | Có đáp án | Search có filter Phần thứ hai | Đúng; citation 1/1 | 2 |
| q13 | Có đáp án | Search | Nêu các loại quyền tài sản thay vì nghĩa vụ khi khai thác; citation 0/1 | 0 |
| q14 | Có đáp án | Search | Đúng; citation 1/1 | 2 |
| q15 | Có đáp án | Search | Đúng cả điều kiện và giới hạn chuyển giao; citation 1/1 | 2 |
| q21 | Thiếu evidence | Search rồi từ chối | Corpus chỉ nêu phải nộp phí/lệ phí, không có số tiền cụ thể | Abstention đúng |
| q22 | Thiếu evidence | Search rồi từ chối | Corpus không có mã biểu mẫu/tờ khai đăng ký | Abstention đúng |

- Lệnh chạy: `python -X utf8 starter\agent_runner.py --split dev --top-k 3 --candidate-k 3 --budget 3000 --max-searches 2 --output starter\results\agent-dev-hybrid-minmax-diagnostic2`.
- Nhãn answer/citation được đánh giá thủ công theo gold và evidence trong corpus; citation ID hợp lệ không tự chứng minh citation hỗ trợ câu trả lời.

### Agent trên test

Baseline TF-IDF và cấu hình Hybrid min-max dùng cùng test split q16–q20 (5 câu có đáp án) và q23–q24 (2 câu thiếu evidence), cùng controller `openai/gpt-4o-mini`, top-3, context budget 3000 ký tự và tối đa 2 lượt search. Baseline không dùng metadata filter; Hybrid dùng filter khi câu hỏi có filter.

| Cấu hình / split | Tool use | Chat calls | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---:|---|---|---|---|---|
| Agent baseline TF-IDF / test | 7/7 (100%) | 18 | 1/1/3; trung bình 1,40/2 | 5/7 (71,4%) | 5/5 (100%) | 2/2 (100%) | 0/5 (0%) |
| Agent Hybrid min-max / test | 7/7 (100%) | 16 | 1/0/4; trung bình 1,60/2 | 5/8 (62,5%) | 5/5 (100%) | 2/2 (100%) | 0/5 (0%) |

| Query ID | Baseline TF-IDF: nhãn/citation | Hybrid: nhãn/citation | Nhận xét |
|---|---|---|---|
| q16 | 2; 1/1 | 2; 1/1 | Cả hai trả lời đúng, citation hỗ trợ. |
| q17 | 0; 0/1 | 2; 0/2 | Baseline thiếu điều kiện và citation không hỗ trợ ngoại lệ miễn phí. Hybrid trả lời đúng với evidence có trong context nhưng hai citation được chọn không hỗ trợ câu trả lời. |
| q18 | 2; 3/3 | 2; 3/3 | Cả hai nêu đúng không cần xin phép, nghĩa vụ trả tiền bản quyền và cách xử lý khi không thỏa thuận được. |
| q19 | 2; 1/1 | 2; 1/1 | Cả hai trả lời đúng thời hạn 50 năm sau năm đồng tác giả cuối cùng chết. |
| q20 | 1; 0/1 | 0; 0/1 | Baseline nêu đúng 50 năm nhưng sai mốc tính; Hybrid nêu sai cả thời hạn lẫn mốc tính. Citation của cả hai không hỗ trợ câu trả lời. |
| q23 | Abstention đúng | Abstention đúng | Corpus không có email hoặc URL chính thức để nộp đơn. |
| q24 | Abstention đúng | Abstention đúng | Corpus không có địa chỉ bưu chính của cơ quan tiếp nhận. |

- Thời gian trung bình mỗi câu: baseline 3,98 giây; Hybrid 5,71 giây. Baseline không gọi embedding; Hybrid gọi embedding 9 lần.
- Hybrid có grounded-answer mean cao hơn nhẹ trên 5 câu có đáp án (1,60 so với 1,40), nhưng citation validity thấp hơn. Bộ test nhỏ, nên đây chỉ là kết quả benchmark hiện tại, không khẳng định ưu thế tổng quát.

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
