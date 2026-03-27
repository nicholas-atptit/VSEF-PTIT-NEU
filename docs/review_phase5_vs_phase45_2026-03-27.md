# Review: Phase 5 (v5.0) so với bản trước (PHASE 4-5 V5.0)

## Bối cảnh kiểm tra
- Nhánh hiện tại: `work`.
- Commit hiện tại: `69d9bae8` (`phase 5 v5.0`).
- Commit trước đó: `3e375d33` (`PHASE 4-5 V5.0`).
- Trạng thái remote: chưa cấu hình `origin` nên **không thể pull** tự động từ server Git.

## Kết luận nhanh
Bản hiện tại **đã mở rộng tính năng** theo hướng active sentiment analysis trong TUI/Signal Generator, nhưng **chưa thật sự hoàn chỉnh ở mức release-ready** vì còn dấu hiệu code debug/tạm thời và thiếu lớp kiểm thử tự động trong môi trường sạch.

## Điểm đã cải thiện so với bản trước
1. **Bổ sung active analysis path** trong `SignalGenerator`:
   - Khi chưa có sentiment và bật `active_analysis=True`, hệ thống sẽ crawl tin mới rồi chạy lại LLM để tạo sentiment.
   - Payload sentiment có phân biệt trạng thái `SUCCESS` / `FALLBACK` / `PENDING`.

2. **TUI có cơ chế on-demand agent**:
   - Tự kích hoạt agent khi cache sentiment thiếu hoặc stale.
   - Cập nhật lại cache file để các thành phần khác đọc được ngay.

3. **LLM intel hỗ trợ provider linh hoạt hơn**:
   - Có luồng riêng cho Gemini native.
   - Có retry khi dính rate limit (429) và fallback model khi 404/not found.

4. **News crawler thêm backward compatibility**:
   - Hỗ trợ alias field (`published_date`) và thêm API `crawl_watchlist`.

## Các điểm chưa hoàn chỉnh / rủi ro cần cải thiện
1. **Không pull được code mới do chưa có remote**
   - Repo hiện chưa gắn remote (`git remote -v` trống), nên không xác nhận được "đã khớp ver mới nhất trên server".
   - Việc so sánh hiện chỉ làm được giữa local commits.

2. **Lẫn file debug/tạm vào codebase**
   - Có dấu hiệu file runtime/debug được commit trong lịch sử gần nhất (`tui_debug.log`, `temp_articles.csv`, `data/.tui_lock`).
   - Khuyến nghị: đưa vào `.gitignore`, dọn tracked artifacts và chuẩn hóa output runtime vào `logs/` không track.

3. **Cần siết lại độ bền luồng crawler/LLM**
   - `crawl_ticker` đã có timeout tổng, nhưng cần thêm kiểm soát timeout tầng I/O riêng, fallback API method nhất quán, và metric rõ ràng cho timeout/error rate.
   - Cơ chế mutate `settings.gemini_model_name` trong fallback model có thể gây side-effect nếu chạy đồng thời nhiều ticker.

4. **Thiếu chứng nhận test trong môi trường sạch**
   - Chạy `pytest -q` hiện fail ngay từ bước collect vì thiếu dependencies (fastapi, pydantic, numpy, sqlalchemy, aiokafka...).
   - Chưa có bằng chứng regression test pass cho Phase 5.

## Đề xuất ưu tiên cải thiện (theo thứ tự)
1. **Thiết lập CI tối thiểu + lock dependencies**
   - Chốt môi trường bằng `requirements`/`poetry.lock` và pipeline test smoke.
   - Mục tiêu: đảm bảo ít nhất test collection pass trong CI sạch.

2. **Dọn hygiene repository**
   - Ignore & untrack file runtime/debug.
   - Tách script test thủ công (như `test_gemini*.py` ở root) vào thư mục tooling hoặc chuyển thành test chuẩn có marker.

3. **Ổn định concurrency & model fallback**
   - Tránh mutate global settings khi fallback model; dùng biến cục bộ theo request.
   - Bổ sung circuit-breaker nhẹ cho provider hay bị 429/404.

4. **Bổ sung acceptance tests cho luồng active analysis**
   - Test case: no cached sentiment -> trigger crawl -> analyze -> state chuyển `PENDING -> SUCCESS/FALLBACK`.

## Trả lời câu hỏi "đã hoàn chỉnh chưa?"
- Nếu tiêu chí là **feature demo**: tương đối ổn hơn bản trước.
- Nếu tiêu chí là **production-ready**: **chưa hoàn chỉnh**, cần xử lý 4 nhóm cải thiện ưu tiên ở trên.
