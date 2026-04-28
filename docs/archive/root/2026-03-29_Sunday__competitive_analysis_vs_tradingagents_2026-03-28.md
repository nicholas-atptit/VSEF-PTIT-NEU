# Comparative Research Report (2026-03-28)
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Historical archive |
| Created / authored | Sunday, 2026-03-29 02:07:33 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-04-28 22:28:23 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `8800ce6e3780c7978856737e70cb5e3b999eacee` |
| Timestamp source | Git history |
| Status | Historical reference |

## Scope
So sánh repo nội bộ **AI_ML_LLM-in-Stock** với:
1. https://github.com/TauricResearch/TradingAgents
2. https://github.com/stefan-jansen/machine-learning-for-trading
3. https://github.com/paperswithbacktest/awesome-systematic-trading

Mục tiêu: đề xuất cải tiến để đạt mức “đẳng cấp sản phẩm/research” gần TradingAgents, và đánh giá khả năng ứng dụng thêm từ 2 repo còn lại.

---

## 1) Snapshot nhanh các repo đối chiếu

### TradingAgents (TauricResearch)
- Định vị: framework **multi-agent LLM cho trading** (analyst team → researcher debate → trader → risk manager → portfolio manager).  
- Mức độ cộng đồng tại thời điểm khảo sát: ~43.2k stars, ~7.9k forks.  
- Có release mới gần đây (v0.2.2, 2026-03), và nhấn mạnh hỗ trợ multi-provider LLM + CLI hoàn chỉnh.

### machine-learning-for-trading (Stefan Jansen)
- Định vị: code đi kèm sách “Machine Learning for Algorithmic Trading” (2nd edition).  
- Mạnh về chiều sâu học thuật/end-to-end workflow theo chương (data → features → ML models → backtest → NLP → deep learning/RL).  
- Mức độ cộng đồng: ~16.9k stars.

### awesome-systematic-trading (paperswithbacktest)
- Định vị: curated list tài nguyên hệ thống trading (libraries, strategies, books, videos, blogs).  
- Tại thời điểm khảo sát: liệt kê ~97 libraries/packages, 40+ strategies, 55 books, 23 videos; ~7.5k stars.

---

## 2) Hiện trạng repo nội bộ (điểm mạnh và khoảng trống)

### Điểm mạnh đã có (rất tốt)
- Kiến trúc full-pipeline production-oriented (ingestion, streaming, storage, ML, LLM, UI), có FastAPI service và nhiều test module.
- LLM pipeline đã có cơ chế structured JSON output, fallback, kill-switch theo confidence, và tích hợp context định lượng + RAG + news.
- Có risk constraints cứng (anti-FOMO, hard stop-loss cap, ATR position sizing).
- Backtest event-driven đã chú ý look-ahead bias (safe RAG theo mốc thời gian), execution costs (fees + slippage), walk-forward chunks.

### Khoảng trống so với TradingAgents
1. **Thiếu explicit multi-agent orchestration graph**: hiện tại thiên về 1 pipeline LLM (single pass) hơn là debate/committee engine có role tách biệt.
2. **Thiếu “explainable decision trail” theo vai trò agent**: chưa có log phân tầng “ai nói gì, phản biện gì, vì sao approve/reject”.
3. **Thiếu benchmark framework chuẩn hóa theo bộ tasks/market regimes**: chưa thấy bộ benchmark giống kiểu research reproducibility để so model/provider/agent-config.
4. **Thiếu package/CLI experience cấp sản phẩm OSS**: dự án mạnh nội bộ, nhưng trải nghiệm cài/chạy benchmark theo chuẩn OSS public chưa rõ ràng bằng TradingAgents.

---

## 3) Đề xuất cải tiến để “tương xứng” với TradingAgents

## A. Nâng kiến trúc: từ LLM pipeline sang Multi-Agent Graph
- Tách thành các role-agent rõ ràng:
  - FundamentalAgent
  - TechnicalAgent
  - SentimentAgent
  - NewsMacroAgent
  - BullResearcher / BearResearcher
  - TraderAgent
  - RiskManagerAgent
  - PortfolioManagerAgent
- Dùng orchestration graph/state machine (LangGraph style hoặc internal graph runner) để:
  1) Parallel evidence collection  
  2) Debate rounds có giới hạn token/time  
  3) Aggregation + scoring + veto  
  4) Final decision + execution plan

**Tác động**: tăng quality của quyết định trong kịch bản dữ liệu mâu thuẫn, tăng khả năng giải thích cho user.

## B. Decision provenance & auditability (bắt buộc nếu scale)
- Sinh “Decision Card” chuẩn JSON/Markdown cho mỗi lệnh:
  - Inputs snapshot (price/features/news IDs)
  - Agent opinions (bull/bear)
  - Risk overrides đã kích hoạt
  - Final rationale + confidence decomposition
- Lưu vào DB để truy xuất hồi cứu, phục vụ model governance.

**Tác động**: dễ debug, dễ đánh giá drift trong quality reasoning, thuận tiện compliance nội bộ.

## C. Benchmark & evaluation harness
- Thiết kế benchmark matrix:
  - Theo market regime: uptrend/downtrend/sideway/high-volatility
  - Theo horizon: intraday/T+1/swing
  - Theo asset buckets: VN30/midcap/illiquid
- Tách metrics:
  - Trading metrics: CAGR, Sharpe, Sortino, MaxDD, turnover, fee-adjusted alpha
  - LLM metrics: consistency score, contradiction rate, refusal rate, latency cost/token
  - Multi-agent metrics: debate gain (improvement sau debate so với single-agent baseline)

**Tác động**: tạo “bằng chứng định lượng” cho mọi thay đổi model/prompt/provider.

## D. Productization (CLI + configs + reproducibility)
- Bổ sung 1 CLI thống nhất:
  - `analyze`, `debate`, `backtest`, `benchmark`, `report`
- Chuẩn hóa config profile:
  - `provider/openai.yaml`, `provider/gemini.yaml`, `provider/ollama.yaml`
  - `strategy/conservative.yaml`, `strategy/aggressive.yaml`
- Export report tự động (HTML/Markdown) sau mỗi run benchmark.

**Tác động**: giảm friction cho team vận hành; mở đường public/demo dễ hơn.

## E. Data & simulation realism
- Bổ sung hạn chế thị trường VN vào simulator:
  - tick size, lot rules, session breaks, biên độ trần/sàn, thanh khoản depth giả lập
- Thêm stress-test events:
  - gap opens, news shock, liquidity dry-up

**Tác động**: kết quả backtest gần thực tế hơn, giảm over-optimistic bias.

---

## 4) Có thể ứng dụng gì từ `machine-learning-for-trading`?

### Ứng dụng trực tiếp (high value)
1. **Quy trình nghiên cứu alpha có cấu trúc**
   - Lấy tư duy chapter-based workflow để chuẩn hóa pipeline nghiên cứu nội bộ.
2. **Feature research framework**
   - Mở rộng thư viện yếu tố (cross-sectional + time-series) và quy trình validation chống data leakage.
3. **Danh mục mô hình đa dạng**
   - Tăng baseline set (linear, tree ensemble, boosting, Bayesian, deep learning, RL) để benchmark công bằng.
4. **NLP/alt-data integration patterns**
   - Dùng kinh nghiệm xử lý text/sec filings/news embeddings để tăng chiều sâu cho contextual signals.

### Cách áp dụng thực tế
- Không nên “copy code nguyên khối”; nên trích xuất phương pháp và checklist đánh giá.
- Ưu tiên viết lại module theo data schema VN và constraints hệ thống hiện có.

---

## 5) Có thể ứng dụng gì từ `awesome-systematic-trading`?

### Vai trò phù hợp
- Repo này là **meta-resource layer**, không phải framework lõi để tích hợp trực tiếp.

### Ứng dụng thực tế
1. Dùng như “radar” để chọn nhanh thư viện/backtester/OMS phù hợp.
2. Dùng danh sách strategy papers để tạo roadmap R&D theo quý.
3. Dùng danh sách sách/khóa học làm chương trình upskill nội bộ cho team.

### Đề xuất cụ thể
- Lập file `docs/tech_radar_systematic_trading.md` và cập nhật theo tháng:
  - Candidate libs
  - Maturity score
  - Integration cost
  - Decision (Adopt/Hold/Reject)

---

## 6) Lộ trình đề xuất 90 ngày

### 0–30 ngày
- Thiết kế `agent contract` + state schema cho multi-agent graph.
- Implement 3 agent đầu tiên: Technical, NewsMacro, RiskManager.
- Tạo Decision Card v1.

### 31–60 ngày
- Thêm Bull/Bear debate rounds + Trader/PortfolioManager agents.
- Dựng benchmark harness v1 với 3 regimes và bộ metrics cốt lõi.
- CLI command `benchmark` + auto report.

### 61–90 ngày
- So sánh đa provider LLM theo cost/performance/latency.
- Backtest realism upgrade (market microstructure VN).
- Công bố internal leaderboard + model governance dashboard.

---

## 7) Kết luận ngắn
- Repo nội bộ đã có nền tảng production rất tốt (data + ML + risk + API).  
- Khoảng cách lớn nhất với TradingAgents nằm ở **multi-agent coordination + benchmark hóa + product UX cho nghiên cứu**.  
- Hai repo còn lại rất đáng dùng như “nguồn phương pháp và radar công nghệ”, đặc biệt để nâng chuẩn nghiên cứu định lượng và chọn công cụ nhanh hơn.
