# Kế hoạch nâng cấp hệ thống (local + Ollama + ứng dụng 3 repo tham chiếu)
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Historical archive |
| Created / authored | Sunday, 2026-03-29 02:17:38 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-04-28 22:28:23 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `8800ce6e3780c7978856737e70cb5e3b999eacee` |
| Timestamp source | Git history |
| Status | Historical reference |

**Ngày lập:** 2026-03-28  
**Mục tiêu:** Nâng cấp hệ thống hiện tại để đạt chuẩn gần TradingAgents, chạy local-first với Ollama, đồng thời hấp thụ phương pháp từ `machine-learning-for-trading` và radar công nghệ từ `awesome-systematic-trading`.

---

## 1) Hiện trạng kỹ thuật quan trọng (để bám vào triển khai)

- Hệ thống đã có cấu hình đa provider LLM, bao gồm `ollama`, `openai`, `groq`, `gemini`.
- Đã có `docker-compose` service `ollama` với port `11434`.
- LLM pipeline đã hỗ trợ route model theo provider và JSON output có kill-switch.
- Đã có nền tảng streaming + risk + backtest để gắn multi-agent vào mà không cần đập đi làm lại.

=> Vì vậy kế hoạch sẽ là **nâng cấp theo lớp (incremental refactor), không rewrite toàn bộ**.

---

## 2) North Star (định nghĩa “xong”)

Sau 90 ngày, hệ thống đạt các tiêu chí:

1. **Local-first**: chạy end-to-end với Ollama không cần cloud API.  
2. **Multi-agent**: có pipeline debate Bull/Bear + Risk veto + Portfolio allocation.  
3. **Reproducible benchmark**: chạy được benchmark theo market regimes với report tự động.  
4. **Decision audit**: mỗi quyết định có Decision Card truy vết đầy đủ.  
5. **Research cadence**: có tech-radar + backlog chiến lược cập nhật định kỳ.

---

## 3) Kế hoạch triển khai theo 4 workstreams

## WS-A: Local Ollama Runtime (hạ tầng chạy local)

### A1. Chuẩn hóa profile cấu hình local
Tạo file `.env.local_ollama` (hoặc profile tương đương) với tối thiểu:

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=ollama
OLLAMA_MODEL_NAME=qwen2.5:7b
OPENAI_API_KEY=
GEMINI_API_KEY=
GROQ_API_KEY=
```

### A2. Chuỗi lệnh khởi động local chuẩn
```bash
# 1) Hạ tầng nền
docker compose up -d timescaledb chromadb zookeeper kafka redis ollama

# 2) Kéo model Ollama
docker exec -it algo_ollama ollama pull qwen2.5:7b

# 3) Activate env và cài libs
python -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt

# 4) Chạy API local với profile Ollama
set -a && source .env.local_ollama && set +a
uvicorn src.api.main:app --host 0.0.0.0 --port 8888
```

### A3. Tiêu chí nghiệm thu WS-A
- `/api/v1/analyze?ticker=SSI` chạy thành công bằng model Ollama.
- Log ghi nhận provider=`ollama` và model local.
- Không gọi API key cloud trong runtime local.

---

## WS-B: Multi-Agent nâng cấp theo TradingAgents (ứng dụng trực tiếp)

## B1. Thiết kế Agent Graph v1
Tạo orchestration layer mới (đề xuất `src/agents/`):

- `technical_agent.py`
- `news_macro_agent.py`
- `bull_researcher.py`
- `bear_researcher.py`
- `risk_manager_agent.py`
- `portfolio_manager_agent.py`
- `orchestrator.py`

Luồng v1:
1. Technical + News/Macro chạy song song.  
2. Bull/Bear nhận evidence và debate 2 rounds.  
3. RiskManager áp veto (volatility shock, confidence floor, conflict flag).  
4. PortfolioManager xuất allocation + order constraints.

## B2. Decision Card (audit trail bắt buộc)
Tạo schema `DecisionCard` lưu:
- metadata run (ticker, timestamp, provider, model)
- evidence IDs (news/document/features snapshot)
- bull thesis / bear thesis
- risk veto flags
- final action + confidence decomposition

Lưu vào DB (Timescale hoặc bảng audit riêng) + export JSON trong `tmp/reports/`.

## B3. Tiêu chí nghiệm thu WS-B
- Có endpoint mới (ví dụ `/api/v2/debate`) trả về Decision Card.
- Có thể replay lại một quyết định từ dữ liệu lịch sử.
- Tối thiểu 3 test case debate pass/fail rõ ràng.

---

## WS-C: Benchmark Harness (ứng dụng từ machine-learning-for-trading)

## C1. Benchmark matrix
Xây bộ benchmark cố định:
- **Regime**: uptrend / downtrend / sideway / high-vol
- **Horizon**: intraday / T+1 / swing
- **Universe**: VN30 / Midcap / Illiquid bucket

## C2. Metrics chuẩn
- Trading: CAGR, Sharpe, Sortino, MaxDD, Win rate, Turnover, Fee-adjusted PnL
- Agent quality: contradiction rate, abstain rate, decision latency, token cost proxy
- Robustness: performance dispersion theo regime

## C3. CLI benchmark
Tạo command (đề xuất):
```bash
python -m src.benchmark.run --profile local_ollama --regime all --horizon swing
```
Output:
- `reports/benchmark_<date>.md`
- `reports/benchmark_<date>.json`

## C4. Tiêu chí nghiệm thu WS-C
- Chạy benchmark 1 lệnh, ra report đầy đủ.
- So sánh được baseline single-agent vs multi-agent.
- Có bảng leaderboard nội bộ theo profile/model.

---

## WS-D: Tech Radar & R&D Pipeline (ứng dụng từ awesome-systematic-trading)

## D1. Thiết lập quy trình “Adopt / Hold / Reject” hàng tháng
Tạo tài liệu `docs/tech_radar_systematic_trading.md` gồm:
- candidate tool/library
- use-case fit với VN market
- integration effort (S/M/L)
- decision + owner + due date

## D2. R&D backlog theo quý
Nguồn ý tưởng: strategy papers, backtester libs, risk engine libs trong awesome list.

Mẫu backlog item:
- Hypothesis
- Required data
- Experiment design
- Success metric
- Go/No-go decision

## D3. Tiêu chí nghiệm thu WS-D
- Mỗi tháng tối thiểu 5 candidates được review.
- Mỗi quý tối thiểu 2 thử nghiệm chiến lược hoàn chỉnh.

---

## 4) Lộ trình thời gian chi tiết

### Phase 1 (Tuần 1–2): ổn định local Ollama + profile hóa
- Hoàn thiện `.env.local_ollama` + scripts khởi động.
- Viết smoke test local cho `/predict`, `/analyze`, `/execute`.

### Phase 2 (Tuần 3–6): multi-agent v1
- Implement WS-B (graph + debate + decision card).
- Bổ sung endpoint API v2.

### Phase 3 (Tuần 7–10): benchmark hóa
- Implement WS-C benchmark runner + report.
- Dựng baseline A/B: single-agent vs multi-agent.

### Phase 4 (Tuần 11–12): hardening + R&D governance
- Tích hợp WS-D tech radar.
- Chốt KPI quý + dashboard theo dõi.

---

## 5) KPI theo dõi (đề xuất)

## KPI vận hành
- API success rate >= 99%
- P95 latency `/analyze` <= 8s (local Ollama 7B)
- Debate pipeline timeout rate <= 2%

## KPI chất lượng quyết định
- Contradiction rate giảm >= 20% so với single-agent baseline
- MaxDD giảm >= 10% ở cùng turnover
- Fee-adjusted Sharpe tăng >= 10%

## KPI sản phẩm/research
- 100% quyết định có Decision Card
- 100% benchmark run có artifact report
- >= 2 experiment chiến lược / quý

---

## 6) Risk register & phương án giảm rủi ro

1. **Ollama quá chậm khi debate nhiều vòng**  
   - Mitigation: giảm context, cache evidence, giới hạn rounds, dùng model nhỏ cho agent phụ.

2. **Agent disagreement quá cao → không ra quyết định**  
   - Mitigation: thêm tie-breaker policy (risk-first), confidence floor, abstain action hợp lệ.

3. **Benchmark tốn tài nguyên & thời gian**  
   - Mitigation: tách quick benchmark (nightly) và full benchmark (weekly).

4. **Overfitting vào 1 regime thị trường**  
   - Mitigation: bắt buộc evaluate đa regime + walk-forward rolling windows.

---

## 7) Checklist triển khai ngay (7 ngày tới)

- [ ] Tạo `.env.local_ollama` và script `scripts/run_local_ollama.sh`
- [ ] Pull model `qwen2.5:7b` cho service `algo_ollama`
- [ ] Tạo skeleton `src/agents/` + orchestrator interface
- [ ] Thiết kế `DecisionCard` schema + persistence
- [ ] Draft benchmark config `configs/benchmark/default.yaml`
- [ ] Tạo tài liệu `docs/tech_radar_systematic_trading.md`

---

## 8) Kết luận điều hành

Đây là kế hoạch có thể triển khai ngay trên codebase hiện tại: giữ nguyên nền tảng production, thêm lớp multi-agent và benchmark để tăng chất lượng quyết định; đồng thời khóa đường vận hành local với Ollama để chủ động chi phí/tốc độ thử nghiệm.
