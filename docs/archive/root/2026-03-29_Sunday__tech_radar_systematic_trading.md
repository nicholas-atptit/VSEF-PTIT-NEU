# Tech Radar & R&D Pipeline cho Systematic Trading (Tháng 3/2026)
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Historical archive |
| Created / authored | Sunday, 2026-03-29 04:02:27 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-04-28 22:28:23 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `8800ce6e3780c7978856737e70cb5e3b999eacee` |
| Timestamp source | Git history |
| Status | Historical reference |

Mục tiêu tài liệu này là duy trì quy trình "Adopt / Hold / Reject" (Áp dụng / Chờ / Loại bỏ) theo chu kỳ định kỳ hàng tháng để thẩm định các công nghệ và chiến thuật từ `awesome-systematic-trading`.

## I. Tech Radar (Thẩm định định kỳ)

### 1. Phân bổ rủi ro (Allocator) - FinRL-Trading (RL Models)
- **Use-case**: Sử dụng PPO/SAC outputs làm Weight Vector cho danh mục VN30.
- **Effort**: Vừa (M) - Cần convert state/action sang custom env.
- **Status**: **ADOPT** (Bắt đầu ở giai đoạn F2 lộ trình 90 ngày).
- **Due Date**: 30/04/2026
- **Owner**: Quant Team

### 2. Time-Series Feature Extraction (tsfresh / catch22)
- **Use-case**: Tự động sinh hàng nghìn feature từ nến để feed vào cây Decision Tree thay vì chart TA truyền thống.
- **Effort**: Cao (L) - Rất tốn RAM và khả năng Overfitting ở VN Market lớn.
- **Status**: **HOLD** (Đợi có report lợi nhuận ổn định hơn).
- **Due Date**: Tái xem xét 06/2026

### 3. Agentic Risk Engine
- **Use-case**: LLM agent quét tin vĩ mô + news flash, kích hoạt chế độ `KILL SWITCH` (ban lệnh Buy, chỉ Sell).
- **Effort**: Vừa (M).
- **Status**: **ADOPT** (Đã code khung trong `risk_manager_agent.py`).
- **Due Date**: 15/04/2026

## II. R&D Pipeline Backlog (Quý 2/2026)

### Thử nghiệm 1: A/B Testing giữa EqualWeightAllocator vs MultiAgent
- **Hypothesis**: MultiAgent Debate sẽ giảm Max Drawdown nhờ Risk Veto so với EqualWeight mù quáng.
- **Data required**: VN30 từ 2021-2025 (~1200 phiên).
- **Metric**: Sortino / MaxDD.

### Thử nghiệm 2: Paper Trading - Độ trượt giá (Slippage) khi dùng Weight-Centric Execution
- **Hypothesis**: Lệnh sinh ra từ hệ số weight của Portfolio Agent sẽ chịu trượt giá <= 0.2% trên tổng giá trị khớp lệnh ở nhóm Midcap.
- **Data required**: Paper trading API (VPS/TCBS) 100 giao dịch.
- **Metric**: Actual vs Backtest Deviation.

--- 
*Tài liệu này nên được bổ sung hàng tháng, bất kỳ công nghệ nào mới từ `machine-learning-for-trading` hay `awesome-systematic-trading` đều phải qua phễu Radar này trước khi viết code Production.*
