# Backtest & Risk Engine Audit

## A. Backtest Framework Audit

### Implementation
- **File**: `src/ml/backtest/paper.py`.
- **Framework**: Event-driven `PaperTradingEngine`.
- **Latency**: Profiled for WebSocket, ML Compute, LLM Inference, and Risk Engine.

### Finding: Simulation Realism (High)
The use of `simulate_execution_cost(entry_price, volume, action)` is a high-quality inclusion in `src/ml/backtest/event_driven.py`.
- **Features**: Fees, slippage calculation based on liquidity/volume.
- **Risk**: If liquidity estimates in `simulate_execution_cost` are static, they may over-estimate fill quality during low-liquidity periods.

---

## B. Logic Validity & Leakage Audit

### Observation: Forward Bias Check
The `PaperTradingEngine` uses a `run_single_cycle` method that loads current market snapshots. 
- **Time Safety**: It loads `ticker` OHLCV at runtime. 
- **Leakage Risk**: If `load_ohlcv_from_db` includes the current day's close *before* the day has ended, there is leakage. 
- **Check**: `_fetch_market_snapshot` uses the canonical `VnstockAdapter`/`vnstock_data` OHLCV path for runtime snapshots.

---

## C. Risk Management Integration

### Audit of `apply_risk_constraints`
- **File**: `src/engine/risk.py`.
- **Logic**: Uses ATR-14 for position sizing (Line 158 of `paper.py`).
- **Kill Switch**: `fomo_check_passed` veto logic correctly implemented (Line 169).

### Finding: Benchmark Alignment (Medium)
The paper trading summary `get_portfolio_summary` (Line 252) reports P&L but lacks a direct beta-adjusted benchmark (VNINDEX) comparison.
- **Problem**: Positive returns in a bull market may mask poor alpha model performance.
- **Recommendation**: Integrate the `VNINDEX` benchmark into the `PaperTradingEngine` for every cycle.

---

## D. Execution Gap Analysis

### Current Status
- **Pros**: Event-driven, realistic costs, risk-aware.
- **Cons**: Lack of walk-forward retraining simulation (models are static).
- **Recommendation**: Implement a `WalkForwardEngine` that retrains models every N sessions during long-span backtests to capture regime shifts.
