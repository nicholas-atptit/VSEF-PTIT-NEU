# 🛡️ Repository Audit Report (Consolidated)

## 1. CURRENT ARCHITECTURE
- **System**: Hybrid Agentic Trading System (v5.3.1).
- **Core Strategy**: Multi-agent consensus (Technical + Sentiment).
- **Design Pattern**: Domain-driven ML pipeline (Features -> Labels -> Trainer -> Inference -> Backtest).

## 2. MAIN ENTRY POINTS
- Data Ingestion: `scripts/sync_all_data.py`
- ML Training: `scripts/train_ml_tickers.py`
- Real-time Inference: `scripts/per_session_predict.py`
- Paper Trading: `src/ml/backtest/paper.py`

## 3. DATA FLOW
- **Ingestion**: `vnstock` API -> TimescaleDB (raw_prices).
- **Processing**: DB -> Ticker CSVs (`data/daily_market_split_data/`).
- **Feature Engineering**: Ticker CSVs -> Feature Matrices (`src/ml/feature_engineering.py`).
- **Training**: Features -> Model Artifacts (`models/<TICKER>/`).

## 4. SAVE/LOAD LOGIC
- **Prices**: Database and local split-CSVs.
- **Models**: Joblib serialization in ticker-named subdirectories.
- **Predictions**: JSON snapshots in `data/latest_predictions.json`.

## 5. COMPLETED COMPONENTS
- [x] Multi-Horizon Feature Engineering (1w, 1m, 6m).
- [x] Advanced Volatility Estimators (Yang-Zhang).
- [x] Time-Series Purged Cross-Validation.
- [x] Event-Driven Paper Trading Simulation with Slippage.

## 6. INCOMPLETE / WEAK COMPONENTS
- [ ] Centralized Factor Registry (Factors are currently ad-hoc).
- [ ] Unit-tested Causal Labels (Labels at risk of Kalman-smoothing bias).
- [ ] Efficient Data I/O (CSV splitting creates disk bottlenecks).

## 7. DUPLICATE / CONFUSING FILES
- `scripts/準備_fundamental_features.py` (Verify usage).
- `scripts/prepare_vip_list.py` (Redundant).
- `scripts/import_historical_csv.py` (Overlaps with `sync_all_data`).

## 8. TECHNICAL DEBT & RISK
- **P0 Risk**: Potential smoothed price target in custom label generation.
- **P1 Debt**: 1600+ ticker model directories degrade repository performance.
- **P2 Debt**: Hardcoded magic numbers scattered in operational scripts.

---
*For detailed breakdowns, see:*
- [Technical Prediction Audit](file:///h:/AI-ML-LLM%20in%20Stock_march26_PTIT_NEU/reports/audits/technical_prediction_audit.md)
- [Data Flow Audit](file:///h:/AI-ML-LLM%20in%20Stock_march26_PTIT_NEU/reports/audits/data_flow_audit.md)
- [Backtest Risk Audit](file:///h:/AI-ML-LLM%20in%20Stock_march26_PTIT_NEU/reports/audits/backtest_risk_audit.md)
- [Repo Structure Audit](file:///h:/AI-ML-LLM%20in%20Stock_march26_PTIT_NEU/reports/audits/repo_structure_audit.md)
