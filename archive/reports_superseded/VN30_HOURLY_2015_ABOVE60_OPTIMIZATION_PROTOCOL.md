# VN30 Hourly 2015 - Above 60% Optimization Protocol

## 1. Target Hierarchy

| Priority | Target | Requirements |
|----------|--------|-------------|
| A | Full-universe global accuracy >60% | All 30 tickers, full evaluation coverage, no filtering |
| B | Model/horizon full-universe >60% | All 30 tickers for specific model/horizon |
| C | Coverage-qualified confidence-filtered >60% | Coverage >= 30%, rows >= 1000, threshold pre-registered |
| D | Ticker/subset >60% | Exploratory only, not headline |

## 2. Leakage Control

- **Training labels**: target_timestamp <= 2024-12-31 23:59:59
- **Validation labels**: 2024-01-01 to 2024-12-31 (for hyperparameter/threshold selection)
- **Final evaluation labels**: 2025-01-01 to 2026-05-14 (untouched until final scoring)
- **Hyperparameter selection**: Only on validation period (2024)
- **Threshold selection**: Only on validation period (2024)
- **Final scoring**: Single pass on evaluation period (2025-2026)

## 3. Validation Split

| Split | Period | Purpose |
|-------|--------|---------|
| Train | 2015-01-01 to 2023-12-31 | Model training |
| Validation | 2024-01-01 to 2024-12-31 | Hyperparameter/threshold selection |
| Evaluation | 2025-01-01 to 2026-05-14 | Final scoring (untouched until end) |

## 4. Candidate Model Families

- LightGBM (classifier)
- XGBoost (classifier)
- Random Forest (classifier)
- Stacking v2 (time-series-safe OOF base predictions only)
- Existing baselines only

## 5. Feature Sets

| Set | Description |
|-----|-------------|
| A | Existing benchmark features (returns, RSI, MACD, volume, time) |
| B | Stock lagged features (lagged returns, rolling vol, rolling momentum, volume change, hour/session) |
| C | Lagged market context (VNINDEX, VN30, HNXINDEX lagged returns, rolling market vol) |
| D | Combined stock + index context (B + C) |

## 6. Horizons

| Horizon | Priority |
|---------|----------|
| h=20 | Primary (current best: 54.58%) |
| h=8 | Secondary |
| h=4 | Tertiary |
| h=1 | Baseline |

## 7. Success Criteria

- **Primary**: Global >60% on final evaluation
- **Secondary**: Coverage-qualified >60% with coverage >= 30% and rows >= 1000
- **Tertiary**: Model/horizon >60% on full universe
- **Report all failures honestly**

## 8. Boundary

- No trading-readiness, profitability, or live deployment claim
- No paper or DOCX generation
- No new data fetching
- No main branch modifications