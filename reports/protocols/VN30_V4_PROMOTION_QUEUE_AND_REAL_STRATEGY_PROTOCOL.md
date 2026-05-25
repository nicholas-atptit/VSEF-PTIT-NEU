# VN30 V4 Promotion Queue And Real Strategy Protocol

## Scope

- VN30 stock hourly diagnostic benchmark only.
- V3 promotion queue is mined before any simulation.
- Frozen candidates are evaluated without changing model parameters, features, horizons, or thresholds.
- Index data is used only as lagged market-context features or baseline VN30 index buy-and-hold comparison.
- Strategy results are offline diagnostics only.
- Out of scope: trading, profitability claim, BUY/SELL, recommendation, live deployment, DOCX, push, merge, tag, VN100.

## Split Discipline

- Train rows require feature_timestamp <= `2023-12-31 23:59:59` and target_timestamp <= `2023-12-31 23:59:59`.
- Validation rows require feature_timestamp and target_timestamp from `2024-01-01 00:00:00` through `2024-12-31 23:59:59`.
- Final rows require feature_timestamp and target_timestamp >= `2025-01-01 00:00:00`.
- Final-ranked v3 candidates remain exploratory unless re-locked or future-blind confirmed.

## Strategy Assumptions

- Initial capital is normalized to 1.0.
- No leverage and no shorting.
- Max positions: [3, 5, 10].
- Equal-weight entries, fixed-horizon exits, cash earns 0.
- Transaction cost bps: [0, 5, 10, 20, 30].
- Slippage bps: [0, 5, 10, 20].
- Templates: long_only_confidence, long_only_market_regime_filter, relative_strength_rotation, cash_when_uncertain, regime_gated_strategy.
