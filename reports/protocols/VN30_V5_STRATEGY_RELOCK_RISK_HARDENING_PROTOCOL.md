# VN30 V5 Strategy Relock Risk Hardening Protocol

## Scope

- VN30 stock hourly strategy diagnostic only.
- Frozen family: calibrated_logistic / absolute_direction / compact_stable_features / h40.
- Threshold candidates: [0.54, 0.545, 0.55, 0.555, 0.56].
- Strategy template: long_only_market_regime_filter.
- Max positions: [3, 5].
- No broad model tuning or final-period claimable ranking.

## Relock Rule

- Selection uses validation-period strategy diagnostics only.
- If no validation variant has sufficient trades and positive baseline-relative return, the run records a non-claimable predeclared fallback and keeps final-period risk rankings exploratory.
- Final-period optimization is not claimable.

## Risk Grid

- max_exposure: [0.5, 0.7, 1.0].
- volatility_filter: ['off', 'high_vol_filter'].
- market_drawdown_filter: ['off', 'on'].
- stop_loss_proxy: none, -3%, -5%, -7%.
- take_profit_proxy: none, +5%, +8%, +10%.
- cooldown_after_loss: [0, 1, 2].
- cost_bps: [0, 5, 10, 20, 30].
- slippage_bps: [0, 5, 10, 20].

## Claim Boundary

- Offline diagnostic simulation only.
- No trading, profitability, BUY/SELL, recommendation, live deployment, DOCX, push, merge, or tag.
