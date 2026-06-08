# VN30 V5 Strategy Relock Risk Hardening Result Summary

## Relock

- Relock status: no_validation_trade_or_positive_baseline_variant.
- Selected threshold: 0.54.
- Selected max positions: 3.
- Selected reporting cost/slippage: 10 bps / 5 bps.
- Selected risk controls: max_exposure=0.7, volatility_filter=high_vol_filter, market_drawdown_filter=on, stop_loss=-0.05, take_profit=0.08, cooldown=1.

## Best Risk-Hardened Final Diagnostic

- Risk variant: `v5risk_00242`.
- Threshold/max positions/max exposure: 0.54 / 3 / 0.5.
- Controls: volatility=off, market_drawdown=off, stop_loss=-0.03, take_profit=none, cooldown=0.
- Cost/slippage: 0 bps / 5 bps.
- Return/Sharpe/drawdown: 61.15% / 1.5797111276768967 / -33.81%.
- Trade count/exposure/turnover: 17 / 0.4998874712200795 / 7.816363555995046.

## Required Answers

1. Threshold selected by relock logic: 0.54.
2. Did risk hardening reduce max drawdown below V4 -38.42%: true.
3. Any variant kept return above buy-and-hold VN30 after costs: true (matching cost/slippage baseline=53.35%).
4. Any variant beat random same-turnover after costs: true (best total-return variant matching random baseline=64.91%; strongest matching-baseline outperformer=`v5risk_00252` at 10 bps / 20 bps, baseline_delta=+0.31 pp).
5. Cost/slippage sensitivity: best=[{'threshold': 0.54, 'max_positions': 3, 'max_exposure': 0.5, 'volatility_filter': 'off', 'market_drawdown_filter': 'off', 'stop_loss_proxy': '-0.03', 'take_profit_proxy': 'none', 'cooldown_after_loss': 0, 'cost_bps': 0, 'slippage_bps': 0, 'total_return': 0.6157475707891447, 'sharpe': 1.5853758680713703, 'max_drawdown': -0.3379268226184635, 'trade_count': 17}], worst=[{'threshold': 0.54, 'max_positions': 3, 'max_exposure': 0.5, 'volatility_filter': 'off', 'market_drawdown_filter': 'off', 'stop_loss_proxy': 'none', 'take_profit_proxy': '0.05', 'cooldown_after_loss': 1, 'cost_bps': 30, 'slippage_bps': 20, 'total_return': -0.030549506619891, 'sharpe': -0.7378503734284314, 'max_drawdown': -0.0678814913294801, 'trade_count': 15}].
6. Performance survives best-trade removal: return_positive=true, sharpe_positive=true.
7. Performance depends on a small number of trades: no.
8. Claimable or exploratory: exploratory_not_claimable; future_blind_required.
9. Paper-safe wording: offline diagnostic only; no trading, profitability, recommendation, live deployment, or claimable strategy claim.

Paper-safe wording:

> VN30 V5 froze the V4 calibrated-logistic compact-stable h40 strategy family and ran validation-only relock auditing plus final-period risk-hardening diagnostics. Because final-period strategy ranking remains exploratory and validation relock did not establish a claimable strategy, no strategy result is claimable. Future-blind confirmation is required before any stronger statement.
