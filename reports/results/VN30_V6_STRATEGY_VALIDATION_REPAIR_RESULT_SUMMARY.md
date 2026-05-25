# VN30 V6 Strategy Validation Repair Result Summary

## Root Cause

- Validation prediction rows: 28830.
- Non-null predicted probabilities: 28830.
- Validation predicted probability range: 0.1731764577829716 to 0.47905665397048364.
- V5 threshold band: 0.54 to 0.56.
- V5 threshold-band max validation trade count: 0.

## Required Answers

1. Why did V5 validation produce zero trades: the validation predicted-probability maximum was below the V5 0.54-0.56 threshold band, so the confidence-threshold step produced zero candidates.
2. Were validation predictions missing, misaligned, or below threshold: predictions were present and aligned; they were below the V5 threshold band.
3. Did market regime or risk filters block all trades: no. The zero happened before those filters; market-regime filtering alone did not cause zero trades.
4. Was there an implementation bug: false. This was a threshold feasibility/protocol issue, not a join or timestamp bug.
5. Can validation relock be repaired: trade generation can be repaired diagnostically by lowering the validation threshold band, but claimable relock is not repaired because the selected feasible validation rows do not beat the strongest validation baseline.
6. Validation-selected diagnostic settings: `v6diag_t0p440__mp5__regoff__voloff__ddoff` threshold=0.44, max_positions=5, market_regime_filter=off, volatility_filter=off, drawdown_filter=off, return=0.98%, trade_count=20, baseline_delta=-19.11 pp.
7. Does any repaired validation strategy beat baseline: false.
8. Does the claim boundary change: no.
9. What should be done next: pre-lock a validation-governed threshold/calibration rule, require >=10 validation trades plus strongest-baseline outperformance, then future-blind confirm.

## Validation Baseline Context

- Strongest validation strategy baseline at diagnostic cost/slippage: [{'baseline_name': 'equal_weight_vn30_stock_basket', 'split': 'validation', 'max_positions': nan, 'total_return': 0.2008753642863744, 'cost_bps': 10.0, 'slippage_bps': 5.0, 'annualized_return': nan, 'sharpe': nan, 'sortino': nan, 'max_drawdown': nan, 'calmar': nan, 'win_rate': nan, 'profit_factor': nan, 'average_trade_return': nan, 'trade_count': nan, 'exposure_ratio': nan, 'turnover': nan, 'cost_adjusted_return': nan}].
- Feasible validation strategies with >=10 trades and positive after-cost return: 48.

Paper-safe wording:

> VN30 V6 found that the V5 validation relock failed because the frozen candidate's validation probability distribution never reached the 0.54-0.56 threshold band. Predictions and split timestamps were present and aligned, and the market-regime/risk filters did not cause the zero-trade outcome. Lower validation thresholds can generate diagnostic trades, but they do not establish a claimable strategy relock because strongest-baseline outperformance is not satisfied. The current claim boundary is unchanged.
