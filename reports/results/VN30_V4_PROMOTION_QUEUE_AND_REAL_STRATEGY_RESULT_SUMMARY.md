# VN30 V4 Promotion Queue And Real Strategy Result Summary

## Promotion Queue Verdict

- 65.51 candidate cluster verdict: stable_candidate_cluster.
- Stable clusters found: 229.
- Isolated one-off clusters found: 24.

## Rolling-Origin Confirmation

- Best rolling-origin candidate: `forced_v3_calibrated_compact_h40__t0p540`.
- Best rolling-origin split: final.
- Accuracy/lift: 65.51% / +14.80 pp.
- Rolling-origin robust candidates: 5.

## Strategy Diagnostics

- Best after-cost strategy: `v4strat_002762`.
- Candidate/template: `forced_v3_calibrated_compact_h40__t0p540` / long_only_market_regime_filter.
- Cost/slippage/max positions: 0 bps / 5 bps / 3.
- Total return: 120.12%.
- Sharpe: 1.0720175442435094.
- Max drawdown: -38.42%.
- Exposure/turnover/trades: 0.699817500375772 / 18.40592597374972 / 17.

## Required Answers

1. 65.51 candidate stable cluster or one-off: stable_candidate_cluster.
2. Any frozen candidate show rolling-origin robustness: true.
3. Best real strategy after costs: `v4strat_002762`.
4. Any strategy beat buy-and-hold VN30: true.
5. Any strategy beat equal-weight VN30: true.
6. Any strategy beat random same-turnover: true.
7. Cost/slippage sensitivity: best grid {'cost_bps': 0.0, 'slippage_bps': 0.0, 'total_return': 1.2122320052784996}; worst grid {'cost_bps': 30.0, 'slippage_bps': 20.0, 'total_return': 1.1035106581753356}.
8. Any result becomes claimable: false.
9. Exploratory results: promotion queue clusters, frozen final-ranked candidates, rolling-origin confirmations, and all strategy diagnostics.
10. Paper-safe claim boundary: offline diagnostic only; no trading, profitability, recommendation, live deployment, or claimable champion replacement is made.

Paper-safe wording:

> VN30 V4 mined the v3 promotion queue and evaluated frozen VN30 hourly candidate signals in offline fixed-horizon long-only strategy diagnostics with explicit cost and slippage assumptions. The current 61.61% strict-replay directional champion is not replaced. Final-ranked and strategy results remain exploratory diagnostics requiring re-lock or future-blind confirmation before any claim.
