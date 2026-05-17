# VN100 Cost and Slippage Validation Report

## Source and Method

- Prediction inputs: official `daily/predicted_vs_actual.csv` and `hourly/predicted_vs_actual.csv`.
- Signal mapping: long when the selected signal predicts upward direction; flat otherwise.
- Return proxy: official target return divided by horizon, so overlapping h-step returns are not compounded as one-period returns.
- Baselines: buy-and-hold, flat/no-trade, always-up, moving-average signal, and previous-direction signal.
- Cost grid: transaction cost bps 5/10/15/20 crossed with slippage bps 5/10/15/20.
- This is a diagnostic proxy using official target returns, not an executable trade simulator with entry/exit prices.

## Model-Signal Results at 10 bps Cost + 10 bps Slippage

| slice_name | row_count | gross_return | net_return | turnover | max_drawdown | profit_factor | win_rate | trade_count | exposure | benchmark_comparison |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hourly_stacking_h1_confidence_0.57 | 2297 | 37.54% | 12.07% | 9.14% | -11.68% | 1.176 | 60.44% | 210 | 9.80% | 18.97% |
| daily_lightgbm_h20_posthoc_bear | 444 | 42.20% | 33.57% | 13.96% | -5.51% | 3.992 | 73.57% | 62 | 70.72% | -5.05% |
| daily_xgboost_h20_posthoc_bear | 444 | 43.66% | 36.20% | 11.71% | -5.50% | 4.182 | 74.19% | 52 | 69.82% | -2.43% |
| full_sweep_daily_stacking_h20_t0.71 | 344 | 70.34% | 67.97% | 3.49% | -11.38% | 2.097 | 62.71% | 12 | 88.08% | 0.55% |
| full_sweep_daily_stacking_h20_t0.70 | 411 | 59.37% | 57.34% | 3.41% | -12.14% | 1.927 | 60.56% | 14 | 87.59% | -4.01% |
| full_sweep_daily_stacking_h20_t0.69 | 482 | 56.34% | 54.55% | 2.90% | -13.15% | 1.784 | 60.05% | 14 | 87.76% | -5.22% |

## Model-Signal Results at 20 bps Cost + 20 bps Slippage

| slice_name | row_count | gross_return | net_return | turnover | max_drawdown | profit_factor | win_rate | trade_count | exposure | benchmark_comparison |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hourly_stacking_h1_confidence_0.57 | 2297 | 37.54% | -8.71% | 9.14% | -16.46% | 0.8455 | 54.22% | 210 | 9.80% | -1.25% |
| daily_lightgbm_h20_posthoc_bear | 444 | 42.20% | 25.45% | 13.96% | -6.33% | 2.951 | 72.29% | 62 | 70.72% | -12.11% |
| daily_xgboost_h20_posthoc_bear | 444 | 43.66% | 29.12% | 11.71% | -6.28% | 3.219 | 72.90% | 52 | 69.82% | -8.45% |
| full_sweep_daily_stacking_h20_t0.71 | 344 | 70.34% | 65.63% | 3.49% | -11.60% | 2.016 | 61.72% | 12 | 88.08% | -0.56% |
| full_sweep_daily_stacking_h20_t0.70 | 411 | 59.37% | 55.33% | 3.41% | -12.41% | 1.849 | 59.44% | 14 | 87.59% | -5.24% |
| full_sweep_daily_stacking_h20_t0.69 | 482 | 56.34% | 52.78% | 2.90% | -13.21% | 1.721 | 59.10% | 14 | 87.76% | -6.32% |

## Readiness Interpretation

- Figure status: ready.
- Positive model-signal net return at the 10/10 bps diagnostic grid: 6 of 6 slices.
- Practical trading readiness is not established because official artifacts still lack execution prices, liquidity filters, fills, and deployment constraints.
- Weak or cost-sensitive rows should be treated as evidence against broad trading-readiness claims.

## Output Artifacts

- Summary CSV: `reports/generated/evidence_gap_closure/vn100_cost_slippage_summary.csv`.
- Trade list CSV: `reports/generated/evidence_gap_closure/vn100_trade_list.csv`.
- Equity curve CSV: `reports/generated/evidence_gap_closure/vn100_equity_curve.csv`.
- Equity curve figure: `reports/generated/evidence_gap_closure/vn100_equity_curve.png`.

## Claim Boundary

This report adds cost/slippage-aware diagnostic artifacts. It does not justify profitability, investment suitability,
or live trading readiness claims.
