# VN30 Hourly 2015 Post-Benchmark Diagnostics Summary

## Base Benchmark Result

- Active universe: VN30 January 2025 review.
- Active ticker count: 30/30 represented.
- Frequency: hourly only.
- Train cutoff: 2024-12-31 23:59:59.
- Evaluation period: 2025-01-01 00:00:00 to 2026-05-14 00:00:00.
- Total predictions: 77,692.
- Global directional accuracy: 51.34%.
- Global 60% threshold: not passed.
- Best base model/horizon: LightGBM horizon 20, 54.58% over 4,599 observations.

## Confidence Sweep Result

- Output: `reports/generated/vn30_hourly_2015_benchmark/confidence/vn30_confidence_sweep_summary.csv`.
- Coverage floors evaluated: 50%, 40%, 30%.
- Best 50% coverage-floor slice: XGBoost h=20, threshold 0.750, 2,345 rows, 50.99% coverage, 57.48% accuracy.
- Best 40% coverage-floor slice: XGBoost h=20, threshold 0.800, 1,840 rows, 40.01% coverage, 57.61% accuracy.
- Best 30% coverage-floor slice: Random Forest h=20, threshold 0.675, 1,583 rows, 34.42% coverage, 59.19% accuracy.
- Coverage-ok slices at or above 60%: 0.
- Interpretation: confidence filtering improves some h=20 slices but does not create a coverage-qualified 60% claim.

## Regime Diagnostics Result

- Output: `reports/generated/vn30_hourly_2015_benchmark/regime/vn30_regime_diagnostics_summary.csv`.
- Regime diagnostic rows: 80.
- Label source: existing benchmark prediction labels only.
- Best row: XGBoost h=20 in `return_regime=bear`, 1,539 observations, 58.93% accuracy.
- Regime rows at or above 60%: 0.
- Interpretation: regime slicing does not produce a 60% diagnostic row. Labels remain benchmark-internal regime proxies.

## Significance Result

- Output: `reports/generated/vn30_hourly_2015_benchmark/significance/vn30_significance_summary.csv`.
- Tests: one-sided binomial/sign tests against 50% directional null by model and horizon.
- Rows tested: 16.
- Unadjusted rows above 50% at alpha=0.05: 6.
- Bonferroni-adjusted rows above 50%: 3.
- Strongest row: LightGBM h=20, 54.58% accuracy, p=2.87e-10.
- Interpretation: h=20 tree-based rows are statistically above 50%, but still below the 60% benchmark threshold.

## Cost/Slippage Proxy Result

- Output: `reports/generated/vn30_hourly_2015_benchmark/cost_slippage/vn30_cost_slippage_proxy_summary.csv`.
- Strategies: long/flat and direction-following long/short.
- Cost grid: 0, 5, 10, 20 bps transaction cost crossed with 0, 5, 10, 20 bps slippage.
- Best 10 bps cost + 10 bps slippage proxy row: Random Forest h=20 long/flat.
- Standard-cost proxy metrics for that row: net return proxy 24,479.55%, max drawdown -79.55%, win rate 33.42%, trade count 522, exposure 60.21%.
- Interpretation: this is a non-executable overlapping-horizon signal proxy. It must not be described as profitability or tradable performance.

## Limitations

- The base benchmark failed the configured 60% global threshold.
- Confidence sweep is post-hoc and did not produce a coverage-qualified 60% slice.
- Regime labels were not newly validated against an independent ex-ante market-regime framework.
- Significance against 50% does not imply a 60% method or economic value.
- Cost/slippage proxy uses prediction rows as signal diagnostics and does not model order book depth, fill probability, liquidity, sizing, taxes, borrow costs, or execution policy.
- No daily data, resampling, new data fetch, new universe, new model family, paper, or DOCX was introduced.

## Paper Claim Boundary Recommendation

The paper can safely state that the VN30 hourly Jan-2025-universe benchmark is complete, covers all 30 tickers, and reaches 51.34% global directional accuracy without passing the 60% threshold. It can conditionally discuss h=20 statistical signal and baseline deltas as diagnostic evidence only. It should not claim trading readiness, profitability, live deployability, full-market suitability, or a stable 60%+ forecasting method.
