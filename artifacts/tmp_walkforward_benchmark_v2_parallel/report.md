# Walk-Forward Forecasting Report

## Experiment Setup
- Ticker universe: ACB, FPT
- Data window: 2018-01-01 through 2025-01-31
- Historical training window baseline: 2018-01-01 through 2024-12-31
- Rolling forecast window: 2025-01-01 through 2025-01-15
- Horizons: short_5d, short_10d, short_20d, short_30d, long_3m, long_6m
- Step sizes completed: 1, 2
- Models actually run: cart, lightgbm, xgboost
- Stacking method used: prequential Ridge regression on prior out-of-sample base-model predictions, pooled by horizon and step size, with mean fallback before enough realized rows exist.
- Actual comparison: each forecast was aligned to the close-to-close realized forward-return window from the feature cutoff date to the horizon target trading date.
- Evaluation-eligible rows: only predictions whose full realized horizon existed inside the fetched 2026-03-31 data boundary were scored; later rows were kept in output tables but excluded from aggregated metrics.

## Model Coverage
- Fetched history coverage by ticker:
  - ACB: source=csv_fallback, rows=1024, available_range=2020-12-21 through 2025-01-24
  - FPT: source=csv_fallback, rows=1024, available_range=2020-12-21 through 2025-01-24
- No algorithms were skipped after environment capability checks.

## Best Performers
- step_size=1: stacking overall RMSE=0.044817, MAE=0.035515, directional_accuracy=0.7222
- step_size=2: stacking overall RMSE=0.042750, MAE=0.034722, directional_accuracy=0.7222
- step_size=1, horizon=long_3m: best RMSE model was `cart` with RMSE=nan
- step_size=1, horizon=long_6m: best RMSE model was `cart` with RMSE=nan
- step_size=1, horizon=short_10d: best RMSE model was `lightgbm` with RMSE=0.057681
- step_size=1, horizon=short_20d: best RMSE model was `cart` with RMSE=nan
- step_size=1, horizon=short_30d: best RMSE model was `cart` with RMSE=nan
- step_size=1, horizon=short_5d: best RMSE model was `cart` with RMSE=0.022946
- step_size=2, horizon=long_3m: best RMSE model was `cart` with RMSE=nan
- step_size=2, horizon=long_6m: best RMSE model was `cart` with RMSE=nan
- step_size=2, horizon=short_10d: best RMSE model was `lightgbm` with RMSE=0.054772
- step_size=2, horizon=short_20d: best RMSE model was `cart` with RMSE=nan
- step_size=2, horizon=short_30d: best RMSE model was `cart` with RMSE=nan
- step_size=2, horizon=short_5d: best RMSE model was `cart` with RMSE=0.021645

## Stacking vs Individual Models
- scope=overall_horizon, step_size=1, horizon=long_3m: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=1, horizon=long_6m: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=1, horizon=short_10d: stacking beat the field on RMSE in 33.33% of pairwise comparisons and on directional accuracy in 100.00%.
- scope=overall_horizon, step_size=1, horizon=short_20d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=1, horizon=short_30d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=1, horizon=short_5d: stacking beat the field on RMSE in 66.67% of pairwise comparisons and on directional accuracy in 100.00%.
- scope=overall_horizon, step_size=2, horizon=long_3m: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=2, horizon=long_6m: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=2, horizon=short_10d: stacking beat the field on RMSE in 66.67% of pairwise comparisons and on directional accuracy in 100.00%.
- scope=overall_horizon, step_size=2, horizon=short_20d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=2, horizon=short_30d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=2, horizon=short_5d: stacking beat the field on RMSE in 66.67% of pairwise comparisons and on directional accuracy in 66.67%.
- scope=ticker, step_size=1, horizon=long_3m: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=1, horizon=long_6m: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=1, horizon=short_10d: stacking beat the field on RMSE in 66.67% of pairwise comparisons and on directional accuracy in 50.00%.
- scope=ticker, step_size=1, horizon=short_20d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=1, horizon=short_30d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=1, horizon=short_5d: stacking beat the field on RMSE in 66.67% of pairwise comparisons and on directional accuracy in 66.67%.
- scope=ticker, step_size=2, horizon=long_3m: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=2, horizon=long_6m: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=2, horizon=short_10d: stacking beat the field on RMSE in 83.33% of pairwise comparisons and on directional accuracy in 50.00%.
- scope=ticker, step_size=2, horizon=short_20d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=2, horizon=short_30d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=2, horizon=short_5d: stacking beat the field on RMSE in 66.67% of pairwise comparisons and on directional accuracy in 33.33%.

## Largest Divergences
- ACB short_10d step=1 prediction_date=2025-01-13: predicted=-0.077601, actual=0.026163, absolute_error=0.103764
- ACB short_10d step=1 prediction_date=2025-01-10: predicted=-0.076909, actual=0.013909, absolute_error=0.090817
- ACB short_10d step=2 prediction_date=2025-01-10: predicted=-0.076909, actual=0.013909, absolute_error=0.090817
- ACB short_10d step=1 prediction_date=2025-01-07: predicted=-0.077730, actual=0.008205, absolute_error=0.085935
- ACB short_10d step=1 prediction_date=2025-01-09: predicted=-0.087356, actual=-0.006223, absolute_error=0.081133
- ACB short_10d step=1 prediction_date=2025-01-08: predicted=-0.082933, actual=-0.001915, absolute_error=0.081018
- ACB short_10d step=2 prediction_date=2025-01-08: predicted=-0.082933, actual=-0.001915, absolute_error=0.081018
- ACB short_10d step=1 prediction_date=2025-01-06: predicted=-0.077842, actual=-0.005722, absolute_error=0.072119
- ACB short_10d step=2 prediction_date=2025-01-06: predicted=-0.077842, actual=-0.005722, absolute_error=0.072119
- ACB short_10d step=2 prediction_date=2025-01-02: predicted=-0.094662, actual=-0.035250, absolute_error=0.059412

## Limitations
- The local price history available in this environment for the requested ticker universe starts on 2020-12-21, so the requested 2018-01-01 history boundary could not be fully satisfied from the accessible data sources.
- The current modern trainer path retrains on each rolling prediction date using only data available up to that feature cutoff; because the accessible history is shorter than requested, the effective expanding window begins at the first available market row for each ticker.
- Long horizons near the end of the 2025-01-01 to 2026-03-31 forecast window lose evaluation coverage because the full realized window does not exist inside the fetched history.
- Strategy metrics are derived from overlapping forecast windows, so Sharpe, Sortino, and CAGR should be treated as technical utility diagnostics rather than tradable production PnL.
- Final stacking output reports return and direction only; no final probability is emitted because the experiment uses a regression meta-learner and avoids over-claiming confidence semantics.

## Output Bundle Paths
- csv_outputs.zip: `artifacts\tmp_walkforward_benchmark_v2_parallel\csv_outputs.zip`
- charts_outputs.zip: `artifacts\tmp_walkforward_benchmark_v2_parallel\charts_outputs.zip`
- report.md: `artifacts\tmp_walkforward_benchmark_v2_parallel\report.md`
