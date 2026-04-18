# Walk-Forward Forecasting Report

## Experiment Setup
- Ticker universe: MSN, MWG, DGC, SSI, FPT, ACB
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
  - DGC: source=csv_fallback, rows=1024, available_range=2020-12-21 through 2025-01-24
  - FPT: source=csv_fallback, rows=1024, available_range=2020-12-21 through 2025-01-24
  - MSN: source=csv_fallback, rows=1024, available_range=2020-12-21 through 2025-01-24
  - MWG: source=csv_fallback, rows=1024, available_range=2020-12-21 through 2025-01-24
  - SSI: source=csv_fallback, rows=1024, available_range=2020-12-21 through 2025-01-24
- No algorithms were skipped after environment capability checks.

## Best Performers
- step_size=1: stacking overall RMSE=0.060321, MAE=0.050430, directional_accuracy=0.5463
- step_size=2: stacking overall RMSE=0.060815, MAE=0.050915, directional_accuracy=0.5926
- step_size=1, horizon=long_3m: best RMSE model was `cart` with RMSE=nan
- step_size=1, horizon=long_6m: best RMSE model was `cart` with RMSE=nan
- step_size=1, horizon=short_10d: best RMSE model was `xgboost` with RMSE=0.063217
- step_size=1, horizon=short_20d: best RMSE model was `cart` with RMSE=nan
- step_size=1, horizon=short_30d: best RMSE model was `cart` with RMSE=nan
- step_size=1, horizon=short_5d: best RMSE model was `stacking_final` with RMSE=0.051873
- step_size=2, horizon=long_3m: best RMSE model was `cart` with RMSE=nan
- step_size=2, horizon=long_6m: best RMSE model was `cart` with RMSE=nan
- step_size=2, horizon=short_10d: best RMSE model was `xgboost` with RMSE=0.063468
- step_size=2, horizon=short_20d: best RMSE model was `cart` with RMSE=nan
- step_size=2, horizon=short_30d: best RMSE model was `cart` with RMSE=nan
- step_size=2, horizon=short_5d: best RMSE model was `lightgbm` with RMSE=0.053487

## Stacking vs Individual Models
- scope=overall_horizon, step_size=1, horizon=long_3m: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=1, horizon=long_6m: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=1, horizon=short_10d: stacking beat the field on RMSE in 33.33% of pairwise comparisons and on directional accuracy in 100.00%.
- scope=overall_horizon, step_size=1, horizon=short_20d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=1, horizon=short_30d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=1, horizon=short_5d: stacking beat the field on RMSE in 100.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=2, horizon=long_3m: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=2, horizon=long_6m: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=2, horizon=short_10d: stacking beat the field on RMSE in 33.33% of pairwise comparisons and on directional accuracy in 100.00%.
- scope=overall_horizon, step_size=2, horizon=short_20d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=2, horizon=short_30d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=2, horizon=short_5d: stacking beat the field on RMSE in 66.67% of pairwise comparisons and on directional accuracy in 66.67%.
- scope=ticker, step_size=1, horizon=long_3m: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=1, horizon=long_6m: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=1, horizon=short_10d: stacking beat the field on RMSE in 55.56% of pairwise comparisons and on directional accuracy in 50.00%.
- scope=ticker, step_size=1, horizon=short_20d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=1, horizon=short_30d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=1, horizon=short_5d: stacking beat the field on RMSE in 50.00% of pairwise comparisons and on directional accuracy in 33.33%.
- scope=ticker, step_size=2, horizon=long_3m: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=2, horizon=long_6m: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=2, horizon=short_10d: stacking beat the field on RMSE in 55.56% of pairwise comparisons and on directional accuracy in 44.44%.
- scope=ticker, step_size=2, horizon=short_20d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=2, horizon=short_30d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=2, horizon=short_5d: stacking beat the field on RMSE in 61.11% of pairwise comparisons and on directional accuracy in 33.33%.

## Largest Divergences
- DGC short_10d step=1 prediction_date=2025-01-13: predicted=-0.117052, actual=0.012715, absolute_error=0.129766
- MWG short_10d step=1 prediction_date=2025-01-02: predicted=0.058544, actual=-0.060536, absolute_error=0.119080
- MWG short_10d step=2 prediction_date=2025-01-02: predicted=0.058544, actual=-0.060536, absolute_error=0.119080
- DGC short_10d step=2 prediction_date=2025-01-10: predicted=-0.129026, actual=-0.010698, absolute_error=0.118328
- DGC short_10d step=1 prediction_date=2025-01-10: predicted=-0.129026, actual=-0.010698, absolute_error=0.118328
- DGC short_10d step=1 prediction_date=2025-01-07: predicted=-0.138533, actual=-0.023100, absolute_error=0.115433
- DGC short_10d step=1 prediction_date=2025-01-03: predicted=-0.170312, actual=-0.058276, absolute_error=0.112036
- ACB short_10d step=1 prediction_date=2025-01-13: predicted=-0.085751, actual=0.026163, absolute_error=0.111914
- MSN short_5d step=1 prediction_date=2025-01-06: predicted=0.050547, actual=-0.060258, absolute_error=0.110805
- MSN short_5d step=2 prediction_date=2025-01-06: predicted=0.050547, actual=-0.060258, absolute_error=0.110805

## Limitations
- The local price history available in this environment for the requested ticker universe starts on 2020-12-21, so the requested 2018-01-01 history boundary could not be fully satisfied from the accessible data sources.
- The current modern trainer path retrains on each rolling prediction date using only data available up to that feature cutoff; because the accessible history is shorter than requested, the effective expanding window begins at the first available market row for each ticker.
- Long horizons near the end of the 2025-01-01 to 2026-03-31 forecast window lose evaluation coverage because the full realized window does not exist inside the fetched history.
- Strategy metrics are derived from overlapping forecast windows, so Sharpe, Sortino, and CAGR should be treated as technical utility diagnostics rather than tradable production PnL.
- Final stacking output reports return and direction only; no final probability is emitted because the experiment uses a regression meta-learner and avoids over-claiming confidence semantics.

## Output Bundle Paths
- csv_outputs.zip: `artifacts\walkforward_six_tickers_validation_20250115\csv_outputs.zip`
- charts_outputs.zip: `artifacts\walkforward_six_tickers_validation_20250115\charts_outputs.zip`
- report.md: `artifacts\walkforward_six_tickers_validation_20250115\report.md`
