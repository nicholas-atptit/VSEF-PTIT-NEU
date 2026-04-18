# Walk-Forward Forecasting Report

## Experiment Setup
- Data window: 2018-01-01 through 2025-01-31
- Historical training window baseline: 2018-01-01 through 2024-12-31
- Rolling forecast window: 2025-01-01 through 2025-01-15
- Horizons: short_5d
- Step sizes completed: 1
- Models actually run: cart, lightgbm, xgboost
- Stacking method used: prequential Ridge regression on prior out-of-sample base-model predictions, with mean fallback before enough realized rows exist.
- Actual comparison: each forecast was aligned to the close-to-close realized forward-return window from the feature cutoff date to the horizon target trading date.
- Evaluation-eligible rows: only predictions whose full realized horizon existed inside the fetched 2026-03-31 data boundary were scored; later rows were kept in output tables but excluded from aggregated metrics.

## Model Coverage
- No algorithms were skipped after environment capability checks.

## Best Performers
- step_size=1: stacking overall RMSE=0.022236, MAE=0.016323, directional_accuracy=0.3000
- step_size=1, horizon=short_5d: best RMSE model was `cart` with RMSE=0.016331
- step_size=1, horizon=short_5d: best RMSE model was `stacking_final` with RMSE=0.022236
- step_size=1, horizon=short_5d: best RMSE model was `lightgbm` with RMSE=0.024046
- step_size=1, horizon=short_5d: best RMSE model was `xgboost` with RMSE=0.027514

## Stacking vs Individual Models
- scope=overall_horizon, step_size=1, horizon=short_5d: stacking beat the field on RMSE in 66.67% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=1, horizon=short_5d: stacking beat the field on RMSE in 66.67% of pairwise comparisons and on directional accuracy in 0.00%.

## Largest Divergences
- ACB short_5d step=1 prediction_date=2025-01-02: predicted=0.015868, actual=-0.031076, absolute_error=0.046944
- ACB short_5d step=1 prediction_date=2025-01-03: predicted=0.013522, actual=-0.025245, absolute_error=0.038768
- ACB short_5d step=1 prediction_date=2025-01-06: predicted=0.007930, actual=-0.015737, absolute_error=0.023667
- ACB short_5d step=1 prediction_date=2025-01-08: predicted=0.005410, actual=-0.013882, absolute_error=0.019293
- ACB short_5d step=1 prediction_date=2025-01-09: predicted=0.006791, actual=-0.004308, absolute_error=0.011099
- ACB short_5d step=1 prediction_date=2025-01-10: predicted=0.006651, actual=-0.004317, absolute_error=0.010967
- ACB short_5d step=1 prediction_date=2025-01-07: predicted=0.005149, actual=-0.001931, absolute_error=0.007080
- ACB short_5d step=1 prediction_date=2025-01-13: predicted=0.007737, actual=0.010174, absolute_error=0.002437
- ACB short_5d step=1 prediction_date=2025-01-15: predicted=0.009841, actual=0.012136, absolute_error=0.002295
- ACB short_5d step=1 prediction_date=2025-01-14: predicted=0.009473, actual=0.010155, absolute_error=0.000682

## Limitations
- Long horizons near the end of the 2025-01-01 to 2026-03-31 forecast window lose evaluation coverage because the full realized window does not exist inside the fetched history.
- Strategy metrics are derived from overlapping forecast windows, so Sharpe, Sortino, and CAGR should be treated as technical utility diagnostics rather than tradable production PnL.
- Final stacking output reports return and direction only; no final probability is emitted because the experiment uses a regression meta-learner and avoids over-claiming confidence semantics.

## Output Bundle Paths
- csv_outputs.zip: `artifacts\tmp_walkforward_benchmark\csv_outputs.zip`
- charts_outputs.zip: `artifacts\tmp_walkforward_benchmark\charts_outputs.zip`
- report.md: `artifacts\tmp_walkforward_benchmark\report.md`
