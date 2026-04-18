# Walk-Forward Forecasting Report

## Experiment Setup
- Ticker universe: MSN
- Historical input window: 2018-01-01 through 2025-01-31
- Historical training window baseline: 2018-01-01 through 2024-12-31
- Rolling forecast window: 2025-01-02 through 2025-01-10
- Horizons: short_5d
- Step sizes completed: 1, 2
- Models actually run: cart
- Stacking method used: prequential Ridge regression on prior out-of-sample base-model predictions, pooled by horizon and step size, with mean fallback before enough realized rows exist.
- Actual data source used: vnstock daily OHLCV with KBS fallback ahead of local CSV fallback when the repo-local CSV history was too short.
- Exact target semantics used for actual comparison: `close[target_date] / close[prediction_date] - 1`, matching `DualModelTrainer._add_targets(...)`.
- Prediction-date semantics: the forecast anchor is the latest feature row date, so `prediction_date` equals the date whose close is used in the forward-return denominator.
- Evaluation-eligible rows: only predictions whose realized `target_date` close existed inside the fetched history were scored; later rows were kept in output tables but excluded from aggregate metrics.

## Model Coverage
- Fetched history coverage by ticker:
  - MSN: source=vnstock_kbs, rows=1765, available_range=2018-01-02 through 2025-01-24
- No algorithms were skipped after environment capability checks.

## Best Performers
- step_size=1: stacking overall RMSE=0.146234, MAE=0.140875, directional_accuracy=0.1429
- step_size=2: stacking overall RMSE=0.150610, MAE=0.141578, directional_accuracy=0.2500
- step_size=1, horizon=short_5d: best RMSE model was `cart` with RMSE=0.146234
- step_size=2, horizon=short_5d: best RMSE model was `cart` with RMSE=0.150610

## Stacking vs Individual Models
- scope=overall_horizon, step_size=1, horizon=short_5d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=2, horizon=short_5d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=1, horizon=short_5d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=2, horizon=short_5d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.

## Largest Divergences
- MSN short_5d step=1 prediction_date=2025-01-02: predicted=0.163894, actual=-0.049435, absolute_error=0.213329
- MSN short_5d step=2 prediction_date=2025-01-02: predicted=0.163894, actual=-0.049435, absolute_error=0.213329
- MSN short_5d step=1 prediction_date=2025-01-08: predicted=0.133846, actual=-0.028065, absolute_error=0.161911
- MSN short_5d step=2 prediction_date=2025-01-08: predicted=0.133846, actual=-0.028065, absolute_error=0.161911
- MSN short_5d step=1 prediction_date=2025-01-09: predicted=0.133846, actual=-0.014859, absolute_error=0.148705
- MSN short_5d step=1 prediction_date=2025-01-03: predicted=0.082220, actual=-0.060258, absolute_error=0.142478
- MSN short_5d step=1 prediction_date=2025-01-07: predicted=0.082220, actual=-0.046407, absolute_error=0.128627
- MSN short_5d step=1 prediction_date=2025-01-06: predicted=0.082220, actual=-0.032738, absolute_error=0.114958
- MSN short_5d step=2 prediction_date=2025-01-06: predicted=0.082220, actual=-0.032738, absolute_error=0.114958
- MSN short_5d step=1 prediction_date=2025-01-10: predicted=0.082220, actual=0.006107, absolute_error=0.076113

## Limitations
- The repo-local daily CSV cache starts on 2020-12-21 for the requested tickers, so this experiment depends on the live vnstock KBS history path for pre-2020 backfill.
- The requested calendar start date was 2018-01-01, but the first tradable session returned for all six tickers was 2018-01-02; January 1 was not a market session.
- The current modern trainer path retrains on each rolling prediction date using only data available up to that prediction-date close.
- Long horizons near the end of the 2025-01-01 to 2026-03-31 forecast window lose evaluation coverage because the full realized window does not exist inside the fetched history.
- Strategy metrics are derived from overlapping forecast windows, so Sharpe, Sortino, and CAGR should be treated as technical utility diagnostics rather than tradable production PnL.
- Final stacking output reports return and direction only; no final probability is emitted because the experiment uses a regression meta-learner and avoids over-claiming confidence semantics.

## Output Paths
- csv/: `outputs\walkforward_all_models_smoke_internal\msn\_combined_internal\csv`
- charts/: `outputs\walkforward_all_models_smoke_internal\msn\_combined_internal\charts`
- report.md: `outputs\walkforward_all_models_smoke_internal\msn\_combined_internal\report.md`
