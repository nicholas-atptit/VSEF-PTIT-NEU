# Walk-Forward Forecasting Report

## Experiment Setup
- Ticker universe: SSI
- Historical input window: 2018-01-01 through 2026-03-31
- Historical training window baseline: 2018-01-01 through 2024-12-31
- Rolling forecast window: 2025-01-01 through 2025-01-20
- Horizons: short_5d, short_10d, short_20d, short_30d, long_3m, long_6m
- Step sizes completed: 1
- Models actually run: cart
- Stacking method used: prequential Ridge regression on prior out-of-sample base-model predictions, pooled by horizon and step size, with mean fallback before enough realized rows exist.
- Actual data source used: vnstock daily OHLCV with KBS fallback ahead of local CSV fallback when the repo-local CSV history was too short.
- Exact target semantics used for actual comparison: `close[target_date] / close[prediction_date] - 1`, matching `DualModelTrainer._add_targets(...)`.
- Prediction-date semantics: the forecast anchor is the latest feature row date, so `prediction_date` equals the date whose close is used in the forward-return denominator.
- Evaluation-eligible rows: only predictions whose realized `target_date` close existed inside the fetched history were scored; later rows were kept in output tables but excluded from aggregate metrics.

## Model Coverage
- Fetched history coverage by ticker:
  - SSI: source=vnstock_vnd, rows=2054, available_range=2018-01-02 through 2026-03-31
- No algorithms were skipped after environment capability checks.

## Best Performers
- step_size=1: stacking overall RMSE=0.228956, MAE=0.183899, directional_accuracy=0.2051
- step_size=1, horizon=long_3m: best RMSE model was `cart` with RMSE=0.225464
- step_size=1, horizon=long_6m: best RMSE model was `cart` with RMSE=0.383689
- step_size=1, horizon=short_10d: best RMSE model was `cart` with RMSE=0.063150
- step_size=1, horizon=short_20d: best RMSE model was `cart` with RMSE=0.127649
- step_size=1, horizon=short_30d: best RMSE model was `cart` with RMSE=0.306150
- step_size=1, horizon=short_5d: best RMSE model was `cart` with RMSE=0.049638

## Stacking vs Individual Models
- scope=overall_horizon, step_size=1, horizon=long_3m: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=1, horizon=long_6m: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=1, horizon=short_10d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=1, horizon=short_20d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=1, horizon=short_30d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=1, horizon=short_5d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=1, horizon=long_3m: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=1, horizon=long_6m: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=1, horizon=short_10d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=1, horizon=short_20d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=1, horizon=short_30d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=1, horizon=short_5d: stacking beat the field on RMSE in 0.00% of pairwise comparisons and on directional accuracy in 0.00%.

## Largest Divergences
- SSI long_6m step=1 prediction_date=2025-01-20: predicted=-0.084759, actual=0.461174, absolute_error=0.545933
- SSI long_6m step=1 prediction_date=2025-01-17: predicted=-0.084895, actual=0.384950, absolute_error=0.469845
- SSI long_6m step=1 prediction_date=2025-01-10: predicted=-0.083215, actual=0.341757, absolute_error=0.424971
- SSI long_6m step=1 prediction_date=2025-01-14: predicted=-0.084312, actual=0.336122, absolute_error=0.420434
- SSI long_6m step=1 prediction_date=2025-01-15: predicted=-0.084531, actual=0.335384, absolute_error=0.419915
- SSI long_6m step=1 prediction_date=2025-01-16: predicted=-0.085038, actual=0.311465, absolute_error=0.396503
- SSI long_6m step=1 prediction_date=2025-01-09: predicted=-0.082462, actual=0.307359, absolute_error=0.389820
- SSI long_6m step=1 prediction_date=2025-01-13: predicted=-0.083970, actual=0.298140, absolute_error=0.382110
- SSI long_6m step=1 prediction_date=2025-01-08: predicted=-0.081441, actual=0.279386, absolute_error=0.360827
- SSI short_30d step=1 prediction_date=2025-01-10: predicted=-0.250894, actual=0.109723, absolute_error=0.360617

## Limitations
- The repo-local daily CSV cache starts on 2020-12-21 for the requested tickers, so this experiment depends on the live vnstock KBS history path for pre-2020 backfill.
- The requested calendar start date was 2018-01-01, but the first tradable session returned for all six tickers was 2018-01-02; January 1 was not a market session.
- The current modern trainer path retrains on each rolling prediction date using only data available up to that prediction-date close.
- Long horizons near the end of the 2025-01-01 to 2026-03-31 forecast window lose evaluation coverage because the full realized window does not exist inside the fetched history.
- Strategy metrics are derived from overlapping forecast windows, so Sharpe, Sortino, and CAGR should be treated as technical utility diagnostics rather than tradable production PnL.
- Final stacking output reports return and direction only; no final probability is emitted because the experiment uses a regression meta-learner and avoids over-claiming confidence semantics.

## Output Paths
- csv/: `outputs\walkforward_all_models\ssi\_combined_internal\csv`
- charts/: `outputs\walkforward_all_models\ssi\_combined_internal\charts`
- report.md: `outputs\walkforward_all_models\ssi\_combined_internal\report.md`
