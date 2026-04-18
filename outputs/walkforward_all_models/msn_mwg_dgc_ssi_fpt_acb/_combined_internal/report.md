# Walk-Forward Forecasting Report

## Experiment Setup
- Ticker universe: MSN, MWG, DGC, SSI, FPT, ACB
- Historical input window: 2018-01-01 through 2026-03-31
- Historical training window baseline: 2018-01-01 through 2024-12-31
- Rolling forecast window: 2025-01-01 through 2026-04-01
- Horizons: short_5d, short_10d, short_20d, short_30d, long_3m, long_6m
- Step sizes completed: 1, 2
- Models actually run: cart
- Stacking method used: prequential Ridge regression on prior out-of-sample base-model predictions, pooled by horizon and step size, with mean fallback before enough realized rows exist.
- Actual data source used: vnstock daily OHLCV with KBS fallback ahead of local CSV fallback when the repo-local CSV history was too short.
- Exact target semantics used for actual comparison: `close[target_date] / close[prediction_date] - 1`, matching `DualModelTrainer._add_targets(...)`.
- Prediction-date semantics: the forecast anchor is the latest feature row date, so `prediction_date` equals the date whose close is used in the forward-return denominator.
- Evaluation-eligible rows: only predictions whose realized `target_date` close existed inside the fetched history were scored; later rows were kept in output tables but excluded from aggregate metrics.

## Model Coverage
- Fetched history coverage by ticker:
  - ACB: source=vnstock_vnd, rows=2051, available_range=2018-01-02 through 2026-03-31
  - DGC: source=vnstock_vnd, rows=2050, available_range=2018-01-02 through 2026-03-31
  - FPT: source=vnstock_vnd, rows=2054, available_range=2018-01-02 through 2026-03-31
  - MSN: source=vnstock_vnd, rows=2054, available_range=2018-01-02 through 2026-03-31
  - MWG: source=vnstock_vnd, rows=2054, available_range=2018-01-02 through 2026-03-31
  - SSI: source=vnstock_vnd, rows=2054, available_range=2018-01-02 through 2026-03-31
- No algorithms were skipped after environment capability checks.

## Best Performers
- step_size=1: stacking overall RMSE=0.183678, MAE=0.117855, directional_accuracy=0.3912
- step_size=2: stacking overall RMSE=0.184480, MAE=0.118573, directional_accuracy=0.3828
- step_size=1, horizon=long_3m: best RMSE model was `stacking_final` with RMSE=0.276892
- step_size=1, horizon=long_6m: best RMSE model was `stacking_final` with RMSE=0.363546
- step_size=1, horizon=short_10d: best RMSE model was `stacking_final` with RMSE=0.080326
- step_size=1, horizon=short_20d: best RMSE model was `stacking_final` with RMSE=0.106738
- step_size=1, horizon=short_30d: best RMSE model was `stacking_final` with RMSE=0.133431
- step_size=1, horizon=short_5d: best RMSE model was `stacking_final` with RMSE=0.053778
- step_size=2, horizon=long_3m: best RMSE model was `stacking_final` with RMSE=0.276374
- step_size=2, horizon=long_6m: best RMSE model was `stacking_final` with RMSE=0.365785
- step_size=2, horizon=short_10d: best RMSE model was `stacking_final` with RMSE=0.081288
- step_size=2, horizon=short_20d: best RMSE model was `stacking_final` with RMSE=0.107347
- step_size=2, horizon=short_30d: best RMSE model was `stacking_final` with RMSE=0.135303
- step_size=2, horizon=short_5d: best RMSE model was `stacking_final` with RMSE=0.054060

## Stacking vs Individual Models
- scope=overall_horizon, step_size=1, horizon=long_3m: stacking beat the field on RMSE in 100.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=1, horizon=long_6m: stacking beat the field on RMSE in 100.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=1, horizon=short_10d: stacking beat the field on RMSE in 100.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=1, horizon=short_20d: stacking beat the field on RMSE in 100.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=1, horizon=short_30d: stacking beat the field on RMSE in 100.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=1, horizon=short_5d: stacking beat the field on RMSE in 100.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=2, horizon=long_3m: stacking beat the field on RMSE in 100.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=2, horizon=long_6m: stacking beat the field on RMSE in 100.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=2, horizon=short_10d: stacking beat the field on RMSE in 100.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=2, horizon=short_20d: stacking beat the field on RMSE in 100.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=2, horizon=short_30d: stacking beat the field on RMSE in 100.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=overall_horizon, step_size=2, horizon=short_5d: stacking beat the field on RMSE in 100.00% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=1, horizon=long_3m: stacking beat the field on RMSE in 66.67% of pairwise comparisons and on directional accuracy in 16.67%.
- scope=ticker, step_size=1, horizon=long_6m: stacking beat the field on RMSE in 50.00% of pairwise comparisons and on directional accuracy in 16.67%.
- scope=ticker, step_size=1, horizon=short_10d: stacking beat the field on RMSE in 83.33% of pairwise comparisons and on directional accuracy in 16.67%.
- scope=ticker, step_size=1, horizon=short_20d: stacking beat the field on RMSE in 83.33% of pairwise comparisons and on directional accuracy in 33.33%.
- scope=ticker, step_size=1, horizon=short_30d: stacking beat the field on RMSE in 83.33% of pairwise comparisons and on directional accuracy in 16.67%.
- scope=ticker, step_size=1, horizon=short_5d: stacking beat the field on RMSE in 83.33% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=2, horizon=long_3m: stacking beat the field on RMSE in 66.67% of pairwise comparisons and on directional accuracy in 16.67%.
- scope=ticker, step_size=2, horizon=long_6m: stacking beat the field on RMSE in 50.00% of pairwise comparisons and on directional accuracy in 16.67%.
- scope=ticker, step_size=2, horizon=short_10d: stacking beat the field on RMSE in 83.33% of pairwise comparisons and on directional accuracy in 16.67%.
- scope=ticker, step_size=2, horizon=short_20d: stacking beat the field on RMSE in 83.33% of pairwise comparisons and on directional accuracy in 33.33%.
- scope=ticker, step_size=2, horizon=short_30d: stacking beat the field on RMSE in 83.33% of pairwise comparisons and on directional accuracy in 0.00%.
- scope=ticker, step_size=2, horizon=short_5d: stacking beat the field on RMSE in 100.00% of pairwise comparisons and on directional accuracy in 0.00%.

## Largest Divergences
- SSI long_6m step=1 prediction_date=2025-04-09: predicted=-0.074098, actual=1.023499, absolute_error=1.097596
- SSI long_6m step=1 prediction_date=2025-04-08: predicted=-0.074098, actual=0.911624, absolute_error=0.985722
- SSI long_6m step=2 prediction_date=2025-04-08: predicted=-0.074098, actual=0.911624, absolute_error=0.985722
- SSI long_6m step=1 prediction_date=2025-04-10: predicted=-0.074098, actual=0.897048, absolute_error=0.971145
- SSI long_6m step=2 prediction_date=2025-04-10: predicted=-0.074098, actual=0.897048, absolute_error=0.971145
- SSI long_3m step=1 prediction_date=2025-06-13: predicted=-0.110599, actual=0.809959, absolute_error=0.920558
- SSI long_6m step=1 prediction_date=2025-04-16: predicted=-0.074098, actual=0.843571, absolute_error=0.917669
- SSI long_6m step=2 prediction_date=2025-04-16: predicted=-0.074098, actual=0.843571, absolute_error=0.917669
- SSI long_3m step=1 prediction_date=2025-06-11: predicted=-0.110537, actual=0.806419, absolute_error=0.916956
- SSI long_3m step=1 prediction_date=2025-06-10: predicted=-0.110199, actual=0.798718, absolute_error=0.908917

## Limitations
- The repo-local daily CSV cache starts on 2020-12-21 for the requested tickers, so this experiment depends on the live vnstock KBS history path for pre-2020 backfill.
- The requested calendar start date was 2018-01-01, but the first tradable session returned for all six tickers was 2018-01-02; January 1 was not a market session.
- The current modern trainer path retrains on each rolling prediction date using only data available up to that prediction-date close.
- Long horizons near the end of the 2025-01-01 to 2026-03-31 forecast window lose evaluation coverage because the full realized window does not exist inside the fetched history.
- Strategy metrics are derived from overlapping forecast windows, so Sharpe, Sortino, and CAGR should be treated as technical utility diagnostics rather than tradable production PnL.
- Final stacking output reports return and direction only; no final probability is emitted because the experiment uses a regression meta-learner and avoids over-claiming confidence semantics.

## Output Paths
- csv/: `outputs\walkforward_all_models\msn_mwg_dgc_ssi_fpt_acb\_combined_internal\csv`
- charts/: `outputs\walkforward_all_models\msn_mwg_dgc_ssi_fpt_acb\_combined_internal\charts`
- report.md: `outputs\walkforward_all_models\msn_mwg_dgc_ssi_fpt_acb\_combined_internal\report.md`
