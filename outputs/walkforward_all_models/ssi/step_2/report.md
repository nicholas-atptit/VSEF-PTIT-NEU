# Walk-Forward Forecasting Report

## Experiment Setup
- Ticker universe: SSI
- Actual data source used: SSI=vnstock_vnd
- 2018-01-01 coverage status: All six tickers have history from the first tradable 2018 session on 2018-01-02 through 2026-03-31. The exact calendar date 2018-01-01 was not a market session.
- Historical input window: 2018-01-01 through 2026-03-31
- Training window: 2018-01-01 through 2024-12-31
- Forecast window: 2025-01-01 through 2025-03-31
- Horizons: short_5d, short_10d, short_20d, short_30d, long_3m, long_6m
- Step size: 2
- All models actually run: cart
- Stacking method used: prequential Ridge regression over prior out-of-sample base-model predictions within the same horizon and step size, with mean fallback before enough realized rows exist.
- Exact target semantics used for actual comparison: `close[target_date] / close[prediction_date] - 1`, matching `DualModelTrainer._add_targets(...)`.
- Evaluation-eligible rows: rows where the realized `target_date` close exists inside fetched history; rows without enough future market data remain in the detailed prediction tables with `evaluation_eligible=False`.

## Per-Ticker Coverage
- SSI: source=vnstock_vnd, rows=2054, available_range=2018-01-02 through 2026-03-31

## Best-Performing Models and Horizons
- Overall stacking for step_size=2: RMSE=0.284965, MAE=0.216568, directional_accuracy=0.3046
- Horizon long_3m: best model by RMSE was `cart` with RMSE=0.235522, MAE=0.232855, directional_accuracy=0.2069
- Horizon long_6m: best model by RMSE was `cart` with RMSE=0.529197, MAE=0.516544, directional_accuracy=0.5862
- Horizon short_10d: best model by RMSE was `stacking_final` with RMSE=0.071486, MAE=0.050710, directional_accuracy=0.4483
- Horizon short_20d: best model by RMSE was `cart` with RMSE=0.203413, MAE=0.173204, directional_accuracy=0.5862
- Horizon short_30d: best model by RMSE was `cart` with RMSE=0.318837, MAE=0.287338, directional_accuracy=0.4138
- Horizon short_5d: best model by RMSE was `stacking_final` with RMSE=0.059720, MAE=0.038754, directional_accuracy=0.4138

## Stacking vs Individual Models
- scope=overall_horizon, horizon=long_3m: stacking beat the field on MAE in 0.00% of pairwise comparisons, on RMSE in 0.00%, and on directional accuracy in 0.00%.
- scope=overall_horizon, horizon=long_6m: stacking beat the field on MAE in 0.00% of pairwise comparisons, on RMSE in 0.00%, and on directional accuracy in 0.00%.
- scope=overall_horizon, horizon=short_10d: stacking beat the field on MAE in 100.00% of pairwise comparisons, on RMSE in 100.00%, and on directional accuracy in 0.00%.
- scope=overall_horizon, horizon=short_20d: stacking beat the field on MAE in 0.00% of pairwise comparisons, on RMSE in 0.00%, and on directional accuracy in 0.00%.
- scope=overall_horizon, horizon=short_30d: stacking beat the field on MAE in 0.00% of pairwise comparisons, on RMSE in 0.00%, and on directional accuracy in 0.00%.
- scope=overall_horizon, horizon=short_5d: stacking beat the field on MAE in 100.00% of pairwise comparisons, on RMSE in 100.00%, and on directional accuracy in 0.00%.
- scope=ticker, horizon=long_3m: stacking beat the field on MAE in 0.00% of pairwise comparisons, on RMSE in 0.00%, and on directional accuracy in 0.00%.
- scope=ticker, horizon=long_6m: stacking beat the field on MAE in 0.00% of pairwise comparisons, on RMSE in 0.00%, and on directional accuracy in 0.00%.
- scope=ticker, horizon=short_10d: stacking beat the field on MAE in 100.00% of pairwise comparisons, on RMSE in 100.00%, and on directional accuracy in 0.00%.
- scope=ticker, horizon=short_20d: stacking beat the field on MAE in 0.00% of pairwise comparisons, on RMSE in 0.00%, and on directional accuracy in 0.00%.
- scope=ticker, horizon=short_30d: stacking beat the field on MAE in 0.00% of pairwise comparisons, on RMSE in 0.00%, and on directional accuracy in 0.00%.
- scope=ticker, horizon=short_5d: stacking beat the field on MAE in 100.00% of pairwise comparisons, on RMSE in 100.00%, and on directional accuracy in 0.00%.

## Where Predictions Diverged Most
- SSI long_6m prediction_date=2025-03-04: predicted=-0.074098, actual=0.605289, absolute_error=0.679387
- SSI long_6m prediction_date=2025-02-28: predicted=-0.074098, actual=0.604546, absolute_error=0.678643
- SSI long_6m prediction_date=2025-03-10: predicted=-0.074098, actual=0.592912, absolute_error=0.667010
- SSI long_6m prediction_date=2025-03-12: predicted=-0.074098, actual=0.569272, absolute_error=0.643370
- SSI long_6m prediction_date=2025-02-26: predicted=-0.074098, actual=0.557027, absolute_error=0.631124
- SSI long_6m prediction_date=2025-03-14: predicted=-0.074098, actual=0.543609, absolute_error=0.617707
- SSI long_6m prediction_date=2025-03-18: predicted=-0.074098, actual=0.522367, absolute_error=0.596465
- SSI long_6m prediction_date=2025-02-10: predicted=-0.077087, actual=0.506036, absolute_error=0.583123
- SSI long_6m prediction_date=2025-03-28: predicted=-0.074098, actual=0.500192, absolute_error=0.574290
- SSI long_6m prediction_date=2025-03-06: predicted=-0.074098, actual=0.494392, absolute_error=0.568490

## Evaluation Coverage
- long_3m: eligible=58, ineligible=0, coverage_ratio=2.0000
- long_6m: eligible=58, ineligible=0, coverage_ratio=2.0000
- short_10d: eligible=58, ineligible=0, coverage_ratio=2.0000
- short_20d: eligible=58, ineligible=0, coverage_ratio=2.0000
- short_30d: eligible=58, ineligible=0, coverage_ratio=2.0000
- short_5d: eligible=58, ineligible=0, coverage_ratio=2.0000

## Limitations
- The repo-local daily CSV cache was insufficient for the requested start date, so the experiment depends on the live vnstock KBS path for historical backfill.
- Long horizons near the end of the requested forecast window are intentionally kept in the outputs but excluded from scored metrics when realized target closes are not yet available.
- Strategy metrics are computed on overlapping forecast windows, so they are technical usefulness diagnostics rather than execution-ready portfolio PnL.
- The final stack is a regression meta-learner, so no final calibrated probability is emitted.

## Output Paths
- csv/: `outputs\walkforward_all_models\ssi\step_2\csv`
- charts/: `outputs\walkforward_all_models\ssi\step_2\charts`
- report.md: `outputs\walkforward_all_models\ssi\step_2\report.md`
