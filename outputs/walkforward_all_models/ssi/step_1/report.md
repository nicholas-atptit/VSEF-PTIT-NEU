# Walk-Forward Forecasting Report

## Experiment Setup
- Ticker universe: SSI
- Actual data source used: SSI=vnstock_vnd
- 2018-01-01 coverage status: All six tickers have history from the first tradable 2018 session on 2018-01-02 through 2026-03-31. The exact calendar date 2018-01-01 was not a market session.
- Historical input window: 2018-01-01 through 2026-03-31
- Training window: 2018-01-01 through 2024-12-31
- Forecast window: 2025-01-01 through 2025-01-20
- Horizons: short_5d, short_10d, short_20d, short_30d, long_3m, long_6m
- Step size: 1
- All models actually run: cart
- Stacking method used: prequential Ridge regression over prior out-of-sample base-model predictions within the same horizon and step size, with mean fallback before enough realized rows exist.
- Exact target semantics used for actual comparison: `close[target_date] / close[prediction_date] - 1`, matching `DualModelTrainer._add_targets(...)`.
- Evaluation-eligible rows: rows where the realized `target_date` close exists inside fetched history; rows without enough future market data remain in the detailed prediction tables with `evaluation_eligible=False`.

## Per-Ticker Coverage
- SSI: source=vnstock_vnd, rows=2054, available_range=2018-01-02 through 2026-03-31

## Best-Performing Models and Horizons
- Overall stacking for step_size=1: RMSE=0.228956, MAE=0.183899, directional_accuracy=0.2051
- Horizon long_3m: best model by RMSE was `cart` with RMSE=0.225464, MAE=0.220336, directional_accuracy=0.4615
- Horizon long_6m: best model by RMSE was `cart` with RMSE=0.383689, MAE=0.372587, directional_accuracy=0.9231
- Horizon short_10d: best model by RMSE was `cart` with RMSE=0.063150, MAE=0.055526, directional_accuracy=0.6154
- Horizon short_20d: best model by RMSE was `cart` with RMSE=0.127649, MAE=0.112338, directional_accuracy=0.7692
- Horizon short_30d: best model by RMSE was `cart` with RMSE=0.306150, MAE=0.296735, directional_accuracy=0.5385
- Horizon short_5d: best model by RMSE was `cart` with RMSE=0.049638, MAE=0.045871, directional_accuracy=0.7692

## Stacking vs Individual Models
- scope=overall_horizon, horizon=long_3m: stacking beat the field on MAE in 0.00% of pairwise comparisons, on RMSE in 0.00%, and on directional accuracy in 0.00%.
- scope=overall_horizon, horizon=long_6m: stacking beat the field on MAE in 0.00% of pairwise comparisons, on RMSE in 0.00%, and on directional accuracy in 0.00%.
- scope=overall_horizon, horizon=short_10d: stacking beat the field on MAE in 0.00% of pairwise comparisons, on RMSE in 0.00%, and on directional accuracy in 0.00%.
- scope=overall_horizon, horizon=short_20d: stacking beat the field on MAE in 0.00% of pairwise comparisons, on RMSE in 0.00%, and on directional accuracy in 0.00%.
- scope=overall_horizon, horizon=short_30d: stacking beat the field on MAE in 0.00% of pairwise comparisons, on RMSE in 0.00%, and on directional accuracy in 0.00%.
- scope=overall_horizon, horizon=short_5d: stacking beat the field on MAE in 0.00% of pairwise comparisons, on RMSE in 0.00%, and on directional accuracy in 0.00%.
- scope=ticker, horizon=long_3m: stacking beat the field on MAE in 0.00% of pairwise comparisons, on RMSE in 0.00%, and on directional accuracy in 0.00%.
- scope=ticker, horizon=long_6m: stacking beat the field on MAE in 0.00% of pairwise comparisons, on RMSE in 0.00%, and on directional accuracy in 0.00%.
- scope=ticker, horizon=short_10d: stacking beat the field on MAE in 0.00% of pairwise comparisons, on RMSE in 0.00%, and on directional accuracy in 0.00%.
- scope=ticker, horizon=short_20d: stacking beat the field on MAE in 0.00% of pairwise comparisons, on RMSE in 0.00%, and on directional accuracy in 0.00%.
- scope=ticker, horizon=short_30d: stacking beat the field on MAE in 0.00% of pairwise comparisons, on RMSE in 0.00%, and on directional accuracy in 0.00%.
- scope=ticker, horizon=short_5d: stacking beat the field on MAE in 0.00% of pairwise comparisons, on RMSE in 0.00%, and on directional accuracy in 0.00%.

## Where Predictions Diverged Most
- SSI long_6m prediction_date=2025-01-20: predicted=-0.084759, actual=0.461174, absolute_error=0.545933
- SSI long_6m prediction_date=2025-01-17: predicted=-0.084895, actual=0.384950, absolute_error=0.469845
- SSI long_6m prediction_date=2025-01-10: predicted=-0.083215, actual=0.341757, absolute_error=0.424971
- SSI long_6m prediction_date=2025-01-14: predicted=-0.084312, actual=0.336122, absolute_error=0.420434
- SSI long_6m prediction_date=2025-01-15: predicted=-0.084531, actual=0.335384, absolute_error=0.419915
- SSI long_6m prediction_date=2025-01-16: predicted=-0.085038, actual=0.311465, absolute_error=0.396503
- SSI long_6m prediction_date=2025-01-09: predicted=-0.082462, actual=0.307359, absolute_error=0.389820
- SSI long_6m prediction_date=2025-01-13: predicted=-0.083970, actual=0.298140, absolute_error=0.382110
- SSI long_6m prediction_date=2025-01-08: predicted=-0.081441, actual=0.279386, absolute_error=0.360827
- SSI short_30d prediction_date=2025-01-10: predicted=-0.250894, actual=0.109723, absolute_error=0.360617

## Evaluation Coverage
- long_3m: eligible=26, ineligible=0, coverage_ratio=2.0000
- long_6m: eligible=26, ineligible=0, coverage_ratio=2.0000
- short_10d: eligible=26, ineligible=0, coverage_ratio=2.0000
- short_20d: eligible=26, ineligible=0, coverage_ratio=2.0000
- short_30d: eligible=26, ineligible=0, coverage_ratio=2.0000
- short_5d: eligible=26, ineligible=0, coverage_ratio=2.0000

## Limitations
- The repo-local daily CSV cache was insufficient for the requested start date, so the experiment depends on the live vnstock KBS path for historical backfill.
- Long horizons near the end of the requested forecast window are intentionally kept in the outputs but excluded from scored metrics when realized target closes are not yet available.
- Strategy metrics are computed on overlapping forecast windows, so they are technical usefulness diagnostics rather than execution-ready portfolio PnL.
- The final stack is a regression meta-learner, so no final calibrated probability is emitted.

## Output Paths
- csv/: `outputs\walkforward_all_models\ssi\step_1\csv`
- charts/: `outputs\walkforward_all_models\ssi\step_1\charts`
- report.md: `outputs\walkforward_all_models\ssi\step_1\report.md`
