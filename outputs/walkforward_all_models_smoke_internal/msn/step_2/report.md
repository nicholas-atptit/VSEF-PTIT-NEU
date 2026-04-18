# Walk-Forward Forecasting Report

## Experiment Setup
- Ticker universe: MSN
- Actual data source used: MSN=vnstock_kbs
- 2018-01-01 coverage status: One or more tickers did not fully span the requested history window; inspect the per-ticker source lines below.
- Historical input window: 2018-01-01 through 2025-01-31
- Training window: 2018-01-01 through 2024-12-31
- Forecast window: 2025-01-02 through 2025-01-10
- Horizons: short_5d
- Step size: 2
- All models actually run: cart
- Stacking method used: prequential Ridge regression over prior out-of-sample base-model predictions within the same horizon and step size, with mean fallback before enough realized rows exist.
- Exact target semantics used for actual comparison: `close[target_date] / close[prediction_date] - 1`, matching `DualModelTrainer._add_targets(...)`.
- Evaluation-eligible rows: rows where the realized `target_date` close exists inside fetched history; rows without enough future market data remain in the detailed prediction tables with `evaluation_eligible=False`.

## Per-Ticker Coverage
- MSN: source=vnstock_kbs, rows=1765, available_range=2018-01-02 through 2025-01-24

## Best-Performing Models and Horizons
- Overall stacking for step_size=2: RMSE=0.150610, MAE=0.141578, directional_accuracy=0.2500
- Horizon short_5d: best model by RMSE was `cart` with RMSE=0.150610, MAE=0.141578, directional_accuracy=0.5000

## Stacking vs Individual Models
- scope=overall_horizon, horizon=short_5d: stacking beat the field on MAE in 0.00% of pairwise comparisons, on RMSE in 0.00%, and on directional accuracy in 0.00%.
- scope=ticker, horizon=short_5d: stacking beat the field on MAE in 0.00% of pairwise comparisons, on RMSE in 0.00%, and on directional accuracy in 0.00%.

## Where Predictions Diverged Most
- MSN short_5d prediction_date=2025-01-02: predicted=0.163894, actual=-0.049435, absolute_error=0.213329
- MSN short_5d prediction_date=2025-01-08: predicted=0.133846, actual=-0.028065, absolute_error=0.161911
- MSN short_5d prediction_date=2025-01-06: predicted=0.082220, actual=-0.032738, absolute_error=0.114958
- MSN short_5d prediction_date=2025-01-10: predicted=0.082220, actual=0.006107, absolute_error=0.076113

## Evaluation Coverage
- short_5d: eligible=8, ineligible=0, coverage_ratio=2.0000

## Limitations
- The repo-local daily CSV cache was insufficient for the requested start date, so the experiment depends on the live vnstock KBS path for historical backfill.
- Long horizons near the end of the requested forecast window are intentionally kept in the outputs but excluded from scored metrics when realized target closes are not yet available.
- Strategy metrics are computed on overlapping forecast windows, so they are technical usefulness diagnostics rather than execution-ready portfolio PnL.
- The final stack is a regression meta-learner, so no final calibrated probability is emitted.

## Output Paths
- csv/: `outputs\walkforward_all_models_smoke_internal\msn\step_2\csv`
- charts/: `outputs\walkforward_all_models_smoke_internal\msn\step_2\charts`
- report.md: `outputs\walkforward_all_models_smoke_internal\msn\step_2\report.md`
