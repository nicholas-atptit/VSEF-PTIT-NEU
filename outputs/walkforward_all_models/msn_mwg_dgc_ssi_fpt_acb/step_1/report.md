# Walk-Forward Forecasting Report

## Experiment Setup
- Ticker universe: MSN, MWG, DGC, SSI, FPT, ACB
- Actual data source used: ACB=vnstock_vnd, DGC=vnstock_vnd, FPT=vnstock_vnd, MSN=vnstock_vnd, MWG=vnstock_vnd, SSI=vnstock_vnd
- 2018-01-01 coverage status: All six tickers have history from the first tradable 2018 session on 2018-01-02 through 2026-03-31. The exact calendar date 2018-01-01 was not a market session.
- Historical input window: 2018-01-01 through 2026-03-31
- Training window: 2018-01-01 through 2024-12-31
- Forecast window: 2025-01-01 through 2026-04-01
- Horizons: short_5d, short_10d, short_20d, short_30d, long_3m, long_6m
- Step size: 1
- All models actually run: cart
- Stacking method used: prequential Ridge regression over prior out-of-sample base-model predictions within the same horizon and step size, with mean fallback before enough realized rows exist.
- Exact target semantics used for actual comparison: `close[target_date] / close[prediction_date] - 1`, matching `DualModelTrainer._add_targets(...)`.
- Evaluation-eligible rows: rows where the realized `target_date` close exists inside fetched history; rows without enough future market data remain in the detailed prediction tables with `evaluation_eligible=False`.

## Per-Ticker Coverage
- ACB: source=vnstock_vnd, rows=2051, available_range=2018-01-02 through 2026-03-31
- DGC: source=vnstock_vnd, rows=2050, available_range=2018-01-02 through 2026-03-31
- FPT: source=vnstock_vnd, rows=2054, available_range=2018-01-02 through 2026-03-31
- MSN: source=vnstock_vnd, rows=2054, available_range=2018-01-02 through 2026-03-31
- MWG: source=vnstock_vnd, rows=2054, available_range=2018-01-02 through 2026-03-31
- SSI: source=vnstock_vnd, rows=2054, available_range=2018-01-02 through 2026-03-31

## Best-Performing Models and Horizons
- Overall stacking for step_size=1: RMSE=0.183678, MAE=0.117855, directional_accuracy=0.3912
- Horizon long_3m: best model by RMSE was `stacking_final` with RMSE=0.276892, MAE=0.214719, directional_accuracy=0.2908
- Horizon long_6m: best model by RMSE was `stacking_final` with RMSE=0.363546, MAE=0.304614, directional_accuracy=0.4139
- Horizon short_10d: best model by RMSE was `stacking_final` with RMSE=0.080326, MAE=0.054695, directional_accuracy=0.4065
- Horizon short_20d: best model by RMSE was `stacking_final` with RMSE=0.106738, MAE=0.081419, directional_accuracy=0.4237
- Horizon short_30d: best model by RMSE was `stacking_final` with RMSE=0.133431, MAE=0.103431, directional_accuracy=0.3521
- Horizon short_5d: best model by RMSE was `stacking_final` with RMSE=0.053778, MAE=0.037933, directional_accuracy=0.4485

## Stacking vs Individual Models
- scope=overall_horizon, horizon=long_3m: stacking beat the field on MAE in 100.00% of pairwise comparisons, on RMSE in 100.00%, and on directional accuracy in 0.00%.
- scope=overall_horizon, horizon=long_6m: stacking beat the field on MAE in 100.00% of pairwise comparisons, on RMSE in 100.00%, and on directional accuracy in 0.00%.
- scope=overall_horizon, horizon=short_10d: stacking beat the field on MAE in 100.00% of pairwise comparisons, on RMSE in 100.00%, and on directional accuracy in 0.00%.
- scope=overall_horizon, horizon=short_20d: stacking beat the field on MAE in 100.00% of pairwise comparisons, on RMSE in 100.00%, and on directional accuracy in 0.00%.
- scope=overall_horizon, horizon=short_30d: stacking beat the field on MAE in 100.00% of pairwise comparisons, on RMSE in 100.00%, and on directional accuracy in 0.00%.
- scope=overall_horizon, horizon=short_5d: stacking beat the field on MAE in 100.00% of pairwise comparisons, on RMSE in 100.00%, and on directional accuracy in 0.00%.
- scope=ticker, horizon=long_3m: stacking beat the field on MAE in 33.33% of pairwise comparisons, on RMSE in 66.67%, and on directional accuracy in 16.67%.
- scope=ticker, horizon=long_6m: stacking beat the field on MAE in 50.00% of pairwise comparisons, on RMSE in 50.00%, and on directional accuracy in 16.67%.
- scope=ticker, horizon=short_10d: stacking beat the field on MAE in 83.33% of pairwise comparisons, on RMSE in 83.33%, and on directional accuracy in 16.67%.
- scope=ticker, horizon=short_20d: stacking beat the field on MAE in 83.33% of pairwise comparisons, on RMSE in 83.33%, and on directional accuracy in 33.33%.
- scope=ticker, horizon=short_30d: stacking beat the field on MAE in 83.33% of pairwise comparisons, on RMSE in 83.33%, and on directional accuracy in 16.67%.
- scope=ticker, horizon=short_5d: stacking beat the field on MAE in 83.33% of pairwise comparisons, on RMSE in 83.33%, and on directional accuracy in 0.00%.

## Where Predictions Diverged Most
- SSI long_6m prediction_date=2025-04-09: predicted=-0.074098, actual=1.023499, absolute_error=1.097596
- SSI long_6m prediction_date=2025-04-08: predicted=-0.074098, actual=0.911624, absolute_error=0.985722
- SSI long_6m prediction_date=2025-04-10: predicted=-0.074098, actual=0.897048, absolute_error=0.971145
- SSI long_3m prediction_date=2025-06-13: predicted=-0.110599, actual=0.809959, absolute_error=0.920558
- SSI long_6m prediction_date=2025-04-16: predicted=-0.074098, actual=0.843571, absolute_error=0.917669
- SSI long_3m prediction_date=2025-06-11: predicted=-0.110537, actual=0.806419, absolute_error=0.916956
- SSI long_3m prediction_date=2025-06-10: predicted=-0.110199, actual=0.798718, absolute_error=0.908917
- SSI long_3m prediction_date=2025-06-12: predicted=-0.110684, actual=0.794897, absolute_error=0.905581
- SSI long_6m prediction_date=2025-04-15: predicted=-0.074098, actual=0.824186, absolute_error=0.898284
- SSI long_6m prediction_date=2025-04-17: predicted=-0.074098, actual=0.811160, absolute_error=0.885258

## Evaluation Coverage
- long_3m: eligible=2916, ineligible=756, coverage_ratio=9.5294
- long_6m: eligible=2160, ineligible=1512, coverage_ratio=7.0588
- short_10d: eligible=3552, ineligible=120, coverage_ratio=11.6078
- short_20d: eligible=3432, ineligible=240, coverage_ratio=11.2157
- short_30d: eligible=3312, ineligible=360, coverage_ratio=10.8235
- short_5d: eligible=3612, ineligible=60, coverage_ratio=11.8039

## Limitations
- The repo-local daily CSV cache was insufficient for the requested start date, so the experiment depends on the live vnstock KBS path for historical backfill.
- Long horizons near the end of the requested forecast window are intentionally kept in the outputs but excluded from scored metrics when realized target closes are not yet available.
- Strategy metrics are computed on overlapping forecast windows, so they are technical usefulness diagnostics rather than execution-ready portfolio PnL.
- The final stack is a regression meta-learner, so no final calibrated probability is emitted.

## Output Paths
- csv/: `outputs\walkforward_all_models\msn_mwg_dgc_ssi_fpt_acb\step_1\csv`
- charts/: `outputs\walkforward_all_models\msn_mwg_dgc_ssi_fpt_acb\step_1\charts`
- report.md: `outputs\walkforward_all_models\msn_mwg_dgc_ssi_fpt_acb\step_1\report.md`
