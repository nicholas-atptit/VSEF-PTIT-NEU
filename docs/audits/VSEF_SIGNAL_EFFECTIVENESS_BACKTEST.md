# VSEF Signal-Effectiveness Backtest

## Purpose

This audit adds a non-invasive signal-effectiveness layer on top of existing forecast outputs. It converts saved prediction rows into investment-style diagnostic labels:

- `BUY`
- `HOLD`
- `AVOID`

The layer does not add model families, change model governance status, fetch live provider data, modify forecast artifacts, or claim trading-performance proof.

## Why Forecast Metrics Are Not Enough

MAE and RMSE measure point-forecast error. Directional accuracy measures whether the predicted sign matched the realized sign. These are useful model diagnostics, but they do not answer whether a forecast can support a selective investment decision.

The signal-effectiveness layer asks a narrower question: when transparent rules say `BUY`, how often is that BUY correct under explicit success and cost assumptions?

## Input Forecast Tables

The module can read CSV outputs already produced by existing workflows:

- fixed forward-return output: `artifacts/backtest_forward_return/<horizon>/predicted_vs_actual.csv`
- all-model walk-forward output: `predictions_detailed.csv`
- stacking output: `stacking_predictions_detailed.csv`

Supported columns are normalized to:

- `ticker`
- `prediction_date`
- `target_date`, when available
- `model_name`
- `horizon`
- `predicted_return`
- `realized_forward_return`
- optional `predicted_direction`
- optional upward probability columns such as `predicted_positive_probability`

If an upward probability column is not present or is all missing, probability-threshold rules are skipped and metadata records that probability calibration is a future task.

## Signal Definitions

Implemented policies:

- `return_threshold`: `BUY` if `predicted_return >= threshold`; `AVOID` if `predicted_return <= -threshold`; otherwise `HOLD`.
- `direction_and_return_threshold`: `BUY` if `predicted_return >= threshold` and predicted direction is positive; `AVOID` if `predicted_return <= -threshold`; otherwise `HOLD`.
- `strict_buy_precision_probe`: reports the full threshold frontier and does not select a threshold using realized future returns.

The current layer does not emit `SELL`; negative forecasts are mapped to `AVOID`.

## BUY Precision Definition

BUY precision is:

```text
successful BUY signals / total BUY signals
```

BUY recall is reported when computable:

```text
successful BUY signals / all successful realized cases under the selected success definition
```

## Success Definitions

Supported BUY correctness definitions:

- `raw_positive`: correct if `realized_forward_return > 0`.
- `cost_adjusted_positive`: correct if `realized_forward_return > estimated_round_trip_cost`.
- `target_return`: correct if `realized_forward_return >= target_return_threshold`.

The default is `cost_adjusted_positive`.

## Cost And Slippage Assumptions

The diagnostic cost model is a simple forward-return adjustment:

```text
estimated_round_trip_cost = 2 * (cost_per_trade + slippage)
net_realized_return_after_costs = realized_forward_return - estimated_round_trip_cost
```

This is intentionally lighter than `strategy_backtest.py`, which fetches execution OHLCV and models entry and exit prices. Signal-effectiveness is a CSV-only diagnostic layer.

## Threshold Grid

Default grids:

- `predicted_return_threshold`: `0.005`, `0.01`, `0.015`, `0.02`, `0.03`, `0.05`
- `cost_per_trade`: `0.001`, `0.002`, `0.003`
- `slippage`: `0.0005`, `0.001`
- `minimum_signal_count`: `30`, `50`, `100`
- `probability_up_threshold`, only when available: `0.55`, `0.60`, `0.65`, `0.70`

## Output Files

The CLI writes:

- `signal_rows.csv`
- `buy_precision_by_model_horizon.csv`
- `precision_coverage_frontier.csv`
- `signal_effectiveness_summary.csv`
- `strategy_proxy_metrics.csv`
- `benchmark_comparison.csv`
- `run_metadata.json`

`buy_precision_by_model_horizon.csv` includes only model-horizon-threshold rows that pass the configured minimum BUY signal count. The full frontier remains available in `precision_coverage_frontier.csv`.

## CLI

Example:

```powershell
C:\Users\luong\.venv\Scripts\python.exe scripts/run_signal_effectiveness_backtest.py `
  --predictions-path outputs\walkforward_all_models\ssi_fpt_bvh_vnm_acb_hpg_vcb_vic\step_1\csv\stacking_predictions_detailed.csv `
  --output-dir outputs\signal_effectiveness\stacking_step1 `
  --models stacking_final `
  --policy strict_buy_precision_probe `
  --success-definition cost_adjusted_positive
```

The script never modifies the input prediction CSV.

## Limitations

- This is diagnostic signal evaluation, not live trading proof.
- The simple net-return adjustment does not model order book depth, liquidity, intraday execution, taxes, borrow constraints, or portfolio sizing.
- Overlapping forecast horizons can inflate apparent opportunity counts.
- Probability thresholds are used only when a usable probability column exists; final stacking currently emits return and direction but no calibrated probability.
- Threshold frontiers are descriptive. They should not be treated as production thresholds without a separate validation design.

## Next Steps

- Run the CLI on ignored walk-forward outputs when available.
- Compare BUY precision by ticker, horizon, and regime.
- Add out-of-period threshold selection if production-style policy selection is needed.
- Consider probability calibration only for models and horizons with governed probability semantics.
