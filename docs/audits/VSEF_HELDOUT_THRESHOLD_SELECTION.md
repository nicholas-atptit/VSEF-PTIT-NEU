# VSEF Held-Out Threshold Selection
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Audit |
| Created / authored | Tuesday, 2026-04-28 15:25:36 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-04-28 22:34:05 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `8800ce6e3780c7978856737e70cb5e3b999eacee` |
| Timestamp source | Git history |
| Status | Active |

## Purpose

This audit adds a held-out threshold-selection protocol on top of the signal-effectiveness layer. The goal is to test whether BUY precision targets survive when thresholds are selected on a separate validation period and then applied unchanged to a later held-out period.

This remains diagnostic evaluation. It does not add model families, change governance status, modify forecast artifacts, or claim trading-performance proof.

## Why The Descriptive Frontier Is Not Enough

The signal-effectiveness frontier reports many threshold, cost, and slippage combinations on the same evaluated rows. That is useful for discovery, but it can overstate policy quality if a threshold is chosen after seeing the same realized returns used for reporting.

Held-out selection separates the task into two periods:

- selection period: choose the threshold using only earlier realized outcomes
- held-out test period: apply the selected threshold unchanged and report BUY precision

## Selection And Test Split Design

Rows are split by normalized `prediction_date`.

The current implementation supports pooled model-horizon selection:

```text
one selected threshold per model_name, horizon, minimum_signal_count
```

Ticker-specific threshold selection is intentionally deferred. Pooled model-horizon selection is the first governed version because it keeps signal counts higher and interpretation simpler.

## Threshold Selection Rule

For each model/horizon/minimum-count group, the selection period evaluates the same threshold grid used by the frontier workflow.

The selected threshold is chosen by:

1. maximize BUY precision subject to the configured minimum BUY signal count
2. tie-break by higher net average return after BUY
3. tie-break by larger BUY signal count

Held-out realized returns are not used during threshold selection.

## Precision Targets

The held-out report evaluates pass/fail targets:

- `0.60`
- `0.65`
- `0.70`

A target passes only when held-out BUY precision is not missing, held-out BUY count is positive, and held-out BUY precision is greater than or equal to the target.

## Outputs

Held-out mode writes:

- `selected_thresholds.csv`
- `heldout_buy_precision.csv`
- `threshold_selection_trace.csv`
- `heldout_signal_rows.csv`
- `precision_target_pass_fail.csv`
- `heldout_strategy_proxy_metrics.csv`
- `run_metadata.json`

`threshold_selection_trace.csv` preserves all selection-period candidates and marks which row was selected. `heldout_signal_rows.csv` contains only held-out test-period signal rows generated from selected thresholds.

## CLI Example

```powershell
C:\Users\luong\.venv\Scripts\python.exe scripts/run_signal_effectiveness_backtest.py `
  --predictions-path outputs\walkforward_15y_broader_ticker_basket_2024_2025\ssi_fpt_bvh_vnm_acb_hpg_vcb_vic\step_1\csv\stacking_predictions_detailed.csv `
  --output-dir outputs\heldout_threshold_selection_15y_broader_stack_step1 `
  --evaluation-mode heldout_threshold_selection `
  --selection-start 2024-01-01 `
  --selection-end 2024-12-31 `
  --test-start 2025-01-01 `
  --test-end 2025-12-31 `
  --models stacking_final `
  --policy strict_buy_precision_probe `
  --success-definition cost_adjusted_positive `
  --precision-targets 0.60,0.65,0.70
```

## Optional Saved-Output Run

The existing ignored broader 15-year fallback stacking output was available and was evaluated with:

- selection period: 2024-01-01 through 2024-12-31
- held-out period: 2025-01-01 through 2025-12-31
- model: `stacking_final`
- policy: `strict_buy_precision_probe`
- success definition: `cost_adjusted_positive`
- default threshold, cost, slippage, and minimum-signal grids

Selected held-out results:

| horizon | minimum BUY count | selected threshold | held-out BUY count | held-out BUY precision | 70% target |
| --- | ---: | ---: | ---: | ---: | --- |
| `short_10d` | 30 | 0.030 | 108 | 0.814815 | pass |
| `short_10d` | 50 | 0.030 | 108 | 0.814815 | pass |
| `short_10d` | 100 | 0.020 | 236 | 0.627119 | fail |
| `short_5d` | 30 | 0.015 | 63 | 0.761905 | pass |
| `short_5d` | 50 | 0.010 | 173 | 0.578035 | fail |
| `short_20d` | 30 | 0.015 | 1034 | 0.581238 | fail |
| `long_3m` | 30 | 0.005 | 1216 | 0.557566 | fail |

Interpretation: the 70% BUY precision target survived held-out testing for some high-threshold, lower minimum-count slices, especially `short_10d` and `short_5d`. It did not survive universally across horizons or stricter minimum-count settings.

Generated outputs remain under ignored `outputs/` and are not committed.

## Rolling Extension

One held-out split is not enough to treat a BUY precision target as stable. Use `--evaluation-mode rolling_heldout_threshold_selection` with multiple chronological split definitions to repeat the same selection/test protocol across folds and summarize threshold stability.

The rolling extension is documented in `docs/audits/VSEF_ROLLING_REGIME_THRESHOLD_SELECTION.md`. Its optional regime diagnostics run only when prediction rows already contain a recognized regime column; this branch does not infer or fabricate regime labels.

## Limitations

- This is still diagnostic signal evaluation, not trading-performance proof.
- Threshold selection currently pools tickers by model and horizon.
- The held-out window can still contain overlapping forecast horizons.
- The simple cost/slippage adjustment is not an execution simulator.
- Passing a precision target in one held-out period is not enough to declare production policy readiness.
- Final stacking has no calibrated probability output, so probability thresholds are skipped for that table.

## Next Steps

- Add ticker-specific selection once signal counts are sufficient.
- Add a governed join layer for regime labels if regime-conditioned rolling evaluation is needed on saved outputs without embedded regime columns.
- Compare selected thresholds against buy-and-hold and flat baselines with consistent assumptions.
