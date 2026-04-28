# VSEF Rolling Regime Threshold Selection

## Purpose

This audit extends the signal-effectiveness and held-out threshold-selection layer with `rolling_heldout_threshold_selection`.

The goal is to test whether BUY precision targets such as `0.60`, `0.65`, and `0.70` survive across multiple chronological selection/test splits. The mode also supports optional regime-conditioned BUY precision diagnostics when prediction rows already contain a safe regime label.

This remains diagnostic evaluation. It does not add model families, change model governance status, modify forecast artifacts, or claim trading-performance proof.

## Why One Held-Out Split Is Not Enough

A single held-out split can show that a threshold survived one later period, but it does not show that the threshold is stable through changing market conditions. Rolling held-out testing repeats the same selection rule across several chronological folds and reports whether selected thresholds and held-out BUY precision remain consistent.

Rolling held-out testing is required before treating a BUY precision target as stable. Regime-conditioned evaluation is needed before deciding whether BUY rules should be active in all market states or only in favorable regimes.

## Rolling Split Design

Rows are filtered by normalized `prediction_date`.

Inline CLI syntax:

```text
--rolling-splits "selection_start:selection_end:test_start:test_end,selection_start:selection_end:test_start:test_end"
```

Each split requires:

```text
selection_start:selection_end:test_start:test_end
```

JSON or CSV split files are also supported with fields or columns:

- `fold_id`, optional
- `selection_start`
- `selection_end`
- `test_start`
- `test_end`

The test window must start after the selection window ends. Test-period realized returns are not used to select thresholds.

## Threshold Selection Rule

For each fold, model, horizon, and minimum BUY count:

1. evaluate the threshold grid on the selection period
2. maximize BUY precision subject to the configured minimum BUY signal count
3. tie-break by higher net average return after BUY
4. tie-break by larger BUY count
5. apply the selected threshold unchanged to the fold test period

Selection remains pooled by `model_name` and `horizon`. Ticker-specific threshold selection remains deferred.

## Output Files

Rolling mode writes:

- `rolling_selected_thresholds.csv`
- `rolling_heldout_buy_precision.csv`
- `rolling_threshold_selection_trace.csv`
- `rolling_heldout_signal_rows.csv`
- `rolling_precision_target_pass_fail.csv`
- `rolling_strategy_proxy_metrics.csv`
- `threshold_stability_summary.csv`
- `run_metadata.json`

If regime labels are available, it also writes:

- `regime_buy_precision_summary.csv`
- `regime_precision_stability_summary.csv`

## Threshold Stability Summary

`threshold_stability_summary.csv` aggregates selected thresholds and held-out precision across folds by `model_name`, `horizon`, and `minimum_signal_count`.

Stability labels are intentionally simple:

- `high`: same threshold selected in at least 80% of folds and at least 3 folds exist
- `medium`: same threshold selected in at least 60% of folds and at least 3 folds exist
- `low`: otherwise

The summary also reports held-out BUY precision mean, standard deviation, min/max, total held-out BUY count, and precision target pass rates for `0.60`, `0.65`, and `0.70`.

## Regime-Conditioned Diagnostics

The rolling mode recognizes existing regime columns:

- `regime`
- `market_regime`
- `regime_label`
- `market_state`

If a regime label is present, the runner preserves it as `regime` and reports BUY precision by fold, model, horizon, regime, and minimum BUY count. It also aggregates regime precision stability across folds with labels:

- `promising`: mean BUY precision is at least `0.70` and total BUY count meets the configured minimum
- `mixed`: mean BUY precision is at least `0.60` but below `0.70`
- `weak`: below `0.60`
- `insufficient`: not enough BUY signals

If no regime column exists, the run does not fail. Metadata records that regime diagnostics were skipped. Use `scripts/join_regime_to_predictions.py` to attach precomputed safe regime labels before rerunning rolling diagnostics. The join layer does not infer regimes and does not prove leakage absence by itself; it records coverage and source-review flags.

## Optional Saved-Output Run

The ignored broader stacking output existed at:

```text
outputs\walkforward_15y_broader_ticker_basket_2024_2025\ssi_fpt_bvh_vnm_acb_hpg_vcb_vic\step_1\csv\stacking_predictions_detailed.csv
```

Command:

```powershell
C:\Users\luong\.venv\Scripts\python.exe scripts/run_signal_effectiveness_backtest.py `
  --predictions-path outputs\walkforward_15y_broader_ticker_basket_2024_2025\ssi_fpt_bvh_vnm_acb_hpg_vcb_vic\step_1\csv\stacking_predictions_detailed.csv `
  --output-dir outputs\rolling_heldout_threshold_selection_15y_broader_stack_step1 `
  --evaluation-mode rolling_heldout_threshold_selection `
  --rolling-splits "2024-01-01:2024-06-30:2024-07-01:2024-12-31,2024-07-01:2024-12-31:2025-01-01:2025-06-30,2025-01-01:2025-06-30:2025-07-01:2025-12-31" `
  --models stacking_final `
  --policy strict_buy_precision_probe `
  --success-definition cost_adjusted_positive `
  --precision-targets 0.60,0.65,0.70
```

Selected rolling result:

| horizon | minimum BUY count | selected thresholds by fold | mean held-out BUY precision | total held-out BUY count | 70% pass rate |
| --- | ---: | --- | ---: | ---: | ---: |
| `long_3m` | 30 | `0.005,0.005,0.015` | 0.558987 | 2042 | 0.000000 |
| `short_10d` | 30 | `0.030,0.005,0.010` | 0.543297 | 1312 | 0.000000 |
| `short_20d` | 30 | `0.015,0.020,0.020` | 0.618828 | 1379 | 0.333333 |
| `short_5d` | 30 | `0.015,0.005,0.005` | 0.442028 | 769 | 0.000000 |

Interpretation: the 70% BUY precision target did not survive across rolling folds. It passed only for `short_20d` in fold 2, where held-out BUY precision was `0.840000` with 150 BUY signals. No regime-conditioned results were produced because the saved prediction output did not contain a recognized regime column. A later join layer can enrich a copy of the prediction file once safe precomputed regime labels are available.

Generated outputs remain under ignored `outputs/` and are not committed.

## Limitations

- Rolling folds can still contain overlapping forecast horizons.
- Regime diagnostics require existing safe regime labels in prediction rows.
- The signal-regime join layer can attach precomputed labels, but it does not infer regimes and does not prove source leakage absence.
- Stability labels are descriptive and intentionally coarse.
- The proxy return metrics are not execution simulation or trading-performance proof.

## Next Steps

- Produce or review a trailing-only regime label source, join it to saved predictions, and rerun rolling diagnostics with regime reporting enabled.
- Compare rolling stability across additional saved walk-forward runs before promoting any BUY precision target.
- Consider ticker-specific selection only after signal counts are sufficient.
