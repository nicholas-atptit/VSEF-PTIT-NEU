# VSEF Signal Regime Join

## Purpose

This audit adds a CSV-only join layer for attaching safe precomputed regime labels to saved prediction outputs before running signal-effectiveness, held-out threshold, or rolling held-out diagnostics.

The previous rolling held-out run showed that the 70% BUY precision target did not survive broadly across chronological folds. Regime-conditioned diagnostics are the next question: whether BUY precision is materially different in bull, bear, sideway, or high-volatility states. That analysis requires regime labels on the prediction rows.

This branch does not infer regimes, train models, fetch provider data, modify input prediction files, or claim trading-performance proof.

## Utility

Module:

```text
src/ml/backtest/signal_regime_join.py
```

CLI:

```text
scripts/join_regime_to_predictions.py
```

The utility reads a saved prediction CSV and a precomputed regime CSV, joins labels onto a copy of the predictions, writes an enriched prediction CSV, and writes a join coverage summary.

The enriched prediction column is always:

```text
regime
```

If the prediction file already contains `regime`, the utility preserves it unless `--overwrite-regime` is passed.

## Supported Join Modes

`date`:

- joins prediction rows by `prediction_date` to a regime date column
- default mode
- appropriate for market-wide labels such as VNINDEX trailing-return regimes

`ticker_date`:

- joins by ticker and date
- appropriate only when the regime source is ticker-specific

The default prediction date column is `prediction_date`. The default regime date column is `date`. The regime label column can be passed explicitly or auto-detected from:

- `regime`
- `market_regime`
- `regime_label`
- `market_state`

## Governance Checks

The summary records:

- prediction row count
- regime row count
- matched prediction rows
- unmatched prediction rows
- matched rate
- min/max prediction date
- min/max regime date
- join mode
- regime label distribution
- duplicate regime key detection
- suspicious future-looking regime source columns

Suspicious column-name patterns include:

- `future`
- `forward`
- `lead`
- `target`
- `realized`
- `actual_return`
- `future_return`

Join governance is classified as:

- `safe_if_regime_source_is_trailing`: schema is valid and no duplicate or suspicious source columns were detected
- `requires_source_review`: duplicate regime keys or suspicious source columns were detected
- `schema_invalid`: required join or regime-label columns are missing

## What This Does Not Prove

The join layer does not prove that a regime source is leakage-free. It records coverage and source-review flags. A regime file is only safe if its labels were generated from information available at or before the prediction date.

The utility does not fabricate missing labels. Unmatched prediction rows are preserved with missing `regime`.

## CLI Example

Date join:

```powershell
C:\Users\luong\.venv\Scripts\python.exe scripts/join_regime_to_predictions.py `
  --predictions-path outputs\walkforward_15y_broader_ticker_basket_2024_2025\ssi_fpt_bvh_vnm_acb_hpg_vcb_vic\step_1\csv\stacking_predictions_detailed.csv `
  --regime-path outputs\safe_regime_labels\market_regimes.csv `
  --output-path outputs\signal_regime_join\stacking_predictions_with_regime.csv `
  --summary-path outputs\signal_regime_join\join_summary.json `
  --join-mode date `
  --regime-date-column date
```

Ticker/date join:

```powershell
C:\Users\luong\.venv\Scripts\python.exe scripts/join_regime_to_predictions.py `
  --predictions-path outputs\walkforward_15y_broader_ticker_basket_2024_2025\ssi_fpt_bvh_vnm_acb_hpg_vcb_vic\step_1\csv\stacking_predictions_detailed.csv `
  --regime-path outputs\safe_regime_labels\ticker_regimes.csv `
  --output-path outputs\signal_regime_join\stacking_predictions_with_ticker_regime.csv `
  --summary-path outputs\signal_regime_join\ticker_join_summary.json `
  --join-mode ticker_date `
  --ticker-column ticker
```

## Rerunning Rolling Diagnostics

After an enriched prediction CSV exists, rerun rolling held-out threshold selection with regime diagnostics:

```powershell
C:\Users\luong\.venv\Scripts\python.exe scripts/run_signal_effectiveness_backtest.py `
  --predictions-path outputs\signal_regime_join\stacking_predictions_with_regime.csv `
  --output-dir outputs\rolling_heldout_threshold_selection_with_regime `
  --evaluation-mode rolling_heldout_threshold_selection `
  --rolling-splits "2024-01-01:2024-06-30:2024-07-01:2024-12-31,2024-07-01:2024-12-31:2025-01-01:2025-06-30,2025-01-01:2025-06-30:2025-07-01:2025-12-31" `
  --models stacking_final `
  --policy strict_buy_precision_probe `
  --success-definition cost_adjusted_positive `
  --precision-targets 0.60,0.65,0.70 `
  --enable-regime-diagnostics
```

## Optional Real-Output Preparation

The broader stacking prediction file exists at:

```text
outputs\walkforward_15y_broader_ticker_basket_2024_2025\ssi_fpt_bvh_vnm_acb_hpg_vcb_vic\step_1\csv\stacking_predictions_detailed.csv
```

No existing safe regime-label CSV was found under the current ignored `outputs/` tree during this branch. The files found were context coverage diagnostics, not regime label sources. Therefore no real enriched prediction file was created and rolling regime diagnostics were not rerun.

## Limitations

- The utility does not infer regimes.
- The utility does not prove the source regime file is leakage-free.
- Duplicate regime keys are reported and the first row is used for the join.
- Missing regime labels are not filled or fabricated.
- Enriched prediction outputs are generated artifacts and should remain uncommitted.

## Next Steps

- Produce a governed trailing-only market regime label table with one row per date.
- Optionally produce ticker-specific trailing regime labels if there is a defensible ticker-level regime definition.
- Join those labels to saved predictions and rerun rolling held-out threshold diagnostics with `--enable-regime-diagnostics`.
