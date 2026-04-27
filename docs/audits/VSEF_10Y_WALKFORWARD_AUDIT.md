# VSEF 10-Year Walk-Forward Audit

Date: 2026-04-27

Branch: `vsef-10y-walkforward-audit`

## Purpose

This audit runs the existing all-model walk-forward evaluation workflow over the requested 2015-01-01 through 2025-12-31 window for a controlled smoke test and a four-ticker baseline.

This is a research and evaluation audit only. It does not add model families, change model governance status, or prove trading-performance improvement.

## Preflight

Starting branch:

```bash
git branch --show-current
git status
```

The new branch was created with:

```bash
git checkout -b vsef-10y-walkforward-audit
```

The repository ignore policy already covers generated outputs and local provider artifacts:

- `outputs/`
- `artifacts/`
- `data/foreign_flow_curated.csv`

Generated output folders and provider-backed CSVs are not intended to be committed.

## Smoke Test

Command:

```bash
python scripts/run_walkforward_all_models_stacking_eval.py --tickers SSI --history-start 2015-01-01 --history-end 2025-12-31 --initial-train-start 2015-01-01 --initial-train-end 2023-12-31 --forecast-start 2024-01-01 --forecast-end 2025-12-31 --horizons short_5d --step-sizes 5 --algorithms cart,xgboost,lightgbm --output-dir outputs/walkforward_10y_smoke_ssi --max-workers 1 --max-depth 3 --meta-min-samples 5 --epochs 1
```

Result: completed with exit code 0.

Output directory:

```text
outputs/walkforward_10y_smoke_ssi/ssi/
```

Smoke data coverage:

| ticker | source | fetched rows | available range | scored observations per model | forecast coverage |
| --- | --- | ---: | --- | ---: | ---: |
| `SSI` | `csv_fallback` | 1256 | 2020-12-21 through 2025-12-31 | 99 | 0.99 |

The intended 2015-01-01 start was not available in the local fallback data. The smoke test therefore validates the workflow on the maximum available local range, not on a full 10-year history.

Smoke metric summary:

| model | observations | MAE | RMSE | directional accuracy | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cart` | 99 | 0.037023 | 0.047227 | 0.5455 | 0.5794 |
| `lightgbm` | 99 | 0.046112 | 0.057481 | 0.6465 | 0.7059 |
| `stacking_final` | 99 | 0.032139 | 0.042405 | 0.4444 | 0.5865 |
| `xgboost` | 99 | 0.049472 | 0.061962 | 0.6061 | 0.6549 |

Smoke context coverage:

| ticker | horizon | fold count | mean breadth missing rate | mean foreign-flow missing rate | warning |
| --- | --- | ---: | ---: | ---: | --- |
| `SSI` | `short_5d` | 100 | 0.0000 | 1.0000 | `weak_coverage` |

## Four-Ticker Baseline

Command:

```bash
python scripts/run_walkforward_all_models_stacking_eval.py --tickers SSI FPT ACB HPG --history-start 2015-01-01 --history-end 2025-12-31 --initial-train-start 2015-01-01 --initial-train-end 2023-12-31 --forecast-start 2024-01-01 --forecast-end 2025-12-31 --horizons short_5d --step-sizes 5 --algorithms cart,xgboost,lightgbm --output-dir outputs/walkforward_10y_baseline --max-workers 1 --max-depth 3 --meta-min-samples 5 --epochs 1
```

Result: completed with exit code 0.

Output directory:

```text
outputs/walkforward_10y_baseline/ssi_fpt_acb_hpg/
```

Generated diagnostics include:

- `csv/summary_by_model.csv`
- `csv/summary_by_ticker.csv`
- `csv/forecast_coverage_summary.csv`
- `csv/context_coverage_summary.csv`
- `csv/feature_governance_review.csv`
- `csv/feature_importance_stability_summary.csv`
- `csv/linear_coefficient_stability_summary.csv`
- `csv/linear_vs_importance_feature_comparison.csv`
- `csv/run_metadata.json`

## Data Coverage

The requested historical input window was 2015-01-01 through 2025-12-31. The local fallback data actually available for every audited ticker starts on 2020-12-21.

| ticker | source | fetched rows | available range | audited |
| --- | --- | ---: | --- | --- |
| `ACB` | `csv_fallback` | 1256 | 2020-12-21 through 2025-12-31 | yes |
| `FPT` | `csv_fallback` | 1256 | 2020-12-21 through 2025-12-31 | yes |
| `HPG` | `csv_fallback` | 1256 | 2020-12-21 through 2025-12-31 | yes |
| `SSI` | `csv_fallback` | 1256 | 2020-12-21 through 2025-12-31 | yes |

The intended 10-year window was not fully available in this local run. The baseline should be read as a maximum-available-window audit over the local CSV fallback cache.

## Model Scope

Models run:

- `cart`
- `xgboost`
- `lightgbm`
- `stacking_final` as the existing walk-forward stack over the base-model outputs

Horizon and step:

| setting | value |
| --- | --- |
| horizon | `short_5d` |
| resolved horizon | 5 trading days |
| step size | 5 |
| max workers | 1 |
| max depth | 3 |
| meta minimum samples | 5 |
| epochs | 1 |

No new model families were added.

## Forecast Coverage

Each ticker/model combination emitted 100 predictions, of which 99 were evaluation-eligible. The last prediction row was kept in detailed outputs but excluded from scored metrics because the realized 5-day target was not available inside the fetched history.

| ticker | model count | total predictions per model | eligible predictions per model | coverage ratio | first prediction | last prediction | last eligible prediction |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `ACB` | 4 | 100 | 99 | 0.99 | 2024-01-02 | 2025-12-26 | 2025-12-19 |
| `FPT` | 4 | 100 | 99 | 0.99 | 2024-01-02 | 2025-12-26 | 2025-12-19 |
| `HPG` | 4 | 100 | 99 | 0.99 | 2024-01-02 | 2025-12-26 | 2025-12-19 |
| `SSI` | 4 | 100 | 99 | 0.99 | 2024-01-02 | 2025-12-26 | 2025-12-19 |

## Baseline Metric Summary

Overall summary by model:

| model | observations | MAE | RMSE | correlation | directional accuracy | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cart` | 396 | 0.034319 | 0.048766 | -0.0420 | 0.4848 | 0.5231 | 0.4789 | 0.5000 |
| `lightgbm` | 396 | 0.040792 | 0.051554 | -0.0543 | 0.5429 | 0.5976 | 0.4601 | 0.5199 |
| `stacking_final` | 396 | 0.027147 | 0.036845 | -0.0462 | 0.5227 | 0.5308 | 0.9718 | 0.6866 |
| `xgboost` | 396 | 0.041556 | 0.052547 | -0.0175 | 0.5051 | 0.5630 | 0.3568 | 0.4368 |

Observations:

- `stacking_final` had the lowest MAE and RMSE in this diagnostic run.
- `lightgbm` had the highest directional accuracy.
- All correlations were near zero or negative, so error and directional metrics should be interpreted conservatively.
- These metrics are model-evaluation diagnostics only, not trading-performance evidence.

## Context Coverage

Four-ticker baseline context coverage:

| ticker | horizon | fold count | mean breadth missing rate | max breadth missing rate | mean foreign-flow missing rate | max foreign-flow missing rate | weak coverage folds | warning |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ACB` | `short_5d` | 100 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 100 | `weak_coverage` |
| `FPT` | `short_5d` | 100 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 100 | `weak_coverage` |
| `HPG` | `short_5d` | 100 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 100 | `weak_coverage` |
| `SSI` | `short_5d` | 100 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 100 | `weak_coverage` |

The baseline was run without `--foreign-flow-path`. Run metadata recorded:

| field | value |
| --- | --- |
| `foreign_flow_path` | null |
| `foreign_flow_path_explicit` | false |
| `source_name` | `foreign_flow.csv` |
| `row_count` | 1 |
| `artifact_validation` | null |

Foreign-flow feature interpretation must remain conservative because fold-level foreign-flow coverage is absent in the baseline.

## Feature Governance

The baseline generated governance diagnostics without changing model governance status.

| category/action | count |
| --- | ---: |
| `safe_trailing` features | 12 |
| `requires_review` features | 9 |
| `keep` | 9 |
| `keep_but_document` | 3 |
| `review_timing` | 9 |

The `requires_review` features were joined breadth/context features whose source-date alignment and missing-context handling remain governance review items.

## Stability Diagnostics

Feature importance stability summary:

| stability level | count |
| --- | ---: |
| `high` | 78 |
| `medium` | 7 |
| `low` | 8 |

Linear coefficient stability summary:

| stability level | count |
| --- | ---: |
| `high` | 15 |
| `medium` | 6 |
| `low` | 18 |

Linear-vs-importance alignment:

| alignment label | count |
| --- | ---: |
| `aligned_stable` | 10 |
| `importance_only` | 21 |

Notable stability observations:

- Several return features were aligned across linear and importance diagnostics, including `bb_width`, `close_return_10d`, `close_to_sma_200`, `dist_ma_20`, `dist_ma_60`, `pct_above_ma20`, `range_20`, `rolling_volatility_60`, and `turnover_ma_60`.
- Some features were stable in tree importance but weak in linear coefficient stability, including `ema_50`, `macd_signal`, and `rolling_max_60`.
- Profit and trend task entries were importance-only in the alignment file because the linear diagnostic summary covers return coefficients.

## Foreign-Flow Coverage Audit

The local provider-backed artifact exists:

```text
data/foreign_flow_curated.csv
```

Coverage audit command:

```bash
python scripts/audit_foreign_flow_coverage.py --tickers SSI,FPT,ACB,HPG --start-date 2015-01-01 --end-date 2025-12-31 --foreign-flow-path data/foreign_flow_curated.csv
```

Result:

| field | value |
| --- | --- |
| file exists | yes |
| source status | `loaded_local_artifact` |
| row count | 68 |
| provider | `vnstock_data` |
| source | `vnstock_data.Trading.foreign_trade` |
| retrieved at | 2026-04-26T16:03:56Z |
| artifact classification | `partial_coverage` |
| requested business dates | 2870 |
| requested ticker/date pairs | 11480 |
| matched ticker/date pairs | 68 |
| requested ticker/date coverage rate | 0.005923 |
| artifact date range | 2025-01-02 through 2025-01-24 |
| real provider evidence | true |
| suitable for foreign feature interpretation | false |
| suitable for performance interpretation | false |

Ticker-level exact-join coverage against local OHLCV dates:

| ticker | requested OHLCV dates | foreign-flow rows | exact join matches | exact join missing | exact join missing ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `SSI` | 1256 | 17 | 17 | 1239 | 0.9865 |
| `FPT` | 1256 | 17 | 17 | 1239 | 0.9865 |
| `ACB` | 1256 | 17 | 17 | 1239 | 0.9865 |
| `HPG` | 1256 | 17 | 17 | 1239 | 0.9865 |

Because the artifact covers only a short January 2025 slice, the optional 10-year foreign-flow walk-forward was skipped. A full run with `--foreign-flow-path data/foreign_flow_curated.csv` would still require conservative interpretation because `foreign_*` coverage is weak for the requested long window.

## Runtime Warnings

Observed runtime warnings and notes:

- `vnstock_dataset_unavailable` with `reason=vnstock_data_not_installed`; OHLCV loaded from CSV fallback.
- `market_proxy_not_found` for `data/market_proxy.csv`.
- Market proxy context was stubbed because no market proxy artifact and no provider fallback were available.
- Sentiment integration was explicitly disabled and stubbed by the pipeline.
- LightGBM repeatedly logged `No further splits with positive gain`.
- scikit-learn logged feature-name warnings for LightGBM predictions.
- SciPy logged ill-conditioned matrix warnings in linear diagnostics.

These warnings did not stop the smoke or baseline run, but they are material limitations for interpretation.

## Validation

Validation commands run after documentation updates:

| command | result |
| --- | --- |
| `python -m pytest tests/ml/test_walk_forward_foreign_flow_path.py -q` | 6 passed |
| `python -m pytest tests/ml/test_foreign_flow_artifact_validation.py -q` | 10 passed |
| `python -m pytest tests/ml/test_foreign_flow_coverage.py -q` | 7 passed |
| `python -m pytest tests/ml -q` | 167 passed, 2 skipped |
| `python -m pytest tests/quant_core -q` | 15 passed |
| `python -m pytest tests/phase1/test_forecast_contracts.py -q` | 3 passed |
| `python -m compileall src` | passed |

Validation warnings were non-fatal and included pytest cache write permission warnings for `.pytest_cache`, LightGBM/scikit-learn feature-name warnings, class-label warnings in ML tests, and logical-core detection fallback from joblib.

## Limitations

- The requested 2015-01-01 start was not available locally. The actual fetched range was 2020-12-21 through 2025-12-31 for all four tickers.
- The audit used local CSV fallback data because `vnstock_data` was unavailable in this environment.
- The default foreign-flow context was effectively unavailable for the audited tickers.
- `data/foreign_flow_curated.csv` is provider-backed but far too sparse for a 2015-2025 interpretation.
- Forecast outputs near the end of the window may be ineligible for scoring when the future target close is not available.
- MAPE and SMAPE can be unstable for return targets near zero; MAE, RMSE, and directional metrics are more useful here.
- Strategy and backtest artifacts generated by the workflow are diagnostics, not execution-ready portfolio evidence.

## Next Recommended Task

Run a dedicated data-availability audit to extend or replace the local daily OHLCV cache before repeating the same walk-forward workflow for a true 10-year window. Foreign-flow should remain a separate governed context audit until long-window provider coverage is available.

## Follow-Up: OHLCV Cache Availability

`docs/audits/VSEF_OHLCV_CACHE_10Y_AVAILABILITY_AUDIT.md` adds the dedicated local cache audit. It confirms that `SSI`, `FPT`, `ACB`, and `HPG` cache files are present but start on 2020-12-21, covering 1256 of 2870 requested business dates in the 2015-01-01 through 2025-12-31 baseline window.
