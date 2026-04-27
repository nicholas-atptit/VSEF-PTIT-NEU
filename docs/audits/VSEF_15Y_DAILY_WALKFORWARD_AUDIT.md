# VSEF 15-Year Daily Walk-Forward Audit

Date: 2026-04-27

Branch: `vsef-15y-daily-walkforward-audit`

## Purpose

This audit stages canonical `vnstock_data` OHLCV for `SSI`, `FPT`, `BVH`, and `VNM`, validates coverage for 2010-01-01 through 2025-12-31, then runs the existing daily walk-forward evaluation with `--step-sizes 1`.

This is research and evaluation evidence only. It is not trading-performance proof, does not add model families, and does not change model governance status.

`--step-sizes 1` means every available trading session in the forecast window, not every calendar day.

## Runtime And Provider

| check | result |
| --- | --- |
| default `python` | `C:\Python\python.exe`, Python 3.13.5 |
| project venv | `C:\Users\luong\.venv\Scripts\python.exe`, Python 3.13.5 |
| default `python` canonical provider | `vnstock_data = None` |
| default `python` alternate provider | `vnstock` available |
| project venv canonical provider | `vnstock_data` available |
| project venv alternate provider | `vnstock` available |
| runtime used for refresh/evaluation/tests | `C:\Users\luong\.venv\Scripts\python.exe` |

The run used canonical `vnstock_data`. The alternate `vnstock` package was not used as a substitute.

## Preflight Findings

- `scripts/extract_daily_csv.py` accepts `--days`, computes `today - days` through today, and writes per-ticker CSVs.
- The extractor appends `_data` to `--output` unless the supplied path already ends in `_data`.
- `scripts/audit_ohlcv_cache_coverage.py` is read-only and accepts `--data-dir`.
- The walk-forward runner previously used provider first, then the tracked default CSV fallback at `data/daily_market_split_data`.
- This branch adds `--ohlcv-data-dir` so staged OHLCV CSVs can be used directly without overwriting tracked cache files.
- If `--ohlcv-data-dir` is supplied, missing directories or missing ticker files fail clearly instead of silently falling back to `data/daily_market_split_data`.

## Staged OHLCV Refresh

Initial console execution failed before fetching due Windows cp1252 output encoding. The command was rerun with UTF-8 stdout and provider network access:

```powershell
$env:PYTHONIOENCODING='utf-8'; C:\Users\luong\.venv\Scripts\python.exe scripts/extract_daily_csv.py --tickers SSI FPT BVH VNM --days 6000 --output tmp/ohlcv_15y_refresh_probe
```

Result: exit code 0.

Staged output directory:

```text
tmp/ohlcv_15y_refresh_probe_data
```

The refresh wrote 16,388 rows across four files. The staged CSVs remain ignored and were not committed.

## OHLCV Coverage Audit

Command:

```powershell
C:\Users\luong\.venv\Scripts\python.exe scripts/audit_ohlcv_cache_coverage.py --tickers SSI,FPT,BVH,VNM --start-date 2010-01-01 --end-date 2025-12-31 --data-dir tmp/ohlcv_15y_refresh_probe_data
```

Top-level result:

| field | value |
| --- | --- |
| requested business-day count | 4,174 |
| data directory | `tmp/ohlcv_15y_refresh_probe_data` |
| missing file count | 0 |
| supporting ticker count | 4 |
| all tickers support requested window | true |
| canonical provider available | true |
| provider fetch attempted by audit script | false |

Per-ticker staged coverage:

| ticker | staged rows | staged min date | staged max date | matched requested business dates | missing business dates | coverage rate | supports requested window |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| `SSI` | 4,097 | 2009-11-23 | 2026-04-24 | 3,993 | 181 | 0.956636 | yes |
| `FPT` | 4,097 | 2009-11-23 | 2026-04-24 | 3,993 | 181 | 0.956636 | yes |
| `BVH` | 4,097 | 2009-11-23 | 2026-04-24 | 3,993 | 181 | 0.956636 | yes |
| `VNM` | 4,097 | 2009-11-23 | 2026-04-24 | 3,993 | 181 | 0.956636 | yes |

The staged data spans the requested 2010-01-01 through 2025-12-31 window for all four tickers. The first actual market session used inside the run was 2010-01-04, and the run clipped each ticker to 3,993 rows through 2025-12-31. The 181 missing generic business dates are consistent with a conservative Monday-Friday denominator that does not model Vietnamese exchange holidays.

## Runner Change

Files changed for first-class staged OHLCV support:

- `scripts/run_walkforward_all_models_stacking_eval.py`
- `src/ml/backtest/walk_forward_all_models_stacking.py`
- `tests/ml/test_walk_forward_ohlcv_data_dir.py`

The run metadata records:

```json
{
  "ohlcv_data_dir": "tmp/ohlcv_15y_refresh_probe_data",
  "ohlcv_data_dir_explicit": true,
  "source_mode": "explicit_csv_dir"
}
```

## Smoke Test

Command:

```powershell
$env:PYTHONIOENCODING='utf-8'; C:\Users\luong\.venv\Scripts\python.exe scripts/run_walkforward_all_models_stacking_eval.py --tickers SSI --history-start 2010-01-01 --history-end 2025-12-31 --initial-train-start 2010-01-01 --initial-train-end 2022-12-31 --forecast-start 2023-01-01 --forecast-end 2025-12-31 --horizons short_5d --step-sizes 1 --algorithms cart,xgboost,lightgbm --ohlcv-data-dir tmp/ohlcv_15y_refresh_probe_data --output-dir outputs/walkforward_15y_daily_smoke_ssi --max-workers 1 --max-depth 3 --meta-min-samples 5 --epochs 1
```

Result: exit code 0.

Output directory:

```text
outputs/walkforward_15y_daily_smoke_ssi/ssi
```

Smoke coverage:

| item | value |
| --- | ---: |
| staged OHLCV rows loaded | 3,993 |
| fetched date range | 2010-01-04 through 2025-12-31 |
| base predictions | 2,244 |
| base eligible predictions | 2,229 |
| stacking predictions | 748 |
| stacking eligible predictions | 743 |
| per model total predictions | 748 |
| per model eligible predictions | 743 |
| coverage ratio | 0.993316 |

Smoke model summary:

| model | observations | MAE | RMSE | directional accuracy | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cart` | 743 | 0.056725 | 0.079135 | 0.553163 | 0.633554 |
| `lightgbm` | 743 | 0.052547 | 0.067854 | 0.524899 | 0.609945 |
| `stacking_final` | 743 | 0.035727 | 0.048860 | 0.554509 | 0.710917 |
| `xgboost` | 743 | 0.059767 | 0.078339 | 0.495289 | 0.558304 |

## Four-Ticker Daily Walk-Forward

Command:

```powershell
$env:PYTHONIOENCODING='utf-8'; C:\Users\luong\.venv\Scripts\python.exe scripts/run_walkforward_all_models_stacking_eval.py --tickers SSI FPT BVH VNM --history-start 2010-01-01 --history-end 2025-12-31 --initial-train-start 2010-01-01 --initial-train-end 2022-12-31 --forecast-start 2023-01-01 --forecast-end 2025-12-31 --horizons short_5d --step-sizes 1 --algorithms cart,xgboost,lightgbm --ohlcv-data-dir tmp/ohlcv_15y_refresh_probe_data --output-dir outputs/walkforward_15y_daily_ssi_fpt_bvh_vnm --max-workers 1 --max-depth 3 --meta-min-samples 5 --epochs 1
```

Result: exit code 0.

Output directory:

```text
outputs/walkforward_15y_daily_ssi_fpt_bvh_vnm/ssi_fpt_bvh_vnm
```

Run data coverage:

| ticker | source | rows loaded | available range |
| --- | --- | ---: | --- |
| `BVH` | `csv_explicit_ohlcv_data_dir` | 3,993 | 2010-01-04 through 2025-12-31 |
| `FPT` | `csv_explicit_ohlcv_data_dir` | 3,993 | 2010-01-04 through 2025-12-31 |
| `SSI` | `csv_explicit_ohlcv_data_dir` | 3,993 | 2010-01-04 through 2025-12-31 |
| `VNM` | `csv_explicit_ohlcv_data_dir` | 3,993 | 2010-01-04 through 2025-12-31 |

Prediction counts:

| scope | total predictions | eligible predictions |
| --- | ---: | ---: |
| base models | 8,976 | 8,916 |
| stacking final | 2,992 | 2,972 |
| each ticker/base-model pair | 748 | 743 |
| each ticker/stacking pair | 748 | 743 |

Forecast coverage was identical for every ticker/model row:

| field | value |
| --- | --- |
| first prediction date | 2023-01-03 |
| last prediction date | 2025-12-31 |
| last eligible prediction date | 2025-12-24 |
| total predictions per ticker/model | 748 |
| eligible predictions per ticker/model | 743 |
| ineligible predictions per ticker/model | 5 |
| coverage ratio | 0.993316 |

Overall model summary:

| model | observations | MAE | RMSE | correlation | directional accuracy | F1 | Brier score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cart` | 2,972 | 0.041548 | 0.059189 | -0.063782 | 0.496972 | 0.500501 | 0.337932 |
| `lightgbm` | 2,972 | 0.038627 | 0.051548 | -0.037671 | 0.508748 | 0.495159 | 0.320461 |
| `stacking_final` | 2,972 | 0.027669 | 0.038603 | -0.068472 | 0.491925 | 0.626053 | n/a |
| `xgboost` | 2,972 | 0.041812 | 0.056113 | -0.047804 | 0.512786 | 0.482487 | 0.323122 |

Ticker-level `stacking_final` summary:

| ticker | observations | MAE | RMSE | correlation | directional accuracy | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BVH` | 743 | 0.028009 | 0.039988 | -0.066548 | 0.492598 | 0.627838 |
| `FPT` | 743 | 0.027134 | 0.036798 | -0.027676 | 0.506057 | 0.618106 |
| `SSI` | 743 | 0.035143 | 0.047482 | -0.088597 | 0.523553 | 0.678182 |
| `VNM` | 743 | 0.020390 | 0.027443 | -0.121414 | 0.445491 | 0.572614 |

These metrics are diagnostic. They do not prove tradable performance or future profitability.

## Context Coverage

| ticker | horizon | folds | mean breadth missing rate | mean foreign-flow missing rate | warning |
| --- | --- | ---: | ---: | ---: | --- |
| `BVH` | `short_5d` | 748 | 0.0000 | 1.0000 | `weak_coverage` |
| `FPT` | `short_5d` | 748 | 0.0000 | 1.0000 | `weak_coverage` |
| `SSI` | `short_5d` | 748 | 0.0000 | 1.0000 | `weak_coverage` |
| `VNM` | `short_5d` | 748 | 0.0000 | 1.0000 | `weak_coverage` |

The context warning is driven by absent usable foreign-flow coverage. Breadth coverage was present in the diagnostic tables.

## Feature Governance And Stability

Feature governance review:

| governance category | action | count |
| --- | --- | ---: |
| `safe_trailing` | `keep` | 12 |
| `requires_review` | `review_timing` | 10 |
| `alias_or_redundant` | `review_redundancy` | 2 |

Feature importance stability:

| stability level | count |
| --- | ---: |
| `high` | 69 |
| `medium` | 30 |
| `low` | 15 |

Linear coefficient stability:

| stability level | count |
| --- | ---: |
| `medium` | 20 |
| `high` | 14 |
| `low` | 8 |

Linear-vs-importance feature comparison:

| alignment label | count |
| --- | ---: |
| `importance_only` | 24 |
| `aligned_stable` | 14 |

No model governance status was changed.

## Foreign-Flow Handling

Foreign-flow was not supplied to the walk-forward command through `--foreign-flow-path`.

Local artifacts were checked:

| artifact | result |
| --- | --- |
| `data/foreign_flow_curated.csv` | provider-backed but partial; 68 rows, 2025-01-02 through 2025-01-24, unsuitable for 2010-2025 interpretation |
| `data/foreign_flow.csv` | one-row fixture for `TEST`, no requested ticker coverage |

Run metadata recorded the default `foreign_flow.csv` context artifact with `row_count=1`, but context diagnostics show `mean_foreign_flow_missing_rate=1.0` for every audited ticker. Foreign-flow features should not be interpreted in this 15-year audit unless a long-window artifact is created later.

Follow-up workflow support is documented in `docs/governance/VSEF_FOREIGN_FLOW_DISABLE_MODE.md`. For repeat long-window audits without a governed foreign-flow artifact, use `--foreign-flow-mode disabled` so the default fixture/cache artifact is not loaded and foreign-flow remains explicitly uninterpreted.

## Runtime Warnings

Observed non-fatal warnings and notes:

- `market_proxy_not_found` for `data/market_proxy.csv`; the runner fetched VNINDEX context through the provider path.
- Sentiment integration was explicitly disabled by the existing pipeline.
- LightGBM repeatedly logged no meaningful features or no additional leaves meeting split requirements during small early folds.
- scikit-learn logged feature-name warnings for LightGBM predictions.
- pandas logged future downcasting warnings during fill operations.
- SciPy logged ill-conditioned matrix warnings during linear diagnostics.
- The first provider-backed smoke attempt inside sandboxed network failed during license verification; rerunning with approved provider network access succeeded.

## Limitations

- The coverage audit uses generic business days, not a Vietnam exchange calendar, so it overstates missing dates around local market holidays.
- OHLCV data are staged provider outputs and were not promoted into the tracked cache.
- The run covers one horizon, `short_5d`; it is not a multi-horizon production model audit.
- The final rows near 2025-12-31 are retained but not scored when the 5-trading-day realized target is outside the fetched history.
- MAPE and SMAPE are unstable for return targets near zero; MAE, RMSE, and directional metrics are more interpretable here.
- Strategy and backtest outputs from the workflow are diagnostics, not execution-ready portfolio evidence.
- Foreign-flow is unavailable for this long window and must not be interpreted.

## Generated Outputs

Generated data and model outputs are intentionally ignored and were not committed:

- `tmp/ohlcv_15y_refresh_probe_data/`
- `outputs/walkforward_15y_daily_smoke_ssi/`
- `outputs/walkforward_15y_daily_ssi_fpt_bvh_vnm/`

## Validation

Validation commands run:

| command | result |
| --- | --- |
| `C:\Users\luong\.venv\Scripts\python.exe -m pytest tests/ml/test_ohlcv_cache_coverage.py -q` | 7 passed |
| `C:\Users\luong\.venv\Scripts\python.exe -m pytest tests/ml/test_walk_forward_ohlcv_data_dir.py -q` | 7 passed |
| `C:\Users\luong\.venv\Scripts\python.exe -m pytest tests/ml -q` | first sandboxed run failed on provider license verification network permission; rerun with provider network access passed: 181 passed, 2 skipped |
| `C:\Users\luong\.venv\Scripts\python.exe -m pytest tests/quant_core -q` | 15 passed |
| `C:\Users\luong\.venv\Scripts\python.exe -m pytest tests/phase1/test_forecast_contracts.py -q` | 3 passed |
| `C:\Users\luong\.venv\Scripts\python.exe -m compileall src` | passed |

Non-fatal validation warnings included pytest cache write permission warnings, an unknown `asyncio_mode` pytest option warning, LightGBM/scikit-learn feature-name warnings, pandas future downcasting warnings, `Series.pct_change` fill-method deprecation warnings, class-label warnings, Pydantic v2 deprecation warnings, provider package update notices, and ill-conditioned matrix warnings in linear diagnostics.

## Next Recommended Task

Create a governed long-window foreign-flow artifact, or rerun the audit with `--foreign-flow-mode disabled` when foreign-flow remains intentionally excluded.
