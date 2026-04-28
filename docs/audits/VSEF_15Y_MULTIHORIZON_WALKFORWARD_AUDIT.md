# VSEF 15-Year Multi-Horizon Walk-Forward Audit

Date: 2026-04-28

Branch: `vsef-15y-multihorizon-audit`

Base commit: `b506d9f89bb7c1c069e26e7d817de0825be50755`

## Purpose

This audit evaluates whether walk-forward behavior changes across forecast horizons for `SSI`, `FPT`, `BVH`, and `VNM` using staged 2010-2025 OHLCV data, daily walk-forward steps, and disabled foreign-flow context.

This is research/evaluation evidence only. It is not trading-performance proof, does not add model families, and does not change model governance status.

## Runtime And Data Source

Python interpreter:

```text
C:\Users\luong\.venv\Scripts\python.exe
```

Python version:

```text
3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]
```

OHLCV source mode:

```text
--ohlcv-data-dir tmp\ohlcv_15y_refresh_probe_data
```

The runner loaded ticker OHLCV from the explicit staged CSV directory. `market_proxy.csv` was not present, so VNINDEX market context was fetched through the existing provider path. Foreign-flow was intentionally disabled with `--foreign-flow-mode disabled`.

## Supported Horizons

The runner accepts comma-separated horizon names through `--horizons`.

Supported runner names inspected in `src/ml/backtest/walk_forward_all_models_stacking.py`:

| requested | used | days | note |
| --- | --- | ---: | --- |
| `short_5d` | `short_5d` | 5 | supported exactly |
| `short_10d` | `short_10d` | 10 | supported exactly |
| `short_20d` | `short_20d` | 20 | supported exactly |
| `medium_60d` | `long_3m` | 63 | `medium_60d` is not supported; `long_3m` is the existing approximate 3-month equivalent |

No horizon names or model logic were changed.

`--step-sizes 1` means the runner evaluates each available trading session in the forecast window, not each calendar day.

## OHLCV Coverage

Coverage command:

```powershell
C:\Users\luong\.venv\Scripts\python.exe scripts/audit_ohlcv_cache_coverage.py --tickers SSI,FPT,BVH,VNM --start-date 2010-01-01 --end-date 2025-12-31 --data-dir tmp\ohlcv_15y_refresh_probe_data
```

Requested business-day count: `4174`.

| ticker | staged rows | file min date | file max date | run-loaded range | matched business dates | missing generic business dates | coverage rate | supports requested window |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| `SSI` | 4097 | 2009-11-23 | 2026-04-24 | 2010-01-04 to 2025-12-31 | 3993 | 181 | 0.956636 | yes |
| `FPT` | 4097 | 2009-11-23 | 2026-04-24 | 2010-01-04 to 2025-12-31 | 3993 | 181 | 0.956636 | yes |
| `BVH` | 4097 | 2009-11-23 | 2026-04-24 | 2010-01-04 to 2025-12-31 | 3993 | 181 | 0.956636 | yes |
| `VNM` | 4097 | 2009-11-23 | 2026-04-24 | 2010-01-04 to 2025-12-31 | 3993 | 181 | 0.956636 | yes |

The missing-day count is based on generic business days rather than a Vietnam exchange calendar.

## Commands

Smoke test command:

```powershell
C:\Users\luong\.venv\Scripts\python.exe scripts\run_walkforward_all_models_stacking_eval.py --tickers SSI --history-start 2010-01-01 --history-end 2025-12-31 --initial-train-start 2010-01-01 --initial-train-end 2022-12-31 --forecast-start 2023-01-01 --forecast-end 2025-12-31 --horizons short_5d,short_10d,short_20d,long_3m --step-sizes 1 --algorithms cart,xgboost,lightgbm --ohlcv-data-dir tmp\ohlcv_15y_refresh_probe_data --foreign-flow-mode disabled --output-dir outputs\walkforward_15y_multihorizon_smoke_ssi --max-workers 1 --max-depth 3 --meta-min-samples 5 --epochs 1
```

Full run command:

```powershell
C:\Users\luong\.venv\Scripts\python.exe scripts\run_walkforward_all_models_stacking_eval.py --tickers SSI FPT BVH VNM --history-start 2010-01-01 --history-end 2025-12-31 --initial-train-start 2010-01-01 --initial-train-end 2022-12-31 --forecast-start 2023-01-01 --forecast-end 2025-12-31 --horizons short_5d,short_10d,short_20d,long_3m --step-sizes 1 --algorithms cart,xgboost,lightgbm --ohlcv-data-dir tmp\ohlcv_15y_refresh_probe_data --foreign-flow-mode disabled --output-dir outputs\walkforward_15y_multihorizon_ssi_fpt_bvh_vnm --max-workers 1 --max-depth 3 --meta-min-samples 5 --epochs 1
```

The commands were run through `cmd /c` with stdout/stderr redirected to ignored logs under `outputs/` to avoid flooding the terminal.

## Run Results

Smoke test:

- Status: completed successfully.
- Output directory: `outputs\walkforward_15y_multihorizon_smoke_ssi\ssi`
- Forecast coverage: `short_5d` 2992 total / 2972 eligible, `short_10d` 2992 / 2952, `short_20d` 2896 / 2816, `long_3m` 2416 / 2164.
- Context coverage: breadth missing rate `0.0`; foreign-flow disabled; warning level `ok` for all horizons.

Full run:

- Status: completed successfully.
- Output directory: `outputs\walkforward_15y_multihorizon_ssi_fpt_bvh_vnm\ssi_fpt_bvh_vnm`
- Combined CSV output directory: `outputs\walkforward_15y_multihorizon_ssi_fpt_bvh_vnm\ssi_fpt_bvh_vnm\_combined_internal\csv`

Foreign-flow metadata in `run_metadata.json`:

```json
{
  "enabled": false,
  "mode": "disabled",
  "path": null,
  "foreign_flow_path": null,
  "foreign_flow_path_explicit": false,
  "source_name": "disabled",
  "source_provenance": "disabled",
  "row_count": 0,
  "artifact_validation": null,
  "reason": "foreign-flow context intentionally disabled"
}
```

Disabled foreign-flow means intentionally excluded, not complete foreign-flow coverage. Foreign-flow features must not be interpreted.

## Prediction Counts

| horizon | base prediction rows | stacking prediction rows | combined rows | base observed rows | stacking observed rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `short_5d` | 8976 | 2992 | 11968 | 8916 | 2972 |
| `short_10d` | 8976 | 2992 | 11968 | 8856 | 2952 |
| `short_20d` | 8688 | 2896 | 11584 | 8448 | 2816 |
| `long_3m` | 7173 | 2391 | 9564 | 6417 | 2139 |

Forecast coverage by horizon:

| horizon | total predictions | eligible predictions | ineligible predictions |
| --- | ---: | ---: | ---: |
| `short_5d` | 11968 | 11888 | 80 |
| `short_10d` | 11968 | 11808 | 160 |
| `short_20d` | 11584 | 11264 | 320 |
| `long_3m` | 9564 | 8556 | 1008 |

Longer horizons lose more eligible rows near the end of 2025 because realized targets require more future trading sessions.

## Model Metrics By Horizon

| horizon | model | observations | MAE | RMSE | correlation | directional accuracy | F1 | Brier score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `short_5d` | `cart` | 2972 | 0.041548 | 0.059189 | -0.063782 | 0.496972 | 0.500501 | 0.337932 |
| `short_5d` | `lightgbm` | 2972 | 0.038627 | 0.051548 | -0.037671 | 0.508748 | 0.495159 | 0.320461 |
| `short_5d` | `stacking_final` | 2972 | 0.027669 | 0.038603 | -0.068472 | 0.491925 | 0.626053 | n/a |
| `short_5d` | `xgboost` | 2972 | 0.041812 | 0.056113 | -0.047804 | 0.512786 | 0.482487 | 0.323122 |
| `short_10d` | `cart` | 2952 | 0.061638 | 0.081398 | -0.128745 | 0.493902 | 0.519305 | 0.393200 |
| `short_10d` | `lightgbm` | 2952 | 0.058106 | 0.074869 | -0.101849 | 0.489499 | 0.496492 | 0.349279 |
| `short_10d` | `stacking_final` | 2952 | 0.040564 | 0.055338 | -0.033149 | 0.509824 | 0.644734 | n/a |
| `short_10d` | `xgboost` | 2952 | 0.059716 | 0.077254 | -0.047748 | 0.489160 | 0.483208 | 0.351632 |
| `short_20d` | `cart` | 2816 | 0.085939 | 0.112797 | -0.113473 | 0.492543 | 0.491278 | 0.434165 |
| `short_20d` | `lightgbm` | 2816 | 0.081734 | 0.108591 | -0.106519 | 0.493253 | 0.501920 | 0.388604 |
| `short_20d` | `stacking_final` | 2816 | 0.061643 | 0.080093 | -0.035315 | 0.488991 | 0.623002 | n/a |
| `short_20d` | `xgboost` | 2816 | 0.083253 | 0.109862 | -0.092116 | 0.489347 | 0.479740 | 0.386134 |
| `long_3m` | `cart` | 2139 | 0.154071 | 0.202510 | -0.083331 | 0.549322 | 0.594958 | 0.419490 |
| `long_3m` | `lightgbm` | 2139 | 0.153814 | 0.203395 | -0.078701 | 0.571763 | 0.625511 | 0.357358 |
| `long_3m` | `stacking_final` | 2139 | 0.124236 | 0.168430 | -0.186810 | 0.458626 | 0.582552 | n/a |
| `long_3m` | `xgboost` | 2139 | 0.156275 | 0.205545 | -0.079354 | 0.576438 | 0.634677 | 0.334699 |

## Best Model By Horizon

| horizon | criterion | best model | value |
| --- | --- | --- | ---: |
| `short_5d` | lowest MAE | `stacking_final` | 0.027669 |
| `short_5d` | lowest RMSE | `stacking_final` | 0.038603 |
| `short_5d` | highest directional accuracy | `xgboost` | 0.512786 |
| `short_5d` | highest F1 | `stacking_final` | 0.626053 |
| `short_10d` | lowest MAE | `stacking_final` | 0.040564 |
| `short_10d` | lowest RMSE | `stacking_final` | 0.055338 |
| `short_10d` | highest directional accuracy | `stacking_final` | 0.509824 |
| `short_10d` | highest F1 | `stacking_final` | 0.644734 |
| `short_20d` | lowest MAE | `stacking_final` | 0.061643 |
| `short_20d` | lowest RMSE | `stacking_final` | 0.080093 |
| `short_20d` | highest directional accuracy | `lightgbm` | 0.493253 |
| `short_20d` | highest F1 | `stacking_final` | 0.623002 |
| `long_3m` | lowest MAE | `stacking_final` | 0.124236 |
| `long_3m` | lowest RMSE | `stacking_final` | 0.168430 |
| `long_3m` | highest directional accuracy | `xgboost` | 0.576438 |
| `long_3m` | highest F1 | `xgboost` | 0.634677 |

## Stacking Ticker-Level Summary

| ticker | horizon | observations | MAE | RMSE | correlation | directional accuracy | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BVH` | `short_5d` | 743 | 0.028009 | 0.039988 | -0.066548 | 0.492598 | 0.627838 |
| `FPT` | `short_5d` | 743 | 0.027134 | 0.036798 | -0.027676 | 0.506057 | 0.618106 |
| `SSI` | `short_5d` | 743 | 0.035143 | 0.047482 | -0.088597 | 0.523553 | 0.678182 |
| `VNM` | `short_5d` | 743 | 0.020390 | 0.027443 | -0.121414 | 0.445491 | 0.572614 |
| `BVH` | `short_10d` | 738 | 0.041791 | 0.058421 | -0.063606 | 0.491870 | 0.641834 |
| `FPT` | `short_10d` | 738 | 0.038317 | 0.051918 | 0.040913 | 0.548780 | 0.663296 |
| `SSI` | `short_10d` | 738 | 0.052321 | 0.068165 | 0.008942 | 0.542005 | 0.677481 |
| `VNM` | `short_10d` | 738 | 0.029827 | 0.038654 | -0.160376 | 0.456640 | 0.594540 |
| `BVH` | `short_20d` | 704 | 0.061831 | 0.079795 | -0.082038 | 0.474432 | 0.621677 |
| `FPT` | `short_20d` | 704 | 0.059423 | 0.074327 | 0.140036 | 0.561080 | 0.686930 |
| `SSI` | `short_20d` | 704 | 0.079584 | 0.103375 | -0.049660 | 0.497159 | 0.617711 |
| `VNM` | `short_20d` | 704 | 0.045734 | 0.055511 | -0.214605 | 0.423295 | 0.561555 |
| `BVH` | `long_3m` | 539 | 0.112011 | 0.126191 | 0.007687 | 0.454545 | 0.591667 |
| `FPT` | `long_3m` | 503 | 0.137562 | 0.181366 | -0.455252 | 0.461233 | 0.577223 |
| `SSI` | `long_3m` | 541 | 0.169103 | 0.238614 | -0.295948 | 0.452865 | 0.596730 |
| `VNM` | `long_3m` | 556 | 0.080375 | 0.092419 | 0.039087 | 0.465827 | 0.562592 |

## Context Coverage

Every ticker/horizon pair had `748` context coverage folds, breadth missing rate `0.0`, unavailable foreign-flow rates by design, zero weak-coverage folds, and `overall_coverage_warning_level = ok`.

`context_coverage_diagnostics.csv` contains disabled foreign-flow markers:

- `foreign_flow_context_mode = disabled`
- `foreign_flow_coverage_status = disabled`

Breadth remains interpretable as joined context coverage. Foreign-flow must not be interpreted.

## Feature Governance

`feature_governance_review.csv` summary:

| category | count |
| --- | ---: |
| `safe_trailing` | 12 |
| `requires_review` | 10 |
| `alias_or_redundant` | 2 |

Recommended actions:

| action | count |
| --- | ---: |
| `keep` | 12 |
| `review_timing` | 10 |
| `review_redundancy` | 2 |

The `requires_review` features are breadth/market context features. No governance status was changed.

## Stability Diagnostics

Feature-importance stability:

| level | count |
| --- | ---: |
| `high` | 288 |
| `medium` | 98 |
| `low` | 70 |

Linear coefficient stability:

| level | count |
| --- | ---: |
| `medium` | 74 |
| `high` | 56 |
| `low` | 38 |

Linear-vs-importance comparison:

| alignment label | count |
| --- | ---: |
| `importance_only` | 95 |
| `aligned_stable` | 55 |
| `unstable_or_missing` | 2 |

## Comparison To Prior `short_5d` Audit

The `short_5d` slice matches the prior single-horizon no-foreign-flow audit in model counts and overall metrics. For example, `summary_by_horizon.csv` reports `short_5d` `stacking_final` MAE `0.027669`, RMSE `0.038603`, directional accuracy `0.491925`, and F1 `0.626053`, matching the single-horizon audit.

The multi-horizon run adds horizon sensitivity:

- Error magnitudes increase as the target horizon lengthens.
- `stacking_final` gives the lowest MAE and RMSE for every evaluated horizon.
- Directional and F1 leaders vary: `xgboost` leads directional accuracy and F1 on `long_3m`, while `stacking_final` leads most short-horizon F1 and all error metrics.
- Longer horizons have fewer eligible observations because more future trading sessions are required for realized targets.

## Interpretation

This audit suggests that model behavior is horizon-sensitive. The stacking layer is strongest on point-error metrics across horizons, but not consistently strongest on directional accuracy. The long-horizon proxy (`long_3m`) changes the directional ranking relative to the short horizons.

These findings are empirical diagnostics only. They do not prove trading performance, causality, or production readiness.

## Runtime Warnings

Observed non-fatal warnings and notes:

- `market_proxy_not_found` for `data/market_proxy.csv`; VNINDEX market context was fetched through the provider path.
- Sentiment integration remained disabled by the existing pipeline.
- LightGBM repeatedly logged no meaningful features, no further positive-gain splits, or no additional leaves meeting split requirements.
- scikit-learn logged feature-name warnings for LightGBM predictions.
- pandas logged future downcasting warnings during fill operations.
- SciPy logged ill-conditioned matrix warnings during linear diagnostics.
- Provider package update notices appeared for `vnstock`, `vnai`, and `vnii`.

## Limitations

- `medium_60d` is not a supported runner horizon name; `long_3m` uses 63 trading days as the existing approximate equivalent.
- Disabled foreign-flow is intentional exclusion, not complete foreign-flow coverage.
- Foreign-flow features must not be interpreted in this audit.
- The coverage audit uses generic business days rather than a Vietnam exchange calendar.
- The run evaluates one model family set only: `cart`, `xgboost`, `lightgbm`, plus the existing stacking layer.
- MAPE and SMAPE are unstable for return targets near zero.
- Strategy and backtest outputs are diagnostics, not execution-ready portfolio evidence.
- Generated outputs and runtime logs under `outputs/` are not committed.

## Validation

Validation used `C:\Users\luong\.venv\Scripts\python.exe`.

| Command | Result |
| --- | ---: |
| `python -m pytest tests/ml/test_foreign_flow_disable_mode.py -q` | 8 passed |
| `python -m pytest tests/ml/test_walk_forward_ohlcv_data_dir.py -q` | 7 passed |
| `python -m pytest tests/ml/test_context_coverage_diagnostics.py -q` | 5 passed |
| `python -m pytest tests/ml -q` | 189 passed, 2 skipped |
| `python -m pytest tests/quant_core -q` | 15 passed |
| `python -m pytest tests/phase1/test_forecast_contracts.py -q` | 3 passed |
| `python -m compileall src` | passed |

Warnings observed during validation were pre-existing pytest cache permission warnings, the unknown `asyncio_mode` pytest config warning, LightGBM/sklearn feature-name warnings, pandas future warnings, SciPy ill-conditioned matrix warnings, and provider package update notices.

## Next Recommended Task

Run the same multi-horizon audit across a broader ticker basket or add a governed long-window foreign-flow artifact before interpreting any foreign-flow feature behavior.

Synthesis report: `docs/reports/VSEF_15Y_DAILY_MULTIHORIZON_TECHNICAL_REPORT.md` consolidates this audit with the 15-year daily and no-foreign-flow audits for supervisor or presentation review.
