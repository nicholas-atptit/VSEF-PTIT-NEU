# VSEF 15-Year Daily Walk-Forward Audit With Foreign-Flow Disabled

Date: 2026-04-28

Branch: `vsef-15y-daily-audit-no-foreign-flow`

Base commit: `9b7a72fd817fd86812c0f8da24d77b2154bdc4b4`

## Purpose

This audit reruns the 15-year daily walk-forward evaluation for `SSI`, `FPT`, `BVH`, and `VNM` using staged OHLCV data and `--foreign-flow-mode disabled`.

The goal is to keep long-window foreign-flow intentionally excluded instead of loading the local default fixture/cache and reporting 100 percent missing foreign-flow coverage. This is research/evaluation evidence only. It is not trading-performance proof, does not add model families, and does not change model governance status.

## Runtime

Python interpreter:

```text
C:\Users\luong\.venv\Scripts\python.exe
```

Python version:

```text
3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]
```

## Command

```powershell
$env:PYTHONIOENCODING='utf-8'; C:\Users\luong\.venv\Scripts\python.exe scripts/run_walkforward_all_models_stacking_eval.py --tickers SSI FPT BVH VNM --history-start 2010-01-01 --history-end 2025-12-31 --initial-train-start 2010-01-01 --initial-train-end 2022-12-31 --forecast-start 2023-01-01 --forecast-end 2025-12-31 --horizons short_5d --step-sizes 1 --algorithms cart,xgboost,lightgbm --ohlcv-data-dir tmp\ohlcv_15y_refresh_probe_data --foreign-flow-mode disabled --output-dir outputs\walkforward_15y_daily_ssi_fpt_bvh_vnm_no_foreign_flow --max-workers 1 --max-depth 3 --meta-min-samples 5 --epochs 1
```

The first full run attempt timed out after one hour before final outputs were written. The rerun completed and emitted all expected outputs. Because the rerun redirected all native streams to an ignored log, PowerShell returned exit code 1 after categorizing native stderr warning records as `NativeCommandError`; the runner log contains the completion marker and the full output set is present.

Output directory:

```text
outputs\walkforward_15y_daily_ssi_fpt_bvh_vnm_no_foreign_flow\ssi_fpt_bvh_vnm
```

Generated outputs and staged CSVs are not committed.

## OHLCV Coverage

Coverage command:

```powershell
C:\Users\luong\.venv\Scripts\python.exe scripts/audit_ohlcv_cache_coverage.py --tickers SSI,FPT,BVH,VNM --start-date 2010-01-01 --end-date 2025-12-31 --data-dir tmp\ohlcv_15y_refresh_probe_data
```

Requested business-day count: `4174`.

| ticker | staged rows | file min date | file max date | run-loaded range | matched business dates | missing business dates | coverage rate | supports requested window |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| `SSI` | 4097 | 2009-11-23 | 2026-04-24 | 2010-01-04 to 2025-12-31 | 3993 | 181 | 0.956636 | yes |
| `FPT` | 4097 | 2009-11-23 | 2026-04-24 | 2010-01-04 to 2025-12-31 | 3993 | 181 | 0.956636 | yes |
| `BVH` | 4097 | 2009-11-23 | 2026-04-24 | 2010-01-04 to 2025-12-31 | 3993 | 181 | 0.956636 | yes |
| `VNM` | 4097 | 2009-11-23 | 2026-04-24 | 2010-01-04 to 2025-12-31 | 3993 | 181 | 0.956636 | yes |

All four tickers support the requested window at the audit threshold. The missing-business-date count is based on generic business days, not a Vietnam exchange calendar.

## Foreign-Flow Disabled Metadata

`run_metadata.json` records:

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

No default `data/foreign_flow.csv` fixture was loaded. Disabled foreign-flow is not complete foreign-flow coverage; foreign-flow features must not be interpreted for this run.

## Prediction Counts

| scope | rows |
| --- | ---: |
| Base model prediction rows | 8976 |
| Stacking prediction rows | 2992 |
| Combined prediction rows | 11968 |
| Base model rows with realized target | 8916 |
| Stacking rows with realized target | 2972 |

Forecast coverage:

| ticker | models | total predictions | eligible predictions | ineligible predictions |
| --- | --- | ---: | ---: | ---: |
| `BVH` | cart, lightgbm, stacking_final, xgboost | 2992 | 2972 | 20 |
| `FPT` | cart, lightgbm, stacking_final, xgboost | 2992 | 2972 | 20 |
| `SSI` | cart, lightgbm, stacking_final, xgboost | 2992 | 2972 | 20 |
| `VNM` | cart, lightgbm, stacking_final, xgboost | 2992 | 2972 | 20 |

Each ticker/model combination produced `748` daily step-1 forecasts; `743` were eligible for evaluation after requiring the realized `short_5d` target.

## Model Metrics

Overall `summary_by_model.csv`:

| model | observations | MAE | RMSE | correlation | directional accuracy | F1 | Brier score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cart` | 2972 | 0.041548 | 0.059189 | -0.063782 | 0.496972 | 0.500501 | 0.337932 |
| `lightgbm` | 2972 | 0.038627 | 0.051548 | -0.037671 | 0.508748 | 0.495159 | 0.320461 |
| `stacking_final` | 2972 | 0.027669 | 0.038603 | -0.068472 | 0.491925 | 0.626053 | n/a |
| `xgboost` | 2972 | 0.041812 | 0.056113 | -0.047804 | 0.512786 | 0.482487 | 0.323122 |

These metrics are evaluation diagnostics, not evidence of deployable trading performance.

## Ticker Metrics

Selected `summary_by_ticker.csv` metrics:

| ticker | model | observations | MAE | RMSE | directional accuracy | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `BVH` | `cart` | 743 | 0.031514 | 0.043254 | 0.495289 | 0.437781 |
| `BVH` | `lightgbm` | 743 | 0.030464 | 0.042853 | 0.471063 | 0.424597 |
| `BVH` | `stacking_final` | 743 | 0.028009 | 0.039988 | 0.492598 | 0.627838 |
| `BVH` | `xgboost` | 743 | 0.033570 | 0.045689 | 0.508748 | 0.480797 |
| `FPT` | `cart` | 743 | 0.048353 | 0.066201 | 0.446837 | 0.451268 |
| `FPT` | `lightgbm` | 743 | 0.038872 | 0.051275 | 0.506057 | 0.471942 |
| `FPT` | `stacking_final` | 743 | 0.027134 | 0.036798 | 0.506057 | 0.618106 |
| `FPT` | `xgboost` | 743 | 0.041776 | 0.052847 | 0.492598 | 0.446402 |
| `SSI` | `cart` | 743 | 0.056725 | 0.079135 | 0.553163 | 0.633554 |
| `SSI` | `lightgbm` | 743 | 0.052547 | 0.067854 | 0.524899 | 0.609945 |
| `SSI` | `stacking_final` | 743 | 0.035143 | 0.047482 | 0.523553 | 0.678182 |
| `SSI` | `xgboost` | 743 | 0.059767 | 0.078339 | 0.495289 | 0.558304 |
| `VNM` | `cart` | 743 | 0.029598 | 0.038696 | 0.492598 | 0.438152 |
| `VNM` | `lightgbm` | 743 | 0.032626 | 0.039485 | 0.532974 | 0.430213 |
| `VNM` | `stacking_final` | 743 | 0.020390 | 0.027443 | 0.445491 | 0.572614 |
| `VNM` | `xgboost` | 743 | 0.032136 | 0.039714 | 0.554509 | 0.414159 |

## Context Coverage

`context_coverage_summary.csv`:

| ticker | folds | mean breadth missing rate | max breadth missing rate | mean foreign-flow missing rate | weak coverage folds | review folds | overall warning |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `BVH` | 748 | 0.0 | 0.0 | n/a | 0 | 0 | `ok` |
| `FPT` | 748 | 0.0 | 0.0 | n/a | 0 | 0 | `ok` |
| `SSI` | 748 | 0.0 | 0.0 | n/a | 0 | 0 | `ok` |
| `VNM` | 748 | 0.0 | 0.0 | n/a | 0 | 0 | `ok` |

`context_coverage_diagnostics.csv` contains `2992` fold-level rows with:

- `foreign_flow_context_mode = disabled`
- `foreign_flow_coverage_status = disabled`
- `coverage_warning_level = ok`

Foreign-flow missing rates are intentionally unavailable. Breadth coverage remains visible and fully measured in this run.

## Comparison To Previous 15-Year Audit

The previous 15-year daily audit used the default foreign-flow loader path. Local `data/foreign_flow.csv` contained only a one-row `TEST` fixture/local artifact, so context diagnostics showed `mean_foreign_flow_missing_rate = 1.0` for every audited ticker.

This rerun uses `--foreign-flow-mode disabled`. The default fixture is not loaded, run metadata marks foreign-flow as disabled, and foreign-flow absence no longer creates a weak-coverage finding by itself. This does not improve or complete foreign-flow coverage. It makes the audit more explicit: foreign-flow is excluded and should not be interpreted.

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

The `requires_review` features are breadth/market context features such as `breadth_member_count`, `breadth_thrust_10`, `declining_share`, `down_volume`, `market_return_60d`, `new_high_low_spread_5`, `pct_above_ma20`, `pct_above_ma50`, `up_down_volume_ratio_5`, and `up_volume`. No model governance status was changed.

## Stability Diagnostics

Feature-importance stability:

| stability level | count |
| --- | ---: |
| `high` | 69 |
| `medium` | 30 |
| `low` | 15 |

Examples with `top_10_ratio = 1.0` include `m_ret_20d`, `ema_50`, `rolling_min_5`, `breadth_member_count`, `breadth_thrust_10`, `market_return_60d`, `new_high_low_spread_5`, `range_20`, and `pct_above_ma20`.

Linear coefficient stability:

| stability level | count |
| --- | ---: |
| `high` | 14 |
| `medium` | 20 |
| `low` | 8 |

High sign-consistency examples include `market_return_60d`, `rolling_volatility_60`, `dist_ma_20`, `close_to_sma_200`, `range_20`, `close_return_10d`, and `bb_width`.

Linear-vs-importance feature comparison:

| alignment label | count |
| --- | ---: |
| `importance_only` | 24 |
| `aligned_stable` | 14 |

## Runtime Warnings

Observed non-fatal warnings and notes:

- `market_proxy_not_found` for `data/market_proxy.csv`; VNINDEX market context was fetched through the provider path.
- Sentiment integration remained disabled by the existing pipeline.
- LightGBM repeatedly logged no meaningful features, no further positive-gain splits, or no additional leaves meeting split requirements.
- scikit-learn logged feature-name warnings for LightGBM predictions.
- pandas logged future downcasting warnings during fill operations.
- SciPy logged ill-conditioned matrix warnings during linear diagnostics.
- Provider package update notices appeared for `vnai` and `vnii`.

## Limitations

- Disabled foreign-flow is intentional exclusion, not complete foreign-flow coverage.
- Foreign-flow features must not be interpreted in this audit.
- The coverage audit uses generic business days rather than a Vietnam exchange calendar.
- The run covers only `short_5d`; it is not a multi-horizon production audit.
- Final forecasts near 2025-12-31 are retained but ineligible when the realized 5-trading-day target is outside the fetched history.
- MAPE and SMAPE are unstable for return targets near zero.
- Strategy and backtest outputs are diagnostics, not execution-ready portfolio evidence.
- The rerun generated output files and a runtime log under `outputs/`; none are committed.

## Validation

Validation commands run:

| command | result |
| --- | --- |
| `C:\Users\luong\.venv\Scripts\python.exe -m pytest tests/ml/test_foreign_flow_disable_mode.py -q` | 8 passed |
| `C:\Users\luong\.venv\Scripts\python.exe -m pytest tests/ml/test_walk_forward_ohlcv_data_dir.py -q` | 7 passed |
| `C:\Users\luong\.venv\Scripts\python.exe -m pytest tests/ml/test_context_coverage_diagnostics.py -q` | 5 passed |
| `C:\Users\luong\.venv\Scripts\python.exe -m pytest tests/ml -q` | 189 passed, 2 skipped |
| `C:\Users\luong\.venv\Scripts\python.exe -m pytest tests/quant_core -q` | 15 passed |
| `C:\Users\luong\.venv\Scripts\python.exe -m pytest tests/phase1/test_forecast_contracts.py -q` | 3 passed |
| `C:\Users\luong\.venv\Scripts\python.exe -m compileall src` | passed |

Non-fatal validation warnings included pytest cache write permission warnings, an unknown `asyncio_mode` pytest option warning, LightGBM/scikit-learn feature-name warnings, pandas future downcasting warnings, `Series.pct_change` fill-method deprecation warnings, class-label warnings, Pydantic v2 deprecation warnings, provider package update notices, and ill-conditioned matrix warnings in linear diagnostics.

## Next Recommended Task

Create and validate a governed long-window foreign-flow artifact if future audits need to interpret foreign-flow. Until then, long-window audits should use `--foreign-flow-mode disabled` and state that foreign-flow is excluded.
