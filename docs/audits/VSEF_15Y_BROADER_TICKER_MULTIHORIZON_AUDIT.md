# VSEF 15-Year Broader-Ticker Multi-Horizon Walk-Forward Audit

## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Audit |
| Created / authored | Tuesday, 2026-04-28 12:33:07 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-04-28 12:33:07 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-15y-broader-ticker-audit` |
| Commit | `1c790843f7b1dddb7c4b13a32797892d9dfffe18` |
| Timestamp source | Git history; existing document date mirrored |

## Purpose

This audit extends the previous 4-ticker 15-year multi-horizon baseline (`SSI`, `FPT`, `BVH`, `VNM`) to a broader requested basket:

```text
SSI FPT BVH VNM ACB HPG MWG DGC VCB VIC
```

The goal is to test whether the prior pattern still appears under a broader Vietnamese equity basket:

- `stacking_final` tends to dominate MAE and RMSE.
- Directional accuracy remains horizon-sensitive.
- XGBoost or LightGBM may outperform stacking on direction for some horizons.
- Foreign-flow remains intentionally excluded unless a governed long-window artifact exists.

This is an empirical evaluation audit only. It does not add model families, change model governance status, or claim trading-performance proof.

## Data Source and Coverage

Runtime and provider:

| item | value |
| --- | --- |
| Python runtime | `C:\Users\luong\.venv\Scripts\python.exe` |
| Python version | `3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]` |
| Canonical provider | `vnstock_data` |
| Alternate provider present | `vnstock` present, not used as a substitute |
| Staged OHLCV path | `tmp\ohlcv_15y_broader_ticker_probe_data` |
| Staged refresh output | 10 CSV files, 38622 rows |
| Requested generic business-day denominator | 4174 days |

The staged provider refresh wrote to ignored `tmp/` only. No staged CSVs were copied into tracked `data/`.

Coverage command:

```powershell
C:\Users\luong\.venv\Scripts\python.exe scripts/audit_ohlcv_cache_coverage.py --tickers SSI,FPT,BVH,VNM,ACB,HPG,MWG,DGC,VCB,VIC --start-date 2010-01-01 --end-date 2025-12-31 --data-dir tmp\ohlcv_15y_broader_ticker_probe_data
```

Coverage result:

| ticker | row count | min date | max date | matched business dates | missing generic business dates | coverage rate | supports requested window |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| `SSI` | 4097 | 2009-11-23 | 2026-04-24 | 3993 | 181 | 0.956636 | yes |
| `FPT` | 4097 | 2009-11-23 | 2026-04-24 | 3993 | 181 | 0.956636 | yes |
| `BVH` | 4097 | 2009-11-23 | 2026-04-24 | 3993 | 181 | 0.956636 | yes |
| `VNM` | 4097 | 2009-11-23 | 2026-04-24 | 3993 | 181 | 0.956636 | yes |
| `ACB` | 4092 | 2009-11-23 | 2026-04-24 | 3988 | 186 | 0.955438 | yes |
| `HPG` | 4097 | 2009-11-23 | 2026-04-24 | 3993 | 181 | 0.956636 | yes |
| `MWG` | 2944 | 2014-07-14 | 2026-04-24 | 2869 | 1305 | 0.687350 | no |
| `DGC` | 2907 | 2014-08-26 | 2026-04-24 | 2832 | 1342 | 0.678486 | no |
| `VCB` | 4097 | 2009-11-23 | 2026-04-24 | 3993 | 181 | 0.956636 | yes |
| `VIC` | 4097 | 2009-11-23 | 2026-04-24 | 3993 | 181 | 0.956636 | yes |

`MWG` and `DGC` do not support the requested 2010-2025 window because their staged histories begin in 2014. They were retained in the coverage audit but excluded from the walk-forward evaluation. The evaluated basket is therefore:

```text
SSI FPT BVH VNM ACB HPG VCB VIC
```

The missing-date count uses generic Monday-Friday business days, not an official Vietnam exchange calendar.

## Evaluation Design

| item | value |
| --- | --- |
| History window | `2010-01-01` to `2025-12-31` |
| Initial training window | `2010-01-01` to `2022-12-31` |
| Preferred forecast window | `2023-01-01` to `2025-12-31` |
| Executed fallback forecast window | `2024-01-01` to `2025-12-31` |
| Step size | `1`, one available trading session per step |
| Horizons | `short_5d`, `short_10d`, `short_20d`, `long_3m` |
| `long_3m` target length | 63 trading days, existing supported proxy for about 3 months |
| Models | CART, XGBoost, LightGBM, existing `stacking_final` |
| Foreign-flow mode | `disabled` |
| Output directory | `outputs\walkforward_15y_broader_ticker_basket_2024_2025` |

The 2-ticker smoke run over 2023-2025 completed but took about 78 minutes. A proportional 8-ticker 2023-2025 run was judged too heavy for this audit turn, so the documented fallback window was used. The fallback keeps the supported 8 tickers and the same four horizons, but shortens the forecast window to 2024-2025.

No new model families were added or evaluated.

## Run Commands

Staged OHLCV refresh:

```powershell
$env:PYTHONIOENCODING='utf-8'; C:\Users\luong\.venv\Scripts\python.exe scripts/extract_daily_csv.py --tickers SSI FPT BVH VNM ACB HPG MWG DGC VCB VIC --days 6000 --output tmp/ohlcv_15y_broader_ticker_probe
```

Smoke command:

```powershell
$env:PYTHONIOENCODING='utf-8'; C:\Users\luong\.venv\Scripts\python.exe scripts/run_walkforward_all_models_stacking_eval.py --tickers SSI FPT --history-start 2010-01-01 --history-end 2025-12-31 --initial-train-start 2010-01-01 --initial-train-end 2022-12-31 --forecast-start 2023-01-01 --forecast-end 2025-12-31 --horizons short_5d,short_10d,short_20d,long_3m --step-sizes 1 --algorithms cart,xgboost,lightgbm --ohlcv-data-dir tmp\ohlcv_15y_broader_ticker_probe_data --foreign-flow-mode disabled --output-dir outputs\walkforward_15y_broader_smoke_ssi_fpt --max-workers 1 --max-depth 3 --meta-min-samples 5 --epochs 1
```

Fallback broader run command:

```powershell
$env:PYTHONIOENCODING='utf-8'; C:\Users\luong\.venv\Scripts\python.exe scripts/run_walkforward_all_models_stacking_eval.py --tickers SSI FPT BVH VNM ACB HPG VCB VIC --history-start 2010-01-01 --history-end 2025-12-31 --initial-train-start 2010-01-01 --initial-train-end 2022-12-31 --forecast-start 2024-01-01 --forecast-end 2025-12-31 --horizons short_5d,short_10d,short_20d,long_3m --step-sizes 1 --algorithms cart,xgboost,lightgbm --ohlcv-data-dir tmp\ohlcv_15y_broader_ticker_probe_data --foreign-flow-mode disabled --output-dir outputs\walkforward_15y_broader_ticker_basket_2024_2025 --max-workers 1 --max-depth 3 --meta-min-samples 5 --epochs 1
```

Run results:

| run | status | output directory | notes |
| --- | --- | --- | --- |
| Smoke `SSI FPT`, 2023-2025 | completed, exit 0 | `outputs\walkforward_15y_broader_smoke_ssi_fpt\ssi_fpt` | about 78 minutes; context coverage ok; foreign-flow disabled |
| Broader fallback, 8 tickers, 2024-2025 | completed, exit 0 | `outputs\walkforward_15y_broader_ticker_basket_2024_2025\ssi_fpt_bvh_vnm_acb_hpg_vcb_vic` | about 3.5 hours; preferred 2023-2025 8-ticker run not executed because measured runtime was too heavy |

Observed runtime warnings included missing local `data\market_proxy.csv`, LightGBM split warnings, feature-name warnings from LightGBM predictions, pandas downcasting deprecation warnings, and ill-conditioned linear diagnostic matrix warnings. These warnings did not stop the run.

## Prediction Coverage

Forecast coverage by horizon for the fallback 8-ticker run:

| horizon | total predictions | eligible predictions | ineligible predictions |
| --- | ---: | ---: | ---: |
| `short_5d` | 15968 | 15808 | 160 |
| `short_10d` | 15968 | 15648 | 320 |
| `short_20d` | 15968 | 15328 | 640 |
| `long_3m` | 15968 | 13952 | 2016 |

Longer horizons have more ineligible rows near the end of 2025 because realized targets require more future trading sessions.

Smoke coverage:

| horizon | total predictions | eligible predictions | ineligible predictions |
| --- | ---: | ---: | ---: |
| `short_5d` | 5984 | 5944 | 40 |
| `short_10d` | 5984 | 5904 | 80 |
| `short_20d` | 5792 | 5632 | 160 |
| `long_3m` | 4680 | 4176 | 504 |

## Model Metrics by Horizon

| horizon | model | observations | MAE | RMSE | correlation | directional accuracy | F1 | Brier score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `short_5d` | `cart` | 3952 | 0.036711 | 0.055281 | -0.013606 | 0.490385 | 0.448823 | 0.344833 |
| `short_5d` | `lightgbm` | 3952 | 0.040198 | 0.057846 | -0.111271 | 0.519231 | 0.430797 | 0.319293 |
| `short_5d` | `stacking_final` | 3952 | 0.028088 | 0.041345 | 0.010164 | 0.504555 | 0.665414 | n/a |
| `short_5d` | `xgboost` | 3952 | 0.042669 | 0.061021 | -0.112169 | 0.523532 | 0.437743 | 0.316786 |
| `short_10d` | `cart` | 3912 | 0.059922 | 0.089147 | -0.235536 | 0.493865 | 0.484912 | 0.377385 |
| `short_10d` | `lightgbm` | 3912 | 0.065053 | 0.094519 | -0.228909 | 0.497955 | 0.423371 | 0.357157 |
| `short_10d` | `stacking_final` | 3912 | 0.041172 | 0.058348 | 0.133228 | 0.518405 | 0.674724 | n/a |
| `short_10d` | `xgboost` | 3912 | 0.066519 | 0.097643 | -0.192131 | 0.497699 | 0.416394 | 0.351393 |
| `short_20d` | `cart` | 3832 | 0.098706 | 0.153536 | -0.324371 | 0.513831 | 0.508573 | 0.396776 |
| `short_20d` | `lightgbm` | 3832 | 0.095877 | 0.141306 | -0.289868 | 0.497129 | 0.490885 | 0.387800 |
| `short_20d` | `stacking_final` | 3832 | 0.062468 | 0.088335 | 0.226906 | 0.537839 | 0.692801 | n/a |
| `short_20d` | `xgboost` | 3832 | 0.095143 | 0.136839 | -0.218969 | 0.499478 | 0.494465 | 0.374204 |
| `long_3m` | `cart` | 3488 | 0.190467 | 0.314755 | -0.395425 | 0.556766 | 0.591653 | 0.403259 |
| `long_3m` | `lightgbm` | 3488 | 0.183751 | 0.305357 | -0.381935 | 0.565080 | 0.620465 | 0.362390 |
| `long_3m` | `stacking_final` | 3488 | 0.138285 | 0.220925 | 0.196127 | 0.546445 | 0.673276 | n/a |
| `long_3m` | `xgboost` | 3488 | 0.182417 | 0.295529 | -0.334971 | 0.551892 | 0.615876 | 0.345464 |

## Best Model by Horizon

| horizon | criterion | best model | value |
| --- | --- | --- | ---: |
| `short_5d` | lowest MAE | `stacking_final` | 0.028088 |
| `short_5d` | lowest RMSE | `stacking_final` | 0.041345 |
| `short_5d` | highest directional accuracy | `xgboost` | 0.523532 |
| `short_5d` | highest F1 | `stacking_final` | 0.665414 |
| `short_10d` | lowest MAE | `stacking_final` | 0.041172 |
| `short_10d` | lowest RMSE | `stacking_final` | 0.058348 |
| `short_10d` | highest directional accuracy | `stacking_final` | 0.518405 |
| `short_10d` | highest F1 | `stacking_final` | 0.674724 |
| `short_20d` | lowest MAE | `stacking_final` | 0.062468 |
| `short_20d` | lowest RMSE | `stacking_final` | 0.088335 |
| `short_20d` | highest directional accuracy | `stacking_final` | 0.537839 |
| `short_20d` | highest F1 | `stacking_final` | 0.692801 |
| `long_3m` | lowest MAE | `stacking_final` | 0.138285 |
| `long_3m` | lowest RMSE | `stacking_final` | 0.220925 |
| `long_3m` | highest directional accuracy | `lightgbm` | 0.565080 |
| `long_3m` | highest F1 | `stacking_final` | 0.673276 |

## Ticker-Level Findings

`stacking_final` by ticker and horizon:

| ticker | horizon | observations | MAE | RMSE | correlation | directional accuracy | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ACB` | `short_5d` | 494 | 0.021176 | 0.030672 | -0.124949 | 0.518219 | 0.680965 |
| `BVH` | `short_5d` | 494 | 0.032587 | 0.045918 | -0.163567 | 0.522267 | 0.686170 |
| `FPT` | `short_5d` | 494 | 0.028958 | 0.039267 | -0.013381 | 0.500000 | 0.650636 |
| `HPG` | `short_5d` | 494 | 0.025759 | 0.036912 | -0.096875 | 0.514170 | 0.675676 |
| `SSI` | `short_5d` | 494 | 0.031778 | 0.043869 | -0.058432 | 0.530364 | 0.693122 |
| `VCB` | `short_5d` | 494 | 0.019180 | 0.027553 | 0.046748 | 0.449393 | 0.597633 |
| `VIC` | `short_5d` | 494 | 0.044912 | 0.065406 | 0.024800 | 0.540486 | 0.701708 |
| `VNM` | `short_5d` | 494 | 0.020355 | 0.027567 | -0.114177 | 0.461538 | 0.627451 |
| `ACB` | `short_10d` | 489 | 0.031442 | 0.042859 | -0.142662 | 0.539877 | 0.698795 |
| `BVH` | `short_10d` | 489 | 0.048723 | 0.065916 | -0.189978 | 0.539877 | 0.701195 |
| `FPT` | `short_10d` | 489 | 0.040104 | 0.053913 | 0.070302 | 0.464213 | 0.598160 |
| `HPG` | `short_10d` | 489 | 0.037080 | 0.048939 | -0.108111 | 0.494888 | 0.660248 |
| `SSI` | `short_10d` | 489 | 0.050184 | 0.066690 | -0.029529 | 0.564417 | 0.713324 |
| `VCB` | `short_10d` | 489 | 0.026942 | 0.036886 | 0.138672 | 0.505112 | 0.650289 |
| `VIC` | `short_10d` | 489 | 0.065535 | 0.092510 | 0.215984 | 0.584867 | 0.738065 |
| `VNM` | `short_10d` | 489 | 0.029367 | 0.037230 | -0.122774 | 0.453988 | 0.620199 |
| `ACB` | `short_20d` | 479 | 0.046382 | 0.059192 | -0.154646 | 0.544885 | 0.705405 |
| `BVH` | `short_20d` | 479 | 0.072420 | 0.091429 | -0.275846 | 0.557411 | 0.715818 |
| `FPT` | `short_20d` | 479 | 0.063590 | 0.078347 | 0.148895 | 0.521921 | 0.676096 |
| `HPG` | `short_20d` | 479 | 0.052874 | 0.066824 | -0.255720 | 0.542797 | 0.702849 |
| `SSI` | `short_20d` | 479 | 0.074956 | 0.101352 | 0.071103 | 0.563674 | 0.706872 |
| `VCB` | `short_20d` | 479 | 0.039301 | 0.051990 | 0.013748 | 0.513570 | 0.661829 |
| `VIC` | `short_20d` | 479 | 0.105861 | 0.154991 | 0.314617 | 0.597077 | 0.736698 |
| `VNM` | `short_20d` | 479 | 0.044356 | 0.054418 | -0.017860 | 0.461378 | 0.631429 |
| `ACB` | `long_3m` | 436 | 0.075691 | 0.100948 | -0.260697 | 0.584862 | 0.737300 |
| `BVH` | `long_3m` | 436 | 0.116914 | 0.133659 | -0.073226 | 0.552752 | 0.692913 |
| `FPT` | `long_3m` | 436 | 0.122258 | 0.165393 | 0.027270 | 0.587156 | 0.666667 |
| `HPG` | `long_3m` | 436 | 0.092291 | 0.126959 | -0.139177 | 0.616972 | 0.734499 |
| `SSI` | `long_3m` | 436 | 0.182043 | 0.254468 | -0.146429 | 0.486239 | 0.525424 |
| `VCB` | `long_3m` | 436 | 0.086215 | 0.106167 | -0.506070 | 0.435780 | 0.607029 |
| `VIC` | `long_3m` | 436 | 0.351589 | 0.483837 | 0.433411 | 0.651376 | 0.766154 |
| `VNM` | `long_3m` | 436 | 0.079276 | 0.093860 | -0.087125 | 0.456422 | 0.605657 |

Average `stacking_final` behavior by ticker:

| ticker | average MAE | average RMSE | average directional accuracy | average F1 |
| --- | ---: | ---: | ---: | ---: |
| `VCB` | 0.042909 | 0.055649 | 0.475964 | 0.629195 |
| `VNM` | 0.043339 | 0.053269 | 0.458332 | 0.621184 |
| `ACB` | 0.043673 | 0.058418 | 0.546961 | 0.705617 |
| `HPG` | 0.052001 | 0.069909 | 0.542207 | 0.693318 |
| `FPT` | 0.063727 | 0.084230 | 0.518322 | 0.647890 |
| `BVH` | 0.067661 | 0.084230 | 0.543077 | 0.699024 |
| `SSI` | 0.084740 | 0.116594 | 0.536174 | 0.659686 |
| `VIC` | 0.141974 | 0.199186 | 0.593452 | 0.735656 |

Point-error behavior is not concentrated in the same tickers as directional behavior. `VCB` and `VNM` have the lowest average stacking errors, while `VIC` has the highest average error but the strongest average directional accuracy and F1. The strongest directional slice is `VIC long_3m` with directional accuracy `0.651376`; the weakest is `VCB long_3m` with directional accuracy `0.435780`.

The broader basket therefore preserves the broad error-vs-direction split, but it also shows that ticker composition can materially shift directional rankings.

## Context Coverage and Governance

Context coverage summary:

| diagnostic | value |
| --- | ---: |
| ticker-horizon rows | 32 |
| fold count per ticker-horizon | 499 |
| mean breadth missing rate | 0.000000 |
| max breadth missing rate | 0.000000 |
| weak coverage fold count | 0 |
| review fold count | 0 |
| warning level | `ok` |

Foreign-flow metadata:

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

Foreign-flow disabled status is intentional exclusion, not complete foreign-flow coverage. Foreign-flow features must not be interpreted.

Feature governance summary:

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

Stability diagnostics:

| stability level | feature importance count | linear coefficient count |
| --- | ---: | ---: |
| `high` | 298 | 55 |
| `medium` | 79 | 65 |
| `low` | 79 | 48 |

Linear-vs-importance alignment:

| alignment label | count |
| --- | ---: |
| `importance_only` | 96 |
| `aligned_stable` | 53 |
| `unstable_or_missing` | 3 |

The `requires_review` features are context/breadth timing features. No governance status was changed.

## Comparison Against 4-Ticker Baseline

The comparison is directionally useful but not perfectly like-for-like because this broader run used the 2024-2025 fallback evaluation window, while the previous 4-ticker baseline used 2023-2025.

Pattern comparison:

| question | 4-ticker baseline | broader 8-supported-ticker fallback |
| --- | --- | --- |
| Did `stacking_final` dominate MAE/RMSE? | yes, all four horizons | yes, all four horizons |
| Did directional accuracy remain horizon-sensitive? | yes | yes |
| Did XGBoost show directional strength? | yes, `short_5d` and `long_3m` | yes on `short_5d`; LightGBM leads `long_3m` direction |
| Did LightGBM show directional strength? | yes on `short_20d` | yes on `long_3m` |
| Did new tickers change the pattern? | not applicable | yes, `VIC` and `ACB`/`HPG` materially change ticker-level direction behavior |
| Was foreign-flow excluded? | yes, disabled mode | yes, disabled mode |

The most stable finding is the MAE/RMSE dominance of `stacking_final`. Directional leadership is less stable. In the broader fallback, `xgboost` leads directional accuracy for `short_5d`, `stacking_final` leads `short_10d` and `short_20d`, and `lightgbm` leads `long_3m`.

This supports the prior caution: a model that reduces point-forecast error is not automatically the strongest directional classifier.

## Interpretation

Expanding the ticker universe does not overturn the main point-error conclusion. `stacking_final` remains the best MAE/RMSE model for every evaluated horizon in the broader supported basket.

The directional story is more sensitive. `xgboost` no longer leads the long-horizon direction metric in the fallback run; `lightgbm` does. `stacking_final` leads directional accuracy at `short_10d` and `short_20d`, but the margins are modest and remain evaluation-window dependent.

Ticker mix matters. Lower-error tickers are not necessarily stronger directional tickers. `VIC` has the highest average stacking error but the best average directional behavior, while `VCB` and `VNM` are low-error but weaker directionally. This argues for regime-aware and ticker-aware slicing before any stronger interpretation.

## Limitations

- The requested basket has 10 tickers, but only 8 support the full 2010-2025 coverage threshold.
- `MWG` and `DGC` were excluded from walk-forward evaluation because their staged histories start in 2014.
- The broader run used the documented 2024-2025 fallback forecast window, not the preferred full 2023-2025 window.
- The coverage denominator uses a generic business-day calendar rather than an official Vietnam exchange calendar.
- Foreign-flow is intentionally disabled and must not be interpreted.
- Only CART, XGBoost, LightGBM, and the existing `stacking_final` layer are evaluated.
- No transaction costs, slippage, liquidity constraints, turnover constraints, or portfolio construction constraints are modeled.
- The outputs are diagnostic artifacts, not trading strategy proof.
- No live-trading claim is made.

## Next Recommended Task

Recommended next task: regime-aware slicing across bull, bear, and sideways regimes.

The broader audit shows that ticker composition changes directional behavior. Regime slicing is the next most useful diagnostic because it can test whether the observed direction sensitivity is tied to market state rather than ticker selection alone.
