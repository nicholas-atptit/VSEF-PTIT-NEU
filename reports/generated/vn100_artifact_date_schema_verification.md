# VN100 Artifact Date and Schema Verification

## Reviewed Artifacts

- `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/run_config.json`
- `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/manifest.json`
- `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/daily/benchmark_summary.json`
- `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/hourly/benchmark_summary.json`
- `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/usable_cache_summary.csv`
- `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/fetch_summary.csv`
- `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/source_health_summary.csv`

## Date Field Findings

| Source | Field | Value |
|---|---|---|
| `run_config.json` | `daily_start` / `daily_end` | 2006-01-01 / 2015-12-31 |
| `run_config.json` | `hourly_start` / `hourly_end` | 2016-01-01 / 2025-12-31 |
| `run_config.json` | `train_cutoff` | 2024-12-31 |
| `run_config.json` | `eval_start` / `eval_end` | 2025-01-01 / 2025-12-31 |
| `run_config.json` | `training_label_cutoff_rule` | `target_timestamp <= train_cutoff` |
| `manifest.json` | `raw_daily_range` | 2006-01-01 / 2015-12-31 |
| `manifest.json` | `raw_hourly_range` | 2016-01-01 / 2025-12-31 |
| `manifest.json` | `effective_training_range.daily` | 2006-01-03 / 2024-12-31 |
| `manifest.json` | `effective_training_range.hourly` | 2024-01-02 / 2024-12-31 |
| `manifest.json` | `effective_evaluation_range.daily` | 2025-01-02 / 2025-12-31 |
| `manifest.json` | `effective_evaluation_range.hourly` | 2025-01-02 / 2025-12-31 |
| `daily/benchmark_summary.json` | `raw_data_start` / `raw_data_end` | 2006-01-03 / 2025-12-31 |
| `daily/benchmark_summary.json` | `effective_train_start` / `effective_train_end` | 2006-01-03 / 2024-12-31 |
| `daily/benchmark_summary.json` | `effective_eval_start` / `effective_eval_end` | 2025-01-02 / 2025-12-31 |
| `hourly/benchmark_summary.json` | `raw_data_start` / `raw_data_end` | 2024-01-02 / 2025-12-31 |
| `hourly/benchmark_summary.json` | `initial_train_start` / `initial_train_end` | 2016-01-01 / 2024-12-31 |
| `hourly/benchmark_summary.json` | `effective_train_start` / `effective_train_end` | 2024-01-02 / 2024-12-31 |
| `hourly/benchmark_summary.json` | `effective_eval_start` / `effective_eval_end` | 2025-01-02 / 2025-12-31 |

## Cache Usability Findings

| Frequency | Benchmark-usable rows in `usable_cache_summary.csv` | Evaluated tickers | Actual/effective range for usable rows |
|---|---:|---|---|
| daily | 0 | none from standalone daily cache rows | none |
| hourly | 7 | ANV, BCM, BID, BMP, BVH, BWE, CII | 2024-01-02 to 2025-12-31 |

The standalone daily cache rows are not marked benchmark-usable because they end before the 2025 evaluation window. The official daily benchmark summary nevertheless reports predictions because the manifest records a hybrid method: daily OHLCV for 2006-2015 plus hourly OHLCV from 2016 onward resampled to daily.

## Schema Findings

| Artifact | Required fields observed | Evidence status |
|---|---|---|
| `daily/predicted_vs_actual.csv`, `hourly/predicted_vs_actual.csv` | `ticker`, `frequency`, `horizon`, `model`, `actual_direction`, `predicted_direction`, `confidence`, `regime`, `volatility_regime`, `is_correct` | enough for ticker concentration and regime diagnostics |
| `confidence_threshold_sweep_summary.csv` | `threshold`, `total_rows`, `evaluated_rows`, `coverage_ratio`, `filtered_accuracy`, `passed_60pct`, `coverage_ok`, `selected_candidate` | enough for hourly coverage review |
| `daily/confidence_threshold_sweep_summary.csv` | header only | missing daily threshold-sweep evidence |
| official prediction and summary artifacts | no transaction-cost, slippage, turnover, drawdown, profit-factor, trade-list, or net-return fields | insufficient for trading-readiness claims |

## Wording Verdict

The statement `Daily historical inputs: 2006-01-01 to 2015-12-31` is artifact-backed only as the raw daily cache request/range. It is misleading if read as the complete daily benchmark input, because official metadata says the daily benchmark uses daily OHLCV for 2006-2015 plus hourly OHLCV from 2016 onward resampled to daily. The research design document has been patched to use this artifact-backed wording.
