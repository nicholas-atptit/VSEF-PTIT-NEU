# VN30 Hourly Coverage Audit 2005-2026

## Scope

- Universe: frozen VN30, exactly 30 tickers from `configs/universes/vn30_constituents_frozen.csv`.
- Frequency: hourly only.
- Historical/training period: 2005-01-01 00:00:00 to 2024-12-31 23:59:59.
- Evaluation/comparison period: 2025-01-01 00:00:00 to 2026-05-31 23:59:59.
- Daily data and daily-to-hourly resampling are not used.

## Summary

- Benchmark-usable tickers: 0 of 30.
- Missing/unusable tickers: 30 of 30.
- Full requested VN30 hourly design feasible: false.

## Benchmark-Usable Tickers

None.

## Missing Reason Concentration

| missing_reason | ticker_count |
| --- | --- |
| training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 | 12 |
| training_start_after_requested_start:2025-10-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-10-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 | 6 |
| training_start_after_requested_start:2026-02-11 14:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2026-02-11 14:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 | 3 |
| training_start_after_requested_start:2024-01-02 10:00:00>2005-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 | 2 |
| training_start_after_requested_start:2025-11-05 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-11-05 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 | 1 |
| training_start_after_requested_start:2025-11-06 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-11-06 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 | 1 |
| training_start_after_requested_start:2025-11-10 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-11-10 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 | 1 |
| training_start_after_requested_start:2025-11-18 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-11-18 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 | 1 |
| training_start_after_requested_start:2025-11-19 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-11-19 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 | 1 |
| training_start_after_requested_start:2025-11-25 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-11-25 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 | 1 |
| training_start_after_requested_start:2025-12-15 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-12-15 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 | 1 |

## Per-Ticker Audit

| ticker | first_available_hourly_timestamp | last_available_hourly_timestamp | hourly_rows | missing_training_coverage | missing_evaluation_coverage | benchmark_usable | missing_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ACB | 2024-01-02 10:00:00 | 2026-05-11 14:00:00 | 1296 | True | True | False | training_start_after_requested_start:2024-01-02 10:00:00>2005-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| BID | 2024-01-02 10:00:00 | 2026-05-11 14:00:00 | 2648 | True | True | False | training_start_after_requested_start:2024-01-02 10:00:00>2005-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| CTG | 2025-11-18 09:00:00 | 2026-05-11 14:00:00 | 575 | True | True | False | training_start_after_requested_start:2025-11-18 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-11-18 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| DGC | 2025-11-25 09:00:00 | 2026-05-11 14:00:00 | 550 | True | True | False | training_start_after_requested_start:2025-11-25 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-11-25 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| FPT | 2025-11-05 09:00:00 | 2026-05-11 14:00:00 | 620 | True | True | False | training_start_after_requested_start:2025-11-05 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-11-05 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| GAS | 2025-10-27 09:00:00 | 2026-05-11 14:00:00 | 655 | True | True | False | training_start_after_requested_start:2025-10-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-10-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| GVR | 2025-10-27 09:00:00 | 2026-05-11 14:00:00 | 655 | True | True | False | training_start_after_requested_start:2025-10-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-10-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| HDB | 2025-11-19 09:00:00 | 2026-05-11 14:00:00 | 570 | True | True | False | training_start_after_requested_start:2025-11-19 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-11-19 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| HPG | 2026-02-11 14:00:00 | 2026-05-11 14:00:00 | 201 | True | True | False | training_start_after_requested_start:2026-02-11 14:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2026-02-11 14:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| LPB | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | True | True | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| MBB | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | True | True | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| MSN | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | True | True | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| MWG | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | True | True | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| PLX | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | True | True | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| SAB | 2025-12-15 09:00:00 | 2026-05-11 14:00:00 | 480 | True | True | False | training_start_after_requested_start:2025-12-15 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-12-15 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| SHB | 2026-02-11 14:00:00 | 2026-05-11 14:00:00 | 276 | True | True | False | training_start_after_requested_start:2026-02-11 14:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2026-02-11 14:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| SSB | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | True | True | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| SSI | 2025-11-10 09:00:00 | 2026-05-11 14:00:00 | 605 | True | True | False | training_start_after_requested_start:2025-11-10 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-11-10 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| STB | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | True | True | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| TCB | 2025-10-27 09:00:00 | 2026-05-11 14:00:00 | 655 | True | True | False | training_start_after_requested_start:2025-10-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-10-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| TPB | 2025-10-27 09:00:00 | 2026-05-11 14:00:00 | 655 | True | True | False | training_start_after_requested_start:2025-10-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-10-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| VCB | 2025-10-27 09:00:00 | 2026-05-11 14:00:00 | 655 | True | True | False | training_start_after_requested_start:2025-10-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-10-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| VHM | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | True | True | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| VIB | 2026-02-11 14:00:00 | 2026-05-11 14:00:00 | 251 | True | True | False | training_start_after_requested_start:2026-02-11 14:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2026-02-11 14:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| VIC | 2025-11-06 09:00:00 | 2026-05-11 14:00:00 | 615 | True | True | False | training_start_after_requested_start:2025-11-06 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-11-06 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| VJC | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | True | True | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| VNM | 2025-10-27 09:00:00 | 2026-05-11 14:00:00 | 655 | True | True | False | training_start_after_requested_start:2025-10-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-10-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| VPB | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | True | True | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| VPL | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 854 | True | True | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| VRE | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | True | True | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |

## Boundary

If fewer than 30 tickers are benchmark-usable, the benchmark and paper must stop before final VN30 claims.
