# VN30 Hourly Benchmark Missing Evidence

## Design

- Universe: frozen VN30, exactly 30 tickers.
- Frequency: hourly only.
- Training-label period: 2005-01-01 00:00:00 to 2024-12-31 23:59:59.
- Evaluation/comparison period: 2025-01-01 00:00:00 to 2026-05-31 23:59:59.
- Leakage rule: training labels require target_timestamp <= train_cutoff.

## Result

- Source script: `run_vn30_hourly_benchmark_2005_2026.py`.
- Benchmark-usable tickers: 0 of 30.
- Failed tickers: 30.
- Requested 2005-2026 hourly design feasible: false.

The VN30 hourly rerun did not achieve full 30-ticker benchmark usability under the requested 2005-2026 hourly design.

## Failed Tickers

| ticker | first_available_hourly_timestamp | last_available_hourly_timestamp | hourly_rows | benchmark_usable | missing_reason |
| --- | --- | --- | --- | --- | --- |
| ACB | 2024-01-02 10:00:00 | 2026-05-11 14:00:00 | 1296 | False | training_start_after_requested_start:2024-01-02 10:00:00>2005-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| BID | 2024-01-02 10:00:00 | 2026-05-11 14:00:00 | 2648 | False | training_start_after_requested_start:2024-01-02 10:00:00>2005-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| CTG | 2025-11-18 09:00:00 | 2026-05-11 14:00:00 | 575 | False | training_start_after_requested_start:2025-11-18 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-11-18 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| DGC | 2025-11-25 09:00:00 | 2026-05-11 14:00:00 | 550 | False | training_start_after_requested_start:2025-11-25 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-11-25 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| FPT | 2025-11-05 09:00:00 | 2026-05-11 14:00:00 | 620 | False | training_start_after_requested_start:2025-11-05 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-11-05 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| GAS | 2025-10-27 09:00:00 | 2026-05-11 14:00:00 | 655 | False | training_start_after_requested_start:2025-10-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-10-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| GVR | 2025-10-27 09:00:00 | 2026-05-11 14:00:00 | 655 | False | training_start_after_requested_start:2025-10-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-10-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| HDB | 2025-11-19 09:00:00 | 2026-05-11 14:00:00 | 570 | False | training_start_after_requested_start:2025-11-19 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-11-19 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| HPG | 2026-02-11 14:00:00 | 2026-05-11 14:00:00 | 201 | False | training_start_after_requested_start:2026-02-11 14:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2026-02-11 14:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| LPB | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| MBB | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| MSN | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| MWG | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| PLX | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| SAB | 2025-12-15 09:00:00 | 2026-05-11 14:00:00 | 480 | False | training_start_after_requested_start:2025-12-15 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-12-15 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| SHB | 2026-02-11 14:00:00 | 2026-05-11 14:00:00 | 276 | False | training_start_after_requested_start:2026-02-11 14:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2026-02-11 14:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| SSB | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| SSI | 2025-11-10 09:00:00 | 2026-05-11 14:00:00 | 605 | False | training_start_after_requested_start:2025-11-10 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-11-10 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| STB | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| TCB | 2025-10-27 09:00:00 | 2026-05-11 14:00:00 | 655 | False | training_start_after_requested_start:2025-10-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-10-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| TPB | 2025-10-27 09:00:00 | 2026-05-11 14:00:00 | 655 | False | training_start_after_requested_start:2025-10-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-10-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| VCB | 2025-10-27 09:00:00 | 2026-05-11 14:00:00 | 655 | False | training_start_after_requested_start:2025-10-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-10-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| VHM | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| VIB | 2026-02-11 14:00:00 | 2026-05-11 14:00:00 | 251 | False | training_start_after_requested_start:2026-02-11 14:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2026-02-11 14:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| VIC | 2025-11-06 09:00:00 | 2026-05-11 14:00:00 | 615 | False | training_start_after_requested_start:2025-11-06 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-11-06 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| VJC | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| VNM | 2025-10-27 09:00:00 | 2026-05-11 14:00:00 | 655 | False | training_start_after_requested_start:2025-10-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-10-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| VPB | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| VPL | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 854 | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |
| VRE | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | False | training_start_after_requested_start:2025-08-27 09:00:00>2005-01-01 00:00:00; no_training_rows_in_requested_window; evaluation_start_after_requested_start:2025-08-27 09:00:00>2025-01-01 00:00:00; evaluation_end_before_requested_end:2026-05-11 14:00:00<2026-05-31 23:59:59 |

## Claim Boundary

- No final VN30 hourly paper claims should be written from this run.
- No daily data, daily-to-hourly resampling, VN100 seven-ticker evidence, or shortened period is accepted as a substitute.
