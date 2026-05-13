# VN30 Hourly Available-Window Audit

## Scope

- Universe: frozen VN30, exactly 30 tickers.
- Frequency: hourly only.
- Source: local hourly files only.
- Daily data, daily-to-hourly resampling, and old VN100 artifacts are not used.

## Direct Answers

- Is a full-30 common hourly window feasible? false.
- Full-30 common window: 2026-02-11 14:00:00 to 2026-05-11 14:00:00; common timestamps: 201.
- If no: the full-30 common window does not meet the minimum train/eval row rule for this available-window study.
- Best available-window design: 27 tickers from 2025-12-15 09:00:00 to 2026-05-11 14:00:00.
- Final paper can proceed: true.

## Coverage Windows

| coverage_floor | ticker_count | window_start | window_end | common_timestamp_count | min_rows_per_ticker | train_rows_per_ticker | eval_rows_per_ticker | valid_split |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 30 | 2026-02-11 14:00:00 | 2026-05-11 14:00:00 | 201 | 201 | 0 | 0 | False |
| 25 | 25 | 2025-11-19 09:00:00 | 2026-05-11 14:00:00 | 570 | 570 | 342 | 228 | True |
| 20 | 20 | 2025-10-27 09:00:00 | 2026-05-11 14:00:00 | 655 | 655 | 393 | 262 | True |
| 15 | 20 | 2025-10-27 09:00:00 | 2026-05-11 14:00:00 | 655 | 655 | 393 | 262 | True |
| 10 | 14 | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 854 | 854 | 512 | 342 | True |

## Per-Ticker Local Hourly Coverage

| ticker | first_available_hourly_timestamp | last_available_hourly_timestamp | hourly_rows | rows_before_2025 | rows_2025_onward |
| --- | --- | --- | --- | --- | --- |
| ACB | 2024-01-02 10:00:00 | 2026-05-11 14:00:00 | 1296 | 431 | 865 |
| BID | 2024-01-02 10:00:00 | 2026-05-11 14:00:00 | 2648 | 1001 | 1647 |
| CTG | 2025-11-18 09:00:00 | 2026-05-11 14:00:00 | 575 | 0 | 575 |
| DGC | 2025-11-25 09:00:00 | 2026-05-11 14:00:00 | 550 | 0 | 550 |
| FPT | 2025-11-05 09:00:00 | 2026-05-11 14:00:00 | 620 | 0 | 620 |
| GAS | 2025-10-27 09:00:00 | 2026-05-11 14:00:00 | 655 | 0 | 655 |
| GVR | 2025-10-27 09:00:00 | 2026-05-11 14:00:00 | 655 | 0 | 655 |
| HDB | 2025-11-19 09:00:00 | 2026-05-11 14:00:00 | 570 | 0 | 570 |
| HPG | 2026-02-11 14:00:00 | 2026-05-11 14:00:00 | 201 | 0 | 201 |
| LPB | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | 0 | 860 |
| MBB | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | 0 | 860 |
| MSN | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | 0 | 860 |
| MWG | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | 0 | 860 |
| PLX | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | 0 | 860 |
| SAB | 2025-12-15 09:00:00 | 2026-05-11 14:00:00 | 480 | 0 | 480 |
| SHB | 2026-02-11 14:00:00 | 2026-05-11 14:00:00 | 276 | 0 | 276 |
| SSB | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | 0 | 860 |
| SSI | 2025-11-10 09:00:00 | 2026-05-11 14:00:00 | 605 | 0 | 605 |
| STB | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | 0 | 860 |
| TCB | 2025-10-27 09:00:00 | 2026-05-11 14:00:00 | 655 | 0 | 655 |
| TPB | 2025-10-27 09:00:00 | 2026-05-11 14:00:00 | 655 | 0 | 655 |
| VCB | 2025-10-27 09:00:00 | 2026-05-11 14:00:00 | 655 | 0 | 655 |
| VHM | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | 0 | 860 |
| VIB | 2026-02-11 14:00:00 | 2026-05-11 14:00:00 | 251 | 0 | 251 |
| VIC | 2025-11-06 09:00:00 | 2026-05-11 14:00:00 | 615 | 0 | 615 |
| VJC | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | 0 | 860 |
| VNM | 2025-10-27 09:00:00 | 2026-05-11 14:00:00 | 655 | 0 | 655 |
| VPB | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | 0 | 860 |
| VPL | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 854 | 0 | 854 |
| VRE | 2025-08-27 09:00:00 | 2026-05-11 14:00:00 | 860 | 0 | 860 |

## Limitations

- This audit uses only the real local hourly data already present in the repository.
- It does not satisfy the 2005-2026 full-history VN30 requirement.
- Any selected subset must be labeled as an available-window VN30 subset unless all 30 tickers are selected.
- The selected split is data-driven because the suggested 2025 evaluation start is not feasible for the local hourly universe.
