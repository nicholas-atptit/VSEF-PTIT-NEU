# VN30 Hourly Listing-Aware Benchmark Missing Evidence

## Decision

The VN30 hourly listing-aware benchmark was not run because the validation gate did not pass.

## Gate

- All 30 frozen VN30 stocks must be usable under the listing-aware rule.
- Minimum training rows per ticker: 1000.
- Minimum evaluation rows per ticker: 100.
- Train cutoff: 2024-12-31 23:59:59.
- Evaluation start: 2025-01-01 00:00:00.
- Requested evaluation end: 2026-05-31 23:59:59.
- Per-ticker start rule: max(first_trading_date, first provider-available hourly timestamp).
- No daily data, daily-to-hourly resampling, VN100 evidence reuse, or fabricated bars.

## Current Status

- Usable VN30 stocks: 0/30.
- actual_eval_end: not available.
- Source script: `run_vn30_hourly_listing_aware_benchmark.py`.

## Failed or Missing Rows

| symbol | asset_type | listing_date_used | requested_start | first_datetime | last_datetime | training_rows | evaluation_rows | benchmark_usable | missing_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACB | stock | 2006-11-21 00:00:00 | 2006-11-21 00:00:00 | 2023-09-11 10:00:00 | 2026-05-13 14:00:00 | 91 | 146 | false | training_rows_below_1000 |
| BID | stock |  | 2005-01-01 00:00:00 | 2023-09-11 10:00:00 | 2026-05-13 14:00:00 | 91 | 156 | false | training_rows_below_1000 |
| CTG | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| DGC | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| FPT | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| GAS | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| GVR | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| HDB | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| HPG | stock | 2007-11-15 00:00:00 | 2007-11-15 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| LPB | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| MBB | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| MSN | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| MWG | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| PLX | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| SAB | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| SHB | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| SSB | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| SSI | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| STB | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| TCB | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| TPB | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| VCB | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| VHM | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| VIB | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| VIC | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| VJC | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| VNM | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| VPB | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| VPL | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| VRE | stock |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| VNINDEX | index |  | 2005-01-01 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| VN30INDEX | index |  | 2012-02-06 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
| VNXALL | index |  | 2016-10-24 00:00:00 |  |  | 0 | 0 | false | cache_file_missing |
