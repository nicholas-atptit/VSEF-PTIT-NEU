# VN30 Hourly vnstock Full Benchmark Missing Evidence

## Decision

The full 2005-2026 VN30 hourly benchmark was not run because the fetched-data validation gate did not pass.

## Required Gate

- All 30 frozen VN30 stocks must be benchmark-usable.
- VNINDEX hourly coverage must be benchmark-usable.
- Training/history: 2005-01-01 00:00:00 to 2024-12-31 23:59:59.
- Evaluation/comparison: 2025-01-01 00:00:00 to 2026-05-31 23:59:59.
- Frequency: hourly only.
- No daily data, no daily-to-hourly resampling, no old VN100 evidence, and no fabricated data.

## Current Validation Snapshot

- Benchmark-usable VN30 stocks: 0/30.
- VNINDEX benchmark-usable: False.
- Benchmark output directory reserved: `outputs/vn30_hourly_vnstock_full_2005_2026_traincutoff`.
- Source script: `run_vn30_hourly_benchmark_2005_2026_from_fetched.py`.

## Failed or Missing Rows

| symbol | asset_type | required_start | required_end | first_datetime | last_datetime | row_count | benchmark_usable | failure_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACB | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| BID | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| CTG | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| DGC | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| FPT | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| GAS | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| GVR | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| HDB | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| HPG | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| LPB | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| MBB | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| MSN | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| MWG | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| PLX | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| SAB | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| SHB | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| SSB | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| SSI | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| STB | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| TCB | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| TPB | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| VCB | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| VHM | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| VIB | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| VIC | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| VJC | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| VNM | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| VPB | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| VPL | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| VRE | stock | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| VNINDEX | index | 2005-01-01 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | required_cache_file_missing |
| VN30INDEX | index | 2012-02-06 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | optional_index_not_fetched_or_unsupported |
| VNXALL | index | 2016-10-24 00:00:00 | 2026-05-31 23:59:59 |  |  | 0 | false | optional_index_not_fetched_or_unsupported |
