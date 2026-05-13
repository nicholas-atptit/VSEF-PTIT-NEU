# VN30 Hourly Listing-Aware Validation

## Gate Decision

- Listing-aware validation gate passed: False.
- Usable VN30 stocks: 0/30.
- actual_eval_end: not available.
- VNINDEX fetched/usable: fetched=False, usable=False.
- VN30INDEX support: False.
- VNXALL support: False.

## Thresholds and Rules

- Minimum training rows per stock: 1000.
- Minimum evaluation rows per stock: 100.
- Per-ticker training start: max(first trading/listing date, first provider-available hourly timestamp).
- Training labels end at: 2024-12-31 23:59:59.
- Evaluation starts at: 2025-01-01 00:00:00.
- Requested evaluation end: 2026-05-31 23:59:59.
- actual_eval_end is computed from available provider timestamps, not assumed future data.
- No daily data, daily-to-hourly resampling, VN100 evidence reuse, or fabricated bars are used.

## Required Failures

| symbol | asset_type | listing_date_used | requested_start | ticker_training_start | first_datetime | last_datetime | training_rows | evaluation_rows | benchmark_usable | missing_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACB | stock | 2006-11-21 00:00:00 | 2006-11-21 00:00:00 | 2023-09-11 10:00:00 | 2023-09-11 10:00:00 | 2026-05-13 14:00:00 | 91 | 146 | false | training_rows_below_1000 |
| BID | stock |  | 2005-01-01 00:00:00 | 2023-09-11 10:00:00 | 2023-09-11 10:00:00 | 2026-05-13 14:00:00 | 91 | 156 | false | training_rows_below_1000 |
| CTG | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| DGC | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| FPT | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| GAS | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| GVR | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| HDB | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| HPG | stock | 2007-11-15 00:00:00 | 2007-11-15 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| LPB | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| MBB | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| MSN | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| MWG | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| PLX | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| SAB | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| SHB | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| SSB | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| SSI | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| STB | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| TCB | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| TPB | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| VCB | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| VHM | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| VIB | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| VIC | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| VJC | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| VNM | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| VPB | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| VPL | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| VRE | stock |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |
| VNINDEX | index |  | 2005-01-01 00:00:00 |  |  |  | 0 | 0 | false | cache_file_missing |

## Per-Symbol Validation

| symbol | asset_type | gate_required | fetched | row_count | first_datetime | last_datetime | training_rows | evaluation_rows | benchmark_usable | missing_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACB | stock | true | true | 237 | 2023-09-11 10:00:00 | 2026-05-13 14:00:00 | 91 | 146 | false | training_rows_below_1000 |
| BID | stock | true | true | 247 | 2023-09-11 10:00:00 | 2026-05-13 14:00:00 | 91 | 156 | false | training_rows_below_1000 |
| CTG | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| DGC | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| FPT | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| GAS | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| GVR | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| HDB | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| HPG | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| LPB | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| MBB | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| MSN | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| MWG | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| PLX | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| SAB | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| SHB | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| SSB | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| SSI | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| STB | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| TCB | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| TPB | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| VCB | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| VHM | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| VIB | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| VIC | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| VJC | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| VNM | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| VPB | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| VPL | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| VRE | stock | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| VNINDEX | index | true | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| VN30INDEX | index | false | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
| VNXALL | index | false | false | 0 |  |  | 0 | 0 | false | cache_file_missing |
