# VN30 Hourly Listing-Aware vnstock Fetch Summary

## Scope

- Universe: frozen VN30 30 tickers.
- Frequency: hourly only.
- Provider path: vnstock_data if importable, otherwise legacy vnstock.
- Raw chunk directory: `data/raw/vnstock_fetch/vn30_hourly_listing_aware`.
- Provider attempt log: `reports/generated/vn30_hourly_listing_aware/fetch/vn30_listing_aware_provider_attempt_log.csv` for completed provider-call runs; persisted raw chunks and normalized cache are summarized separately after interrupted runs.
- Per-ticker start rule: max(first trading/listing date, first provider-available hourly timestamp).
- Missing pre-listing hours are not required, filled, or synthesized.

## Package Detection

| package | installed | version | origin |
| --- | --- | --- | --- |
| vnstock_data | false |  |  |
| vnstock | true | 3.5.0 | __init__.py |

## Gate Snapshot

- Usable VN30 stocks from fetch summary: 0/30.
- actual_eval_end candidate: not available.
- VNINDEX fetched/usable: fetched=False, usable=False.
- VN30INDEX support: False.
- VNXALL support: False.

## Per-Symbol Summary

| ticker | asset_type | listing_date_used | requested_start | provider | total_rows | first_datetime | last_datetime | training_rows_before_cutoff | evaluation_rows_after_2025_01_01 | usable | missing_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACB | stock | 2006-11-21 00:00:00 | 2006-11-21 00:00:00 | vnstock/KBS;vnstock/VCI | 237 | 2023-09-11 10:00:00 | 2026-05-13 14:00:00 | 91 | 146 | false | training_rows_below_1000 |
| BID | stock |  | 2005-01-01 00:00:00 | vnstock/KBS;vnstock/VCI | 247 | 2023-09-11 10:00:00 | 2026-05-13 14:00:00 | 91 | 156 | false | training_rows_below_1000 |
| CTG | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| DGC | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| FPT | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| GAS | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| GVR | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| HDB | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| HPG | stock | 2007-11-15 00:00:00 | 2007-11-15 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| LPB | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| MBB | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| MSN | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| MWG | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| PLX | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| SAB | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| SHB | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| SSB | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| SSI | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| STB | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| TCB | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| TPB | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| VCB | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| VHM | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| VIB | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| VIC | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| VJC | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| VNM | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| VPB | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| VPL | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| VRE | stock |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed; training_rows_below_1000; evaluation_rows_below_100 |
| VNINDEX | index |  | 2005-01-01 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed |
| VN30INDEX | index |  | 2012-02-06 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed |
| VNXALL | index |  | 2016-10-24 00:00:00 |  | 0 |  |  | 0 | 0 | false | no_provider_hourly_rows_observed |

## Failure Preview

| ticker | asset_type | chunk_start | chunk_end | chunk_level | failure_reason |
| --- | --- | --- | --- | --- | --- |
| CTG | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| DGC | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| FPT | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| GAS | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| GVR | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| HDB | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| HPG | stock | 2007-11-15 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| LPB | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| MBB | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| MSN | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| MWG | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| PLX | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| SAB | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| SHB | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| SSB | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| SSI | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| STB | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| TCB | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| TPB | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| VCB | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| VHM | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| VIB | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| VIC | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| VJC | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| VNM | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| VPB | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| VPL | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| VRE | stock | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| VNINDEX | index | 2005-01-01 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| VN30INDEX | index | 2012-02-06 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
| VNXALL | index | 2016-10-24 00:00:00 | 2026-05-14 23:59:59 | summarize_existing_only | no normalized listing-aware cache file exists |
