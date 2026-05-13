# VN30 Hourly vnstock Fetched Data Validation

## Gate Decision

- Full fetched stock+VNINDEX gate passed: False.
- Benchmark-usable VN30 stocks: 0/30.
- VNINDEX benchmark-usable: False.
- VN30INDEX support/usable if fetched: supported=false, usable=false.
- VNXALL support/usable if fetched: supported=false, usable=false.

## Required Coverage

- VN30 stocks: 2005-01-01 00:00:00 to 2026-05-31 23:59:59; train cutoff 2024-12-31 23:59:59.
- VNINDEX: 2005-01-01 00:00:00 to 2026-05-31 23:59:59.
- VN30INDEX optional context if fetched: 2012-02-06 00:00:00 to 2026-05-31 23:59:59.
- VNXALL optional context if fetched: 2016-10-24 00:00:00 to 2026-05-31 23:59:59.
- Common evaluation/comparison window: 2025-01-01 00:00:00 to 2026-05-31 23:59:59.
- Optional VN30INDEX/VNXALL absence does not fail the stock+VNINDEX gate.

## Required Failures

| symbol | asset_type | row_count | first_datetime | last_datetime | training_rows | evaluation_rows | benchmark_usable | failure_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACB | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| BID | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| CTG | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| DGC | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| FPT | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| GAS | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| GVR | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| HDB | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| HPG | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| LPB | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| MBB | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| MSN | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| MWG | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| PLX | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| SAB | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| SHB | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| SSB | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| SSI | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| STB | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| TCB | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| TPB | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| VCB | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| VHM | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| VIB | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| VIC | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| VJC | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| VNM | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| VPB | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| VPL | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| VRE | stock | 0 |  |  | 0 | 0 | false | required_cache_file_missing |
| VNINDEX | index | 0 |  |  | 0 | 0 | false | required_cache_file_missing |

## Per-Symbol Validation

| symbol | asset_type | gate_required | row_count | first_datetime | last_datetime | benchmark_usable | failure_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ACB | stock | true | 0 |  |  | false | required_cache_file_missing |
| BID | stock | true | 0 |  |  | false | required_cache_file_missing |
| CTG | stock | true | 0 |  |  | false | required_cache_file_missing |
| DGC | stock | true | 0 |  |  | false | required_cache_file_missing |
| FPT | stock | true | 0 |  |  | false | required_cache_file_missing |
| GAS | stock | true | 0 |  |  | false | required_cache_file_missing |
| GVR | stock | true | 0 |  |  | false | required_cache_file_missing |
| HDB | stock | true | 0 |  |  | false | required_cache_file_missing |
| HPG | stock | true | 0 |  |  | false | required_cache_file_missing |
| LPB | stock | true | 0 |  |  | false | required_cache_file_missing |
| MBB | stock | true | 0 |  |  | false | required_cache_file_missing |
| MSN | stock | true | 0 |  |  | false | required_cache_file_missing |
| MWG | stock | true | 0 |  |  | false | required_cache_file_missing |
| PLX | stock | true | 0 |  |  | false | required_cache_file_missing |
| SAB | stock | true | 0 |  |  | false | required_cache_file_missing |
| SHB | stock | true | 0 |  |  | false | required_cache_file_missing |
| SSB | stock | true | 0 |  |  | false | required_cache_file_missing |
| SSI | stock | true | 0 |  |  | false | required_cache_file_missing |
| STB | stock | true | 0 |  |  | false | required_cache_file_missing |
| TCB | stock | true | 0 |  |  | false | required_cache_file_missing |
| TPB | stock | true | 0 |  |  | false | required_cache_file_missing |
| VCB | stock | true | 0 |  |  | false | required_cache_file_missing |
| VHM | stock | true | 0 |  |  | false | required_cache_file_missing |
| VIB | stock | true | 0 |  |  | false | required_cache_file_missing |
| VIC | stock | true | 0 |  |  | false | required_cache_file_missing |
| VJC | stock | true | 0 |  |  | false | required_cache_file_missing |
| VNM | stock | true | 0 |  |  | false | required_cache_file_missing |
| VPB | stock | true | 0 |  |  | false | required_cache_file_missing |
| VPL | stock | true | 0 |  |  | false | required_cache_file_missing |
| VRE | stock | true | 0 |  |  | false | required_cache_file_missing |
| VNINDEX | index | true | 0 |  |  | false | required_cache_file_missing |
| VN30INDEX | index | false | 0 |  |  | false | optional_index_not_fetched_or_unsupported |
| VNXALL | index | false | 0 |  |  | false | optional_index_not_fetched_or_unsupported |
