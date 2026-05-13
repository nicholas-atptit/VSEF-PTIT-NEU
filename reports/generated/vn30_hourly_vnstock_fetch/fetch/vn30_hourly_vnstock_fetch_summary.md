# VN30 Hourly vnstock Fetch Summary

## Scope

- Frozen VN30 stocks: 30.
- Required market index: VNINDEX.
- Optional exact-code index probes: VN30INDEX, VNXALL.
- Raw chunk directory: `data/raw/vnstock_fetch/vn30_hourly_2005_2026`.
- Frequency: hourly only.
- Daily data and daily-to-hourly resampling are not used.
- Missing bars are not forward-filled or synthesized.

## Gate Snapshot

- Benchmark-candidate VN30 stocks: 0/30.
- VNINDEX benchmark-candidate: False.
- Total chunk failures: 33.

## Per-Symbol Summary

| symbol | asset_type | provider_used | chunks_attempted | chunks_succeeded | chunks_failed | total_rows | first_datetime | last_datetime | benchmark_candidate | failure_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACB | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| BID | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| CTG | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| DGC | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| FPT | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| GAS | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| GVR | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| HDB | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| HPG | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| LPB | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| MBB | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| MSN | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| MWG | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| PLX | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| SAB | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| SHB | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| SSB | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| SSI | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| STB | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| TCB | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| TPB | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| VCB | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| VHM | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| VIB | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| VIC | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| VJC | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| VNM | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| VPB | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| VPL | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| VRE | stock |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| VNINDEX | index |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| VN30INDEX | index |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |
| VNXALL | index |  | 1 | 0 | 1 | 0 |  |  | false | required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence |

## Failure Preview

| symbol | asset_type | chunk_start | chunk_end | chunk_level | failure_reason |
| --- | --- | --- | --- | --- | --- |
| ACB | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| BID | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| CTG | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| DGC | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| FPT | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| GAS | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| GVR | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| HDB | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| HPG | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| LPB | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| MBB | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| MSN | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| MWG | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| PLX | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| SAB | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| SHB | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| SSB | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| SSI | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| STB | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| TCB | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| TPB | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| VCB | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| VHM | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| VIB | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| VIC | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| VJC | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| VNM | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| VPB | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| VPL | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| VRE | stock | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| VNINDEX | index | 2005-01-01 | 2005-01-07 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| VN30INDEX | index | 2012-02-06 | 2012-02-12 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
| VNXALL | index | 2016-10-24 | 2016-10-30 | required_start_probe | required_start_probe_failed: Supported sources: KBS, VCI, MSN, FMP. Got: MAS |
