# VN30 Stock Hourly 2015 Gateway Fetch Summary

- Provider path: `src.data.providers.vn_price_gateway.fetch_price_history`.
- Data start: `2015-01-01`.
- Frequency: `1H` only.
- Daily data used: no.
- Resampling used: no.

| ticker | fetched | rows | first | last | provider | source | frequency | stopped_by_runtime_cap | stopped_reason |
|---|---:|---:|---|---|---|---|---|---:|---|
| `ACB` | false | 0 |  |  | `` | `` | `` | true | `max_runtime_seconds=14400` |
| `BID` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `CTG` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `DGC` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `FPT` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `GAS` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `GVR` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `HDB` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `HPG` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `LPB` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `MBB` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `MSN` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `MWG` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `PLX` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `SAB` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `SHB` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `SSB` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `SSI` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `STB` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `TCB` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `TPB` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `VCB` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `VHM` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `VIB` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `VIC` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `VJC` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `VNM` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `VPB` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `VPL` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `VRE` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |

## Resume Commands

- `ACB`: `<repo-approved-venv-python> scripts\research\fetch_vn30_stocks_hourly_gateway_2015.py --ticker ACB --start 2015-01-01 --end auto --chunk-days 5 --resume --max-runtime-seconds 14400`
