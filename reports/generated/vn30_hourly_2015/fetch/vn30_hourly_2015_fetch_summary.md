# VN30 Stock Hourly 2015 Listing-Aware Reverse Gateway Fetch Summary

- Provider path: `src.data.providers.vn_price_gateway.fetch_price_history`.
- Effective start rule: `max(2015-01-01, first_trading_date)`.
- Direction: reverse, provider-current/latest available timestamp back to effective start.
- Frequency: `1H` only.
- Daily data used: no.
- Resampling used: no.

| ticker | effective_start | fetched | rows | first | last | stopped_by_runtime_cap | stopped_reason |
|---|---|---:|---:|---|---|---:|---|
| `ACB` | 2015-01-01 | true | 1496 | 2023-09-11 10:00:00 | 2026-05-14 00:00:00 | true | `max_runtime_seconds=14400` |
| `BID` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `CTG` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `DGC` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `FPT` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `GAS` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `GVR` | 2018-03-21 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `HDB` | 2018-01-05 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `HPG` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `LPB` | 2017-10-05 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `MBB` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `MSN` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `MWG` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `PLX` | 2017-04-21 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `SAB` | 2016-12-06 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `SHB` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `SSB` | 2021-03-24 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `SSI` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `STB` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `TCB` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `TPB` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `VCB` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `VHM` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `VIB` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `VIC` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `VJC` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `VNM` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `VPB` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `VPL` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |
| `VRE` | 2015-01-01 | false | 0 |  |  | false | `not_started_runtime_cap` |

## Resume Commands

- `ACB`: `<repo-approved-venv-python> scripts\research\fetch_vn30_stocks_hourly_gateway_2015.py --ticker ACB --direction reverse --start 2015-01-01 --end auto --year-first --monthly-fallback --daily-fallback --resume --max-runtime-seconds 14400`
