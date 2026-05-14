# Index Hourly 2015 Gateway Fetch Summary

- Provider path: `src.data.providers.vn_price_gateway.fetch_price_history`.
- Data start: `2015-01-01`.
- Frequency: `1H` only.
- Daily data used: no.
- Resampling used: no.

| index_code | fetched | rows | first | last | provider | source | frequency | stopped_by_runtime_cap | stopped_reason |
|---|---:|---:|---|---|---|---|---|---:|---|
| `VNINDEX` | false | 0 |  |  | `` | `` | `` | true | `max_runtime_seconds=7200` |
| `HNXINDEX` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `UPCOMINDEX` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `VN30` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `HNX30` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |
| `VN100` | false | 0 |  |  | `` | `` | `` | false | `not_started_runtime_cap` |

## Resume Commands

- `VNINDEX`: `<repo-approved-venv-python> scripts\research\fetch_supported_indices_hourly_gateway_2015.py --index-code VNINDEX --start 2015-01-01 --end auto --chunk-days 5 --resume --max-runtime-seconds 7200`
