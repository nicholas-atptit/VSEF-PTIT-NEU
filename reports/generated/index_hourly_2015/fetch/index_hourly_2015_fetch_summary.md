# Index Hourly 2015 Reverse Gateway Fetch Summary

- Provider path: `src.data.providers.vn_price_gateway.fetch_price_history`.
- Direction: reverse, provider-current/latest available timestamp back to configured start.
- Frequency: `1H` only.
- Daily data used: no.
- Resampling used: no.

| index_code | fetched | rows | first | last | stopped_by_runtime_cap | stopped_reason |
|---|---:|---:|---|---|---:|---|
| `VNINDEX` | true | 994 | 2022-05-19 00:00:00 | 2026-05-14 00:00:00 | true | `max_runtime_seconds=7200` |
| `HNXINDEX` | false | 0 |  |  | false | `not_started_runtime_cap` |
| `UPCOMINDEX` | false | 0 |  |  | false | `not_started_runtime_cap` |
| `VN30` | false | 0 |  |  | false | `not_started_runtime_cap` |
| `HNX30` | false | 0 |  |  | false | `not_started_runtime_cap` |
| `VN100` | false | 0 |  |  | false | `not_started_runtime_cap` |

## Resume Commands

- `VNINDEX`: `<repo-approved-venv-python> scripts\research\fetch_supported_indices_hourly_gateway_2015.py --index-code VNINDEX --direction reverse --start 2015-01-01 --end auto --year-first --monthly-fallback --daily-fallback --resume --max-runtime-seconds 7200`
