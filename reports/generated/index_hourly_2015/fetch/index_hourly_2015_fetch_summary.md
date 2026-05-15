# Index Hourly 2015 Reverse Gateway Fetch Summary

- Provider path: `src.data.providers.vn_price_gateway.fetch_price_history`.
- Direction: reverse, provider-current/latest available timestamp back to configured start.
- Frequency: `1H` only.
- Daily data used: no.
- Resampling used: no.

| index_code | fetched | rows | first | last | stopped_by_runtime_cap | stopped_reason |
|---|---:|---:|---|---|---:|---|
| `VN100` | true | 1570 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | true | `max_runtime_seconds=7200` |

## Resume Commands

- `VN100`: `<repo-approved-venv-python> scripts\research\fetch_supported_indices_hourly_gateway_2015.py --index-code VN100 --direction reverse --start 2015-01-01 --end auto --year-first --monthly-fallback --daily-fallback --resume --max-runtime-seconds 7200`
