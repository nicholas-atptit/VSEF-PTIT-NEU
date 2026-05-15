# VN30 Stock Hourly 2015 Listing-Aware Reverse Gateway Fetch Summary

- Provider path: `src.data.providers.vn_price_gateway.fetch_price_history`.
- Effective start rule: `max(2015-01-01, first_trading_date)`.
- Direction: reverse, provider-current/latest available timestamp back to effective start.
- Frequency: `1H` only.
- Daily data used: no.
- Resampling used: no.

| ticker | effective_start | fetched | rows | first | last | stopped_by_runtime_cap | stopped_reason |
|---|---|---:|---:|---|---|---:|---|
| `ACB` | 2015-01-01 | true | 1497 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | true | `max_runtime_seconds=14400` |

## Resume Commands

- `ACB`: `<repo-approved-venv-python> scripts\research\fetch_vn30_stocks_hourly_gateway_2015.py --ticker ACB --direction reverse --start 2015-01-01 --end auto --year-first --monthly-fallback --daily-fallback --resume --max-runtime-seconds 14400`
