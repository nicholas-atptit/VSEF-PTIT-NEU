# VN30 Stock Hourly 2015 Adaptive Reverse Gateway Fetch Summary

- Provider path: `src.data.providers.vn_price_gateway.fetch_price_history`.
- Direction: reverse from provider-current/latest available timestamp to effective start.
- Effective start rule: `max(2015-01-01, first_trading_date)`.
- Chunk strategy: yearly first, then quarterly/monthly/5-day/1-day only when broader chunks fail.
- Frequency: `1H` only.
- Daily data used: no.
- Resampling used: no.

| ticker | effective_start | rows | first | last | train_rows | eval_rows | usable_candidate | skipped | chunks_attempted | stopped_by_runtime_cap |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| `BVH` | 2015-01-01 | 1456 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1321 | 135 | true | true | 0 | false |
