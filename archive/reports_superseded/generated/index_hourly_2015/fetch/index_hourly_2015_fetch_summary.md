# Index Hourly 2015 Adaptive Reverse Gateway Fetch Summary

- Provider path: `src.data.providers.vn_price_gateway.fetch_price_history`.
- Direction: reverse from provider-current/latest available timestamp to configured start.
- Chunk strategy: yearly first, then quarterly/monthly/5-day/1-day only when broader chunks fail.
- Frequency: `1H` only.
- Daily data used: no.
- Resampling used: no.

| index_code | rows | first | last | usable_candidate | skipped | chunks_attempted | stopped_by_runtime_cap |
|---|---:|---|---|---:|---:|---:|---:|
| `VNINDEX` | 995 | 2022-05-19 00:00:00 | 2026-05-15 00:00:00 | true | true | 0 | false |
| `HNXINDEX` | 994 | 2022-05-19 00:00:00 | 2026-05-14 00:00:00 | true | true | 0 | false |
| `UPCOMINDEX` | 994 | 2022-05-19 00:00:00 | 2026-05-15 00:00:00 | true | true | 0 | false |
| `VN30` | 995 | 2022-05-19 00:00:00 | 2026-05-15 00:00:00 | true | true | 0 | false |
| `HNX30` | 962 | 2022-07-04 00:00:00 | 2026-05-15 00:00:00 | true | true | 0 | false |
| `VN100` | 1570 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | true | true | 0 | false |
