# Phase F1.5 Narrow Forecast Rehab Scope

| dimension | in-scope | comparator-only | baseline-only | out/de-emphasized | reason |
| --- | --- | --- | --- | --- | --- |
| `ticker_group` | `small_banks` |  |  | `mixed_large_cap`, `vn100_subset` | F1 showed the usable edge cluster is concentrated in small banks and degrades sharply when generalized too early. |
| `horizons` | `5`, `10` |  |  | `1` | F1 showed horizon 10 was the least-bad slice and horizon 5 remained materially better than horizon 1. |
| `feature_families` | `tech_core_v1`, `compact_v1`, `compact_plus_longlag_v1`, `compact_plus_longlag_v2` |  |  | `technical_plus_market`, `technical_plus_market_plus_sector`, `current_full`, `short_lag` | F1 favored technical-heavy and compact families over broader context expansion. |
| `model_families` | `lightgbm`, `random_forest`, `xgboost` | `ets`, `sarimax` | `naive`, `moving_average` | `linear`, `ridge`, `lasso` | Tree models led the best-slice cluster, ETS stayed useful as a benchmark, SARIMAX remained usable, and the linear family no longer justified primary scope. |
| `target_framing` | `forward_return`, `direction_binary` |  |  | `forward_log_return`, `future_realized_volatility` | F1 left forward return versus direction unresolved, while the other targets did not justify widening the rehab surface. |

## Narrowing Notes

- `tech_core_v1` is a compressed version of the old `technical_core` family. It keeps medium-horizon trend, volatility, liquidity, and foreign-flow state while removing redundant raw counts.
- `compact_v1` is the cleaned `reduced_compact` baseline. It drops the raw breadth-volume columns and keeps the more interpretable ratio and long-memory trend proxies.
- `compact_plus_longlag_v1` adds 60-day trend and market-memory overlays.
- `compact_plus_longlag_v2` adds lagged return and momentum persistence overlays.
- The Phase 2.6 execution policy remains fixed. Phase F1.5 tests whether the forecast layer improves when the search space is narrowed, not whether policy complexity should expand.
