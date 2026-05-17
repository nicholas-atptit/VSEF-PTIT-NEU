# BCM and VIB Daily Data Diagnosis

- Created at UTC: 2026-05-17
- Track: VN30 Daily 2015
- Universe: VN30 January 2025 review (30 tickers)

## Summary

Both BCM and VIB were missing from the daily cache after the initial fetch run. The provider (vnstock_data/KBS) returned raw data for both tickers, but the canonical gateway's OHLCV validation rejected them due to a small number of rows with inconsistent OHLCV geometry.

## BCM

| Field | Value |
|---|---|
| Cache file exists (before recovery) | No |
| Cache file exists (after recovery) | Yes |
| Raw row count from provider | 2033 |
| Recovered row count (after filtering) | 2026 |
| First date | 2018-02-21 |
| Last date | 2026-05-15 |
| Columns | datetime, ticker, open, high, low, close, volume, provider, source, frequency |
| Provider/Source | vnstock_data / KBS |
| Training rows | 1440 |
| Validation rows | 249 |
| Final evaluation rows | 336 |
| Reason unusable (before recovery) | Gateway normalization rejected all data due to 7 rows with OHLCV consistency violations |
| Issue type | Provider data quality issue |

### Problematic rows (7 total)

- **high < max(open, close)**: 5 rows
  - 2019-03-11: open=22.84, high=19.42, low=19.42, close=22.84, volume=3 (clear data error)
  - 2019-09-17: open=27.91, high=28.46, low=27.82, close=28.56
  - 2019-10-14: open=27.73, high=27.73, low=27.64, close=27.82
  - 2019-12-16: open=25.79, high=25.98, low=25.52, close=26.62
  - 2019-12-19: open=26.34, high=26.90, low=26.25, close=27.17
- **low > min(open, close)**: 2 rows
  - 2020-01-02: open=27.36, high=27.36, low=27.36, close=27.27
  - 2020-02-18: open=24.04, high=24.04, low=24.04, close=23.67

### Resolution

Fetch script updated with fallback: when gateway normalization fails due to OHLCV geometry violations, raw data is fetched directly and problematic rows are filtered out. 2026 of 2033 rows retained (99.7%).

## VIB

| Field | Value |
|---|---|
| Cache file exists (before recovery) | No |
| Cache file exists (after recovery) | Yes |
| Raw row count from provider | 2324 |
| Recovered row count (after filtering) | 2322 |
| First date | 2017-01-09 |
| Last date | 2026-05-15 |
| Columns | datetime, ticker, open, high, low, close, volume, provider, source, frequency |
| Provider/Source | vnstock_data / KBS |
| Training rows | 1736 |
| Validation rows | 249 |
| Final evaluation rows | 336 |
| Reason unusable (before recovery) | Gateway normalization rejected all data due to 2 rows with OHLCV consistency violations |
| Issue type | Provider data quality issue |

### Problematic rows (2 total)

- **low > min(open, close)**: 2 rows
  - 2019-12-31: open=3.90, high=3.90, low=3.85, close=3.83
  - 2020-02-18: open=4.05, high=4.07, low=4.05, close=3.96

### Resolution

Same fallback as BCM. 2322 of 2324 rows retained (99.9%).

## Root Cause

The canonical gateway (`vn_price_gateway.py`) raises `ValueError` when ANY row fails OHLCV consistency checks, rather than filtering out bad rows. This is correct behavior for strict validation but caused complete data loss for tickers with a tiny fraction of problematic rows (<0.3%).

## Recovery Method

Updated `fetch_vn30_daily_gateway_2015.py` with a fallback path:
1. Try canonical gateway first (strict validation)
2. If gateway fails due to OHLCV geometry violations, fetch raw data directly from provider
3. Filter out rows that fail consistency checks
4. Save remaining rows if >= 100 rows remain

## Post-Recovery Status

- Daily universe: 30/30 tickers usable
- Benchmark rerun: yes
- No hourly resampling used
- No trading/profitability claims made
