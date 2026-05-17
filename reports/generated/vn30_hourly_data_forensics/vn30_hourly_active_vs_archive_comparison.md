# VN30 Hourly Active vs Archive Data Comparison

## Summary
- **Active cache**: `data/market_cache/vnstock_data/`
- **Raw fetch**: `data/raw/vnstock_fetch/`
- **Archive snapshots**: `archive/generated_data_snapshots/`
- **Outputs**: `outputs/`

## VN30 Stock Tickers

### Active Cache (hourly_2015)
- **Files**: 32 CSV files (30 VN30 + DGC + VPL)
- **Date range**: 2023-09-11 to 2026-05-15
- **Earliest**: 2023-09-11 (ALL tickers)
- **Latest**: 2026-05-15

### Raw Fetch (vn30_hourly_2015)
- **Files**: 128 CSV files (32 tickers x 4 year chunks)
- **Chunks**: 2023, 2024, 2025, 2026
- **Date range**: 2023-09-11 to 2026-05-16
- **Earliest**: 2023-09-11

### Archive Snapshots
- **Pre-benchmark snapshot** (2026-05-14): Contains `hourly_listing_aware` data
- **Date range**: 2023-09-11 to 2026-05-13
- **No hourly_2015 folder in archive**

### Comparison
| Location | Earliest | Latest | Has Pre-2023 |
|----------|----------|--------|--------------|
| Active cache (stocks) | 2023-09-11 | 2026-05-15 | NO |
| Raw fetch (stocks) | 2023-09-11 | 2026-05-16 | NO |
| Archive (stocks) | 2023-09-11 | 2026-05-13 | NO |

**Conclusion**: NO 2015-2022 stock data exists anywhere. All locations agree on 2023-09-11 as the earliest date.

## Index Data

### Active Cache (indices/hourly_2015)
- **Files**: 6 CSV files (VNINDEX, VN30, HNXINDEX, HNX30, UPCOMINDEX, VN100)
- **Date range**: 2022-05-19 to 2026-05-15
- **Earliest**: 2022-05-19

### Raw Fetch (index_hourly_2015)
- **Files**: 24 CSV files (6 indices x 4 year chunks: 2022-2026)
- **Date range**: 2022-05-19 to 2026-05-15
- **Earliest**: 2022-05-19

### Archive Snapshots
- **Pre-benchmark snapshot**: Contains `indices/hourly` folder
- **Date range**: 2022-05-19 to 2026-05-13
- **Earliest**: 2022-05-19

### Comparison
| Location | Earliest | Latest | Has Pre-2023 |
|----------|----------|--------|--------------|
| Active cache (indices) | 2022-05-19 | 2026-05-15 | YES (2022) |
| Raw fetch (indices) | 2022-05-19 | 2026-05-15 | YES (2022) |
| Archive (indices) | 2022-05-19 | 2026-05-13 | YES (2022) |

**Conclusion**: Index data exists from 2022-05-19, but NOT from 2015-2021.

## Key Findings
1. **NO 2015-2021 data exists anywhere** for stocks or indices.
2. **Index data starts 2022-05-19** (not 2015).
3. **Stock data starts 2023-09-11** (not 2015).
4. **Active cache is NOT missing data** that exists elsewhere - all locations have the same date ranges.
5. **The "hourly_2015" naming is a design target**, not actual data availability.
