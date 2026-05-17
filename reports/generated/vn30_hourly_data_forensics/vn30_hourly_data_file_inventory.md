# VN30 Hourly Data File Inventory

## Summary
- **Total CSV files scanned**: 811
- **Active cache files**: 41
- **Raw fetch files**: 167
- **Archive snapshot files**: 109
- **Output files**: 494
- **Earliest timestamp found**: 2022-05-19 00:00:00
- **Latest timestamp found**: 2026-05-15 00:00:00
- **Files with pre-2023 data**: 15
- **Files with 2015-2022 data**: 15

## Key Finding
**NO 2015-2022 hourly stock data exists anywhere in the repository.**
- Earliest stock data: 2023-09-11 (all locations)
- Earliest index data: 2022-05-19 (VNINDEX and related indices)
- The "hourly_2015" naming is a design target, NOT actual data availability.

## Location Breakdown
### Active Cache (data/market_cache/vnstock_data/)
- VN30 stocks: 32 files
- Indices: 7 files
- Date range: 2022-05-19 00:00:00 to 2026-05-15 00:00:00

### Raw Fetch (data/raw/vnstock_fetch/)
- VN30 stocks: 132 files
- Indices: 29 files
- Date range: 2022-05-19 00:00:00 to 2026-05-15 00:00:00

### Archive Snapshots
- Files: 109
- Date range: 2022-05-19 09:00:00 to 2026-05-13 15:00:00

## Files with Pre-2023 Data
- data/market_cache/vnstock_data/indices/hourly_2015/HNX30.csv: 2022-07-04 00:00:00 to 2026-05-15 00:00:00
- data/market_cache/vnstock_data/indices/hourly_2015/HNXINDEX.csv: 2022-05-19 00:00:00 to 2026-05-14 00:00:00
- data/market_cache/vnstock_data/indices/hourly_2015/UPCOMINDEX.csv: 2022-05-19 00:00:00 to 2026-05-15 00:00:00
- data/market_cache/vnstock_data/indices/hourly_2015/VN30.csv: 2022-05-19 00:00:00 to 2026-05-15 00:00:00
- data/market_cache/vnstock_data/indices/hourly_2015/VNINDEX.csv: 2022-05-19 00:00:00 to 2026-05-15 00:00:00
- data/raw/vnstock_fetch/index_hourly_2015/HNX30/HNX30_20220101_20221231.csv: 2022-07-04 00:00:00 to 2022-12-30 00:00:00
- data/raw/vnstock_fetch/index_hourly_2015/HNXINDEX/HNXINDEX_20220101_20221231.csv: 2022-05-19 00:00:00 to 2022-12-30 00:00:00
- data/raw/vnstock_fetch/index_hourly_2015/UPCOMINDEX/UPCOMINDEX_20220101_20221231.csv: 2022-05-19 00:00:00 to 2022-12-30 00:00:00
- data/raw/vnstock_fetch/index_hourly_2015/VN30/VN30_20220101_20221231.csv: 2022-05-19 00:00:00 to 2022-12-30 00:00:00
- data/raw/vnstock_fetch/index_hourly_2015/VNINDEX/VNINDEX_20220101_20221231.csv: 2022-05-19 00:00:00 to 2022-12-30 00:00:00
- archive/generated_data_snapshots/vn30_hourly_pre_benchmark_20260514_062528/data/market_cache/vnstock_data/indices/hourly/HNX30.csv: 2022-07-04 09:00:00 to 2026-05-13 15:00:00
- archive/generated_data_snapshots/vn30_hourly_pre_benchmark_20260514_062528/data/market_cache/vnstock_data/indices/hourly/HNXINDEX.csv: 2022-05-19 09:00:00 to 2026-05-13 15:00:00
- archive/generated_data_snapshots/vn30_hourly_pre_benchmark_20260514_062528/data/market_cache/vnstock_data/indices/hourly/UPCOMINDEX.csv: 2022-05-19 09:00:00 to 2026-05-13 15:00:00
- archive/generated_data_snapshots/vn30_hourly_pre_benchmark_20260514_062528/data/market_cache/vnstock_data/indices/hourly/VN30.csv: 2022-05-19 09:00:00 to 2026-05-13 15:00:00
- archive/generated_data_snapshots/vn30_hourly_pre_benchmark_20260514_062528/data/market_cache/vnstock_data/indices/hourly/VNINDEX.csv: 2022-05-19 09:00:00 to 2026-05-13 15:00:00

## Files with 2015-2022 Data
- data/market_cache/vnstock_data/indices/hourly_2015/HNX30.csv: 2022-07-04 00:00:00 to 2026-05-15 00:00:00
- data/market_cache/vnstock_data/indices/hourly_2015/HNXINDEX.csv: 2022-05-19 00:00:00 to 2026-05-14 00:00:00
- data/market_cache/vnstock_data/indices/hourly_2015/UPCOMINDEX.csv: 2022-05-19 00:00:00 to 2026-05-15 00:00:00
- data/market_cache/vnstock_data/indices/hourly_2015/VN30.csv: 2022-05-19 00:00:00 to 2026-05-15 00:00:00
- data/market_cache/vnstock_data/indices/hourly_2015/VNINDEX.csv: 2022-05-19 00:00:00 to 2026-05-15 00:00:00
- data/raw/vnstock_fetch/index_hourly_2015/HNX30/HNX30_20220101_20221231.csv: 2022-07-04 00:00:00 to 2022-12-30 00:00:00
- data/raw/vnstock_fetch/index_hourly_2015/HNXINDEX/HNXINDEX_20220101_20221231.csv: 2022-05-19 00:00:00 to 2022-12-30 00:00:00
- data/raw/vnstock_fetch/index_hourly_2015/UPCOMINDEX/UPCOMINDEX_20220101_20221231.csv: 2022-05-19 00:00:00 to 2022-12-30 00:00:00
- data/raw/vnstock_fetch/index_hourly_2015/VN30/VN30_20220101_20221231.csv: 2022-05-19 00:00:00 to 2022-12-30 00:00:00
- data/raw/vnstock_fetch/index_hourly_2015/VNINDEX/VNINDEX_20220101_20221231.csv: 2022-05-19 00:00:00 to 2022-12-30 00:00:00
- archive/generated_data_snapshots/vn30_hourly_pre_benchmark_20260514_062528/data/market_cache/vnstock_data/indices/hourly/HNX30.csv: 2022-07-04 09:00:00 to 2026-05-13 15:00:00
- archive/generated_data_snapshots/vn30_hourly_pre_benchmark_20260514_062528/data/market_cache/vnstock_data/indices/hourly/HNXINDEX.csv: 2022-05-19 09:00:00 to 2026-05-13 15:00:00
- archive/generated_data_snapshots/vn30_hourly_pre_benchmark_20260514_062528/data/market_cache/vnstock_data/indices/hourly/UPCOMINDEX.csv: 2022-05-19 09:00:00 to 2026-05-13 15:00:00
- archive/generated_data_snapshots/vn30_hourly_pre_benchmark_20260514_062528/data/market_cache/vnstock_data/indices/hourly/VN30.csv: 2022-05-19 09:00:00 to 2026-05-13 15:00:00
- archive/generated_data_snapshots/vn30_hourly_pre_benchmark_20260514_062528/data/market_cache/vnstock_data/indices/hourly/VNINDEX.csv: 2022-05-19 09:00:00 to 2026-05-13 15:00:00
