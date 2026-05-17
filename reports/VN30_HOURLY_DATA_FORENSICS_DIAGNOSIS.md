# VN30 Hourly Data Forensics Diagnosis

## 1. Is the data actually deleted, or just not in the active validated cache?

**The data is NOT deleted.** It exists in multiple locations:
- Active cache: `data/market_cache/vnstock_data/` (41 CSV files)
- Raw fetch: `data/raw/vnstock_fetch/` (167 CSV files)
- Archive snapshots: `archive/generated_data_snapshots/` (145 CSV files)
- Outputs: `outputs/` (458 CSV files)

Total: **811 CSV files** found across all locations.

## 2. Does any 2015–2022 hourly data exist anywhere in the repo/worktree/archive?

**NO for stocks.** No VN30 stock hourly data exists from 2015-2022 anywhere.
- Earliest stock data: **2023-09-11** (all locations agree)

**PARTIAL for indices.** Index hourly data exists from 2022-05-19, but NOT from 2015-2021.
- Earliest index data: **2022-05-19** (VNINDEX, VN30, HNXINDEX, HNX30, UPCOMINDEX, VN100)
- No index data exists from 2015-2021

**15 files have 2022 data** (all index files). **Zero files have 2015-2021 data.**

## 3. Which directory currently has the earliest hourly data?

| Directory | Earliest Timestamp | Type |
|-----------|-------------------|------|
| `data/market_cache/vnstock_data/indices/hourly_2015/` | 2022-05-19 | Index |
| `data/raw/vnstock_fetch/index_hourly_2015/` | 2022-05-19 | Index |
| `archive/.../indices/hourly/` | 2022-05-19 | Index |
| `data/market_cache/vnstock_data/vn30/hourly_2015/` | 2023-09-11 | Stock |
| `data/raw/vnstock_fetch/vn30_hourly_2015/` | 2023-09-11 | Stock |

**Earliest overall**: 2022-05-19 (index data only)

## 4. Which tickers/index codes have the longest coverage?

- **Indices** (VNINDEX, VN30, HNXINDEX, HNX30, UPCOMINDEX, VN100): ~4 years (2022-05-19 to 2026-05-15)
- **VN30 stocks** (32 tickers): ~2.7 years (2023-09-11 to 2026-05-15)

## 5. Did cleanup/reset scripts remove active cache paths?

**No evidence of deletion.** The active cache files exist and contain the same data as raw fetch and archive snapshots. The data was never fetched for 2015-2022 because the vendor (KBS/VCI via vnstock) does not provide hourly data before 2022-05-19 for indices and 2023-09-11 for stocks.

## 6. Were archive snapshots created before reset?

**Yes.** Two archive snapshots exist:
- `vn30_hourly_pre_benchmark_20260514_062528/` - Created before benchmark run
- `vn30_jan2025_readiness_audit_refactor_20260516_164954/` - Created during readiness audit

Both contain the same data as the active cache (2022-05-19 earliest for indices, 2023-09-11 for stocks).

## 7. Can data be restored from archive/raw/cache?

**For existing data (2022-2026): YES.** The data exists in multiple locations and can be copied between them.

**For missing data (2015-2021): NO.** This data was never fetched from the vendor. It would need to be re-fetched, but the vendor may not have hourly data that far back.

## 8. Which files should be copied back if restoration is needed?

If the active cache needs to be restored from raw fetch:
```
# Restore index data
Copy: data/raw/vnstock_fetch/index_hourly_2015/*.csv
To:   data/market_cache/vnstock_data/indices/hourly_2015/

# Restore stock data (merge year chunks)
Copy: data/raw/vnstock_fetch/vn30_hourly_2015/{TICKER}/*.csv
Merge and save to: data/market_cache/vnstock_data/vn30/hourly_2015/{TICKER}.csv
```

## 9. Which data is not recoverable from local files and would need refetch/vendor data?

- **2015-2021 hourly stock data**: Does not exist anywhere. Would need vendor refetch.
- **2015-2022 hourly index data** (before 2022-05-19): Does not exist anywhere. Would need vendor refetch.
- **Any data before vendor's earliest available timestamp**: Cannot be recovered.

## 10. What should NOT be done next?

1. **DO NOT** assume "hourly_2015" means data from 2015 exists. It's a design target, not reality.
2. **DO NOT** attempt to restore 2015-2022 data from Git - it was never tracked.
3. **DO NOT** delete raw fetch chunks - they are the source for active cache.
4. **DO NOT** rely on rolling validation windows before 2023 for stocks or before 2022 for indices.
5. **DO NOT** refetch without checking vendor availability first - the vendor may not have hourly data before 2022/2023.

## Restoration Plan (If Needed)

**Do NOT execute automatically.** This is a plan only.

### Step 1: Verify raw fetch integrity
```bash
# Check raw fetch files exist
ls data/raw/vnstock_fetch/vn30_hourly_2015/*/
ls data/raw/vnstock_fetch/index_hourly_2015/*/
```

### Step 2: Rebuild active cache from raw fetch
- Merge year chunks per ticker into single CSV files
- Save to `data/market_cache/vnstock_data/vn30/hourly_2015/`
- Save indices to `data/market_cache/vnstock_data/indices/hourly_2015/`

### Step 3: Verify rebuilt cache
- Check row counts match raw fetch totals
- Check first/last timestamps match
- Run canonical evaluator to verify

### Step 4: Accept limitation
- Acknowledge that 2015-2022 stock data does not exist
- Adjust validation windows to use available data (2023-2024 for stocks, 2022-2024 for indices)

## Conclusion

**The data was NOT deleted.** It exists in multiple locations. The "missing" 2015-2022 data was never fetched because the vendor does not provide hourly data that far back. The "hourly_2015" naming convention is a design target, not a reflection of actual data availability. The earliest available hourly data is 2022-05-19 for indices and 2023-09-11 for stocks.
