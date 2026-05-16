# VN30 Hourly 2015 Requirement Audit

## Verdict
- Full requirement satisfied: false.
- Indices ready: true (6/6).
- Stock universe ready: false (29/30 usable).
- Full 30 VN30 ready: false.
- Benchmark readiness: false.
- Blocking ticker(s): VPL.

## Universe Checks
- Frozen VN30 ticker count: 30.
- Validation ticker count: 30.
- Active universe changed: false.
- Tickers in validation but not frozen universe: none.
- Frozen tickers missing from validation: none.
- BSR in active validation: false.
- BSR active cache file exists: false.
- Extra symbols present: false.
- Unsupported index codes present: none.

## Provider And Frequency Policy
- Gateway policy: pass.
- Provider column exists in all active cache files: true.
- Source column exists in all active cache files: true.
- Frequency column exists in all active cache files: true.
- Frequency values observed: 1H.
- Daily data detected: false.
- Resampling detected: false.
- Raw provider bypass detected: false.

## VPL Blocking Diagnosis
- Cache exists: true.
- Total rows: 176.
- First timestamp: 2025-08-27 00:00:00.
- Last timestamp: 2026-05-15 00:00:00.
- Rows before 2024-12-31: 0.
- Rows from 2025-01-01 onward: 176.
- Effective start: 2015-01-01.
- First trading date: missing.
- Fetch stopped early: false.
- Fetch chunks attempted/with rows/empty/failed: 12/2/10/0.
- Validation failure: training_rows_below_1000.
- Diagnosis: provider coverage/listing window does not provide enough pre-2025 training rows in current cache; missing listing-date metadata; validation threshold is doing its intended blocking role.
- Audit classification: provider coverage/listing-window limitation plus missing listing-date metadata; not a cache/write issue, not a stopped-early issue, and the 1000-row threshold is the active requirement rather than an arbitrary relaxation target.

## Stock Audit
| ticker | frozen | first_trading_date | effective_start | cache | rows | first | last | freq | train_rows | eval_rows | usable | failure_reason |
|---|---:|---|---|---:|---:|---|---|---|---:|---:|---:|---|
| `ACB` | yes | 2006-11-21 | 2015-01-01 | yes | 1497 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 176 | yes |  |
| `BID` | yes | 2014-01-24 | 2015-01-01 | yes | 1455 | 2023-09-11 10:00:00 | 2026-05-14 00:00:00 | 1H | 1321 | 134 | yes |  |
| `CTG` | yes | 2009-07-16 | 2015-01-01 | yes | 1440 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 119 | yes |  |
| `DGC` | yes |  | 2015-01-01 | yes | 1435 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 114 | yes |  |
| `FPT` | yes | 2006-12-13 | 2015-01-01 | yes | 1449 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 128 | yes |  |
| `GAS` | yes | 2012-05-21 | 2015-01-01 | yes | 1456 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 135 | yes |  |
| `GVR` | yes | 2018-03-21 | 2018-03-21 | yes | 1456 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 135 | yes |  |
| `HDB` | yes | 2018-01-05 | 2018-01-05 | yes | 1439 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 118 | yes |  |
| `HPG` | yes | 2007-11-15 | 2015-01-01 | yes | 1592 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 271 | yes |  |
| `LPB` | yes | 2017-10-05 | 2017-10-05 | yes | 1590 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 269 | yes |  |
| `MBB` | yes | 2011-11-01 | 2015-01-01 | yes | 1497 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 176 | yes |  |
| `MSN` | yes | 2009-11-05 | 2015-01-01 | yes | 1497 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 176 | yes |  |
| `MWG` | yes | 2014-07-14 | 2015-01-01 | yes | 1497 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 176 | yes |  |
| `PLX` | yes | 2017-04-21 | 2017-04-21 | yes | 1497 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 176 | yes |  |
| `SAB` | yes | 2016-12-06 | 2016-12-06 | yes | 1421 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 100 | yes |  |
| `SHB` | yes | 2009-04-20 | 2015-01-01 | yes | 1620 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 299 | yes |  |
| `SSB` | yes | 2021-03-24 | 2021-03-24 | yes | 1497 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 176 | yes |  |
| `SSI` | yes | 2006-12-15 | 2015-01-01 | yes | 1446 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 125 | yes |  |
| `STB` | yes | 2006-07-12 | 2015-01-01 | yes | 1497 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 176 | yes |  |
| `TCB` | yes |  | 2015-01-01 | yes | 1456 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 135 | yes |  |
| `TPB` | yes |  | 2015-01-01 | yes | 1456 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 135 | yes |  |
| `VCB` | yes |  | 2015-01-01 | yes | 1456 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 135 | yes |  |
| `VHM` | yes |  | 2015-01-01 | yes | 1497 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 176 | yes |  |
| `VIB` | yes |  | 2015-01-01 | yes | 1602 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 281 | yes |  |
| `VIC` | yes |  | 2015-01-01 | yes | 1448 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 127 | yes |  |
| `VJC` | yes |  | 2015-01-01 | yes | 1497 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 176 | yes |  |
| `VNM` | yes |  | 2015-01-01 | yes | 1456 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 135 | yes |  |
| `VPB` | yes |  | 2015-01-01 | yes | 1589 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 268 | yes |  |
| `VPL` | yes |  | 2015-01-01 | yes | 176 | 2025-08-27 00:00:00 | 2026-05-15 00:00:00 | 1H | 0 | 176 | no | training_rows_below_1000 |
| `VRE` | yes |  | 2015-01-01 | yes | 1497 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | 1321 | 176 | yes |  |

## Index Audit
| index | cache | rows | first | last | freq | usable | failure_reason |
|---|---:|---:|---|---|---|---:|---|
| `VNINDEX` | yes | 995 | 2022-05-19 00:00:00 | 2026-05-15 00:00:00 | 1H | yes |  |
| `HNXINDEX` | yes | 994 | 2022-05-19 00:00:00 | 2026-05-14 00:00:00 | 1H | yes |  |
| `UPCOMINDEX` | yes | 994 | 2022-05-19 00:00:00 | 2026-05-15 00:00:00 | 1H | yes |  |
| `VN30` | yes | 995 | 2022-05-19 00:00:00 | 2026-05-15 00:00:00 | 1H | yes |  |
| `HNX30` | yes | 962 | 2022-07-04 00:00:00 | 2026-05-15 00:00:00 | 1H | yes |  |
| `VN100` | yes | 1570 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 | 1H | yes |  |

## Final Readiness Conclusion
- indices_ready: true.
- stock_universe_ready: false.
- full_30_vn30_ready: false.
- benchmark_readiness: false.
- reason_if_no: usable_tickers=29/30.
- Benchmark was run: no.
- Model training was run: no.
- Paper/DOCX generated: no.
- Daily data used: no.
- Resampling used: no.
