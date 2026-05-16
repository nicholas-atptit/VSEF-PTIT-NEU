# VN30 Hourly 2015 Benchmark Readiness

- Active universe: VN30 January 2025 review universe.
- Active universe source: HOSE January 2025 VN30 review.
- Active universe effective period: 03/02/2025 to 01/08/2025.
- Active universe count: 30/30.
- Active universe tickers: ACB, BCM, BID, BVH, CTG, FPT, GAS, GVR, HDB, HPG, LPB, MBB, MSN, MWG, PLX, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VRE.
- Active universe includes: BCM, BVH.
- Active universe excludes: BSR, DGC, VPL.
- Benchmark can proceed: false.
- Benchmark command path exists: false (`scripts\research\run_vn30_hourly_benchmark_2015_from_gateway.py`).
- Fetched required tickers: 30/30.
- Usable required tickers: 30/30.
- Usable required indices: 6/6.
- Missing/unusable tickers: none.
- Validation extras outside active universe: none.
- Confirmed listing-date tickers: ACB, BID, CTG, FPT, GAS, GVR, HDB, HPG, LPB, MBB, MSN, MWG, PLX, SAB, SHB, SSB, SSI, STB.
- Tickers needing listing-date verification: BCM, BVH, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VRE.
- VNINDEX usable: true.
- VN30 index usable: true.
- Training period claim: `2015-01-01 to 2024-12-31`.
- Training period actual available: `2023-09-11 10:00:00 to 2024-12-31`.
- Evaluation period claim: `2025-01-01 to provider-current/latest available timestamp`.
- Evaluation period actual available: `2025-01-01 to 2026-05-14 00:00:00`.
- Data availability disclosure: 2015 design window with provider-available hourly data beginning on 2023-09-11 10:00:00.
- Benchmark was run: no.
- Model training was run: no.
- Paper/DOCX generated: no.
- Daily data used: no.
- Resampling used: no.

## Decision

Benchmark must not proceed yet.

## Blocking Reasons

- benchmark_command_missing=scripts\research\run_vn30_hourly_benchmark_2015_from_gateway.py

## Warnings

- benchmark design requested from 2015, but actual hourly availability begins at provider first timestamp

## Per-Ticker Actual Timestamps

| ticker | first datetime | last datetime |
|---|---|---|
| `ACB` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `BCM` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `BID` | 2023-09-11 10:00:00 | 2026-05-14 00:00:00 |
| `BVH` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `CTG` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `FPT` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `GAS` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `GVR` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `HDB` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `HPG` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `LPB` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `MBB` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `MSN` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `MWG` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `PLX` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `SAB` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `SHB` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `SSB` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `SSI` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `STB` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `TCB` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `TPB` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `VCB` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `VHM` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `VIB` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `VIC` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `VJC` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `VNM` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `VPB` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
| `VRE` | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |

## Effective Starts

| ticker | effective_start |
|---|---|
| `ACB` | 2015-01-01 |
| `BCM` | 2015-01-01 |
| `BID` | 2015-01-01 |
| `BVH` | 2015-01-01 |
| `CTG` | 2015-01-01 |
| `FPT` | 2015-01-01 |
| `GAS` | 2015-01-01 |
| `GVR` | 2018-03-21 |
| `HDB` | 2018-01-05 |
| `HPG` | 2015-01-01 |
| `LPB` | 2017-10-05 |
| `MBB` | 2015-01-01 |
| `MSN` | 2015-01-01 |
| `MWG` | 2015-01-01 |
| `PLX` | 2017-04-21 |
| `SAB` | 2016-12-06 |
| `SHB` | 2015-01-01 |
| `SSB` | 2021-03-24 |
| `SSI` | 2015-01-01 |
| `STB` | 2015-01-01 |
| `TCB` | 2015-01-01 |
| `TPB` | 2015-01-01 |
| `VCB` | 2015-01-01 |
| `VHM` | 2015-01-01 |
| `VIB` | 2015-01-01 |
| `VIC` | 2015-01-01 |
| `VJC` | 2015-01-01 |
| `VNM` | 2015-01-01 |
| `VPB` | 2015-01-01 |
| `VRE` | 2015-01-01 |

## Index Usability

| index | usable | rows | first | last |
|---|---:|---:|---|---|
| `VNINDEX` | true | 995 | 2022-05-19 00:00:00 | 2026-05-15 00:00:00 |
| `HNXINDEX` | true | 994 | 2022-05-19 00:00:00 | 2026-05-14 00:00:00 |
| `UPCOMINDEX` | true | 994 | 2022-05-19 00:00:00 | 2026-05-15 00:00:00 |
| `VN30` | true | 995 | 2022-05-19 00:00:00 | 2026-05-15 00:00:00 |
| `HNX30` | true | 962 | 2022-07-04 00:00:00 | 2026-05-15 00:00:00 |
| `VN100` | true | 1570 | 2023-09-11 10:00:00 | 2026-05-15 00:00:00 |
