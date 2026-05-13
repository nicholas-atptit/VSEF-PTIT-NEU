# VN30 Hourly Available-Window Design Decision

## Selected Design

- Selected tickers: ACB, BID, CTG, DGC, FPT, GAS, GVR, HDB, LPB, MBB, MSN, MWG, PLX, SAB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIC, VJC, VNM, VPB, VPL, VRE.
- Excluded tickers: HPG, SHB, VIB.
- Training start: 2025-12-15 09:00:00.
- Training cutoff: 2026-03-13 11:00:00.
- Evaluation start: 2026-03-13 13:00:00.
- Evaluation end: 2026-05-11 14:00:00.
- Frequency: hourly only.
- Minimum row count: 480 common hourly timestamps per selected ticker window.
- Claim boundary, full VN30 representativeness: false.
- Final paper can proceed: true.

## Selection Rationale

- Selection rule: priority: 30 valid, then >=25 valid, then >=20 valid; within priority maximize ticker count then common timestamps.
- Minimum split rule: common timestamps >= 370, train rows >= 250, eval rows >= 100.
- Claim boundary: The study is an hourly available-window VN30 subset analysis rather than a full-constituent VN30 historical evaluation..
- The suggested 2025-01-01 evaluation start is not feasible for the local hourly universe because most tickers start in 2025 or 2026.

## Exclusions

| ticker | reason |
| --- | --- |
| HPG | first_available_after_selected_window_start:2026-02-11 14:00:00>2025-12-15 09:00:00 |
| SHB | first_available_after_selected_window_start:2026-02-11 14:00:00>2025-12-15 09:00:00 |
| VIB | first_available_after_selected_window_start:2026-02-11 14:00:00>2025-12-15 09:00:00 |

## Evidence Boundary

- This is not a 2005-2026 full-history VN30 benchmark.
- Daily data is not used.
- VN100 evidence is not reused.
- No data is fabricated.
