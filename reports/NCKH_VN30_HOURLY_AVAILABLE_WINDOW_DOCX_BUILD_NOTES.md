# VN30 Hourly Available-Window DOCX Build Notes

## Source Markdown

- `reports/NCKH_FULL_PAPER_DRAFT_VN30_HOURLY_AVAILABLE_WINDOW_V1_WITH_FIGURES.md`

## Selected Design

- Selected tickers: ACB, BID, CTG, DGC, FPT, GAS, GVR, HDB, LPB, MBB, MSN, MWG, PLX, SAB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIC, VJC, VNM, VPB, VPL, VRE.
- Excluded tickers: HPG, SHB, VIB.
- Training window: 2025-12-15 09:00:00 to 2026-03-13 11:00:00.
- Evaluation window: 2026-03-13 13:00:00 to 2026-05-11 14:00:00.
- Final paper can proceed: true.

## Market Index Context

- `VNINDEX`, `VN30INDEX`, and `VNXALL` index context should be included only if exact-code local hourly index data overlaps the available-window design.
- For the full 2005-2026 design, `VN30INDEX` is not required before `2012-02-06 00:00:00` and `VNXALL` is not required before `2016-10-24 00:00:00`.
- The full-design comparison/evaluation window remains aligned from `2025-01-01 00:00:00` to `2026-05-31 23:59:59`.
- Missing pre-start `VN30INDEX`/`VNXALL` rows are not a readiness failure.

## Artifact Directories

- Tables: `reports/generated/vn30_hourly_available_window/paper_tables`.
- Figures: `reports/generated/vn30_hourly_available_window/paper_figures`.
- Notes: `reports/generated/vn30_hourly_available_window/paper_notes`.

## Citation Styles Needed

- Vietnamese APA.
- Vietnamese IEEE.
- English APA.
- English IEEE.

## Expected DOCX Outputs If Final Paper Exists

- `NCKH_VN30_HOURLY_AVAILABLE_WINDOW_FULL_PAPER_VI_APA.docx`
- `NCKH_VN30_HOURLY_AVAILABLE_WINDOW_FULL_PAPER_VI_IEEE.docx`
- `NCKH_VN30_HOURLY_AVAILABLE_WINDOW_FULL_PAPER_EN_APA.docx`
- `NCKH_VN30_HOURLY_AVAILABLE_WINDOW_FULL_PAPER_EN_IEEE.docx`
