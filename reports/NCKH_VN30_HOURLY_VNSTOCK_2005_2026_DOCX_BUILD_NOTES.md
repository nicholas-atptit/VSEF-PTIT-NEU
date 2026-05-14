# VN30 Hourly vnstock 2005-2026 DOCX Build Notes

> Superseded: the active data-readiness design now starts at `2015-01-01` and uses provider-current/latest available timestamps for the evaluation end. See `reports/VN30_HOURLY_2015_DATA_READINESS_PLAN.md`.

## Source Markdown

- Final paper was not written because the fetched-data validation or benchmark gate did not pass.

## Design

- Universe: frozen VN30 tickers from `configs/universes/vn30_constituents_frozen.csv`.
- Frequency: hourly only.
- Superseded training/history period: 2005-01-01 00:00:00 to 2024-12-31 23:59:59.
- Superseded evaluation/comparison period: 2025-01-01 00:00:00 to 2026-05-31 23:59:59.
- Data source: vnstock/vnstock_data fetched normalized cache.
- Daily data and daily-to-hourly resampling are excluded.
- Old VN100 evidence is excluded.

## Validation Snapshot

- Benchmark-usable VN30 stocks: 0/30.
- VNINDEX benchmark-usable: False.
- VN30INDEX and VNXALL are optional context indices in this track; unsupported exact codes do not fail the stock+VNINDEX gate.

## Artifact Directories

- Fetch reports: `reports/generated/vn30_hourly_vnstock_fetch`.
- Full diagnostics: `reports/generated/vn30_hourly_vnstock_full`.
- Benchmark outputs: `outputs/vn30_hourly_vnstock_full_2005_2026_traincutoff`.

## Expected DOCX Outputs If Paper Exists

- `NCKH_VN30_HOURLY_VNSTOCK_2005_2026_FULL_PAPER_VI_APA.docx`
- `NCKH_VN30_HOURLY_VNSTOCK_2005_2026_FULL_PAPER_VI_IEEE.docx`
- `NCKH_VN30_HOURLY_VNSTOCK_2005_2026_FULL_PAPER_EN_APA.docx`
- `NCKH_VN30_HOURLY_VNSTOCK_2005_2026_FULL_PAPER_EN_IEEE.docx`
