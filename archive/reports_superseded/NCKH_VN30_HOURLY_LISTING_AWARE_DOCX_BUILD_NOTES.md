# VN30 Hourly Listing-Aware DOCX Build Notes

## Source Markdown

- Final paper was not written because the benchmark gate did not pass.

## Design

- Study: VN30 hourly listing-aware historical benchmark.
- Universe: all 30 frozen VN30 tickers.
- Per-ticker start rule: max(first trading/listing date, first provider-available hourly timestamp).
- Training labels end: 2024-12-31 23:59:59.
- Evaluation starts: 2025-01-01 00:00:00.
- actual_eval_end: not available.
- VNINDEX is market context if fetched and validated; VN30INDEX and VNXALL are optional exact-code probes.
- Old VN100 evidence, daily data, daily-to-hourly resampling, and fabricated bars are excluded.

## Validation Snapshot

- Usable VN30 stocks: 0/30.

## Artifact Directories

- Reports: `reports/generated/vn30_hourly_listing_aware`.
- Benchmark outputs: `outputs/vn30_hourly_listing_aware_traincutoff`.

## Expected DOCX Outputs If Paper Exists

- `NCKH_VN30_HOURLY_LISTING_AWARE_FULL_PAPER_VI_APA.docx`
- `NCKH_VN30_HOURLY_LISTING_AWARE_FULL_PAPER_VI_IEEE.docx`
- `NCKH_VN30_HOURLY_LISTING_AWARE_FULL_PAPER_EN_APA.docx`
- `NCKH_VN30_HOURLY_LISTING_AWARE_FULL_PAPER_EN_IEEE.docx`
