# VN30 Daily 2015 Benchmark Audit

- Created at UTC: `2026-05-17T09:23:24+00:00`.
- Output directory: `outputs/vn30_daily_2015_benchmark`.
- Audit passed: yes.
- Checks passed: 13. Warnings: 0. Failures: 0.

## Claim Boundary

- No trading-readiness claim.
- No profitability claim.
- No cost/slippage claim.
- No paper or DOCX claim.
- Daily track is separate from hourly track; no mixing.

## Checks

| check | status | severity | details |
| --- | --- | --- | --- |
| output_directory_exists | PASS | info | outputs/vn30_daily_2015_benchmark |
| run_config_exists | PASS | info | outputs/vn30_daily_2015_benchmark/run_config.json |
| manifest_exists | PASS | info | outputs/vn30_daily_2015_benchmark/manifest.json |
| predicted_vs_actual_exists_non_empty | PASS | info | outputs/vn30_daily_2015_benchmark/daily/predicted_vs_actual.csv |
| accuracy_summary_exists_non_empty | PASS | info | outputs/vn30_daily_2015_benchmark/daily/accuracy_summary.csv |
| all_usable_tickers_represented | PASS | info | missing=[]; usable=28 |
| frequency_daily_only | PASS | info | daily data track; no hourly mixing |
| train_end_matches | PASS | info | 2023-12-31 23:59:59 |
| eval_start_matches | PASS | info | 2025-01-01 00:00:00 |
| no_hourly_resampling | PASS | info | daily track does not resample to hourly |
| no_trading_claims | PASS | info | no trading-readiness, profitability, or live-deployment claims |
| baseline_comparison_exists | PASS | info | baseline files must exist |
| predicted_vs_actual_required_columns | PASS | info | columns=['datetime', 'ticker', 'y_true', 'y_pred'] |
