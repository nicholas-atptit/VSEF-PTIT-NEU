# VN30 Hourly 2015 Jan-2025 Benchmark Audit

- Created at UTC: `2026-05-16T14:08:13+00:00`.
- Output directory: `outputs/vn30_hourly_2015_jan2025_benchmark`.
- Audit passed: yes.
- Checks passed: 21.
- Warnings: 0.
- Failures: 0.

## Claim Boundary

- No trading-readiness claim.
- No profitability claim.
- No cost/slippage claim.
- No paper or DOCX claim.

## Checks

| check | status | severity | details |
| --- | --- | --- | --- |
| output_directory_exists | PASS | info | outputs/vn30_hourly_2015_jan2025_benchmark |
| run_config_exists | PASS | info | outputs/vn30_hourly_2015_jan2025_benchmark/run_config.json |
| manifest_exists | PASS | info | outputs/vn30_hourly_2015_jan2025_benchmark/manifest.json |
| predicted_vs_actual_exists_non_empty | PASS | info | outputs/vn30_hourly_2015_jan2025_benchmark/hourly/predicted_vs_actual.csv |
| all_30_active_tickers_represented | PASS | info | missing=[] |
| frequency_hourly_only | PASS | info | prediction frequency must be hourly |
| train_cutoff_matches | PASS | info | 2024-12-31 23:59:59 |
| eval_start_matches | PASS | info | 2025-01-01 00:00:00 |
| eval_end_matches_readiness | PASS | info | run_config=2026-05-14 00:00:00; expected=2026-05-14 00:00:00 |
| no_daily_or_resampled_markers | PASS | info | daily/resampled flags must remain false and frequency must remain 1H/hourly |
| no_vn100_model_evidence_reused | PASS | info | VN100 index rows, if present, must be readiness-only and not model input |
| no_old_2005_2006_output_reused | PASS | info | outputs/vn30_hourly_2015_jan2025_benchmark |
| predicted_vs_actual_required_columns | PASS | info | missing=[]; has_time_column=True; confidence_supported=True |
| benchmark_summary_prediction_counts | PASS | info | summary_n=77692; actual_n=77692 |
| baseline_comparison_exists | PASS | info | baseline_summary and baseline_delta_summary must be non-empty |
| significance_generated | PASS | info | outputs/vn30_hourly_2015_jan2025_benchmark/hourly/significance_summary.csv |
| regime_accuracy_generated | PASS | info | outputs/vn30_hourly_2015_jan2025_benchmark/hourly/regime_accuracy_summary.csv |
| model_error_generated | PASS | info | outputs/vn30_hourly_2015_jan2025_benchmark/hourly/model_error_summary.csv |
| source_health_generated | PASS | info | outputs/vn30_hourly_2015_jan2025_benchmark/hourly/source_health_summary.csv |
| run_log_generated | PASS | info | outputs/vn30_hourly_2015_jan2025_benchmark/hourly/benchmark_run_log.md |
| claim_boundary_generated | PASS | info | No trading-readiness, profitability, cost/slippage, paper, or DOCX claim is made by this audit. |
