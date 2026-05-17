# VN30 Daily 2015 Target60 Audit

- Created at UTC: `2026-05-17T09:50:53+00:00`.
- Output directory: `outputs/vn30_daily_2015_target60_optimization`.
- Audit passed: yes.
- Checks passed: 15. Warnings: 1. Failures: 0.
- Target60 passed: no.

## Claim Boundary

- No trading-readiness claim.
- No profitability claim.
- No cost/slippage claim.
- No paper or DOCX claim.
- Daily track is separate from hourly track; no mixing.
- No daily-to-hourly resampling.

## Checks

| check | status | severity | details |
| --- | --- | --- | --- |
| output_directory_exists | PASS | info | outputs/vn30_daily_2015_target60_optimization |
| run_config_exists | PASS | info | outputs/vn30_daily_2015_target60_optimization/run_config.json |
| manifest_exists | PASS | info | outputs/vn30_daily_2015_target60_optimization/manifest.json |
| daily_target60_manifest_exists | PASS | info | outputs/vn30_daily_2015_target60_optimization/daily_target60_manifest.json |
| validation_candidate_results_exists_non_empty | PASS | info | outputs/vn30_daily_2015_target60_optimization/daily/validation_candidate_results.csv |
| final_candidate_results_exists_non_empty | PASS | info | outputs/vn30_daily_2015_target60_optimization/daily/final_candidate_results.csv |
| daily_only | PASS | info | daily track only; no hourly data used |
| no_hourly_resampling | PASS | info | no daily-to-hourly resampling |
| active_universe_30 | PASS | info | universe size=30 |
| active_ticker_count_30 | PASS | info | active_ticker_count=30 |
| final_coverage_1.0 | PASS | info | final_coverage=1.0 |
| validation_only_selection | PASS | info | candidates selected on validation accuracy only |
| validation_accuracies_present | PASS | info | all validation accuracies present |
| no_trading_claims | PASS | info | no trading-readiness, profitability, or live-deployment claims |
| candidates_60_file_exists | PASS | info | outputs/vn30_daily_2015_target60_optimization/daily/daily_60_candidates.csv |
| target60_passed | WARN | warn | target60_passed=False; 60 candidates=False |
