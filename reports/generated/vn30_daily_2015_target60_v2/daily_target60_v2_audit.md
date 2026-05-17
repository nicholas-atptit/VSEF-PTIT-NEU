# VN30 Daily 2015 Target60 V2 Audit

- Created at UTC: `2026-05-17T10:15:24+00:00`.
- Output directory: `outputs/vn30_daily_2015_target60_v2`.
- Audit passed: yes.
- Checks passed: 17. Warnings: 1. Failures: 0.
- Target60 passed: no.

## Claim Boundary

- No trading-readiness claim.
- No profitability claim.
- No cost/slippage claim.
- No paper or DOCX claim.
- Daily track is separate from hourly track; no mixing.
- No daily-to-hourly resampling.
- No abstention.
- No ticker subset.

## Checks

| check | status | severity | details |
| --- | --- | --- | --- |
| output_directory_exists | PASS | info | outputs/vn30_daily_2015_target60_v2 |
| run_config_exists | PASS | info | outputs/vn30_daily_2015_target60_v2/run_config.json |
| manifest_exists | PASS | info | outputs/vn30_daily_2015_target60_v2/manifest.json |
| daily_target60_v2_manifest_exists | PASS | info | outputs/vn30_daily_2015_target60_v2/daily_target60_v2_manifest.json |
| rolling_validation_results_exists_non_empty | PASS | info | outputs/vn30_daily_2015_target60_v2/daily/rolling_validation_results.csv |
| final_candidate_results_exists_non_empty | PASS | info | outputs/vn30_daily_2015_target60_v2/daily/final_candidate_results.csv |
| daily_only | PASS | info | daily track only; no hourly data used |
| no_hourly_resampling | PASS | info | no daily-to-hourly resampling |
| active_universe_30 | PASS | info | universe size=30 |
| active_ticker_count_30 | PASS | info | active_ticker_count=30 |
| final_coverage_1.0 | PASS | info | final_coverage=1.0 |
| no_abstention | PASS | info | all candidates predict all final rows; no abstention |
| no_ticker_subset | PASS | info | full 30/30 universe used |
| validation_only_selection | PASS | info | threshold and candidate selected on rolling validation only |
| final_eval_scoring_only | PASS | info | final evaluation used only for scoring, never for selection |
| no_leakage | PASS | info | no future values in features; no label leakage |
| no_trading_claims | PASS | info | no trading-readiness, profitability, or live-deployment claims |
| target60_passed | WARN | warn | target60_passed=False; 60 candidates=False |
