# VN30 Hourly 2015 - Horizon & Relative Target Audit

- Generated: 2026-05-16T23:16:50+00:00
- Audit: 13 pass, 1 warn, 0 fail

## Audit Results

| check | status | severity | details |
| --- | --- | --- | --- |
| output_dir_exists | PASS | info | outputs/vn30_hourly_2015_horizon_relative_target_experiments |
| manifest_exists | PASS | info | manifest present |
| no_daily_data | PASS | info | hourly only by design |
| no_resampling | PASS | info | no resampling by design |
| no_data_fetch | PASS | info | no data fetch by design |
| universe_unchanged | PASS | info | VN30 Jan 2025 unchanged |
| all_30_tickers | PASS | info | all 30 tickers included by design |
| validation_only_selection | PASS | info | all target/threshold selection on 2024 validation |
| final_eval_scoring_only | PASS | info | final eval used only for scoring |
| no_final_label_tuning | PASS | info | no threshold tuning on final labels |
| no_leakage | PASS | info | no leakage indicators |
| baseline_60_pass | PASS | info | Best global: 60.22% |
| final_65_pass | WARN | warn | Best coverage-qualified: 62.73% |
| claim_level | PASS | info | global_full_universe |

## Summary

- Baseline >=60: PASS (best global: 60.22%)
- Final >=65: FAIL (best coverage-qualified: 62.73%)
- Claim level: global_full_universe

## Best Global Candidate

- Target: absolute, Market: 
- Horizon: 60, Model: random_forest
- Accuracy: 60.22%
- Coverage: 100.00%
- Rows: 3474

## Best Coverage-Qualified Candidate

- Target: absolute, Market: 
- Horizon: 60, Model: random_forest
- Accuracy: 62.73%
- Coverage: 61.17%
- Rows: 2125

## Selected Policy (Validation)

- Target: relative_vnindex, Model: lightgbm
- Horizon: 40, Val Accuracy: 97.16%

## Boundary

- No trading-readiness, profitability, or live deployment claim.
