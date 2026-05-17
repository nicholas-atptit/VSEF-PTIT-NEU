# VN30 Hourly 2015 - Target Redesign Audit

- Generated: 2026-05-16T22:46:36+00:00
- Audit: 12 pass, 2 warn, 0 fail

## Audit Results

| check | status | severity | details |
| --- | --- | --- | --- |
| output_dir_exists | PASS | info | outputs/vn30_hourly_2015_target_redesign_experiments |
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
| baseline_60_pass | WARN | warn | Best global: 55.76% |
| final_65_pass | WARN | warn | Best coverage-qualified: 58.22% |
| claim_level | PASS | info | exploratory |

## Summary

- Baseline >=60: FAIL (best global: 55.76%)
- Final >=65: FAIL (best coverage-qualified: 58.22%)
- Claim level: exploratory

## Best Global Candidate

- Target: noise_band, Threshold: 0.001
- Model: xgboost, Horizon: 8
- Accuracy: 55.76%
- Coverage: 98.33%
- Rows: 4950

## Best Coverage-Qualified Candidate

- Target: quantile, Threshold: 0.4
- Model: xgboost, Horizon: 8
- Accuracy: 58.22%
- Coverage: 74.31%
- Rows: 3741

## Selected Policy (Validation)

- Target: binary, Model: random_forest
- Horizon: 4, Val Accuracy: 54.34%

## Boundary

- No trading-readiness, profitability, or live deployment claim.
