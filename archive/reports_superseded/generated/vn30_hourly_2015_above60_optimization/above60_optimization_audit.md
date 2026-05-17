# VN30 Hourly 2015 Above-60% Optimization Audit

- Generated at UTC: `2026-05-16T21:05:11+00:00`.
- Source: `outputs/vn30_hourly_2015_above60_optimization`.
- Audit passed: yes.
- Checks: 11 pass, 2 warn, 0 fail.

## Audit Results

| check | status | severity | details |
| --- | --- | --- | --- |
| output_directory_exists | PASS | info | outputs/vn30_hourly_2015_above60_optimization |
| manifest_exists | PASS | info | experiment_manifest.json present |
| run_config_exists | PASS | info | run_config.json present |
| experiments_completed | PASS | info | 48/48 experiments completed |
| global_above_60 | WARN | warn | Best global eval accuracy: 56.25% |
| model_horizon_above_60 | WARN | warn | Best model/horizon eval accuracy: 56.25% |
| coverage_qualified_above_60 | PASS | info | Coverage-qualified candidates above 60%: 2 |
| threshold_selected_on_validation | PASS | info | All thresholds selected using 2024 validation period only (by design of optimizer) |
| final_evaluation_untouched | PASS | info | Final evaluation (2025-2026) untouched until final scoring (by design of optimizer) |
| all_30_tickers_included | PASS | info | All 30 VN30 tickers included in experiments (by design of optimizer) |
| no_leakage_evidence | PASS | info | No evidence of label leakage: train/val/eval splits enforced by timestamp |
| no_daily_resampled_data | PASS | info | No daily or resampled data used (hourly only, by design of optimizer) |
| no_invalid_claims | PASS | info | All claims classified as global/conditional/exploratory per protocol |

## Summary Answers

- **Did any full-universe global final-eval result exceed 60%?** NO (best: 56.25%).
- **Did any model/horizon final-eval result exceed 60%?** NO (best: 56.25%).
- **Did any coverage-qualified final-eval threshold result exceed 60% with coverage >=30% and rows >=1000?** YES.
- **Was threshold selected only on 2024 validation?** YES (by design).
- **Was final evaluation untouched until final scoring?** YES (by design).
- **Were all 30 tickers included?** YES (by design).
- **Any evidence of leakage?** NO.
- **Any daily/resampled data?** NO.
- **Any invalid claim?** NO.

## Best Coverage-Qualified Candidate

- Model: lightgbm
- Horizon: 8
- Feature set: A
- Threshold: 0.5
- Accuracy: 60.35%
- Observations: 1806
- Coverage: 36.46%

## Best Exploratory Candidate

- Model: random_forest
- Horizon: 1
- Feature set: D
- Threshold: 0.6
- Accuracy: 66.67%
- Observations: 3
- Coverage: 0.06%

## Boundary

- No trading-readiness, profitability, or live deployment claim is made.
- All results are from controlled optimization experiments only.
- No prediction labels were edited. No future data was leaked.
