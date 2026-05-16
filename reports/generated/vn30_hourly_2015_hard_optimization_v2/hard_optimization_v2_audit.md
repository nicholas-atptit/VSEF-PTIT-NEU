# VN30 Hourly 2015 - Hard Optimization v2 Audit

- Generated: 2026-05-16T22:06:15+00:00
- Audit: 11 pass, 2 warn, 0 fail

## Audit Results

| check | status | severity | details |
| --- | --- | --- | --- |
| output_dir_exists | PASS | info | outputs/vn30_hourly_2015_hard_optimization_v2 |
| manifest_exists | PASS | info | manifest present |
| no_daily_data | PASS | info | hourly only by design |
| no_resampling | PASS | info | no resampling by design |
| all_30_tickers | PASS | info | all 30 tickers included by design |
| universe_unchanged | PASS | info | VN30 Jan 2025 unchanged |
| validation_only_selection | PASS | info | all selection on 2024 validation |
| final_eval_scoring_only | PASS | info | final eval used only for scoring |
| no_final_label_tuning | PASS | info | no threshold tuning on final labels |
| no_leakage | PASS | info | no leakage indicators |
| baseline_60_pass | WARN | warn | Best global: 54.63% |
| final_65_pass | WARN | warn | Best coverage-65: 0.00% |
| claim_level | PASS | info | exploratory |

## Summary

- Baseline >=60: FAIL (best global: 54.63%)
- Final >=65: FAIL (best coverage-65: 0.00%)
- Claim level: exploratory

## Boundary

- No trading-readiness, profitability, or live deployment claim.
