# VN30 Hourly 2015 - Final65 Focus v3 Audit

- Generated: 2026-05-17T01:28:03+00:00
- Audit: 12 pass, 1 warn, 0 fail

## Audit Results

| check | status | severity | details |
| --- | --- | --- | --- |
| output_dir_exists | PASS | info | outputs/vn30_hourly_2015_final65_focus_v3 |
| manifest_exists | PASS | info | manifest present |
| canonical_evaluator_used | PASS | info | Version: canonical_v1.0.0 |
| no_daily_data | PASS | info | hourly only by design |
| no_resampling | PASS | info | no resampling by design |
| no_data_fetch | PASS | info | no data fetch by design |
| universe_unchanged | PASS | info | VN30 Jan 2025 unchanged |
| baseline60_retained | PASS | info | RF h=60 60.31% baseline retained |
| validation_only_selection | PASS | info | all policy selection on 2024 validation |
| final_eval_scoring_only | PASS | info | final eval used only for scoring |
| no_final_label_tuning | PASS | info | no threshold/ticker/model selection on final labels |
| no_leakage | PASS | info | no leakage indicators |
| final65_pass | WARN | warn | Best final65: 58.32% |

## Summary

- Final65 >=65: FAIL (best: 58.32%)
- Gap to 65: 6.68%

## Best Final65 Candidate

- Policy: market_regime_abstention
- Accuracy: 58.32%
- Coverage: 100.00%
- Rows: 4074
- Active Tickers: 30

## Selected Policy

| policy_id | policy_type | validation_accuracy | validation_coverage | validation_rows | final_accuracy | final_coverage | final_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| meta_label_0.5 | meta_label_abstention | 0.56917 | 0.3517 | 10561 | 0.61284 | 0.2523 | 1028 |

## Boundary

- No trading-readiness, profitability, or live deployment claim.
