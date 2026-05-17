# VN30 Hourly 2015 - Full Tuning Sweep Audit

- Generated: 2026-05-17T01:17:04+00:00
- Audit: 12 pass, 1 warn, 0 fail

## Audit Results

| check | status | severity | details |
| --- | --- | --- | --- |
| output_dir_exists | PASS | info | outputs/vn30_hourly_2015_full_tuning_sweep |
| manifest_exists | PASS | info | manifest present |
| canonical_evaluator_used | PASS | info | Version: canonical_v1.0.0 |
| no_daily_data | PASS | info | hourly only by design |
| no_resampling | PASS | info | no resampling by design |
| no_data_fetch | PASS | info | no data fetch by design |
| universe_unchanged | PASS | info | VN30 Jan 2025 unchanged |
| validation_only_selection | PASS | info | all policy selection on 2024 validation |
| final_eval_scoring_only | PASS | info | final eval used only for scoring |
| no_final_label_tuning | PASS | info | no threshold/ticker/model selection on final labels |
| no_leakage | PASS | info | no leakage indicators |
| baseline60_pass | PASS | info | Best baseline60: 60.31% |
| final65_pass | WARN | warn | Best final65: 61.48% |

## Summary

- Baseline >=60: PASS (best: 60.31%)
- Final >=65: FAIL (best: 61.48%)
- Gap to 60: -0.31%
- Gap to 65: 3.52%

## Best Global Candidate

- Candidate: random_forest_h60_absolute_p1_ticker_50
- Model: random_forest, Horizon: 60
- Target: absolute, Policy: per_ticker_whitelist
- Accuracy: 60.31%
- Coverage: 100.00%
- Rows: 3474

## Best Coverage-Qualified Candidate

- Candidate: random_forest_h40_absolute_p2_conf_0.5
- Model: random_forest, Horizon: 40
- Target: absolute, Policy: confidence_abstention
- Accuracy: 61.48%
- Coverage: 31.54%
- Rows: 1285

## Selected Policy

| candidate_id | policy_type | model | horizon | target_type | validation_accuracy | final_accuracy | final_coverage | final_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost_h8_relative_vnindex_p0_ticker_50 | per_ticker_whitelist | xgboost | 8 | relative_vnindex | 0.995305 | 0.504172 | 1.0 | 5034 |

## RF h=60 Consistency

- Canonical RF h=60: 60.31% (from consistency audit)
- Baseline60 status: PASS under canonical evaluator

## Boundary

- No trading-readiness, profitability, or live deployment claim.
