# VN30 Hourly 2015 - RF h=60 Final65 Focus Audit

- Generated: 2026-05-17T00:33:49+00:00
- Audit: 10 pass, 1 warn, 0 fail

## Audit Results

| check | status | severity | details |
| --- | --- | --- | --- |
| output_dir_exists | PASS | info | outputs/vn30_hourly_2015_rf_h60_final65_focus |
| manifest_exists | PASS | info | manifest present |
| no_daily_data | PASS | info | hourly only by design |
| no_resampling | PASS | info | no resampling by design |
| no_data_fetch | PASS | info | no data fetch by design |
| universe_unchanged | PASS | info | VN30 Jan 2025 unchanged |
| validation_only_selection | PASS | info | all threshold/policy selection on 2024 validation |
| final_eval_scoring_only | PASS | info | final eval used only for scoring |
| no_final_label_tuning | PASS | info | no threshold tuning on final labels |
| no_leakage | PASS | info | no leakage indicators |
| final65_pass | WARN | warn | Best final65: 59.70% |

## Summary

- Final65 >=65: FAIL (best: 59.70%)
- Gap to 65: 5.30%

## Best Final65 Candidate

- Experiment: platt_calibration
- Threshold: 
- Accuracy: 59.70%
- Coverage: 100.00%
- Rows: 3474

## Selected Policies

| experiment | threshold | val_accuracy | val_coverage | val_rows |
| --- | --- | --- | --- | --- |
| platt_calibration |  | 0.49334 | 1.0 | 30030 |

## Boundary

- No trading-readiness, profitability, or live deployment claim.
