# VN30 Hourly 2015 - RF h=60 Final65 Router v2 Audit

- Generated: 2026-05-17T00:47:45+00:00
- Audit: 11 pass, 1 warn, 0 fail

## Audit Results

| check | status | severity | details |
| --- | --- | --- | --- |
| output_dir_exists | PASS | info | outputs/vn30_hourly_2015_rf_h60_final65_router_v2 |
| manifest_exists | PASS | info | manifest present |
| no_daily_data | PASS | info | hourly only by design |
| no_resampling | PASS | info | no resampling by design |
| no_data_fetch | PASS | info | no data fetch by design |
| universe_unchanged | PASS | info | VN30 Jan 2025 unchanged |
| rf_h60_baseline_retained | PASS | info | RF h=60 absolute direction retained |
| validation_only_selection | PASS | info | all policy selection on 2024 validation |
| final_eval_scoring_only | PASS | info | final eval used only for scoring |
| no_final_label_tuning | PASS | info | no threshold/ticker selection on final labels |
| no_leakage | PASS | info | no leakage indicators |
| final65_pass | WARN | warn | Best final65: 59.87% |

## Summary

- Final65 >=65: FAIL (best: 59.87%)
- Gap to 65: 5.13%

## Best Final65 Candidate

- Policy: per_ticker_whitelist
- Threshold: 
- Accuracy: 59.87%
- Coverage: 100.00%
- Rows: 3474
- Active Tickers: 30

## Selected Policy

| policy_type | threshold | validation_accuracy | validation_coverage | validation_rows |
| --- | --- | --- | --- | --- |
| per_ticker_whitelist |  | 0.492374 | 1.0 | 30030 |

## Boundary

- No trading-readiness, profitability, or live deployment claim.
