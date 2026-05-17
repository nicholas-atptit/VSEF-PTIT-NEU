# VN30 Hourly 2015 - All-Model Final65 Router Audit

- Generated: 2026-05-17T00:59:38+00:00
- Audit: 11 pass, 1 warn, 0 fail

## Audit Results

| check | status | severity | details |
| --- | --- | --- | --- |
| output_dir_exists | PASS | info | outputs/vn30_hourly_2015_all_model_final65_router |
| manifest_exists | PASS | info | manifest present |
| no_daily_data | PASS | info | hourly only by design |
| no_resampling | PASS | info | no resampling by design |
| no_data_fetch | PASS | info | no data fetch by design |
| universe_unchanged | PASS | info | VN30 Jan 2025 unchanged |
| baseline60_retained | PASS | info | RF h=60 60.22% baseline retained |
| validation_only_selection | PASS | info | all policy selection on 2024 validation |
| final_eval_scoring_only | PASS | info | final eval used only for scoring |
| no_final_label_tuning | PASS | info | no threshold/ticker/model selection on final labels |
| no_leakage | PASS | info | no leakage indicators |
| final65_pass | WARN | warn | Best final65: 59.64% |

## Summary

- Final65 >=65: FAIL (best: 59.64%)
- Gap to 65: 5.36%
- Global full-universe >=65: NO
- Exploratory >=65: YES

## Best Final65 Candidate

- Policy: per_ticker_whitelist
- Model: random_forest, Horizon: 60
- Target: absolute
- Accuracy: 59.64%
- Coverage: 100.00%
- Rows: 3474
- Active Tickers: 30

## Selected Policy

| policy_id | policy_type | model | horizon | target_type | validation_accuracy | final_accuracy | final_coverage | final_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost_h8_relative_vnindex_ticker_50 | per_ticker_whitelist | xgboost | 8 | relative_vnindex | 0.995305 | 0.504172 | 1.0 | 5034 |

## Boundary

- No trading-readiness, profitability, or live deployment claim.
