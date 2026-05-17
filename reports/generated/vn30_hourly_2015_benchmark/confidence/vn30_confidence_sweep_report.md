# VN30 Hourly 2015 Confidence Sweep Diagnostics

## Source

- Prediction artifact: `outputs/vn30_hourly_2015_jan2025_benchmark/hourly/predicted_vs_actual.csv`.
- Frequency: hourly only.
- Threshold grid: 0.500 to 0.900 in 0.025 increments.
- Coverage floors: 50%, 40%, 30%.
- This is a post-hoc diagnostic. It does not create a new global benchmark pass.

## Best Diagnostic Slice By Coverage Floor

| coverage_floor | model | horizon | threshold | evaluated_rows | coverage_ratio | filtered_accuracy | passed_60pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50.00% | xgboost | 20 | 0.750 | 2345 | 50.99% | 57.48% | no |
| 40.00% | xgboost | 20 | 0.800 | 1840 | 40.01% | 57.61% | no |
| 30.00% | random_forest | 20 | 0.675 | 1583 | 34.42% | 59.19% | no |

## Boundary

- Sweep rows generated: 816.
- Rows with filtered accuracy at or above 60% and coverage floor satisfied: 0.
- Confidence slices are conditional diagnostics only and must be reported with coverage and post-hoc limitations.
- No trading-readiness, profitability, or stable 60%+ method claim is made.
