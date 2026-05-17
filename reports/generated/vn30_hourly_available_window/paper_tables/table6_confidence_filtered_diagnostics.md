# Table 6: Confidence-filtered diagnostics

| frequency | model | horizon | threshold | evaluated_rows | coverage_ratio | filtered_accuracy | passed_60pct | ticker_count | ticker_concentration_warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hourly | random_forest | 1 | 57.00% | 2688 | 58.69% | 58.85% | False | 27 | no |
| hourly | random_forest | 1 | 60.00% | 1966 | 42.93% | 60.27% | True | 27 | no |
| hourly | random_forest | 1 | 62.00% | 1526 | 33.32% | 61.73% | True | 27 | no |
| hourly | xgboost | 1 | 79.00% | 924 | 20.17% | 63.96% | True | 27 | no |

## Note

Confidence-filtered diagnostics from available-window hourly predictions.
