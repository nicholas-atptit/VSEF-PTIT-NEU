# VN30 Hourly Available-Window Confidence Sweep Report

## Source

- Prediction artifact: `outputs/vn30_hourly_available_window_benchmark/hourly/predicted_vs_actual.csv`.
- Frequency: hourly only.
- Study: VN30 hourly available-window.
- Threshold grid: 0.50 to 0.90.
- Coverage floors: 50%, 40%, 30%, 20%.

## Best Candidate by Coverage Floor

| coverage_floor | candidate | evaluated_rows | coverage_ratio | filtered_accuracy | passed_60pct | ticker_count | top_ticker | top1_prediction_share | top3_prediction_share | ticker_concentration_warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| >= 50% | hourly random_forest h=1 threshold 0.57 | 2688 | 58.69% | 58.85% | False | 27 | VPL | 6.40% | 15.85% | no |
| >= 40% | hourly random_forest h=1 threshold 0.60 | 1966 | 42.93% | 60.27% | True | 27 | VPL | 7.73% | 17.70% | no |
| >= 30% | hourly random_forest h=1 threshold 0.62 | 1526 | 33.32% | 61.73% | True | 27 | VPL | 9.24% | 20.12% | no |
| >= 20% | hourly xgboost h=1 threshold 0.79 | 924 | 20.17% | 63.96% | True | 27 | VPL | 7.47% | 21.10% | no |

## Boundary

- Sweep rows evaluated: 656.
- Coverage floors evaluated: 50%, 40%, 30%, 20%.
- Rows with ticker concentration warnings: 96.
- A filtered slice is diagnostic only and does not establish trading readiness.
