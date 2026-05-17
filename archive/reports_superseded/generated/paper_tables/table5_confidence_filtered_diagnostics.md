# Table 5: Confidence-Filtered Strategy Diagnostics

| scope | frequency | model | horizon | threshold | total_rows | evaluated_rows | coverage_ratio | filtered_accuracy | passed_60pct | selected_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_at_coverage_floor_50% | hourly | stacking | 1 | 0.55 | 7339 | 3782 | 0.5153290639051642 | 0.5817028027498677 | false | false |
| best_at_coverage_floor_40% | hourly | stacking | 1 | 0.56 | 7339 | 3019 | 0.41136394604169507 | 0.5905929115601193 | false | false |
| best_at_coverage_floor_30% | hourly | stacking | 1 | 0.57 | 7339 | 2297 | 0.3129854203569969 | 0.6003482803656944 | true | true |
| selected_candidate | hourly | stacking | 1 | 0.57 | 7339 | 2297 | 0.3129854203569969 | 0.6003482803656944 | true | true |
| best_default_filter_daily | daily | random_forest | 20 | 0.55 | 1592 | 1375 | 0.8636934673366834 | 0.5716363636363636 | false | false |
| best_default_filter_hourly | hourly | stacking | 1 | 0.55 | 7339 | 3782 | 0.5153290639051642 | 0.5817028027498677 | false | false |

## Note

- Source artifact: confidence_threshold_sweep_summary.csv; daily/hourly confidence_filter_summary.csv.
- Claim supported: The selected hourly stacking h=1 threshold 0.57 slice passes 60% only at about 31.30% coverage.
- Limitation: Daily threshold-sweep rows are missing; available sweep rows cover hourly stacking h=1.
- Status: partial.
