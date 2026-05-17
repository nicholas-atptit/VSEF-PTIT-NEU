# VN100 Full Confidence Sweep Report

## Source

- Artifact directory: `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff`.
- Artifact mode: `current_official`.
- The sweep is derived from existing `predicted_vs_actual.csv` files; no model training or benchmark rerun was performed.

## Coverage

- Prediction rows used: 154,048.
- Available frequency/model/horizon combinations swept: 32.
- Thresholds swept: 0.50 to 0.90 in 0.01 increments.
- Daily confidence sweep rows now exist in this v2 derived artifact: yes.
- Official daily threshold-sweep source rows remain: 0 data rows.
- Figure status: ready.

## Best Candidates by Coverage Floor

| coverage_floor | candidate | evaluated_rows | coverage_ratio | filtered_accuracy | passed_60pct |
| --- | --- | --- | --- | --- | --- |
| >= 50% | daily xgboost h=20 threshold 0.86 | 844 | 53.02% | 60.55% | True |
| >= 40% | daily xgboost h=20 threshold 0.90 | 714 | 44.85% | 60.78% | True |
| >= 30% | daily stacking h=20 threshold 0.69 | 482 | 30.28% | 62.03% | True |
| >= 20% | daily stacking h=20 threshold 0.71 | 344 | 21.61% | 64.53% | True |

## Rows Passing 60% at >=20% Coverage

| frequency | model | horizon | threshold | evaluated_rows | coverage_ratio | filtered_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| daily | xgboost | 20 | 85.00% | 878 | 55.15% | 60.48% |
| daily | xgboost | 20 | 86.00% | 844 | 53.02% | 60.55% |
| daily | xgboost | 20 | 87.00% | 811 | 50.94% | 60.54% |
| daily | xgboost | 20 | 88.00% | 777 | 48.81% | 60.62% |
| daily | xgboost | 20 | 89.00% | 749 | 47.05% | 60.75% |
| daily | xgboost | 20 | 90.00% | 714 | 44.85% | 60.78% |
| hourly | stacking | 1 | 57.00% | 2297 | 31.30% | 60.03% |
| daily | stacking | 20 | 69.00% | 482 | 30.28% | 62.03% |
| daily | stacking | 20 | 70.00% | 411 | 25.82% | 62.77% |
| hourly | lightgbm | 1 | 73.00% | 1810 | 24.66% | 60.11% |
| hourly | stacking | 1 | 58.00% | 1808 | 24.64% | 60.73% |
| hourly | xgboost | 1 | 70.00% | 1718 | 23.41% | 60.24% |
| daily | stacking | 10 | 67.00% | 370 | 22.34% | 60.81% |
| daily | stacking | 20 | 71.00% | 344 | 21.61% | 64.53% |
| hourly | xgboost | 1 | 71.00% | 1530 | 20.85% | 60.52% |
| hourly | lightgbm | 1 | 75.00% | 1478 | 20.14% | 60.22% |

## Selected Candidate Concentration

| coverage_floor | candidate | ticker_count | top_ticker | top1_prediction_share | top3_prediction_share | assessment |
| --- | --- | --- | --- | --- | --- | --- |
| 50% | daily xgboost h=20 threshold 0.86 | 7 | BWE | 21.68% | 54.50% | low |
| 40% | daily xgboost h=20 threshold 0.90 | 7 | BWE | 24.37% | 58.96% | low |
| 30% | daily stacking h=20 threshold 0.69 | 6 | ANV | 31.95% | 69.29% | moderate |
| 20% | daily stacking h=20 threshold 0.71 | 6 | ANV | 41.57% | 80.81% | high |

## Interpretation

The v2 sweep closes the missing daily/model/horizon threshold-row gap at the diagnostic level because
it derives thresholds from official prediction rows. It remains a derived analysis, not a fresh official
benchmark rerun. Single-window and seven-ticker limitations still apply.

## Claim Boundary

- A row passing 60% after confidence filtering is a conditional diagnostic, not a global benchmark pass.
- Coverage below broad-market levels should be described as selective signal coverage.
- This report does not establish trading readiness or full VN100 representativeness.
