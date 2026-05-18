# VN30 Hourly Selected Candidate Rolling Stability Result

## Reproduction Status

- Row-level predictions regenerated: yes.
- Regeneration purpose: reproduce the already selected candidate only to save final-window row-level predictions.
- New model selection: no.
- New tuning: no.
- Broad benchmark sweep: no.
- Data fetch: no.
- Confidence abstention: no.
- Ticker subset: no.
- Top-k/ranking: no.
- DOCX/paper rewrite: no.
- Trading/profitability/live-deployment claim: no.

## Fixed Candidate

- Model: L2 Logistic (`l2_logistic`).
- Horizon: h=40.
- Feature set: `feature_set_C_closest`.
- Threshold: 0.50.
- Setup: Track A canonical-like VN30 stock-only hourly.
- Selected candidate changed: no.

## Reproduction Summary

- Original final accuracy: 61.51%.
- Reproduced final accuracy: 61.51%.
- Reproduction difference: 0.000000 percentage points.
- Original final rows: 4,074.
- Reproduced final rows: 4,074.
- Rows difference: 0.
- Original majority baseline: 50.44%.
- Reproduced majority baseline: 50.44%.
- Original lift vs majority baseline: +11.07 percentage points.
- Reproduced lift vs majority baseline: +11.07 percentage points.
- Reproduction status: passed.

Primary reproduction files:

- `outputs/vn30_hourly_selected_l2_logistic_h40_row_predictions/row_predictions.csv`
- `outputs/vn30_hourly_selected_l2_logistic_h40_row_predictions/selected_candidate_reproduction_summary.csv`
- `outputs/vn30_hourly_selected_l2_logistic_h40_row_predictions/selected_candidate_reproduction_manifest.json`
- `outputs/vn30_hourly_selected_l2_logistic_h40_row_predictions/selected_candidate_reproduction_log.md`

## Rolling Stability Summary

Rolling rows are sorted by `datetime`, then `ticker`. Rolling majority baseline is the majority-class rate inside each rolling window, so rolling lift differs from the fixed global train-majority baseline lift.

| Window | Windows | Min Accuracy | Max Accuracy | Mean Accuracy | Median Accuracy | Endpoint Accuracy | Below 50% | Below 55% | Below 60% | Longest Below-60 Episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 rows | 3,825 | 7.60% | 92.80% | 63.14% | 63.60% | 48.00% | 663 | 916 | 1,339 | 666 windows |
| 500 rows | 3,575 | 28.40% | 89.40% | 63.03% | 62.80% | 65.00% | 557 | 736 | 1,259 | 685 windows |
| 1000 rows | 3,075 | 41.50% | 81.90% | 63.30% | 62.10% | 76.20% | 452 | 691 | 1,087 | 813 windows |

Longest below-60 episodes by endpoint order:

- 250-row windows: row 462 to row 1127, endpoint dates 2025-04-28 to 2025-09-12.
- 500-row windows: row 647 to row 1331, endpoint dates 2025-06-13 to 2025-10-01.
- 1000-row windows: row 1000 to row 1812, endpoint dates 2025-08-29 to 2025-11-06.

## Rolling Lift Versus Rolling Majority

| Window | Min Lift | Max Lift | Mean Lift | Median Lift | Endpoint Lift |
| --- | ---: | ---: | ---: | ---: | ---: |
| 250 rows | -84.80pp | +12.40pp | -11.38pp | -4.00pp | -4.00pp |
| 500 rows | -65.40pp | +19.20pp | -8.95pp | -2.80pp | +0.00pp |
| 1000 rows | -29.10pp | +19.30pp | -0.92pp | +0.00pp | +0.00pp |

## Time And Ticker Summaries

- Monthly slices: 15.
- Monthly mean accuracy: 53.71%.
- Monthly median accuracy: 59.60%.
- Months below 60%: 8.
- Quarterly slices: 5.
- Quarterly mean accuracy: 56.29%.
- Quarterly median accuracy: 54.60%.
- Quarters below 60%: 3.
- Ticker-level 250-row rolling windows: 51.
- Ticker-level 500-row rolling windows: 0.
- Ticker-level 1000-row rolling windows: 0.

Generated rolling audit files:

- `reports/generated/vn30_hourly_selected_candidate_rolling/rolling_accuracy_250.csv`
- `reports/generated/vn30_hourly_selected_candidate_rolling/rolling_accuracy_500.csv`
- `reports/generated/vn30_hourly_selected_candidate_rolling/rolling_accuracy_1000.csv`
- `reports/generated/vn30_hourly_selected_candidate_rolling/expanding_accuracy.csv`
- `reports/generated/vn30_hourly_selected_candidate_rolling/rolling_lift_vs_majority.csv`
- `reports/generated/vn30_hourly_selected_candidate_rolling/rolling_stability_summary.csv`
- `reports/generated/vn30_hourly_selected_candidate_rolling/rolling_stability_report.md`
- `reports/generated/vn30_hourly_selected_candidate_rolling/fig_rolling_250_accuracy.png`
- `reports/generated/vn30_hourly_selected_candidate_rolling/fig_rolling_500_accuracy.png`
- `reports/generated/vn30_hourly_selected_candidate_rolling/fig_rolling_1000_accuracy.png`
- `reports/generated/vn30_hourly_selected_candidate_rolling/fig_expanding_accuracy.png`
- `reports/generated/vn30_hourly_selected_candidate_rolling/fig_rolling_lift_vs_majority.png`

## Interpretation Boundary

The final score is 9.63 percentage points above validation accuracy; therefore, the result must be interpreted cautiously.

The row-level reproduction passed, but the rolling windows show substantial time variation and many windows below 60%. This does not upgrade the result to target62 or final65 evidence.

No trading, profitability, investment-recommendation, or live-deployment claim is made.
