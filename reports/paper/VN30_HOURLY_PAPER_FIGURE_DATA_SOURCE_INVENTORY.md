# VN30 Hourly Paper Figure Data Source Inventory

All numeric paper tables and figures are derived from repository artifacts. No market-data fetch, benchmark run, or model training is performed by the paper table/figure builders.

## Inventory

| artifact_file | metric_extracted | exact_value | source_type | source_method | row_level_predictions_exist | figure_generation_basis |
| --- | --- | --- | --- | --- | --- | --- |
| reports/generated/vn30_hourly_target62_paper_ready_stability/global_summary.csv | selected candidate identifiers | {'model': 'l2_logistic', 'horizon': 40, 'feature_set': 'feature_set_C_closest', 'threshold': 0.5} | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/global_summary.csv | final_accuracy | 0.6151202749140894 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/global_summary.csv | total_rows | 4074 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/global_summary.csv | correct_predictions | 2506 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/global_summary.csv | incorrect_predictions | 1568 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/global_summary.csv | majority_baseline_accuracy | 0.5044182621502209 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/global_summary.csv | delta_vs_majority_baseline | 0.1107020127638684 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/global_summary.csv | delta_vs_60_43 | 0.0108001963672067 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/global_summary.csv | delta_vs_60_31 | 0.0120202749140894 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/global_summary.csv | pass_60 | True | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/global_summary.csv | pass_60_43 | True | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/global_summary.csv | pass_62 | False | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/global_summary.csv | pass_65 | False | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/global_summary.csv | validation_accuracy | 0.5188145188145188 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/global_summary.csv | validation_final_gap | 0.0963057560995706 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/global_summary.csv | validation_final_mismatch | high_positive_final_gap | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/global_summary.csv | paper_ready_claim_level | improved_baseline60 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/paper_ready_summary_table.csv | mean_ticker_accuracy | 0.6525663158826698 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/paper_ready_summary_table.csv | median_ticker_accuracy | 0.6544117647058824 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/paper_ready_summary_table.csv | mean_month_accuracy | 0.5371200256074595 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/paper_ready_summary_table.csv | median_month_accuracy | 0.5959780621572212 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/paper_ready_summary_table.csv | mean_quarter_accuracy | 0.5629015712724363 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/paper_ready_summary_table.csv | median_quarter_accuracy | 0.5459770114942529 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/paper_ready_summary_table.csv | bootstrap_ci_low | 0.5464561460060171 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/paper_ready_summary_table.csv | bootstrap_ci_high | 0.6895665825681563 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/paper_ready_summary_table.csv | significance_result | significant_vs_50_and_majority_baseline | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/paper_ready_summary_table.csv | ticker_stability_classification | ticker_moderately_stable | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/paper_ready_summary_table.csv | time_stability_classification | time_concentrated_or_mixed | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/paper_ready_summary_table.csv | regime_stability_classification | regime_unstable | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/paper_ready_summary_table.csv | stability_classification | concentrated_or_mixed | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/paper_ready_summary_table.csv | claim_level | improved_baseline60 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/paper_ready_summary_table.csv | target62_claim | False | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/paper_ready_summary_table.csv | final65_claim | False | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_track_a_target62_validation_safe/selected_candidate_summary.csv | selected summary validation_accuracy | 0.5188145188145188 | CSV | parsed | no | summary data |
| outputs/vn30_hourly_track_a_target62_validation_safe/selected_candidate_summary.csv | output mirror selected summary validation_accuracy | 0.5188145188145188 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_track_a_target62_validation_safe/selected_candidate_summary.csv | selected summary final_accuracy | 0.6151202749140894 | CSV | parsed | no | summary data |
| outputs/vn30_hourly_track_a_target62_validation_safe/selected_candidate_summary.csv | output mirror selected summary final_accuracy | 0.6151202749140894 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_track_a_target62_validation_safe/selected_candidate_summary.csv | selected summary final_rows | 4074 | CSV | parsed | no | summary data |
| outputs/vn30_hourly_track_a_target62_validation_safe/selected_candidate_summary.csv | output mirror selected summary final_rows | 4074 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_track_a_target62_validation_safe/selected_candidate_summary.csv | selected summary active_ticker_count | 30 | CSV | parsed | no | summary data |
| outputs/vn30_hourly_track_a_target62_validation_safe/selected_candidate_summary.csv | output mirror selected summary active_ticker_count | 30 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_track_a_target62_validation_safe/selected_candidate_summary.csv | selected summary claim_level | exploratory_baseline60 | CSV | parsed | no | summary data |
| outputs/vn30_hourly_track_a_target62_validation_safe/selected_candidate_summary.csv | output mirror selected summary claim_level | exploratory_baseline60 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_track_a_target62_validation_safe/run_config.json | baseline_logistic_h40 | 0.6043200785468826 | JSON | parsed | no | summary data |
| reports/generated/vn30_hourly_track_a_target62_validation_safe/run_config.json | historical_rf_h60 | 0.6031 | JSON | parsed | no | summary data |
| reports/generated/vn30_hourly_track_a_target62_validation_safe/run_config.json | selection rule uses final accuracy | False | JSON | parsed | no | summary data |
| reports/generated/vn30_hourly_rf_h60_reproduction/rf_h60_reproduction_summary.csv | historical_rf_h60_accuracy detail | 0.603051 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/bootstrap_ci.csv | bootstrap_source | ticker_weighted_resample | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/bootstrap_ci.csv | iterations | 20000 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/bootstrap_ci.csv | ci_low | 0.5464561460060171 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/bootstrap_ci.csv | ci_high | 0.6895665825681563 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/bootstrap_ci.csv | bootstrap_mean | 0.6168505477664373 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/bootstrap_ci.csv | standard_error | 0.0076231071723507 | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/significance_tests.csv | significance tests | [{'test': 'vs_50_percent', 'successes': 2506, 'n': 4074, 'null_p': 0.5, 'z_score': 14.695769314290136, 'p_value_two_sided': 6.861477056908793e-49, 'result': 'significant'}, {'test': 'vs_majority_baseline', 'successes': 2506, 'n': 4074, 'null_p': 0.5044182621502209, 'z_score': 14.132304346885764, 'p_value_two_sided': 2.401553237484099e-45, 'result': 'significant'}] | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/validation_final_mismatch.csv | validation-final mismatch row | {'validation_accuracy': 0.5188145188145188, 'final_accuracy': 0.6151202749140894, 'validation_final_gap': 0.0963057560995706, 'overfit_risk': 'low', 'validation_final_mismatch': 'high_positive_final_gap', 'validation_baseline_accuracy': 0.4878787878787878, 'final_baseline_accuracy': 0.5044182621502209, 'final_delta_vs_baseline': 0.1107020127638684} | CSV | parsed | no | summary data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/by_ticker.csv | ticker slice rows used | 30 | CSV | parsed | no | summary slice data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/by_month.csv | monthly slice rows used | 15 | CSV | parsed | no | summary slice data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/by_quarter.csv | quarterly slice rows used | 5 | CSV | parsed | no | summary slice data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/by_regime.csv | regime slice rows used | 3 | CSV | parsed | no | summary slice data |
| reports/generated/vn30_hourly_target62_paper_ready_stability/paper_ready_figure_index.csv | paper-ready figure source index | [{'figure_id': 'fig_ticker_accuracy', 'source_csv': 'by_ticker.csv', 'description': 'Ticker-level final accuracy and majority-baseline lift'}, {'figure_id': 'fig_month_accuracy', 'source_csv': 'by_month.csv', 'description': 'Monthly final accuracy and majority-baseline lift'}, {'figure_id': 'fig_quarter_accuracy', 'source_csv': 'by_quarter.csv', 'description': 'Quarterly final accuracy and majority-baseline lift'}, {'figure_id': 'fig_regime_accuracy', 'source_csv': 'by_regime.csv', 'description': 'Regime-level final accuracy and majority-baseline lift'}, {'figure_id': 'fig_monthly_expanding', 'source_csv': 'rolling_monthly_expanding.csv', 'description': 'Monthly expanding accuracy'}, {'figure_id': 'fig_quarterly_expanding', 'source_csv': 'rolling_quarterly_expanding.csv', 'description': 'Quarterly expanding accuracy'}, {'figure_id': 'fig_rolling_rows', 'source_csv': 'rolling_accuracy_250/500/1000.csv', 'description': 'Unavailable: row-level predictions not saved'}] | CSV | parsed | no | summary metadata |
| reports/generated/vn30_hourly_target62_paper_ready_stability/rolling_accuracy_250.csv | row-level rolling status 250 | row_level_predictions_not_saved; audit does not regenerate predictions or train models | CSV | parsed | no | not generated; row-level predictions unavailable |
| reports/generated/vn30_hourly_target62_paper_ready_stability/rolling_accuracy_500.csv | row-level rolling status 500 | row_level_predictions_not_saved; audit does not regenerate predictions or train models | CSV | parsed | no | not generated; row-level predictions unavailable |
| reports/generated/vn30_hourly_target62_paper_ready_stability/rolling_accuracy_1000.csv | row-level rolling status 1000 | row_level_predictions_not_saved; audit does not regenerate predictions or train models | CSV | parsed | no | not generated; row-level predictions unavailable |
| reports/generated/vn30_hourly_2015_benchmark_readiness/vn30_2015_benchmark_readiness_manifest.json | hourly stock actual data window | 2023-09-11 10:00:00 to 2026-05-15 00:00:00 | JSON | parsed | no | summary data |
| reports/generated/vn30_hourly_2015_benchmark_readiness/vn30_2015_benchmark_readiness_manifest.json | hourly usable stock count | 30 | JSON | parsed | no | summary data |
| reports/generated/vn30_daily_2015/vn30_daily_2015_readiness.csv | daily stock readiness rows | 30 | CSV | parsed | no | data-scope summary |
| reports/generated/index_benchmark/index_data_scope_audit.csv | index data-scope rows | 100 | CSV | parsed | no | data-scope summary |
| reports/generated/vn30_hourly_data_forensics/vn30_hourly_data_file_inventory.md | hourly forensics limitation | no 2015-2022 hourly stock data exists anywhere in the repository | MD | manually read from MD | no | data-scope limitation text |

## Row-Level Prediction Availability

- Target62 selected-candidate row-level predictions: no.
- The target62 paper-ready audit records rolling 250/500/1000 row accuracy as unavailable because row-level predictions were not saved.
- Separate OOF/final prediction artifacts exist for a different true-stacking workflow and are not used for target62 figures.

## Missing Items

- External literature-comparison values are not verified by current repository artifacts.
- Target62 row-level prediction records are not saved, so rolling 250/500/1000 row figures and correctness-over-time plots are unsupported.
