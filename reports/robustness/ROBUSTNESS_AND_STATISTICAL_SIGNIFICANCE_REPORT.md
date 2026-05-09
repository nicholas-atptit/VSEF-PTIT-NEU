# Robustness and Statistical Significance Report

## 1. Executive summary

Phase 6 adds robustness and statistical-significance evidence from local artifacts. DM tests attempted 464 aligned model/baseline comparisons, with 354 significant at 5% and 360 significant at 10%. The observed difference is not statistically significant under this test; it should not be treated as robust evidence of superiority.
Window sensitivity has 1 exact local source setting(s) and 2 requested setting(s) without exact local rerun evidence, so window robustness remains constrained. The result is sensitive to configuration choices and should be treated as exploratory.
At 0 total bps, at least one diagnostic basket row has non-positive net average return. The diagnostic edge weakens or disappears under cost/slippage assumptions, so the result should not be interpreted as executable strategy evidence.
Bootstrap generated 108 CI rows; 31 key return/hit-ratio rows are wide or overlap cautious thresholds. The confidence interval is wide, so the estimate is uncertain and should be interpreted cautiously.

## 2. Phase 6 objective

The objective is to test whether earlier VSEF findings are stable across alternative settings and whether observed differences are statistically meaningful rather than random variation.

## 3. Relation to Phase 0-5

- Phase 0 froze v1 governance, provider boundaries, and diagnostic-only constraints.
- Phase 1 standardized experiment execution and artifact layout.
- Phase 2 found that forecasting models do not consistently beat simple baselines on MAE/RMSE.
- Phase 3 found weak aggregate improvement from risk-aware ranking.
- Phase 4 supported regime dependence and the no-universal-best-model thesis.
- Phase 5 found strongest feature contribution evidence for rolling mean features, with mixed evidence elsewhere.

## 4. Robustness design

- Train/test window sensitivity uses `EXP-RB-001` and only exact local source artifacts for requested train/test splits.
- Universe sensitivity uses `EXP-RB-002` ticker groups over forecasting-core metrics.
- Cost/slippage sensitivity uses `EXP-RB-003` diagnostic basket period returns.
- Diebold-Mariano tests use `EXP-ST-001` aligned model/baseline forecast errors.
- Bootstrap confidence intervals use `EXP-ST-002` basket period returns with fixed seed reproducibility.

## 5. Window robustness results

Exact source settings with computed rows: 1. Missing exact settings: 2.
Ranking stability and metric variance cannot be claimed across the missing windows because those train/test reruns are not present as exact local artifacts.
| setting_id | ticker | horizon | model_name | model_type | metric_name | metric_value | rank | robustness_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| window_2023_train_2024_test_h1 | ACB | 1 | ets | model | directional_accuracy | 0.527132 | 1 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | lightgbm | model | directional_accuracy | 0.44186 | 2 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | moving_average_rule | baseline | directional_accuracy | 0.44186 | 3 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | sarimax | model | directional_accuracy | 0.44186 | 4 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | xgboost | model | directional_accuracy | 0.44186 | 5 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | persistence | baseline | directional_accuracy | 0.108527 | 6 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | zero_return | baseline | directional_accuracy | 0.108527 | 7 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | persistence | baseline | mae | 0.147597 | 1 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | zero_return | baseline | mae | 0.147597 | 2 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | moving_average_rule | baseline | mae | 0.234589 | 3 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | ets | model | mae | 0.550385 | 4 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | lightgbm | model | mae | 1.18948 | 5 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | xgboost | model | mae | 1.28213 | 6 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | sarimax | model | mae | 1.12475e+91 | 7 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | persistence | baseline | mape | 0.712166 | 1 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | zero_return | baseline | mape | 0.712166 | 2 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | moving_average_rule | baseline | mape | 1.12911 | 3 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | ets | model | mape | 2.64898 | 4 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | lightgbm | model | mape | 5.675 | 5 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |
| window_2023_train_2024_test_h1 | ACB | 1 | xgboost | model | mape | 6.12133 | 6 | computed_from_source_predictions:EXP-FC-003;exact_train_test_window_match |

## 6. Universe robustness results

Universe grouping preserves the Phase 2 baseline-competitiveness caveat where baselines remain first-ranked in MAE/RMSE rows. Where models lead, the result is still a grouped diagnostic result rather than investment value.
| universe_group | mae_rmse_winner_rows | baseline_winners | model_winners | baseline_competitiveness_note |
| --- | --- | --- | --- | --- |
| banking | 6 | 4 | 2 | Baseline competitiveness persists. |
| diversified_core | 30 | 28 | 2 | Baseline competitiveness persists. |
| industrial_materials | 12 | 12 | 0 | Baseline competitiveness persists. |
| large_cap_core | 18 | 18 | 0 | Baseline competitiveness persists. |

## 7. Cost/slippage sensitivity

Gross averages decline mechanically after bps deductions. The result is a retrospective diagnostic sensitivity table, not executable strategy evidence.
| total_cost_bps | rows | mean_gross_average | mean_net_average | nonpositive_net_rows | mean_net_hit_ratio |
| --- | --- | --- | --- | --- | --- |
| 0 | 18 | 0.00227948 | 0.00227948 | 6 | 0.531194 |
| 10 | 18 | 0.00227948 | 0.00127948 | 7 | 0.522237 |
| 20 | 18 | 0.00227948 | 0.000279478 | 11 | 0.497896 |
| 40 | 18 | 0.00227948 | -0.00172052 | 11 | 0.457551 |
| 100 | 18 | 0.00227948 | -0.00772052 | 18 | 0.330242 |

## 8. Diebold-Mariano test results

Comparisons: 464. Significant at 5%: 354. Significant at 10%: 360. Non-significant at 10%: 64.
The observed difference is not statistically significant under this test; it should not be treated as robust evidence of superiority.
| source_experiment | ticker | horizon | model_name | baseline_name | loss | dm_statistic | p_value | effect_size | warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-FC-001 | ACB | 1 | sarimax | persistence | squared |  |  |  | invalid_or_zero_loss_differential_variance |
| EXP-FC-001 | ACB | 1 | sarimax | persistence | absolute | 1.21417 | 0.226919 | 0.106486 |  |
| EXP-FC-001 | ACB | 1 | sarimax | zero_return | squared |  |  |  | invalid_or_zero_loss_differential_variance |
| EXP-FC-001 | ACB | 1 | sarimax | zero_return | absolute | 1.21417 | 0.226919 | 0.106486 |  |
| EXP-FC-001 | ACB | 1 | ets | persistence | squared | 9.67092 | 6.08769e-17 | 0.84817 |  |
| EXP-FC-001 | ACB | 1 | ets | persistence | absolute | 12.4849 | 6.86161e-24 | 1.09497 |  |
| EXP-FC-001 | ACB | 1 | ets | zero_return | squared | 9.67092 | 6.08769e-17 | 0.84817 |  |
| EXP-FC-001 | ACB | 1 | ets | zero_return | absolute | 12.4849 | 6.86161e-24 | 1.09497 |  |
| EXP-FC-001 | ACB | 1 | xgboost | persistence | squared | 15.2295 | 1.59926e-30 | 1.33568 |  |
| EXP-FC-001 | ACB | 1 | xgboost | persistence | absolute | 25.3391 | 1.02296e-51 | 2.22232 |  |
| EXP-FC-001 | ACB | 1 | xgboost | zero_return | squared | 15.2295 | 1.59926e-30 | 1.33568 |  |
| EXP-FC-001 | ACB | 1 | xgboost | zero_return | absolute | 25.3391 | 1.02296e-51 | 2.22232 |  |
| EXP-FC-001 | ACB | 1 | lightgbm | persistence | squared | 14.3007 | 2.61111e-28 | 1.25422 |  |
| EXP-FC-001 | ACB | 1 | lightgbm | persistence | absolute | 23.3716 | 5.1296e-48 | 2.04976 |  |
| EXP-FC-001 | ACB | 1 | lightgbm | zero_return | squared | 14.3007 | 2.61111e-28 | 1.25422 |  |
| EXP-FC-001 | ACB | 1 | lightgbm | zero_return | absolute | 23.3716 | 5.1296e-48 | 2.04976 |  |
| EXP-FC-001 | DGC | 1 | sarimax | persistence | squared | 1.50114 | 0.135783 | 0.131654 |  |
| EXP-FC-001 | DGC | 1 | sarimax | persistence | absolute | 1.48587 | 0.139773 | 0.130315 |  |
| EXP-FC-001 | DGC | 1 | sarimax | zero_return | squared | 1.50114 | 0.135783 | 0.131654 |  |
| EXP-FC-001 | DGC | 1 | sarimax | zero_return | absolute | 1.48587 | 0.139773 | 0.130315 |  |

## 9. Bootstrap confidence interval results

Return intervals that cross zero and hit-ratio intervals that cross 0.5 weaken superiority claims.
| metric_name | rows | mean_estimate | mean_ci_width | min_sample_size | warning_rows |
| --- | --- | --- | --- | --- | --- |
| average_realized_return | 18 | 0.00227948 | 0.00746006 | 112 | 0 |
| cvar_95 | 18 | -0.0484247 | 0.0339761 | 112 | 0 |
| hit_ratio | 18 | 0.531194 | 0.177742 | 112 | 0 |
| max_drawdown | 18 | -0.196246 | 0.255569 | 112 | 0 |
| return_volatility_proxy | 18 | 0.0824132 | 0.379216 | 112 | 0 |
| var_95 | 18 | -0.0284329 | 0.0258164 | 112 | 0 |

## 10. Statistical interpretation

Important differences are statistically supported only where p-values and interval evidence support them. Non-significant DM results remain exploratory.
Wide or overlapping bootstrap intervals indicate uncertainty and should constrain claims.
The prior baseline-competitiveness, weak aggregate risk-aware improvement, regime-dependence, and mixed feature-evidence conclusions remain best described as mixed diagnostic evidence rather than investment evidence.

## 11. Limitations

- Small sample sizes may affect statistical power.
- Overlapping forecast horizons can induce autocorrelation; DM tests use a Newey-West style adjustment but remain approximate.
- Forecast errors and returns may be non-normal.
- Bootstrap intervals assume the sampled period-return rows are representative of the local diagnostic artifact.
- Cost modeling uses simplified bps deductions from diagnostic period returns.
- Local artifact availability constrains reproducibility; missing exact train/test reruns are disclosed instead of imputed.

## 12. Acceptance criteria table

| criterion | exists |
| --- | --- |
| configs/experiments/EXP-RB-001.yaml | True |
| configs/experiments/EXP-RB-002.yaml | True |
| configs/experiments/EXP-RB-003.yaml | True |
| configs/experiments/EXP-ST-001.yaml | True |
| configs/experiments/EXP-ST-002.yaml | True |
| src/ml/statistics/dm_test.py | True |
| src/ml/statistics/bootstrap_eval.py | True |
| reports/robustness/window_robustness.csv | True |
| reports/robustness/universe_robustness.csv | True |
| reports/robustness/cost_sensitivity.csv | True |
| reports/robustness/dm_test_results.csv | True |
| reports/robustness/bootstrap_ci.csv | True |
| reports/robustness/ROBUSTNESS_AND_STATISTICAL_SIGNIFICANCE_REPORT.md | True |

## 13. Diagnostic-only disclaimer

All Phase 6 outputs are robustness and statistical research artifacts only. They are not BUY / SELL / HOLD advice, capital allocation guidance, broker execution instructions, portfolio recommendations, statistical proof of future profitability, or proof of guaranteed profitable trading.

## Source configs

| experiment_id | loaded |
| --- | --- |
| EXP-RB-001 | True |
| EXP-RB-002 | True |
| EXP-RB-003 | True |
| EXP-ST-001 | True |
| EXP-ST-002 | True |

## Summary artifacts

| section | item | value | note |
| --- | --- | --- | --- |
| window | settings_with_exact_source_rows | 1 | Only exact local train/test artifacts are used for window metrics. |
| window | settings_with_missing_exact_source | 2 | The result is sensitive to configuration choices and should be treated as exploratory. |
| window | mae_mean_performance | 1.07119e+89 | Mean MAE across exact-source window rows. |
| window | mae_worst_case_performance | 1.12475e+91 | Worst-case MAE across exact-source window rows; lower is better. |
| window | mae_metric_variance | 1.19336e+180 | Variance across available exact-source MAE rows, not across missing window reruns. |
| window | mae_rank_stability_std | 2 | Rank spread across available exact-source MAE rows; full window stability needs missing reruns. |
| universe | banking_baseline_winners_mae_rmse | 4 | 4 of 6 MAE/RMSE first-rank rows are baselines. |
| universe | banking_mae_mean_performance | 5.35597e+89 | Mean MAE across this universe group. |
| universe | banking_mae_worst_case_performance | 1.12475e+91 | Worst-case MAE across this universe group; lower is better. |
| universe | banking_mae_metric_variance | 5.73729e+180 | Metric variance across ticker/horizon/model rows in this universe group. |
| universe | diversified_core_baseline_winners_mae_rmse | 28 | 28 of 30 MAE/RMSE first-rank rows are baselines. |
| universe | diversified_core_mae_mean_performance | 1.07119e+89 | Mean MAE across this universe group. |
| universe | diversified_core_mae_worst_case_performance | 1.12475e+91 | Worst-case MAE across this universe group; lower is better. |
| universe | diversified_core_mae_metric_variance | 1.19336e+180 | Metric variance across ticker/horizon/model rows in this universe group. |
| universe | industrial_materials_baseline_winners_mae_rmse | 12 | 12 of 12 MAE/RMSE first-rank rows are baselines. |
| universe | industrial_materials_mae_mean_performance | 3.1658 | Mean MAE across this universe group. |
| universe | industrial_materials_mae_worst_case_performance | 21.5742 | Worst-case MAE across this universe group; lower is better. |
| universe | industrial_materials_mae_metric_variance | 25.1342 | Metric variance across ticker/horizon/model rows in this universe group. |
| universe | large_cap_core_baseline_winners_mae_rmse | 18 | 18 of 18 MAE/RMSE first-rank rows are baselines. |
| universe | large_cap_core_mae_mean_performance | 6.26152e+24 | Mean MAE across this universe group. |
| universe | large_cap_core_mae_worst_case_performance | 3.94378e+26 | Worst-case MAE across this universe group; lower is better. |
| universe | large_cap_core_mae_metric_variance | 2.42959e+51 | Metric variance across ticker/horizon/model rows in this universe group. |
| cost | cost_0bps_nonpositive_net_average_rows | 6 | The diagnostic edge weakens or disappears under cost/slippage assumptions, so the result should not be interpreted as executable strategy evidence. |
| cost | cost_10bps_nonpositive_net_average_rows | 11 | The diagnostic edge weakens or disappears under cost/slippage assumptions, so the result should not be interpreted as executable strategy evidence. |
| cost | cost_20bps_nonpositive_net_average_rows | 11 | The diagnostic edge weakens or disappears under cost/slippage assumptions, so the result should not be interpreted as executable strategy evidence. |
| cost | cost_50bps_nonpositive_net_average_rows | 18 | The diagnostic edge weakens or disappears under cost/slippage assumptions, so the result should not be interpreted as executable strategy evidence. |
| cost | cost_5bps_nonpositive_net_average_rows | 7 | The diagnostic edge weakens or disappears under cost/slippage assumptions, so the result should not be interpreted as executable strategy evidence. |

## Statistical significance summary

| section | item | value | note |
| --- | --- | --- | --- |
| dm_test | comparison_count | 464 | Model/baseline/loss comparisons attempted. |
| dm_test | significant_05_count | 354 | Significant at p < 0.05; interpret with sample and multiple-comparison caution. |
| dm_test | significant_10_count | 360 | Significant at p < 0.10; weaker evidence than 5%. |
| dm_test | non_significant_10_count | 64 | The observed difference is not statistically significant under this test; it should not be treated as robust evidence of superiority. |
| dm_test | warning_count | 40 | Warnings include missing aligned rows, small samples, or invalid variance. |
| bootstrap | ci_rows | 108 | Bootstrap intervals computed from basket period returns. |
| bootstrap | wide_or_overlapping_key_ci_rows | 31 | The confidence interval is wide, so the estimate is uncertain and should be interpreted cautiously. |

## Effect size summary

| model_name | baseline_name | loss | comparison_count | mean_effect_size | median_effect_size | mean_loss_diff | median_p_value | significant_05_count | significant_10_count | warning_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ets | persistence | absolute | 29 | 1.58704 | 1.16893 | 6.27904 | 6.86161e-24 | 29 | 29 | 0 |
| ets | persistence | squared | 29 | 1.2383 | 0.888426 | 132.108 | 1.74092e-16 | 29 | 29 | 0 |
| ets | zero_return | absolute | 29 | 1.58704 | 1.16893 | 6.27904 | 6.86161e-24 | 29 | 29 | 0 |
| ets | zero_return | squared | 29 | 1.2383 | 0.888426 | 132.108 | 1.74092e-16 | 29 | 29 | 0 |
| lightgbm | persistence | absolute | 29 | 1.62059 | 1.66817 | 5.19127 | 6.73155e-25 | 28 | 28 | 0 |
| lightgbm | persistence | squared | 29 | 1.08026 | 1.1734 | 94.663 | 8.14252e-16 | 27 | 27 | 0 |
| lightgbm | zero_return | absolute | 29 | 1.62059 | 1.66817 | 5.19127 | 6.73155e-25 | 28 | 28 | 0 |
| lightgbm | zero_return | squared | 29 | 1.08026 | 1.1734 | 94.663 | 8.14252e-16 | 27 | 27 | 0 |
| sarimax | persistence | absolute | 29 | 0.124979 | 0.118401 | 1.12475e+90 | 0.226919 | 6 | 7 | 9 |
| sarimax | persistence | squared | 29 | 0.120955 | 0.130371 | 1.11965e+183 | 0.177053 | 3 | 3 | 11 |
| sarimax | zero_return | absolute | 29 | 0.124979 | 0.118401 | 1.12475e+90 | 0.226919 | 6 | 7 | 9 |
| sarimax | zero_return | squared | 29 | 0.120955 | 0.130371 | 1.11965e+183 | 0.177053 | 3 | 3 | 11 |
| xgboost | persistence | absolute | 29 | 1.72626 | 1.7218 | 5.42234 | 7.45043e-27 | 28 | 29 | 0 |
| xgboost | persistence | squared | 29 | 1.13834 | 1.22984 | 98.2496 | 4.88575e-19 | 27 | 28 | 0 |
| xgboost | zero_return | absolute | 29 | 1.72626 | 1.7218 | 5.42234 | 7.45043e-27 | 28 | 29 | 0 |
| xgboost | zero_return | squared | 29 | 1.13834 | 1.22984 | 98.2496 | 4.88575e-19 | 27 | 28 | 0 |

## Charts

- generated:window_robustness_sample_size.png
- generated:universe_robustness_baseline_win_rate.png
- generated:cost_sensitivity_net_average.png
- generated:dm_test_pvalues_histogram.png
- generated:bootstrap_ci_key_metrics.png
- generated:effect_size_mean_by_comparison.png
