# Phase 6 Robustness and Statistical Significance

This document indexes the Phase 6 configs and generated review artifacts.

## Configs

- `configs/experiments/EXP-RB-001.yaml`
- `configs/experiments/EXP-RB-002.yaml`
- `configs/experiments/EXP-RB-003.yaml`
- `configs/experiments/EXP-ST-001.yaml`
- `configs/experiments/EXP-ST-002.yaml`

## Report artifacts

- `reports\robustness\ROBUSTNESS_AND_STATISTICAL_SIGNIFICANCE_REPORT.md`
- `reports\robustness\window_robustness.csv`
- `reports\robustness\universe_robustness.csv`
- `reports\robustness\cost_sensitivity.csv`
- `reports\robustness\dm_test_results.csv`
- `reports\robustness\bootstrap_ci.csv`

## Robustness summary

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

## Statistical summary

| section | item | value | note |
| --- | --- | --- | --- |
| dm_test | comparison_count | 464 | Model/baseline/loss comparisons attempted. |
| dm_test | significant_05_count | 354 | Significant at p < 0.05; interpret with sample and multiple-comparison caution. |
| dm_test | significant_10_count | 360 | Significant at p < 0.10; weaker evidence than 5%. |
| dm_test | non_significant_10_count | 64 | The observed difference is not statistically significant under this test; it should not be treated as robust evidence of superiority. |
| dm_test | warning_count | 40 | Warnings include missing aligned rows, small samples, or invalid variance. |
| bootstrap | ci_rows | 108 | Bootstrap intervals computed from basket period returns. |
| bootstrap | wide_or_overlapping_key_ci_rows | 31 | The confidence interval is wide, so the estimate is uncertain and should be interpreted cautiously. |

All Phase 6 outputs are robustness and statistical research artifacts only. They are not BUY / SELL / HOLD advice, capital allocation guidance, broker execution instructions, portfolio recommendations, statistical proof of future profitability, or proof of guaranteed profitable trading.
