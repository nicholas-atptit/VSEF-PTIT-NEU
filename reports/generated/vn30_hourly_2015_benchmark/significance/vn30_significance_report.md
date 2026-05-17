# VN30 Hourly 2015 Significance Diagnostics

## Source

- Prediction artifact: `outputs/vn30_hourly_2015_jan2025_benchmark/hourly/predicted_vs_actual.csv`.
- Null hypothesis: directional accuracy is 50%.
- Test: one-sided binomial/sign test by model and horizon.
- Multiple-testing limitation: 16 model/horizon rows are tested, so unadjusted p-values are diagnostic only.

## Results

| model | horizon | observations | accuracy | p_value | statistically_above_50pct | bonferroni_significant |
| --- | --- | --- | --- | --- | --- | --- |
| lightgbm | 20 | 4599 | 54.58% | 2.86944e-10 | yes | yes |
| xgboost | 20 | 4599 | 54.42% | 1.04544e-09 | yes | yes |
| random_forest | 20 | 4599 | 54.03% | 2.39687e-08 | yes | yes |
| random_forest | 1 | 4907 | 51.89% | 0.00430758 | yes | no |
| random_forest | 4 | 4993 | 51.39% | 0.0254054 | yes | no |
| lightgbm | 8 | 4924 | 51.40% | 0.0254418 | yes | no |
| xgboost | 1 | 4907 | 51.13% | 0.0581673 | no | no |
| xgboost | 8 | 4924 | 51.10% | 0.063645 | no | no |
| lightgbm | 1 | 4907 | 51.05% | 0.0726782 | no | no |
| lightgbm | 4 | 4993 | 50.89% | 0.106494 | no | no |
| stacking | 1 | 4907 | 50.76% | 0.145395 | no | no |
| random_forest | 8 | 4924 | 50.65% | 0.184646 | no | no |
| xgboost | 4 | 4993 | 50.61% | 0.197908 | no | no |
| stacking | 4 | 4993 | 50.53% | 0.230896 | no | no |
| stacking | 8 | 4924 | 48.84% | 0.949382 | no | no |
| stacking | 20 | 4599 | 48.55% | 0.975925 | no | no |

## Boundary

- Unadjusted rows statistically above 50% at alpha=0.05: 6.
- Bonferroni-adjusted rows statistically above 50%: 3.
- Statistical evidence against a 50% null is not a trading-readiness or profitability claim.
