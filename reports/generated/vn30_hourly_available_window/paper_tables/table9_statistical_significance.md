# Table 9: Statistical significance

| frequency | model | horizon | n_obs | accuracy | null_accuracy | binomial_p_value | bootstrap_ci_low | bootstrap_ci_high | significant_at_5pct | significant_at_10pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hourly | random_forest | 1 | 4580 | 56.11% | 50.00% | 0.00% | 54.65% | 57.51% | True | True |
| hourly | stacking | 1 | 4580 | 55.24% | 50.00% | 0.00% | 53.84% | 56.68% | True | True |
| hourly | xgboost | 1 | 4580 | 54.56% | 50.00% | 0.00% | 53.10% | 55.94% | True | True |
| hourly | lightgbm | 1 | 4580 | 54.39% | 50.00% | 0.00% | 52.82% | 55.79% | True | True |
| hourly | xgboost | 4 | 4888 | 52.60% | 50.00% | 0.01% | 51.15% | 54.03% | True | True |
| hourly | xgboost | 8 | 4879 | 52.45% | 50.00% | 0.03% | 51.04% | 53.82% | True | True |
| hourly | lightgbm | 4 | 4888 | 52.37% | 50.00% | 0.05% | 50.94% | 53.87% | True | True |
| hourly | random_forest | 4 | 4888 | 51.92% | 50.00% | 0.37% | 50.45% | 53.36% | True | True |
| hourly | random_forest | 8 | 4879 | 51.40% | 50.00% | 2.58% | 50.05% | 52.84% | True | True |
| hourly | stacking | 4 | 4888 | 51.19% | 50.00% | 5.00% | 49.75% | 52.68% | True | True |
| hourly | lightgbm | 8 | 4879 | 51.06% | 50.00% | 7.21% | 49.56% | 52.57% | False | True |
| hourly | stacking | 8 | 4879 | 48.37% | 50.00% | 98.90% | 47.02% | 49.83% | False | False |
| hourly | stacking | 20 | 4592 | 47.65% | 50.00% | 99.93% | 46.25% | 49.06% | False | False |
| hourly | lightgbm | 20 | 4592 | 47.58% | 50.00% | 99.95% | 46.17% | 48.98% | False | False |
| hourly | xgboost | 20 | 4592 | 46.49% | 50.00% | 100.00% | 45.08% | 48.02% | False | False |
| hourly | random_forest | 20 | 4592 | 45.51% | 50.00% | 100.00% | 44.10% | 46.89% | False | False |

## Note

Statistical significance diagnostics for available-window hourly predictions.
