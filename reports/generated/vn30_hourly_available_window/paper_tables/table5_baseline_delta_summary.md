# Table 5: Baseline delta summary

| frequency | model | horizon | baseline | model_accuracy | baseline_accuracy | accuracy_delta | model_n_obs | baseline_n_obs | model_better_than_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hourly | random_forest | 1 | always_up | 56.11% | 44.80% | 11.31% | 4580 | 4580 | True |
| hourly | stacking | 1 | always_up | 55.24% | 44.80% | 10.44% | 4580 | 4580 | True |
| hourly | xgboost | 1 | always_up | 54.56% | 44.80% | 9.76% | 4580 | 4580 | True |
| hourly | lightgbm | 1 | always_up | 54.39% | 44.80% | 9.59% | 4580 | 4580 | True |
| hourly | random_forest | 1 | previous_direction | 56.11% | 49.13% | 6.99% | 4580 | 4580 | True |
| hourly | random_forest | 1 | moving_average_signal | 56.11% | 49.85% | 6.27% | 4580 | 4580 | True |
| hourly | random_forest | 1 | random_seeded_direction | 56.11% | 49.85% | 6.27% | 4580 | 4580 | True |
| hourly | stacking | 1 | previous_direction | 55.24% | 49.13% | 6.11% | 4580 | 4580 | True |
| hourly | xgboost | 8 | moving_average_signal | 52.45% | 46.85% | 5.60% | 4879 | 4879 | True |
| hourly | xgboost | 1 | previous_direction | 54.56% | 49.13% | 5.44% | 4580 | 4580 | True |
| hourly | stacking | 1 | moving_average_signal | 55.24% | 49.85% | 5.39% | 4580 | 4580 | True |
| hourly | stacking | 1 | random_seeded_direction | 55.24% | 49.85% | 5.39% | 4580 | 4580 | True |
| hourly | lightgbm | 1 | previous_direction | 54.39% | 49.13% | 5.26% | 4580 | 4580 | True |
| hourly | xgboost | 4 | moving_average_signal | 52.60% | 47.87% | 4.73% | 4888 | 4888 | True |
| hourly | xgboost | 1 | random_seeded_direction | 54.56% | 49.85% | 4.72% | 4580 | 4580 | True |
| hourly | xgboost | 1 | moving_average_signal | 54.56% | 49.85% | 4.72% | 4580 | 4580 | True |
| hourly | random_forest | 8 | moving_average_signal | 51.40% | 46.85% | 4.55% | 4879 | 4879 | True |
| hourly | lightgbm | 1 | moving_average_signal | 54.39% | 49.85% | 4.54% | 4580 | 4580 | True |
| hourly | lightgbm | 1 | random_seeded_direction | 54.39% | 49.85% | 4.54% | 4580 | 4580 | True |
| hourly | lightgbm | 4 | moving_average_signal | 52.37% | 47.87% | 4.50% | 4888 | 4888 | True |

## Note

Hourly model-versus-baseline directional-accuracy deltas.
