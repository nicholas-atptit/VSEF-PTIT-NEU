# Table 2: Model and Baseline List

| type | name | frequency | horizons | role |
| --- | --- | --- | --- | --- |
| model | lightgbm | daily/hourly | daily=[1, 5, 10, 20]; hourly=[1, 4, 8, 20] | machine-learning classifier in official benchmark |
| model | random_forest | daily/hourly | daily=[1, 5, 10, 20]; hourly=[1, 4, 8, 20] | machine-learning classifier in official benchmark |
| model | stacking | daily/hourly | daily=[1, 5, 10, 20]; hourly=[1, 4, 8, 20] | machine-learning classifier in official benchmark |
| model | xgboost | daily/hourly | daily=[1, 5, 10, 20]; hourly=[1, 4, 8, 20] | machine-learning classifier in official benchmark |
| baseline | always_up | daily/hourly | same evaluated horizons where baseline rows exist | directional comparison baseline |
| baseline | moving_average_signal | daily/hourly | same evaluated horizons where baseline rows exist | directional comparison baseline |
| baseline | previous_direction | daily/hourly | same evaluated horizons where baseline rows exist | directional comparison baseline |
| baseline | random_seeded_direction | daily/hourly | same evaluated horizons where baseline rows exist | directional comparison baseline |

## Note

- Source artifact: run_config.json; daily/hourly baseline_summary.csv.
- Claim supported: The study compares LightGBM, XGBoost, random forest, stacking, and simple directional baselines.
- Limitation: No new model families are introduced by this artifact pack.
- Status: ready.
