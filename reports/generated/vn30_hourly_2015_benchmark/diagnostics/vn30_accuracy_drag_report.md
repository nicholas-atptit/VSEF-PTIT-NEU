# VN30 Hourly 2015 - Accuracy Drag Diagnosis Report

- Generated at UTC: `2026-05-16T14:53:19+00:00`.
- Source: `outputs/vn30_hourly_2015_jan2025_benchmark/hourly/predicted_vs_actual.csv`.
- Global accuracy: 51.34%.
- Total incorrect predictions: 37807.

## Which Tickers Drag Accuracy Below 60%?

### Worst 10 Tickers by Accuracy

| ticker | n_obs | accuracy | correct | incorrect | incorrect_contribution_to_global |
| --- | --- | --- | --- | --- | --- |
| FPT | 1868 | 37.47% | 700 | 1168 | 3.09% |
| VIC | 1872 | 42.68% | 799 | 1073 | 2.84% |
| GAS | 1992 | 45.23% | 901 | 1091 | 2.89% |
| SHB | 4384 | 45.67% | 2002 | 2382 | 6.30% |
| VNM | 1976 | 45.70% | 903 | 1073 | 2.84% |
| SSI | 1804 | 47.67% | 860 | 944 | 2.50% |
| PLX | 2616 | 48.93% | 1280 | 1336 | 3.53% |
| VJC | 2644 | 49.81% | 1317 | 1327 | 3.51% |
| VRE | 2644 | 49.96% | 1321 | 1323 | 3.50% |
| VHM | 2636 | 50.30% | 1326 | 1310 | 3.46% |

### Best 10 Tickers by Accuracy

| ticker | n_obs | accuracy | correct | incorrect | incorrect_contribution_to_global |
| --- | --- | --- | --- | --- | --- |
| SSB | 2476 | 58.48% | 1448 | 1028 | 2.72% |
| MBB | 2580 | 56.86% | 1467 | 1113 | 2.94% |
| MSN | 2600 | 55.46% | 1442 | 1158 | 3.06% |
| BCM | 5004 | 54.64% | 2734 | 2270 | 6.00% |
| LPB | 4080 | 54.61% | 2228 | 1852 | 4.90% |
| BID | 1972 | 54.36% | 1072 | 900 | 2.38% |
| BVH | 1976 | 54.30% | 1073 | 903 | 2.39% |
| MWG | 2624 | 54.15% | 1421 | 1203 | 3.18% |
| HDB | 1696 | 53.83% | 913 | 783 | 2.07% |
| ACB | 2552 | 53.45% | 1364 | 1188 | 3.14% |

### Top 5 Tickers by Incorrect Contribution to Global

| ticker | n_obs | accuracy | incorrect | incorrect_contribution_to_global |
| --- | --- | --- | --- | --- |
| SHB | 4384 | 45.67% | 2382 | 6.30% |
| BCM | 5004 | 54.64% | 2270 | 6.00% |
| VIB | 4128 | 51.87% | 1987 | 5.26% |
| HPG | 4032 | 50.92% | 1979 | 5.23% |
| VPB | 3956 | 51.97% | 1900 | 5.03% |

## Which Horizons Are Strongest?

### Model/Horizon Accuracy (aggregate)

| model | horizon | n_obs | accuracy | incorrect_contribution_to_global | class_0_ratio |
| --- | --- | --- | --- | --- | --- |
| random_forest | 0 | 19423 | 51.95% | 24.68% | 0.4894 |
| lightgbm | 0 | 19423 | 51.93% | 24.69% | 0.4894 |
| xgboost | 0 | 19423 | 51.77% | 24.78% | 0.4894 |
| stacking | 0 | 19423 | 49.69% | 25.84% | 0.4894 |

## Is h=20 Consistently Strongest?

**YES.** h=20 is consistently the strongest horizon across all base models:

- lightgbm h=20: 54.58% (4599 obs)
- random_forest h=20: 54.03% (4599 obs)
- xgboost h=20: 54.42% (4599 obs)

## Does Model Ensemble/Stacking Help or Hurt?

### Stacking vs Base Models by Horizon

| horizon | stacking | lightgbm | xgboost | random_forest | best_base | stacking_vs_best |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 50.76% | 51.05% | 51.13% | 51.89% | 51.89% | -1.12% |
| 4 | 50.53% | 50.89% | 50.61% | 51.39% | 51.39% | -0.86% |
| 8 | 48.84% | 51.40% | 51.10% | 50.65% | 51.40% | -2.56% |
| 20 | 48.55% | 54.58% | 54.42% | 54.03% | 54.58% | -6.02% |

**Stacking hurts performance** relative to the best base model across most horizons.
The meta-learner appears to dilute the signal from LightGBM/XGBoost at h=20.

## Model Disagreement Analysis

- Full model agreement rate: 52.10%.
- Disagreement rate: 47.90%.

## Do Errors Cluster in Specific Tickers/Periods?

- Model disagreement may indicate uncertain predictions.
- When all models agree: 10119 instances.
- When models disagree: 9304 instances.

## Whether a Filtered Deployment-Like Candidate Exists

Based on this diagnostic analysis:

- **Best single model/horizon:** lightgbm h=20 at 54.58%.
- This is below 60% globally.
- A filtered deployment candidate would require confidence thresholding or ticker subsetting.
- Any such candidate must be labeled as conditional/exploratory, not global.

## Summary of Accuracy Drag

- Global accuracy: 51.34% (target: 60%).
- Gap to target: 8.66%.
- Primary drags: all models hover near 51% globally.
- h=20 is the strongest horizon (~54-55% for LightGBM/XGBoost, ~54% for RF).
- Stacking underperforms base models at h=20.
- No single ticker, regime, or time slice lifts the global average above 60%.
- Confidence filtering can produce >60% slices but with reduced coverage.

## Boundary

- No trading-readiness, profitability, or live deployment claim is made.
- This is a post-hoc diagnostic analysis of existing benchmark outputs.
- No prediction labels were edited. No future data was leaked.
