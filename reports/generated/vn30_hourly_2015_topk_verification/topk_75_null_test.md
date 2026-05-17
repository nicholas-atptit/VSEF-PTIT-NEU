# VN30 Hourly 2015 - Top-K 75% Null/Permutation Test Report

## Observed Result
- Observed precision@10: 74.46%
- Horizon: h=120
- Model: LightGBM
- Events: 216

## Null Test 1: Random Top-K Baseline
- Random mean: 77.36%
- Random std: 0.95%
- Random min: 74.29%
- Random max: 80.36%
- Empirical p-value: 0.999000
- Significantly above random: NO

## Null Test 2: Score Shuffle Permutation
- Score shuffle mean: 77.38%
- Score shuffle std: 0.89%
- Empirical p-value: 0.998000
- Significantly above shuffled scores: NO

## Null Test 3: Label Shuffle Permutation
- Label shuffle mean: 77.42%
- Label shuffle std: 0.91%
- Empirical p-value: 1.000000
- Significantly above shuffled labels: NO

## Conclusion
- The observed precision@10 is NOT statistically significantly above random selection.
- The model scores contain NO predictive signal beyond random.
- The result is NOT robust to label permutation.

## Interpretation
The result may be due to random chance.
The model scores may not contain meaningful signal.
