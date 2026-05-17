# Table 3: Global Benchmark Results

| frequency | overall_accuracy | n_predictions | best_model_accuracy | best_model | best_model_horizon | best_baseline_accuracy | best_model_delta_vs_best_baseline | passed_60pct_global | evaluated_tickers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| daily | 0.531873 | 26104 | 0.569724 | random_forest | 20 | 0.545226 | 0.0244975 | false | ANV, BCM, BID, BMP, BVH, BWE, CII |
| hourly | 0.512857 | 127944 | 0.555934 | stacking | 1 | 0.505447 | 0.0504874 | false | ANV, BCM, BID, BMP, BVH, BWE, CII |

## Note

- Source artifact: daily/hourly benchmark_summary.json; daily/hourly classification_accuracy_summary.csv.
- Claim supported: The official benchmark produced nonzero predictions but did not pass the global 60% threshold.
- Limitation: Single official 2025 window with seven evaluated tickers.
- Status: ready.
