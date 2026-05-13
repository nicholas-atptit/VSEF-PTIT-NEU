# VN100 Ticker Concentration Summary

## Source

- Official artifact directory: `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff`.
- Prediction inputs: `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/daily/predicted_vs_actual.csv` and `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/hourly/predicted_vs_actual.csv`.
- Selected confidence candidate source: `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff/confidence_threshold_sweep_summary.csv`.
- Contribution share is the share of prediction rows within each diagnostic scope.
- Positive excess share is the share of correct predictions above a 50% null, counted only where a ticker has positive excess.

## Evaluated Tickers

ANV, BCM, BID, BMP, BVH, BWE, CII

## Scope Concentration

| scope | tickers | predictions | accuracy | top ticker | top1 prediction share | top3 prediction share | top positive-edge ticker | top1 positive-edge share | top3 positive-edge share | prediction-count assessment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_regime_daily_lightgbm_h20_bear | 7 | 444 | 69.59% | BVH | 19.37% | 51.13% | CII | 27.01% | 58.05% | low |
| best_regime_hourly_stacking_h1_high_volatility | 7 | 2012 | 56.51% | BID | 16.30% | 46.92% | BCM | 24.43% | 59.16% | low |
| global_daily | 7 | 26104 | 53.19% | BMP | 14.47% | 43.15% | BID | 31.46% | 75.20% | low |
| global_hourly | 7 | 127944 | 51.29% | BMP | 14.73% | 43.56% | BID | 38.14% | 74.95% | low |
| selected_confidence_hourly_stacking_h1_t0.57 | 5 | 2297 | 60.03% | BID | 31.17% | 79.49% | BCM | 29.93% | 77.66% | high |

## Selected Finding Assessment

- best_regime_daily_lightgbm_h20_bear: prediction-count concentration is low (top ticker BVH at 19.37%; top three at 51.13%).
- best_regime_hourly_stacking_h1_high_volatility: prediction-count concentration is low (top ticker BID at 16.30%; top three at 46.92%).
- selected_confidence_hourly_stacking_h1_t0.57: prediction-count concentration is high (top ticker BID at 31.17%; top three at 79.49%).

## Selected Scope Ticker Rows

| scope | ticker | predictions | accuracy | contribution share | excess correct vs 50% | positive excess share |
| --- | --- | --- | --- | --- | --- | --- |
| best_regime_daily_lightgbm_h20_bear | BVH | 86 | 63.95% | 19.37% | 12.000000 | 13.79% |
| best_regime_daily_lightgbm_h20_bear | ANV | 74 | 58.11% | 16.67% | 6.000000 | 6.90% |
| best_regime_daily_lightgbm_h20_bear | CII | 67 | 85.07% | 15.09% | 23.500000 | 27.01% |
| best_regime_daily_lightgbm_h20_bear | BMP | 63 | 71.43% | 14.19% | 13.500000 | 15.52% |
| best_regime_daily_lightgbm_h20_bear | BID | 60 | 70.00% | 13.51% | 12.000000 | 13.79% |
| best_regime_daily_lightgbm_h20_bear | BWE | 51 | 62.75% | 11.49% | 6.500000 | 7.47% |
| best_regime_daily_lightgbm_h20_bear | BCM | 43 | 81.40% | 9.68% | 13.500000 | 15.52% |
| best_regime_hourly_stacking_h1_high_volatility | BID | 328 | 56.10% | 16.30% | 20.000000 | 15.27% |
| best_regime_hourly_stacking_h1_high_volatility | BMP | 320 | 55.31% | 15.90% | 17.000000 | 12.98% |
| best_regime_hourly_stacking_h1_high_volatility | BCM | 296 | 60.81% | 14.71% | 32.000000 | 24.43% |
| best_regime_hourly_stacking_h1_high_volatility | BVH | 289 | 55.36% | 14.36% | 15.500000 | 11.83% |
| best_regime_hourly_stacking_h1_high_volatility | CII | 285 | 57.89% | 14.17% | 22.500000 | 17.18% |
| best_regime_hourly_stacking_h1_high_volatility | BWE | 266 | 58.65% | 13.22% | 23.000000 | 17.56% |
| best_regime_hourly_stacking_h1_high_volatility | ANV | 228 | 50.44% | 11.33% | 1.000000 | 0.76% |
| selected_confidence_hourly_stacking_h1_t0.57 | BID | 716 | 57.96% | 31.17% | 57.000000 | 24.73% |
| selected_confidence_hourly_stacking_h1_t0.57 | CII | 660 | 58.03% | 28.73% | 53.000000 | 22.99% |
| selected_confidence_hourly_stacking_h1_t0.57 | BCM | 450 | 65.33% | 19.59% | 69.000000 | 29.93% |
| selected_confidence_hourly_stacking_h1_t0.57 | ANV | 334 | 62.57% | 14.54% | 42.000000 | 18.22% |
| selected_confidence_hourly_stacking_h1_t0.57 | BMP | 137 | 56.93% | 5.96% | 9.500000 | 4.12% |

## Interpretation Boundary

These diagnostics address concentration of the official prediction rows and positive directional edge. They do not establish trading profitability, cost-adjusted returns, or stability beyond the official artifact window.
