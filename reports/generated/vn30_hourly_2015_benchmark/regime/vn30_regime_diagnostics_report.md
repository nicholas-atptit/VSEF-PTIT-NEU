# VN30 Hourly 2015 Regime Diagnostics

## Source

- Prediction artifact: `outputs/vn30_hourly_2015_jan2025_benchmark/hourly/predicted_vs_actual.csv`.
- Frequency: hourly only.
- Regime source: existing labels already emitted by the base benchmark predictions.
- No new regime labels were created or fabricated.

## Top Regime Diagnostic Rows

| regime_dimension | regime | model | horizon | observations | accuracy | passed_60pct |
| --- | --- | --- | --- | --- | --- | --- |
| return_regime | bear | xgboost | 20 | 1539 | 58.93% | no |
| volatility_regime | low_volatility | xgboost | 20 | 520 | 58.65% | no |
| return_regime | bear | lightgbm | 20 | 1539 | 58.35% | no |
| volatility_regime | low_volatility | lightgbm | 20 | 520 | 58.27% | no |
| volatility_regime | low_volatility | stacking | 4 | 745 | 57.85% | no |
| volatility_regime | low_volatility | random_forest | 8 | 685 | 57.66% | no |
| return_regime | sideways | xgboost | 20 | 1231 | 57.60% | no |
| volatility_regime | low_volatility | stacking | 8 | 685 | 56.79% | no |
| volatility_regime | low_volatility | random_forest | 20 | 520 | 56.73% | no |
| return_regime | sideways | lightgbm | 20 | 1231 | 56.21% | no |
| return_regime | sideways | random_forest | 20 | 1231 | 56.05% | no |
| return_regime | bear | random_forest | 20 | 1539 | 56.01% | no |
| volatility_regime | low_volatility | stacking | 20 | 520 | 55.00% | no |
| volatility_regime | high_volatility | random_forest | 20 | 2591 | 54.19% | no |
| volatility_regime | low_volatility | xgboost | 8 | 685 | 54.16% | no |
| return_regime | bear | lightgbm | 8 | 1573 | 53.97% | no |
| volatility_regime | low_volatility | lightgbm | 8 | 685 | 53.72% | no |
| volatility_regime | high_volatility | lightgbm | 20 | 2591 | 53.42% | no |
| volatility_regime | low_volatility | random_forest | 4 | 745 | 53.29% | no |
| volatility_regime | high_volatility | xgboost | 20 | 2591 | 53.11% | no |

## Boundary

- Regime diagnostic rows generated: 80.
- Rows at or above 60% accuracy: 0.
- These are benchmark-internal regime slices, not an independently validated market-regime system.
- No trading-readiness or profitability claim is made.
