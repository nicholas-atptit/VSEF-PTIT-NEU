# VN30 Hourly Available-Window Ex-Ante Regime Validation Report

## Source

- Prediction artifact: `outputs/vn30_hourly_available_window_benchmark/hourly/predicted_vs_actual.csv`.
- Frequency: hourly only.
- Study: VN30 hourly available-window.
- Ex-ante proxy rule: labels use shifted prior actual returns only.
- Rolling window: 20; minimum prior observations: 5.

## Aggregate Regime Diagnostics

| regime_source | model | horizon | regime | observation_count | accuracy | passed_60pct | passed_63pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exante_proxy | lightgbm | 1 | insufficient_history | 135 | 81.48% | True | True |
| exante_proxy | lightgbm | 1 | sideways | 4445 | 53.57% | False | False |
| exante_proxy | lightgbm | 4 | bear | 54 | 50.00% | False | False |
| exante_proxy | lightgbm | 4 | bull | 40 | 37.50% | False | False |
| exante_proxy | lightgbm | 4 | insufficient_history | 135 | 63.70% | True | True |
| exante_proxy | lightgbm | 4 | sideways | 4659 | 52.20% | False | False |
| exante_proxy | lightgbm | 8 | bear | 133 | 46.62% | False | False |
| exante_proxy | lightgbm | 8 | bull | 281 | 45.20% | False | False |
| exante_proxy | lightgbm | 8 | insufficient_history | 135 | 60.00% | True | False |
| exante_proxy | lightgbm | 8 | sideways | 4330 | 51.29% | False | False |
| exante_proxy | lightgbm | 20 | bear | 385 | 44.68% | False | False |
| exante_proxy | lightgbm | 20 | bull | 779 | 38.13% | False | False |
| exante_proxy | lightgbm | 20 | insufficient_history | 135 | 48.15% | False | False |
| exante_proxy | lightgbm | 20 | sideways | 3293 | 50.14% | False | False |
| exante_proxy | random_forest | 1 | insufficient_history | 135 | 87.41% | True | True |
| exante_proxy | random_forest | 1 | sideways | 4445 | 55.16% | False | False |
| exante_proxy | random_forest | 4 | bear | 54 | 50.00% | False | False |
| exante_proxy | random_forest | 4 | bull | 40 | 37.50% | False | False |
| exante_proxy | random_forest | 4 | insufficient_history | 135 | 72.59% | True | True |
| exante_proxy | random_forest | 4 | sideways | 4659 | 51.47% | False | False |
| exante_proxy | random_forest | 8 | bear | 133 | 51.88% | False | False |
| exante_proxy | random_forest | 8 | bull | 281 | 45.20% | False | False |
| exante_proxy | random_forest | 8 | insufficient_history | 135 | 61.48% | True | False |
| exante_proxy | random_forest | 8 | sideways | 4330 | 51.48% | False | False |
| exante_proxy | random_forest | 20 | bear | 385 | 44.16% | False | False |
| exante_proxy | random_forest | 20 | bull | 779 | 38.90% | False | False |
| exante_proxy | random_forest | 20 | insufficient_history | 135 | 51.85% | False | False |
| exante_proxy | random_forest | 20 | sideways | 3293 | 46.98% | False | False |
| exante_proxy | stacking | 1 | insufficient_history | 135 | 63.70% | True | True |
| exante_proxy | stacking | 1 | sideways | 4445 | 54.98% | False | False |
| exante_proxy | stacking | 4 | bear | 54 | 29.63% | False | False |
| exante_proxy | stacking | 4 | bull | 40 | 55.00% | False | False |
| exante_proxy | stacking | 4 | insufficient_history | 135 | 68.15% | True | True |
| exante_proxy | stacking | 4 | sideways | 4659 | 50.91% | False | False |
| exante_proxy | stacking | 8 | bear | 133 | 44.36% | False | False |
| exante_proxy | stacking | 8 | bull | 281 | 42.35% | False | False |
| exante_proxy | stacking | 8 | insufficient_history | 135 | 45.93% | False | False |
| exante_proxy | stacking | 8 | sideways | 4330 | 48.96% | False | False |
| exante_proxy | stacking | 20 | bear | 385 | 54.03% | False | False |
| exante_proxy | stacking | 20 | bull | 779 | 38.51% | False | False |
| ... |  |  |  |  |  |  |  |

## Boundary

- Per-ticker regime rows: 2400.
- Per-ticker rows passing 63%: 612.
- Ex-ante proxy labels avoid current/future target leakage but remain diagnostics.
