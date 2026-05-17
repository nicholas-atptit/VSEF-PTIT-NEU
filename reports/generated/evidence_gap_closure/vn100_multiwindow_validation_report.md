# VN100 Multi-Window Validation Report

## Source

- Official 2025 artifact directory: `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff`.
- Requested windows: 2022, 2023, 2024, and 2025.
- No heavy benchmark rerun was performed.

## Window Availability

| eval_year | available | note |
| --- | --- | --- |
| 2022 | False | missing official prediction artifacts |
| 2023 | False | missing official prediction artifacts |
| 2024 | False | missing official prediction artifacts |
| 2025 | True | official artifact available |

## Stability Matrix

| signal | 2022_status | 2023_status | 2024_status | 2025_status | 2025_accuracy | stable_across_windows |
| --- | --- | --- | --- | --- | --- | --- |
| global_daily | unavailable | unavailable | unavailable | available | 0.5318725099601593 | not_established |
| global_hourly | unavailable | unavailable | unavailable | available | 0.5128571875195398 | not_established |
| hourly_stacking_h1_confidence_0.57 | unavailable | unavailable | unavailable | available | 0.6003482803656944 | not_established |
| daily_lightgbm_h20_posthoc_bear | unavailable | unavailable | unavailable | available | 0.6959459459459459 | not_established |
| daily_xgboost_h20_posthoc_bear | unavailable | unavailable | unavailable | available | 0.6914414414414415 | not_established |

## Required Answers

- Stable signals across windows: not established; only the 2025 official window has prediction artifacts.
- Selected confidence result persistence: not established beyond 2025.
- Bear-regime diagnostic persistence: not established beyond 2025.
- Global benchmark pass in any available window: no, the available 2025 daily and hourly global summaries do not pass 60%.
- Claim-boundary impact: unchanged. Multi-window evidence remains a major missing-evidence item.

## Missing Evidence

- Official 2022 prediction and benchmark artifacts with train_cutoff=2021-12-31.
- Official 2023 prediction and benchmark artifacts with train_cutoff=2022-12-31.
- Official 2024 prediction and benchmark artifacts with train_cutoff=2023-12-31.
- Recomputed confidence, regime, baseline, significance, and concentration diagnostics for each window.
