# VN30 Hourly 2015 Jan-2025 Benchmark Result Summary

- Active universe source: VN30 January 2025 review.
- Active ticker count: 30.
- Active ticker list: ACB, BCM, BID, BVH, CTG, FPT, GAS, GVR, HDB, HPG, LPB, MBB, MSN, MWG, PLX, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VRE.
- Train period: 2015-01-01 00:00:00 to 2024-12-31 23:59:59.
- Evaluation period: 2025-01-01 00:00:00 to 2026-05-14 00:00:00.
- Models run: lightgbm, random_forest, stacking, xgboost.
- Baselines run: always_up, moving_average_signal, previous_direction, random_seeded_direction.
- Total predictions: 77692.
- Global directional accuracy: 51.34%.
- Global benchmark pass: no.
- Audit passed: yes.
- Benchmark run: yes.
- Model training run: yes, only inside benchmark workflow.
- Data fetch run: no.
- Daily data used: no.
- Resampling used: no.
- Paper/DOCX generated: no.

## Accuracy By Model And Horizon

| model | horizon | n_obs | accuracy |
| --- | --- | --- | --- |
| lightgbm | 20 | 4599 | 54.58% |
| xgboost | 20 | 4599 | 54.42% |
| random_forest | 20 | 4599 | 54.03% |
| random_forest | 1 | 4907 | 51.89% |
| lightgbm | 8 | 4924 | 51.40% |
| random_forest | 4 | 4993 | 51.39% |
| xgboost | 1 | 4907 | 51.13% |
| xgboost | 8 | 4924 | 51.10% |
| lightgbm | 1 | 4907 | 51.05% |
| lightgbm | 4 | 4993 | 50.89% |
| stacking | 1 | 4907 | 50.76% |
| random_forest | 8 | 4924 | 50.65% |
| xgboost | 4 | 4993 | 50.61% |
| stacking | 4 | 4993 | 50.53% |
| stacking | 8 | 4924 | 48.84% |
| stacking | 20 | 4599 | 48.55% |

## Best Model/Horizon

- Model: lightgbm.
- Horizon: 20.
- Accuracy: 54.58%.
- Observations: 4599.

## Baseline Deltas

| model | horizon | baseline | model_accuracy | baseline_accuracy | delta |
| --- | --- | --- | --- | --- | --- |
| lightgbm | 20 | previous_direction | 54.58% | 48.16% | 6.41% |
| xgboost | 20 | previous_direction | 54.42% | 48.16% | 6.26% |
| random_forest | 20 | previous_direction | 54.03% | 48.16% | 5.87% |
| lightgbm | 20 | moving_average_signal | 54.58% | 48.86% | 5.72% |
| xgboost | 20 | moving_average_signal | 54.42% | 48.86% | 5.57% |
| random_forest | 20 | moving_average_signal | 54.03% | 48.86% | 5.18% |
| lightgbm | 20 | random_seeded_direction | 54.58% | 50.58% | 4.00% |
| xgboost | 20 | random_seeded_direction | 54.42% | 50.58% | 3.85% |
| random_forest | 4 | moving_average_signal | 51.39% | 47.73% | 3.67% |
| random_forest | 20 | random_seeded_direction | 54.03% | 50.58% | 3.46% |
| lightgbm | 8 | moving_average_signal | 51.40% | 48.19% | 3.21% |
| lightgbm | 4 | moving_average_signal | 50.89% | 47.73% | 3.16% |

## Limitations

- The global benchmark did not meet the configured threshold.
- The run is a directional classification benchmark, not a trading system.
- No transaction cost, slippage, capital allocation, or execution diagnostics were run.
- Index caches were inspected for readiness only and were not model feature inputs.
- The evaluation uses the validated gateway cache with the Jan-2025 frozen VN30 universe only.

## Claim Boundary

- No trading-readiness claim.
- No profitability claim.
- No cost/slippage claim.
- No paper or DOCX generated.
