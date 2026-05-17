# VN30 Daily 2015 Benchmark Result Summary

- Active universe source: VN30 January 2025 review.
- Usable ticker count: 30.
- Usable ticker list: ACB, BCM, BID, BVH, CTG, FPT, GAS, GVR, HDB, HPG, LPB, MBB, MSN, MWG, PLX, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VRE.
- Excluded tickers: none (BCM and VIB recovered).
- Train period: 2015-01-01 to 2023-12-31 23:59:59.
- Validation period: 2024-01-01 00:00:00 to 2024-12-31 23:59:59.
- Evaluation period: 2025-01-01 00:00:00 to latest available.
- Models run: random_forest, xgboost, lightgbm.
- Horizons: [1, 5, 10, 20, 60].
- Total experiments: 15.
- Audit passed: yes.

## Accuracy By Model And Horizon

| model | horizon | val_accuracy | final_accuracy | final_rows | claim_level |
| --- | --- | --- | --- | --- | --- |
| random_forest | 1 | 53.52% | 54.15% | 10050 | failed |
| xgboost | 1 | 55.51% | 55.84% | 10050 | failed |
| lightgbm | 1 | 55.59% | 55.41% | 10050 | failed |
| random_forest | 5 | 49.29% | 51.73% | 9930 | failed |
| xgboost | 5 | 52.09% | 53.93% | 9930 | failed |
| lightgbm | 5 | 51.69% | 53.20% | 9930 | failed |
| random_forest | 10 | 47.83% | 51.54% | 9780 | failed |
| xgboost | 10 | 49.69% | 53.49% | 9780 | failed |
| lightgbm | 10 | 48.59% | 52.59% | 9780 | failed |
| random_forest | 20 | 49.27% | 51.14% | 9480 | failed |
| xgboost | 20 | 50.45% | 53.11% | 9480 | failed |
| lightgbm | 20 | 50.45% | 53.44% | 9480 | failed |
| random_forest | 60 | 52.04% | 53.25% | 8280 | failed |
| xgboost | 60 | 55.09% | 54.83% | 8280 | failed |
| lightgbm | 60 | 53.47% | 53.38% | 8280 | failed |

## Best Model/Horizon

- Model: xgboost.
- Horizon: 1 trading days.
- Validation accuracy: 55.51%.
- Final accuracy: 55.84%.
- Final rows: 10050.
- Claim level: failed.

## Baseline Delta

| model | horizon | model_accuracy | baseline_accuracy | delta |
| --- | --- | --- | --- | --- |
| xgboost | 1 | 55.84% | 50.00% | 5.84% |

## Limitations

- Daily track is separate from hourly track; results are not directly comparable.
- No hourly data exists for 2015-2022; daily data used instead.
- No daily-to-hourly resampling was performed.
- The run is a directional classification benchmark, not a trading system.
- No transaction cost, slippage, capital allocation, or execution diagnostics were run.

## Claim Boundary

- No trading-readiness claim.
- No profitability claim.
- No cost/slippage claim.
- No paper or DOCX generated.
