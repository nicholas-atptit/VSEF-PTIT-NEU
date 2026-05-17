# VN30 Daily 2015 Benchmark Result Summary

- Active universe source: VN30 January 2025 review.
- Usable ticker count: 28.
- Usable ticker list: ACB, BID, BVH, CTG, FPT, GAS, GVR, HDB, HPG, LPB, MBB, MSN, MWG, PLX, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIC, VJC, VNM, VPB, VRE.
- Excluded tickers (no data): BCM, VIB.
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
| random_forest | 1 | 53.77% | 53.82% | 9380 | failed |
| xgboost | 1 | 55.73% | 55.81% | 9380 | failed |
| lightgbm | 1 | 55.16% | 55.39% | 9380 | failed |
| random_forest | 5 | 48.59% | 51.35% | 9268 | failed |
| xgboost | 5 | 51.10% | 52.51% | 9268 | failed |
| lightgbm | 5 | 51.41% | 53.14% | 9268 | failed |
| random_forest | 10 | 47.34% | 52.02% | 9128 | failed |
| xgboost | 10 | 49.76% | 52.88% | 9128 | failed |
| lightgbm | 10 | 49.17% | 52.51% | 9128 | failed |
| random_forest | 20 | 50.39% | 50.42% | 8848 | failed |
| xgboost | 20 | 51.23% | 52.98% | 8848 | failed |
| lightgbm | 20 | 49.13% | 51.21% | 8848 | failed |
| random_forest | 60 | 51.46% | 52.69% | 7728 | failed |
| xgboost | 60 | 54.41% | 55.14% | 7728 | failed |
| lightgbm | 60 | 53.79% | 54.45% | 7728 | failed |

## Best Model/Horizon

- Model: xgboost.
- Horizon: 1 trading days.
- Validation accuracy: 55.73%.
- Final accuracy: 55.81%.
- Final rows: 9380.
- Claim level: failed.

## Baseline Delta

| model | horizon | model_accuracy | baseline_accuracy | delta |
| --- | --- | --- | --- | --- |
| xgboost | 1 | 55.81% | 50.00% | 5.81% |

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
