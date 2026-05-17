# Daily Accuracy Drag Report

- Created at UTC: `2026-05-17T09:59:50+00:00`.
- Model: LightGBM daily_cross h=40.
- Overall accuracy: 57.58%.
- Overall class balance (positive rate): 56.25%.
- Final rows: 8880.

## Accuracy by Ticker (sorted ascending)

| ticker | n_rows | n_correct | accuracy | class_balance |
|---|---|---|---|---|
| VIC | 296 | 106 | 35.81% | 0.9020 |
| VJC | 296 | 111 | 37.50% | 0.5946 |
| VIB | 296 | 119 | 40.20% | 0.3885 |
| ACB | 296 | 122 | 41.22% | 0.4493 |
| BVH | 296 | 122 | 41.22% | 0.6385 |
| VPB | 296 | 139 | 46.96% | 0.4932 |
| GAS | 296 | 140 | 47.30% | 0.5338 |
| SSB | 296 | 145 | 48.99% | 0.3412 |
| CTG | 296 | 159 | 53.72% | 0.5878 |
| MSN | 296 | 162 | 54.73% | 0.5169 |
| BID | 296 | 163 | 55.07% | 0.5236 |
| MWG | 296 | 167 | 56.42% | 0.7534 |
| TPB | 296 | 169 | 57.09% | 0.4392 |
| VNM | 296 | 169 | 57.09% | 0.5980 |
| TCB | 296 | 178 | 60.14% | 0.6318 |
| LPB | 296 | 178 | 60.14% | 0.6250 |
| STB | 296 | 181 | 61.15% | 0.7703 |
| SHB | 296 | 182 | 61.49% | 0.5878 |
| MBB | 296 | 182 | 61.49% | 0.6419 |
| VHM | 296 | 184 | 62.16% | 0.7905 |
| SSI | 296 | 185 | 62.50% | 0.5203 |
| HPG | 296 | 188 | 63.51% | 0.5980 |
| BCM | 296 | 189 | 63.85% | 0.4189 |
| SAB | 296 | 190 | 64.19% | 0.4527 |
| VCB | 296 | 203 | 68.58% | 0.4662 |
| VRE | 296 | 209 | 70.61% | 0.6959 |
| HDB | 296 | 211 | 71.28% | 0.6182 |
| FPT | 296 | 218 | 73.65% | 0.3446 |
| PLX | 296 | 219 | 73.99% | 0.4662 |
| GVR | 296 | 223 | 75.34% | 0.4865 |

## Worst 5 Tickers (dragging accuracy)

- **VIC**: 35.81% (296 rows, class_balance=0.9020)
- **VJC**: 37.50% (296 rows, class_balance=0.5946)
- **VIB**: 40.20% (296 rows, class_balance=0.3885)
- **ACB**: 41.22% (296 rows, class_balance=0.4493)
- **BVH**: 41.22% (296 rows, class_balance=0.6385)

## Best 5 Tickers (supporting accuracy)

- **VRE**: 70.61% (296 rows, class_balance=0.6959)
- **HDB**: 71.28% (296 rows, class_balance=0.6182)
- **FPT**: 73.65% (296 rows, class_balance=0.3446)
- **PLX**: 73.99% (296 rows, class_balance=0.4662)
- **GVR**: 75.34% (296 rows, class_balance=0.4865)

## Accuracy by Time

| year | month | n_rows | n_correct | accuracy | class_balance |
|---|---|---|---|---|---|
| 2025 | 1 | 510 | 337 | 66.08% | 0.8118 |
| 2025 | 2 | 600 | 317 | 52.83% | 0.2717 |
| 2025 | 3 | 630 | 293 | 46.51% | 0.2587 |
| 2025 | 4 | 600 | 356 | 59.33% | 0.8167 |
| 2025 | 5 | 600 | 404 | 67.33% | 0.9633 |
| 2025 | 6 | 630 | 514 | 81.59% | 0.9508 |
| 2025 | 7 | 690 | 523 | 75.80% | 0.8029 |
| 2025 | 8 | 630 | 354 | 56.19% | 0.5206 |
| 2025 | 9 | 600 | 251 | 41.83% | 0.2683 |
| 2025 | 10 | 690 | 289 | 41.88% | 0.3899 |
| 2025 | 11 | 600 | 278 | 46.33% | 0.7517 |
| 2025 | 12 | 690 | 370 | 53.62% | 0.6522 |
| 2026 | 1 | 600 | 309 | 51.50% | 0.1417 |
| 2026 | 2 | 450 | 301 | 66.89% | 0.2444 |
| 2026 | 3 | 360 | 217 | 60.28% | 0.5000 |

## Confusion Matrix

| | Predicted 0 | Predicted 1 |
|---|---|---|
| Actual 0 | 1810 | 1692 |
| Actual 1 | 2075 | 3303 |

## Observations

- Accuracy std across tickers: 0.1111
- Worst ticker accuracy: 35.81% (VIC)
- Best ticker accuracy: 75.34% (GVR)
- Accuracy range: 39.53%
- Tickers below 50%: 8 (VIC, VJC, VIB, ACB, BVH, VPB, GAS, SSB)
- Periods below 50%: 4
  - 2025-03: 46.51%
  - 2025-09: 41.83%
  - 2025-10: 41.88%
  - 2025-11: 46.33%

## Conclusion

- Overall accuracy 57.58% is 2.42% below the 60% target.
- Accuracy varies across tickers (std=0.1111).
- 8 tickers are below 50% accuracy, contributing to the drag.
- No hourly data used. Daily-only analysis.