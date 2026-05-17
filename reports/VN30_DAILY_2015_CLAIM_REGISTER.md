# VN30 Daily 2015 Claim Register

- Created at UTC: 2026-05-17
- Track: VN30 Daily 2015
- Branch: research/vn100-evidence-hardening-v1
- Tags: nckh/vn30-daily-2015-benchmark-v1, nckh/vn30-daily-2015-bcm-vib-recovery-v1

## Universe

- Source: VN30 January 2025 review
- Ticker count: 30/30 usable
- Excluded: none (BCM and VIB recovered)
- Frequency: daily only
- Period: 2015-01-01 (or first trading date) to latest available
- Evaluation: 2025-01-01 to latest available

## Best Result

- Model: XGBoost
- Horizon: h=1 trading day
- Validation accuracy: 55.51% (7,500 rows, 2024)
- Final accuracy: 55.84% (10,050 rows, 2025-01-01+)
- Full universe: yes (30/30)
- Full coverage: yes
- Claim level: failed (below 60% threshold)

## All Results (Final Evaluation)

| Model | Horizon | Final Accuracy | Final Rows | Claim Level |
|---|---|---|---|---|
| random_forest | 1 | 54.15% | 10,050 | failed |
| xgboost | 1 | 55.84% | 10,050 | failed |
| lightgbm | 1 | 55.41% | 10,050 | failed |
| random_forest | 5 | 51.73% | 9,930 | failed |
| xgboost | 5 | 53.93% | 9,930 | failed |
| lightgbm | 5 | 53.20% | 9,930 | failed |
| random_forest | 10 | 51.54% | 9,780 | failed |
| xgboost | 10 | 53.49% | 9,780 | failed |
| lightgbm | 10 | 52.59% | 9,780 | failed |
| random_forest | 20 | 51.14% | 9,480 | failed |
| xgboost | 20 | 53.11% | 9,480 | failed |
| lightgbm | 20 | 53.44% | 9,480 | failed |
| random_forest | 60 | 53.25% | 8,280 | failed |
| xgboost | 60 | 54.83% | 8,280 | failed |
| lightgbm | 60 | 53.38% | 8,280 | failed |

## Claim Boundary

- No trading-readiness claim is made.
- No profitability claim is made.
- No cost/slippage claim is made.
- No paper or DOCX generated.
- Daily track is separate from hourly track; results are not directly comparable.
- No daily-to-hourly resampling was performed.
- No hourly claims made from daily data.

## Audit Status

- Audit: passed
- All 30 tickers represented in outputs
- No hourly resampling markers present
- No trading/profitability claims in outputs

## Recovery Notes

- BCM and VIB were initially missing due to provider data quality issues (7 and 2 bad rows respectively out of 2000+).
- Recovery: fetch script updated with fallback to filter bad rows and retain valid data.
- Post-recovery: 30/30 tickers usable, benchmark rerun.
