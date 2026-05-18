# VN30 Daily 2015 Target60 Result Summary

- Created at UTC: 2026-05-17
- Track: VN30 Daily 2015
- Branch: research/vn100-evidence-hardening-v1
- Tag: nckh/vn30-daily-2015-target60-optimization-v1

## Previous Best

- Model: XGBoost
- Horizon: h=1
- Final accuracy: 55.84%
- Final rows: 10,050

## New Best (Target60 Optimization)

- Model: LightGBM
- Feature set: daily_cross (extended + cross-sectional rank)
- Horizon: h=40 trading days
- Validation accuracy: 52.95% (rolling 2021-2024)
- Final accuracy: 57.58%
- Final rows: 8,880
- Final coverage: 1.0 (30/30 tickers)
- Claim level: failed

## Target60 Result

- Target: >=60% pooled overall directional accuracy
- **Target60 passed: no**
- Gap to 60%: **2.42 percentage points**

## Optimization Scope

- Total experiments: 315
- Completed: 105
- Skipped: 210 (insufficient data for some horizon/feature combinations)
- Models: XGBoost, LightGBM, Random Forest
- Feature sets: daily_basic (37 features), daily_extended (105 features), daily_cross (108 features)
- Hyperparameter configs: 5 per model
- Horizons: 1, 3, 5, 10, 20, 40, 60

## Top 5 Results (by Final Accuracy)

| Model | Feature Set | Horizon | Val Accuracy | Final Accuracy | Final Rows |
|---|---|---|---|---|---|
| lightgbm | daily_cross | 40 | 52.95% | 57.58% | 8,880 |
| lightgbm | daily_extended | 40 | 52.66% | 57.50% | 8,880 |
| lightgbm | daily_extended | 40 | 52.81% | 55.83% | 8,880 |
| lightgbm | daily_cross | 40 | 53.09% | 55.70% | 8,880 |
| lightgbm | daily_extended | 60 | 52.70% | 55.42% | 8,280 |

## Validation Method

- Rolling validation: 2021, 2022, 2023, 2024
- Training: all data before each validation year
- Selection: best candidate chosen on mean validation accuracy only
- Final evaluation: 2025-01-01 to latest available
- No final-label tuning

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
- No hourly claim made from daily data.
