# VN30 Daily 2015 Target60 V2 Result Summary

- Created at UTC: 2026-05-17
- Track: VN30 Daily 2015
- Branch: research/vn100-evidence-hardening-v1
- Tag: nckh/vn30-daily-2015-target60-v2

## Previous Best

- Model: LightGBM
- Feature set: daily_cross
- Horizon: h=40
- Final accuracy: 57.58%
- Final rows: 8,880

## New Best (Target60 V2 Optimization)

- Model: LightGBM
- Feature set: volatility_normalized
- Horizon: h=50 trading days
- Decision threshold: 0.525
- Rolling validation mean accuracy: 52.70% (2021-2024)
- Final accuracy: 57.45%
- Final rows: 8,580
- Final coverage: 1.0 (30/30 tickers)
- Claim level: failed

## Target60 V2 Result

- Target: >=60% pooled overall directional accuracy
- **Target60 passed: no**
- Gap to 60%: **2.55 percentage points**

## Optimization Scope

- Base experiments: 240 (3 models x 4-5 hp configs x 4 horizons x 5 feature sets)
- Total candidates (with thresholds): 3,120 (240 x 13 thresholds)
- Completed: 100
- Skipped: 140 (insufficient data for some horizon/feature combinations)
- Models: XGBoost, LightGBM, Random Forest
- Feature sets: daily_cross, daily_cross_plus_market, daily_cross_interactions, volatility_normalized, momentum_volatility_interaction
- Horizons: 30, 40, 50, 60
- Thresholds: 0.35 to 0.65 by 0.025
- 60% candidates: 0

## Best by Stability Score (validation-selected)

- Model: LightGBM
- Feature set: daily_cross
- Horizon: h=40
- Threshold: 0.50
- Stability score: 0.5216
- Final accuracy: 53.92%

## Top 5 Results (by Final Accuracy)

| Model | Feature Set | Horizon | Threshold | Val Mean | Final Accuracy | Final Rows |
|---|---|---|---|---|---|---|
| lightgbm | volatility_normalized | 50 | 0.525 | 52.70% | 57.45% | 8,580 |
| lightgbm | daily_cross_interactions | 50 | 0.500 | 52.54% | 57.33% | 8,580 |
| lightgbm | daily_cross | 50 | 0.500 | 52.62% | 57.25% | 8,580 |
| lightgbm | daily_cross_interactions | 40 | 0.500 | 52.47% | 57.20% | 8,880 |
| lightgbm | daily_cross | 40 | 0.500 | 52.49% | 57.19% | 8,880 |

## Validation Method

- Rolling validation: 2021, 2022, 2023, 2024
- Training: all data before each validation year
- Threshold selection: best threshold by stability_score (mean - std/2) on rolling validation
- Candidate selection: by stability_score
- Final evaluation: 2025-01-01 to latest available (used only for scoring)
- No final-label tuning

## Comparison with V1

| Metric | V1 Best | V2 Best |
|---|---|---|
| Model | LightGBM | LightGBM |
| Feature set | daily_cross | volatility_normalized |
| Horizon | h=40 | h=50 |
| Threshold | 0.50 (default) | 0.525 |
| Final accuracy | 57.58% | 57.45% |
| Final rows | 8,880 | 8,580 |
| Gap to 60% | 2.42pp | 2.55pp |

V2 did not improve over V1. The best V2 result (57.45%) is slightly below the V1 best (57.58%).

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
