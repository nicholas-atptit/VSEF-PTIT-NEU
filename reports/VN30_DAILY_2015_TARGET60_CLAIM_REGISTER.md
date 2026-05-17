# VN30 Daily 2015 Target60 Claim Register

- Created at UTC: 2026-05-17
- Track: VN30 Daily 2015
- Branch: research/vn100-evidence-hardening-v1
- Tag: nckh/vn30-daily-2015-target60-optimization-v1

## Universe

- Source: VN30 January 2025 review
- Ticker count: 30/30 usable
- Excluded: none
- Frequency: daily only
- Period: 2015-01-01 (or first trading date) to latest available
- Evaluation: 2025-01-01 to latest available

## Target

- Target: >=60% pooled overall directional accuracy
- Target60 passed: **no**

## Best Result

- Model: LightGBM
- Feature set: daily_cross (extended + cross-sectional rank)
- Horizon: h=40 trading days
- Validation accuracy: 52.95% (rolling 2021-2024, 29,885 rows)
- Final accuracy: 57.58% (8,880 rows, 2025-01-01+)
- Full universe: yes (30/30)
- Full coverage: yes (1.0)
- Claim level: failed (below 60% threshold)
- Gap to 60%: 2.42 percentage points

## Previous Best (for comparison)

- Model: XGBoost
- Horizon: h=1
- Final accuracy: 55.84% (10,050 rows)

## Optimization Scope

- Total experiments: 315
- Completed: 105
- Skipped: 210
- Models: XGBoost, LightGBM, Random Forest
- Feature sets: daily_basic, daily_extended, daily_cross
- Hyperparameter configs: 5 per model
- Horizons: 1, 3, 5, 10, 20, 40, 60
- 60% candidates: 0

## Validation Method

- Rolling validation across 2021, 2022, 2023, 2024
- Training: all data before each validation year
- Selection: best candidate chosen on mean validation accuracy only
- Final evaluation: 2025-01-01 to latest available
- No final-label tuning

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
- Daily-only: yes
- No hourly data: yes
- No daily-to-hourly resampling: yes
- Active universe 30/30: yes
- Validation-only selection: yes
- No leakage: yes
