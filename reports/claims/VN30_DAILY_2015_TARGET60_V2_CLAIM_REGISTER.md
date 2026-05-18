# VN30 Daily 2015 Target60 V2 Claim Register

- Created at UTC: 2026-05-17
- Track: VN30 Daily 2015
- Branch: research/vn100-evidence-hardening-v1
- Tag: nckh/vn30-daily-2015-target60-v2

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

## Best Result (V2)

- Model: LightGBM
- Feature set: volatility_normalized
- Horizon: h=50 trading days
- Decision threshold: 0.525
- Rolling validation mean accuracy: 52.70% (2021-2024)
- Final accuracy: 57.45% (8,580 rows, 2025-01-01+)
- Full universe: yes (30/30)
- Full coverage: yes (1.0)
- Claim level: failed (below 60% threshold)
- Gap to 60%: 2.55 percentage points

## Previous Best (V1, for comparison)

- Model: LightGBM
- Feature set: daily_cross
- Horizon: h=40
- Decision threshold: 0.50 (default)
- Final accuracy: 57.58% (8,880 rows)

## Optimization Scope

- Base experiments: 240
- Total candidates (with thresholds): 3,120
- Completed: 100
- Skipped: 140
- Models: XGBoost, LightGBM, Random Forest
- Feature sets: daily_cross, daily_cross_plus_market, daily_cross_interactions, volatility_normalized, momentum_volatility_interaction
- Horizons: 30, 40, 50, 60
- Thresholds: 0.35 to 0.65 by 0.025
- 60% candidates: 0

## Validation Method

- Rolling validation across 2021, 2022, 2023, 2024
- Training: all data before each validation year
- Threshold selection: best threshold by stability_score (mean - std/2) on rolling validation
- Candidate selection: by stability_score
- Final evaluation: 2025-01-01 to latest available
- No final-label tuning

## Safe Claims

- Daily 30/30 full-coverage result: **yes** (LightGBM volatility_normalized h=50 at 57.45%)
- Daily target60 passed: **no** (57.45% < 60%)

## Unsafe Claims (must not be made)

- Using hourly result as daily result: **no hourly data used**
- Daily-to-hourly resampling: **not performed**
- Ticker subset as full VN30: **full 30/30 used**
- Abstention result as overall full coverage: **no abstention; all rows predicted**
- Trading/profitability/live deployment: **no such claims made**

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
- Full final coverage: yes
- No abstention: yes
- No ticker subset: yes
- Threshold selected using validation only: yes
- Final evaluation used only for scoring: yes
- Leakage detected: no
