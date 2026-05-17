# VN30 Hourly 2015 - Target 60/65 Result Summary

## Current Baseline Benchmark: 51.34%

## Current Optimized Result Before This Phase: 60.35% conditional (LightGBM h=8, threshold 0.5)

## New Baseline-v2 Result

- Best full-universe global accuracy: **55.20%** (XGBoost h=8, feature set A)
- Best threshold-filtered eval accuracy: **56.76%** (XGBoost h=8, feature set C, threshold 0.625)
- Baseline target >=60: **FAIL**
- Gap to 60: **4.80 percentage points**

## New Final-v2 Result

- Best final threshold accuracy: **56.76%** (XGBoost h=8, feature set C, threshold 0.625, 66.39% coverage, 3,342 rows)
- Final target >=65 global: **FAIL**
- Final target >=65 coverage-qualified: **FAIL**
- Gap to 65: **8.24 percentage points**

## Best Valid Candidate

| Model | Horizon | Feature Set | Threshold | Eval Accuracy | Coverage | Rows | Claim Level |
|-------|---------|-------------|-----------|---------------|----------|------|-------------|
| XGBoost | 8 | C (A + index) | 0.625 | 56.76% | 66.39% | 3,342 | conditional_coverage_qualified |

## Whether Headline Is Allowed

**NO.** No valid >=60% global or >=65% coverage-qualified result exists.

## Gap Remaining

- Baseline benchmark: 51.34%
- Previous optimization v1: 60.35% conditional (LightGBM h=8, threshold 0.5, 36.46% coverage)
- Baseline-v2: 55.20% global, 56.76% threshold-filtered
- Final-v2: 56.76% best candidate (XGBoost h=8, threshold 0.625, 66.39% coverage)
- Gap to 60% global: 4.80 pp
- Gap to 65% coverage-qualified: 8.24 pp

## What Can Be Claimed

- XGBoost h=8 with threshold 0.625 achieves 56.76% directional accuracy on 66.39% of predictions (3,342 rows) during 2025-2026 evaluation.
- Threshold selected on 2024 validation only.
- All 30 VN30 tickers included.
- No label leakage, no daily data, no resampling.

## What Cannot Be Claimed

- No >=60% global accuracy
- No >=65% accuracy at any coverage level
- No trading-readiness, profitability, or live deployment capability

## Boundary

- No trading-readiness, profitability, or live deployment claim is made.
- No paper or DOCX generated.
- No prediction labels were edited. No future data was leaked.