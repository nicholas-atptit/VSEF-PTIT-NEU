# VN30 Hourly 2015 - Top-K 75% Verification Report

## Executive Summary
- **Original claim**: LightGBM h=40 k=10 achieves 75.54% precision@10
- **Finding**: DISCREPANCY - 75.54% is from h=120, NOT h=40
- **Recomputed h=120 precision@10**: 74.82%
- **Metric computation correct**: YES
- **Leakage detected**: NO
- **Overfitting risk**: HIGH
- **Random baseline**: 77.36%
- **Momentum baseline**: 74.46%
- **Model lift over random**: -2.54%
- **Temporal robustness**: STABLE
- **Concentration risk**: NO
- **Final decision**: use_with_caution

## Critical Finding: Configuration Discrepancy
The original summary stated "Best: lightgbm, h=40, k=10" but the actual 75.54% precision@10
comes from lightgbm h=120 k=10. This is a significant reporting error.

- h=40 k=10 actual precision@10: 38.01%
- h=120 k=10 actual precision@10: 74.82%

## Metric Computation
- Mean precision@10: 74.82%
- Median precision@10: 80.00%
- Event-weighted precision@10: 74.82%
- Hit rate@10: 100.00%
- Events: 56
- Difference from original: 0.71%

## Baseline Comparison
- Random baseline mean: 77.36% +/- 0.95%
- Momentum baseline: 74.46%
- Model lift over random: -2.54%
- Significantly above random: NO

## Temporal Robustness
- Monthly precision std: 17.71%
- Temporal stability: STABLE

## Ticker Concentration
- Top 5 selected tickers share: 46.25%
- Top 5 correct tickers share: 39.11%
- Concentration risk: NO

## Leakage Audit
- Suspicious features found: 0
- None

## Decision
use_with_caution

## Notes
- Hit rate@10 is likely trivially easy (near 100% across most configs)
- Only 56 events in final eval for h=120 - result may not be robust
- Configuration discrepancy must be corrected before any use
