# VN30 Hourly 2015 - Hard Optimization v2 Result Summary

## Previous Results

- Baseline benchmark: 51.34%
- Previous best global: 55.20% (XGBoost h=8, baseline-v2)
- Previous best conditional: 60.35% (LightGBM h=8, threshold 0.5, optimization v1)

## Targets

- Baseline target: 60%
- Final target: 65%

## New Hard Optimization v2 Results

### Best Global Result

- **54.63%** (Weighted ensemble, equal weight, h=8, feature set C)
- Baseline 60 target: **FAIL**
- Gap to 60: **5.37 percentage points**

### Best Coverage-Qualified Result

- **None** at >=65% with coverage >=30% and rows >=1000
- Best conditional: XGBoost h=8, calibration threshold, 56.13% accuracy, 79.06% coverage, 3,980 rows
- Final 65 target: **FAIL**
- Gap to 65: **8.87 percentage points**

### Strategy Results

| Strategy | Best Accuracy | Coverage | Rows | Pass 60 |
|----------|--------------|----------|------|---------|
| Per-ticker models | 51.10% | 100% | 5,006 | No |
| Weighted ensemble (equal) | 54.63% | 100% | 5,034 | No |
| Weighted ensemble (val-weighted) | 54.63% | 100% | 5,034 | No |
| Router (per-ticker) | 51.10% | 100% | 5,006 | No |
| Meta-labeling | 56.13% | 79.06% | 3,980 | No |
| Calibration threshold | 56.13% | 79.06% | 3,980 | No |

## What Failed

- No strategy achieved >=60% global accuracy
- No strategy achieved >=65% coverage-qualified accuracy
- Per-ticker models performed worse than global models (51.10% vs 54.63%)
- Router did not improve over per-ticker (same 51.10%)
- Meta-labeling provided modest improvement (56.13%) but still below 60%
- Calibration threshold selection did not push any result above 60%

## Next Possible Directions

1. **Deeper feature engineering**: interaction terms, non-linear transformations
2. **More aggressive model tuning**: larger hyperparameter grids, Bayesian optimization
3. **Alternative model families**: neural networks, gradient boosting variants
4. **Longer training history**: if more data becomes available
5. **Different target definitions**: alternative labeling schemes
6. **Regime-specific models**: separate models for bull/bear/sideways regimes

## Boundary

- No trading-readiness, profitability, or live deployment claim is made.
- No paper or DOCX generated.
- No prediction labels were edited. No future data was leaked.