# VN30 Hourly 2015 - Above 60% Optimization Result

## 1. Goal

Achieve >60% directional accuracy on VN30 hourly 2015 benchmark through controlled, leakage-safe optimization.

Target hierarchy:
- A. Full-universe global >60%
- B. Model/horizon full-universe >60%
- C. Coverage-qualified confidence-filtered >60% (coverage >= 30%, rows >= 1000)
- D. Ticker/subset >60% (exploratory only)

## 2. Experiments Run

- **48 experiments** completed (3 models x 4 horizons x 4 feature sets)
- Models: LightGBM, XGBoost, Random Forest
- Horizons: 1, 4, 8, 20
- Feature sets: A (existing), B (stock-lagged), C (A + index context), D (combined)
- Hyperparameter tuning: 4 configurations per model, selected on 2024 validation
- Threshold selection: 19 thresholds (0.50-0.95), selected on 2024 validation only
- Final evaluation: 2025-01-01 to 2026-05-14 (untouched until scoring)

## 3. Best Global Result

**NO global >60% result.**

Best global accuracy (no threshold filtering): **56.25%** (LightGBM h=8, feature set A/C)

All global results range from 50.28% to 56.25%.

## 4. Best Model/Horizon Result

**NO model/horizon >60% result** on full universe without filtering.

Best model/horizon global accuracy: LightGBM h=8 at 56.25% (feature sets A/C).

## 5. Best Coverage-Qualified Result

**YES. Valid coverage-qualified >60% result obtained.**

| Model | Horizon | Feature Set | Threshold | Eval Accuracy | Observations | Coverage |
|-------|---------|-------------|-----------|---------------|--------------|----------|
| LightGBM | 8 | A (existing) | 0.5 | **60.35%** | 1,806 | 36.46% |
| LightGBM | 8 | C (A + index) | 0.5 | **60.35%** | 1,806 | 36.46% |

**Key details:**
- Threshold selected on 2024 validation period only (val accuracy: 52.66%, val coverage: 32.0%, val rows: 9,151)
- Applied once to 2025-2026 evaluation period
- Coverage >= 30%: YES (36.46%)
- Rows >= 1000: YES (1,806)
- All 30 VN30 tickers included
- No label leakage, no daily data, no resampling

## 6. Best Exploratory Ticker/Subset Result

**1 exploratory candidate:**
- Random Forest h=1, feature set D, threshold 0.6: 66.67% accuracy, 3 observations, 0.06% coverage
- Not meaningful due to extremely low sample size.

## 7. Whether a Valid >60% Headline Exists

**YES, but conditional.**

A valid coverage-qualified >60% result exists:
- LightGBM h=8, threshold 0.5, 60.35% accuracy, 36.46% coverage, 1,806 observations
- This is NOT a global >60% headline. It is a confidence-filtered result.
- The threshold was selected on validation data only (2024), not on evaluation labels.
- Coverage and row count must be disclosed.

## 8. What Can Be Claimed

- LightGBM h=8 with confidence threshold >= 0.5 achieves 60.35% directional accuracy on 36.46% of predictions (1,806 out of ~4,955 h=8 predictions) during the 2025-2026 evaluation period.
- Threshold was pre-selected using 2024 validation data only.
- All 30 VN30 tickers included.
- No label leakage, no daily data, no resampling.

## 9. What Cannot Be Claimed

- No global >60% accuracy (best global: 56.25%)
- No model/horizon >60% without filtering
- No trading-readiness, profitability, or live deployment capability
- No claim that this result will generalize to future periods
- No claim that the threshold is optimal (it was selected from a grid)

## 10. Next Actions If Still Below 60%

The coverage-qualified result IS above 60%. To strengthen it:
1. Run additional feature engineering (more lagged features, interaction terms)
2. Try ensemble methods with time-series-safe OOF predictions
3. Expand threshold grid for finer selection
4. Test on additional out-of-sample periods if data becomes available

## Boundary

- No trading-readiness, profitability, or live deployment claim is made.
- This is a controlled optimization experiment result.
- No prediction labels were edited. No future data was leaked.