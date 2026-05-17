# VN30 Hourly 2015 - Above 60% Result Summary

## What Passed 60%

### Global Model/Horizon
**NO.** No global model/horizon combination exceeds 60% accuracy.
- Best: LightGBM h=20 at 54.58% (4599 obs)

### Confidence-Filtered Slices
**YES, conditionally.**
- Random Forest h=20, conf>=0.775: 61.93% accuracy, 549 obs, 11.94% coverage
- Random Forest h=20, conf>=0.70: 61.71% accuracy, 1251 obs, 27.20% coverage
- Random Forest h=20, conf>=0.725: 61.31% accuracy, 964 obs, 20.96% coverage
- XGBoost h=20, conf>=0.875: 61.04% accuracy, 1060 obs, 23.05% coverage

### Ticker-Level Slices
**YES, exploratory.**
- 160 ticker-level candidates pass 60%
- Best: LightGBM h=8, ACB at 61.73% (162 obs)
- Other notable: LightGBM h=20, SSB at 61.54% (various obs)

### Regime-Level Slices
**NO.** No regime-level rows exceed 60%.
- Best: XGBoost h=20, bear regime at 58.93% (1539 obs)
- Best: LightGBM h=20, low_volatility at 58.27% (520 obs)

## What Did Not Pass 60%

- All global model/horizon combinations (best: 54.58%)
- All regime-level combinations (best: ~59%)
- Stacking ensemble (underperforms base models at h=20: 48.55%)
- Any slice with >= 30% coverage AND >= 1000 rows AND > 60% accuracy

## Which Candidates Are Legitimate

**Conditional claims (42 candidates):**
- Random Forest h=20 with confidence >= 0.70-0.775
- XGBoost h=20 with confidence >= 0.875
- These are legitimate but require coverage disclosure (11-27% coverage)

**Exploratory observations (1528 candidates):**
- Ticker-level slices (post-hoc)
- High-confidence, low-coverage slices
- Combined filter slices (confidence + ticker)

## Which Are Post-Hoc Only

**All confidence-filtered, ticker-level, and combined slices are post-hoc.**
The thresholds and slices were identified after seeing the evaluation results.
They cannot be presented as pre-registered methods without separate validation.

## What Must Be Rerun to Make a 60% Claim Stronger

1. **Pre-registered confidence thresholds**: Select thresholds on training/validation period only, then apply to evaluation period.
2. **Model tuning**: Tune LightGBM/XGBoost/RF at h=20 using walk-forward validation.
3. **Feature engineering**: Add lagged returns, rolling volatility, momentum, volume features.
4. **Stacking v2**: Fix the stacking meta-learner (currently hurts by 6% at h=20).

## Whether the Project Currently Has a Valid "Above 60%" Headline

**NO.**

The current benchmark does not contain any global >60% result. The best global accuracy is 54.58% (LightGBM h=20).

Conditional confidence-filtered slices exist at 61-62% but with 11-27% coverage, which is below the 30% coverage floor for a strong conditional claim.

## Key Findings from Diagnostics

- **Worst tickers**: FPT (37.47%), VIC (42.68%), GAS (45.23%)
- **Best tickers**: SSB (58.48%), MBB (56.86%), MSN (55.46%)
- **h=20 is consistently strongest** across all base models
- **Stacking hurts** relative to best base model at all horizons (-1.12% to -6.02%)
- **Model disagreement rate**: 47.90% (models agree on only 52.10% of predictions)
- **Gap to 60%**: 8.66 percentage points

## Boundary

- No trading-readiness, profitability, or live deployment claim is made.
- This summary is based on existing benchmark outputs only.
- No prediction labels were edited. No future data was leaked.
- No new data was fetched.