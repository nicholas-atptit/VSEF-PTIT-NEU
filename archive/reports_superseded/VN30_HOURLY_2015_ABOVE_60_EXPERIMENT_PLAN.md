# VN30 Hourly 2015 - Above 60% Experiment Plan

## Context

- Current global accuracy: 51.34%
- Best base model/horizon: LightGBM h=20 at 54.58%
- Best confidence slice: Random Forest h=20, threshold 0.675, 59.19%, 34.42% coverage
- No global model/horizon combination exceeds 60%
- All 30 VN30 tickers active, 77,692 total predictions
- Evaluation period: 2025-01-01 to 2026-05-14
- Train cutoff: 2024-12-31 23:59:59

## A. Threshold/Calibration Experiments

### A1. Confidence Calibration
- Apply Platt scaling or isotonic regression to model probabilities
- Use only train/validation period (pre-2025) for calibration fitting
- Apply calibrated thresholds to evaluation period
- Report coverage and accuracy at calibrated thresholds

### A2. Probability Thresholding
- Pre-register threshold selection using training/validation only
- Do not select thresholds based on evaluation labels
- Test thresholds: 0.55, 0.60, 0.65, 0.70, 0.75
- Report coverage floor compliance (>= 30%) and row count (>= 1000)

### A3. Coverage Floors
- Minimum coverage: 30% of parent model/horizon set
- Minimum rows: 1000
- Report all slices that meet both floors and exceed 60% accuracy

## B. Model Tuning

### B1. LightGBM h=20 Tuning
- Tune: num_leaves, learning_rate, n_estimators, min_child_samples, max_depth
- Use time-series-safe walk-forward validation
- No hyperparameter search on evaluation labels
- Compare tuned vs untuned on evaluation period

### B2. XGBoost h=20 Tuning
- Tune: max_depth, learning_rate, n_estimators, subsample, colsample_bytree
- Same walk-forward validation protocol
- Compare tuned vs untuned

### B3. Random Forest h=20 Tuning
- Tune: n_estimators, max_depth, min_samples_split, min_samples_leaf
- Same walk-forward validation protocol
- Compare tuned vs untuned

### B4. Stacking v2
- Use only out-of-fold time-series-safe base predictions
- Meta-learner: logistic regression or simple weighted average
- No leakage from evaluation period into meta-features
- Compare stacking v2 vs stacking v1 vs best base model

## C. Feature Engineering

### C1. Lagged Returns
- Add lagged return features: lag_1, lag_4, lag_8, lag_20
- All lags computed from historical data only
- No future values used

### C2. Rolling Volatility
- Add rolling volatility features: roll_vol_20, roll_vol_60
- Computed from historical data only

### C3. Rolling Momentum
- Add rolling momentum features: mom_20, mom_60, mom_120
- Computed from historical data only

### C4. Volume Features
- Add volume ratio, volume momentum features
- Computed from historical data only

### C5. Market-Index Context
- Use existing validated hourly index cache (VNINDEX, HNXINDEX, etc.)
- Add index return, index volatility as features
- Lagged index features only (no future index values)

## D. Ticker/Slice Handling

### D1. Per-Ticker Strength Report
- Report per-ticker accuracy from existing benchmark
- Identify strongest and weakest tickers
- Do not drop weak tickers from global claim

### D2. Tradable Diagnostic Subset
- Allow subset analysis only as conditional/exploratory
- Clearly label as subset, not full VN30
- Report subset composition and rationale

## E. Validation

### E1. Walk-Forward Split
- Maintain existing walk-forward evaluation protocol
- Train on data before forecast chunk start
- Evaluate on forecast chunk only
- No train/eval leakage

### E2. No Post-Hoc Threshold on Final Eval
- Threshold selection must use train/validation only
- Separate validation slice before final eval if possible
- If no separate validation exists, use time-based split within training period

### E3. Label Leakage Prevention
- No training on evaluation labels
- No feature computation using future values
- No threshold selection using evaluation accuracy

## F. Success Criteria

### F1. Global >60%
- All 30 active tickers
- Full evaluation coverage
- All model/horizon combinations
- No filtering or subsetting

### F2. Coverage-Qualified >60%
- Coverage >= 30% of parent set
- Rows >= 1000
- Confidence threshold pre-registered
- Clearly labeled as conditional

### F3. Ticker/Subset >60%
- Clearly labeled as subset
- Subset composition disclosed
- Not presented as full VN30 result
- Labeled as exploratory/diagnostic

## Risk Assessment

- **Low risk**: Confidence thresholding with pre-registered thresholds
- **Medium risk**: Model tuning with walk-forward validation
- **Medium risk**: Feature engineering with lagged values only
- **Higher risk**: Stacking v2 (requires careful OOF implementation)

## Timeline

1. Run audit and diagnostics (Phase 1-2)
2. Review results and select experiments
3. Implement safe experiments (Phase 4)
4. Report results (Phase 5-6)

## Boundary

- No trading-readiness, profitability, or live deployment claim
- No paper or DOCX generation
- No new data fetching
- No main branch modifications
- All experiments under separate output directory