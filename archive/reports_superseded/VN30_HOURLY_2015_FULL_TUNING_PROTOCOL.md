# VN30 Hourly 2015 - Full Tuning Protocol

## 1. Candidate Models
- LightGBM
- XGBoost
- Random Forest
- Stacking v2 if time-series-safe

## 2. Candidate Horizons
- h=4, h=8, h=20, h=40, h=60, h=80, h=120

## 3. Candidate Targets
- absolute direction
- absolute direction with noise band
- relative_to_vn30
- relative_to_vnindex
- relative_to_vn30 noise band
- relative_to_vnindex noise band
- quantile top/bottom event target
- three-class up/flat/down target if already supported safely
- meta-label target if leakage-safe

## 4. Candidate Feature Sets
- existing benchmark features
- stock lagged features
- market-index lagged context
- combined stock + market
- interaction features: stock return minus market return, volatility-normalized return, volume shock, momentum x volatility, market trend x stock momentum
- time features: hour/session if meaningful and already hourly

## 5. Tuning
- Bounded hyperparameter search
- Validation-only selection
- No final-label tuning
- Deterministic seeds
- Keep runtime bounded and checkpointed

## 6. Candidate Policies
- Confidence threshold
- Per-ticker threshold
- Per-ticker model routing
- Model/horizon routing
- Target routing
- Validation-weighted ensemble
- Per-ticker ensemble
- Regime-aware router using lagged/ex-ante regime features only
- Meta-label abstention

## 7. Success Criteria
- baseline_60_pass: full universe, no confidence filter, all 30 tickers, accuracy >=60
- final_65_pass: accuracy >=65, coverage >=30, rows >=1000, selected on validation only
- exploratory: accuracy >=65 but coverage <30 or rows <1000

## 8. Boundary
- No trading-readiness, profitability, or live deployment claim.
- All selection on 2024 validation only, leakage-safe.
