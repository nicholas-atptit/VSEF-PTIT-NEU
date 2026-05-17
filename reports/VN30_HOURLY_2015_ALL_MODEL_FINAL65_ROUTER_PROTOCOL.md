# VN30 Hourly 2015 - All-Model Final65 Router Protocol

## 1. Goal
- Search for a valid final65 candidate across all eligible models/horizons/targets.
- RF h=60 baseline60 remains locked but is not the only candidate.
- Target: final accuracy >=65%, coverage >=30%, rows >=1000.

## 2. Candidate Models
- LightGBM
- XGBoost
- Random Forest
- Stacking if already implemented safely with time-series-safe out-of-fold predictions

## 3. Candidate Horizons
- h=8
- h=20
- h=40
- h=60
- h=80
- h=120

## 4. Candidate Targets
- absolute direction
- relative_to_vn30
- relative_to_vnindex
- noise-band absolute direction
- noise-band relative target
- quantile/event target if already implemented safely

## 5. Candidate Feature Sets
- existing
- stock_lagged
- stock_lagged_plus_market
- combined

## 6. Candidate Policies

### A. Confidence Abstention
- Thresholds 0.50 to 0.95 by 0.025.
- Filter out predictions below threshold.

### B. Per-Ticker Whitelist
- Selected using validation accuracy only.
- Minimum validation rows per ticker: 50, 100, 150.

### C. Ticker + Confidence Policy
- Combine per-ticker filtering with confidence abstention.

### D. Model/Horizon Router
- Choose best model/horizon per ticker using validation only.
- Apply fixed router to final eval.

### E. Target Router
- Choose target type per ticker/model/horizon using validation only.
- Apply fixed policy to final eval.

### F. Regime Abstention/Router
- Use lagged/ex-ante market features only.
- No future regime labels.

### G. Ensemble/Router
- Validation-weighted ensemble.
- Per-ticker validation-weighted ensemble.
- Per-horizon validation-weighted ensemble.
- No final eval weighting.

## 7. Selection Rule
- Select policies using 2024 validation only.
- Primary objective: maximize validation accuracy subject to validation coverage >=30% and validation rows >=1000.
- Tie-breakers: higher validation coverage, then simpler policy, then higher validation rows.
- Apply selected policy once to final eval.

## 8. Claim Levels
- `final65_coverage_qualified`: final accuracy >=65, coverage >=30%, rows >=1000, validation-selected.
- `exploratory`: accuracy >=65 but coverage <30 or rows <1000.
- `invalid`: selected using final labels.
- `failed`: no candidate reaches target.

## 9. Forbidden
- Final evaluation label selection.
- Post-hoc final-only filtering.
- Dropping weak tickers and calling it full VN30.
- Trading/profitability claims.
- Paper/DOCX generation.

## 10. Boundary
- No trading-readiness, profitability, or live deployment claim.
- All selection on 2024 validation only, leakage-safe.
