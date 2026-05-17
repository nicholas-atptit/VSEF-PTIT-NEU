# VN30 Hourly 2015 - Final65 Focus v3 Protocol

## 1. Locked Baseline
- **Model**: Random Forest
- **Horizon**: h=60
- **Target**: Absolute direction
- **Final accuracy**: 60.31%
- **Coverage**: 100%
- **Rows**: 3,474
- This remains the valid baseline60 result.

## 2. Focus Candidate
- **Model**: Random Forest
- **Horizon**: h=40
- **Target**: Absolute direction
- **Policy**: Confidence abstention
- **Current best**: 61.48%
- **Coverage**: 31.5%
- **Rows**: 1,285
- **Gap to 65**: 3.52 percentage points

## 3. Candidate Policies

### A. Refined Confidence Threshold
- Thresholds: 0.50 to 0.975 by 0.01
- Selected using validation only
- Goal: find optimal threshold that maximizes accuracy while maintaining coverage >=30% and rows >=1000

### B. Per-Ticker Confidence Threshold
- Each ticker threshold selected on validation only
- Minimum validation rows per ticker: 50, 100, 150
- Final policy must still meet global coverage >=30% and rows >=1000

### C. Ticker Whitelist + Confidence
- Select tickers and threshold using validation only
- Must disclose active ticker count
- If active ticker count < 30, claim must be conditional/subset, not full universe

### D. Market-Regime Abstention
- Use only lagged/ex-ante market features
- Candidate filters:
  - Lagged VN30 return state
  - Lagged VNINDEX return state
  - Rolling market volatility bucket
  - Previous-session trend state
- No future regime labels

### E. Meta-Label Abstention
- Base model: RF h=40
- Meta-label predicts whether forecast is likely correct
- Train meta-labeler using train/validation only
- Select abstention threshold on 2024 validation only
- Final eval only for scoring

### F. Hybrid Policy
- Combine confidence + ticker + lagged market-regime filters
- Only if validation rows remain sufficient

## 4. Claim Levels
- `final65_coverage_qualified`: accuracy >=65, coverage >=30%, rows >=1000, validation-selected
- `exploratory`: accuracy >=65 but coverage <30% or rows <1000
- `failed`: no candidate reaches final65 target
- `invalid`: selected using final labels

## 5. Forbidden
- Final-eval label selection
- Post-hoc final-only filtering
- Hiding coverage
- Trading/profitability claims
- Paper/DOCX generation

## 6. Selection Rule
- Select policy using 2024 validation only
- Primary objective: maximize validation accuracy subject to validation coverage >=30% and validation rows >=1000
- Tie-breakers: higher validation coverage, higher validation rows, simpler policy
- Apply selected policy once to final evaluation

## 7. Boundary
- No trading-readiness, profitability, or live deployment claim
- All selection on 2024 validation only, leakage-safe
- Canonical evaluator v1.0.0 used for all metrics
