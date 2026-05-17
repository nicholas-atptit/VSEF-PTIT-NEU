# VN30 Hourly 2015 - RF h=60 Final65 Router v2 Protocol

## 1. Goal
- Reach final accuracy >=65%.
- Coverage must be >=30%.
- Rows must be >=1000.
- Policy must be selected on 2024 validation only.

## 2. Candidate Policies

### A. Confidence Abstention
- Select confidence/probability thresholds using validation only.
- Thresholds: 0.50 to 0.95 by 0.025.
- Filter out predictions below threshold.

### B. Per-Ticker Whitelist
- Select tickers where RF h60 validation accuracy is strong.
- Minimum validation rows per ticker: 50, 100, 150.
- Apply selected ticker whitelist to final eval.
- Must report active ticker count and coverage.

### C. Ticker + Confidence Policy
- Select ticker whitelist and confidence threshold using validation only.
- Combine per-ticker filtering with confidence abstention.

### D. Market Regime Abstention
- Use only ex-ante/lagged market features.
- Candidate filters:
  - Lagged VN30/VNINDEX return regime
  - Rolling market volatility regime
  - Previous-session trend state
- No future regime labels.

### E. Ticker + Regime + Confidence Router
- Only if validation rows remain sufficient.
- Must satisfy coverage >=30% and rows >=1000 for headline.

## 3. Claim Levels
- If >=65 and coverage >=30 and rows >=1000: `conditional_final65_candidate`
- If coverage <30 or rows <1000: `exploratory_only`
- If selected using final labels: `invalid`

## 4. Forbidden
- Final evaluation label selection
- Post-hoc final-only filtering
- Dropping tickers and calling it full VN30
- Trading/profitability claims

## 5. Selection Rule
- Select policy using validation only.
- Primary objective: maximize validation accuracy subject to validation coverage >=30% and validation rows >=1000.
- Tie-break: higher validation coverage.
- Apply selected policy once to final evaluation.
- Also report all candidate policies, but mark post-hoc/exploratory properly.

## 6. Boundary
- No trading-readiness, profitability, or live deployment claim.
- No paper/DOCX generated.
- All selection on 2024 validation only, leakage-safe.
