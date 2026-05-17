# VN30 Hourly 2015 - Target 60/65 Protocol

## 1. Baseline Target

- Full active VN30 universe (30 tickers)
- Full final evaluation window (2025-01-01 to 2026-05-14)
- No confidence filter, no ticker subset, no regime subset
- Target accuracy >= 60%
- Label: baseline_60_pass

## 2. Final Target

- Prefer full-universe final accuracy >= 65%
- If not possible, allow coverage-qualified:
  - accuracy >= 65%
  - coverage >= 30%
  - rows >= 1000
  - threshold selected using 2024 validation only
  - final 2025-2026 labels used only once for scoring
- Anything below coverage 30% is exploratory only
- Label: final_65_global_pass / final_65_coverage_qualified_pass / exploratory_only

## 3. Validation Design

| Split | Period | Purpose |
|-------|--------|---------|
| Train | 2015-01-01 to 2023-12-31 | Model training |
| Validation | 2024-01-01 to 2024-12-31 | Hyperparameter/threshold selection |
| Final evaluation | 2025-01-01 to 2026-05-14 | Scoring only |

## 4. Success Labels

- baseline_60_pass
- final_65_global_pass
- final_65_coverage_qualified_pass
- exploratory_only

## 5. Boundary

- No trading-readiness, profitability, or live deployment claim
- No paper or DOCX generation
- No new data fetching
- No main branch modifications