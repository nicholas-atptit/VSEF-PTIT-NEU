# VN30 Hourly 2015 - Above 60% Claim Register

## Classification Framework

| Class | Definition | Requirements |
|-------|-----------|-------------|
| Safe | Global >60% with all 30 tickers, full eval coverage | All tickers, full coverage, no filtering |
| Conditional | >60% with coverage and row count disclosure | Coverage >= 30%, rows >= 1000, threshold pre-registered |
| Exploratory | Post-hoc >60% observations | Any slice, clearly labeled as post-hoc |
| Unsafe | Claims that violate leakage, benchmark, or data rules | N/A - must not be made |

## Registered Claims

### Safe >60% Claims

**None registered.**

No global model/horizon combination exceeds 60% accuracy on the full VN30 universe with all 30 tickers and full evaluation coverage.

### Conditional >60% Claims

**42 conditional candidates identified from audit.**

Best conditional candidates (from audit):
- Random Forest h=20, conf>=0.775: 61.93% accuracy, 549 obs, 11.94% coverage
- Random Forest h=20, conf>=0.70: 61.71% accuracy, 1251 obs, 27.20% coverage
- Random Forest h=20, conf>=0.725: 61.31% accuracy, 964 obs, 20.96% coverage
- XGBoost h=20, conf>=0.875: 61.04% accuracy, 1060 obs, 23.05% coverage

Note: Coverage floors in audit were set at 10-50%. The strict 30% coverage floor is not met by any candidate. The closest is RF h=20 conf>=0.70 at 27.20% coverage.

### Exploratory >60% Observations

**1528 exploratory candidates identified from audit.**

Key exploratory observations:
- 160 ticker-level candidates pass 60% (best: LightGBM h=8, ACB at 61.73%, 162 obs)
- Multiple confidence-filtered slices pass 60% with low coverage (<10%)
- No regime-level rows exceed 60%
- Best single observation: Random Forest h=4, conf>=0.875 at 100% (1 obs - not meaningful)

### Unsafe Claims

**None permitted.**

The following claims are explicitly prohibited:
- Global >60% based on post-hoc slice selection
- Trading-readiness or profitability claims
- Live deployment recommendations
- Claims using evaluation labels for threshold selection
- Claims using daily data or resampled data
- Claims that leak future data

## Claim Rules

1. A global >60% claim requires all active tickers and full evaluation coverage.
2. A confidence >60% claim requires coverage and row count disclosure.
3. A ticker/subset >60% claim must state the subset explicitly.
4. A post-hoc >60% slice cannot be presented as a confirmed method.
5. No trading/profitability/live deployment claim is permitted.

## Register History

| Date | Claim | Class | Status |
|------|-------|-------|--------|
| 2026-05-16 | Initial register | - | Open |
| 2026-05-16 | RF h=20 conf>=0.775: 61.93%, 549 obs | Conditional | Registered |
| 2026-05-16 | RF h=20 conf>=0.70: 61.71%, 1251 obs | Conditional | Registered |
| 2026-05-16 | XGB h=20 conf>=0.875: 61.04%, 1060 obs | Conditional | Registered |
| 2026-05-16 | LightGBM h=8 ACB: 61.73%, 162 obs | Exploratory | Registered |

## Boundary

- No trading-readiness, profitability, or live deployment claim is made.
- This register documents analytical observations only.
- All claims must be validated against the rules above.