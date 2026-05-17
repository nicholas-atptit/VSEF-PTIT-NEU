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

**2 conditional candidates registered (from optimization v1).**

| Model | Horizon | Feature Set | Threshold | Eval Accuracy | Observations | Coverage | Val Accuracy | Val Coverage |
|-------|---------|-------------|-----------|---------------|--------------|----------|-------------|--------------|
| LightGBM | 8 | A (existing) | 0.5 | 60.35% | 1,806 | 36.46% | 52.66% | 32.0% |
| LightGBM | 8 | C (A + index) | 0.5 | 60.35% | 1,806 | 36.46% | 52.66% | 32.0% |

**Notes:**
- Threshold selected on 2024 validation period only.
- Applied once to 2025-2026 evaluation period.
- Coverage >= 30%: YES (36.46%)
- Rows >= 1000: YES (1,806)
- All 30 VN30 tickers included.
- Feature set C (index context) does not improve over feature set A.

### Exploratory >60% Observations

**From audit v1 (previous):**
- 160 ticker-level candidates pass 60% (best: LightGBM h=8, ACB at 61.73%, 162 obs)
- Multiple confidence-filtered slices pass 60% with low coverage (<10%)

**From optimization v1:**
- Random Forest h=1, feature set D, threshold 0.6: 66.67% accuracy, 3 obs, 0.06% coverage (not meaningful)

### Unsafe Claims

**None permitted.**

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
| 2026-05-17 | LGBM h=8, threshold 0.5: 60.35%, 1806 obs, 36.46% cov | Conditional | Registered (optimization v1) |

## Boundary

- No trading-readiness, profitability, or live deployment claim is made.
- This register documents analytical observations only.
- All claims must be validated against the rules above.