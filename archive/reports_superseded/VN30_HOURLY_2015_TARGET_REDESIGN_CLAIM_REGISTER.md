# VN30 Hourly 2015 - Target Redesign Claim Register

## Classification Framework

| Class | Definition |
|-------|-----------|
| safe | Full-universe >=60% or >=65%, no filter |
| conditional | >=60% or >=65% with coverage/rows disclosure |
| exploratory | Post-hoc observations, below targets |
| unsafe | Violates rules |

## Registered Claims

### Baseline 60 Safe Claims

**None.** No full-universe result reached >=60%.

### Final 65 Safe Claims

**None.** No result reached >=65%.

### Conditional Claims

**None.** No result reached >=60% or >=65% with coverage qualification.

### Exploratory Claims

| Target Type | Threshold | Model | Horizon | Accuracy | Coverage | Rows |
|-------------|-----------|-------|---------|----------|----------|------|
| Quantile | 0.40 | XGBoost | 8 | 58.22% | 74.3% | 3,741 |
| Quantile | 0.35 | XGBoost | 8 | 57.75% | 71.1% | 3,581 |
| Quantile | 0.30 | XGBoost | 8 | 57.25% | 67.9% | 3,420 |
| Noise-band | 0.50% | XGBoost | 8 | 56.62% | 91.6% | 4,610 |
| Vol-adjusted | 0.3 | XGBoost | 8 | 56.51% | 87.2% | 4,387 |
| Noise-band | 0.10% | XGBoost | 8 | 55.76% | 98.3% | 4,950 |
| Three-class | 0.10% | XGBoost | 8 | 55.76% | 98.3% | 4,950 |
| Binary | - | XGBoost | 8 | 54.67% | 100% | 5,034 |

### Unsafe Claims

**None permitted.**

## Unsafe Always

- Profitability claims
- Trading readiness claims
- Live deployment claims
- Full global 65 if only coverage-qualified
- Stable universal performance if result exists only under one target definition

## Register History

| Date | Claim | Class | Status |
|------|-------|-------|--------|
| 2026-05-17 | Target redesign v1 | - | All failed |

## Boundary

- No trading-readiness, profitability, or live deployment claim.
