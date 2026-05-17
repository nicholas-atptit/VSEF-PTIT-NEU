# VN30 Hourly 2015 - Target 60/65 Hard Optimization Claim Register

## Classification Framework

| Class | Definition |
|-------|-----------|
| safe | Full-universe >=60% or >=65%, no filter |
| conditional | >=60% or >=65% with coverage/rows disclosure |
| exploratory | Post-hoc observations, low coverage |
| unsafe | Violates rules |

## Registered Claims

### Baseline 60 Safe Claims

**None.** No full-universe result reached >=60%.

### Final 65 Safe Claims

**None.** No result reached >=65%.

### Conditional Claims

| Strategy | Model | Horizon | Accuracy | Coverage | Rows |
|----------|-------|---------|----------|----------|------|
| Calibration threshold | XGBoost | 8 | 56.13% | 79.06% | 3,980 |
| Calibration threshold | LightGBM | 8 | 55.46% | 67.98% | 3,422 |
| Calibration threshold | XGBoost | 20 | 53.59% | 61.32% | 2,866 |
| Calibration threshold | LightGBM | 20 | 52.92% | 67.76% | 3,167 |
| Calibration threshold | LightGBM | 4 | 52.44% | 69.07% | 3,560 |

### Exploratory Claims

- Per-ticker models: 51.10% global (worse than global models)
- Router: 51.10% global (same as per-ticker)
- Weighted ensemble: 54.63% global (best global but below 60%)

### Unsafe Claims

**None permitted.**

## Unsafe Always

- Profitability claims
- Trading readiness claims
- Live deployment claims
- Full global 65 if only coverage-qualified
- Stable universal 65 if only one validation-selected result

## Register History

| Date | Claim | Class | Status |
|------|-------|-------|--------|
| 2026-05-17 | Hard opt v2 | - | All failed |

## Boundary

- No trading-readiness, profitability, or live deployment claim.