# VN30 Hourly 2015 - All-Model Final65 Router Claim Register

## Classification Framework

| Class | Definition |
|-------|-----------|
| safe | Full-universe >=60% or >=65%, no filter |
| conditional | >=60% or >=65% with coverage/rows disclosure |
| exploratory | Post-hoc observations, below targets |
| unsafe | Violates rules |

## Registered Claims

### Baseline 60 Safe Claims

**Retained from previous tag.**
- RF h=60 absolute direction: 60.22%, 100% coverage, 3,474 rows

### Final 65 Safe Claims

**None.** No result reached >=65%.

### Conditional Claims

**None.** No result reached >=65% with coverage qualification.

### Exploratory Claims

| Policy Type | Model | Horizon | Target | Final Accuracy | Coverage | Rows |
|-------------|-------|---------|--------|----------------|----------|------|
| Per-ticker whitelist | Random Forest | 60 | Absolute | 59.64% | 100.0% | 3,474 |
| Per-ticker whitelist | LightGBM | 60 | Absolute | 59.12% | 100.0% | 3,474 |
| Per-ticker whitelist | Random Forest | 40 | Absolute | 58.17% | 100.0% | 4,074 |
| Confidence abstention | XGBoost | 8 | Absolute | 57.99% | 31.8% | 1,602 |
| Confidence abstention | XGBoost | 8 | Absolute | 57.70% | 34.4% | 1,733 |
| Per-ticker whitelist | XGBoost | 60 | Absolute | 57.63% | 100.0% | 3,474 |

### Unsafe Claims

**None permitted.**

## Unsafe Always

- Global 65 if only conditional policy passes
- Trading readiness
- Profitability
- Live deployment
- Final65 claim if coverage <30% or rows <1000
- Policy selected using final labels
- High-accuracy low-coverage results presented as meaningful

## Register History

| Date | Claim | Class | Status |
|------|-------|-------|--------|
| 2026-05-17 | All-model final65 router | exploratory | Final65 FAIL |

## Boundary

- No trading-readiness, profitability, or live deployment claim.
