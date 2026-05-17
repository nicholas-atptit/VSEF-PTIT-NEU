# VN30 Hourly 2015 - Horizon & Relative Target Claim Register

## Classification Framework

| Class | Definition |
|-------|-----------|
| safe | Full-universe >=60% or >=65%, no filter |
| conditional | >=60% or >=65% with coverage/rows disclosure |
| exploratory | Post-hoc observations, below targets |
| unsafe | Violates rules |

## Registered Claims

### Baseline 60 Safe Claims

**PASS.** One full-universe result reached >=60%.

| Target Type | Model | Horizon | Accuracy | Coverage | Rows |
|-------------|-------|---------|----------|----------|------|
| Absolute direction | Random Forest | 60 | 60.22% | 100.0% | 3,474 |

### Final 65 Safe Claims

**None.** No result reached >=65%.

### Conditional Claims

| Target Type | Model | Horizon | Accuracy | Coverage | Rows |
|-------------|-------|---------|----------|----------|------|
| Absolute direction | Random Forest | 60 | 62.73% | 61.2% | 2,125 |
| Absolute direction | Random Forest | 60 | 62.67% | 52.0% | 1,808 |
| Absolute direction | Random Forest | 60 | 62.35% | 69.0% | 2,396 |
| Absolute direction | Random Forest | 60 | 62.05% | 42.4% | 1,473 |
| Absolute direction | Random Forest | 60 | 61.75% | 76.8% | 2,669 |

### Exploratory Claims

- Relative-to-VN30 targets did not outperform absolute targets
- Relative-to-VNINDEX targets showed similar performance to VN30
- Longer horizons (h=80, h=120) showed diminishing returns
- Noise-band filtering on relative targets provided marginal improvement
- Confidence filtering improved accuracy at coverage cost

### Unsafe Claims

**None permitted.**

## Unsafe Always

- Profitability claims
- Trading readiness claims
- Live deployment claims
- Full global 65 if only coverage-qualified
- Comparing relative target directly to absolute target without stating target change
- Claiming stable universal performance based on single result

## Register History

| Date | Claim | Class | Status |
|------|-------|-------|--------|
| 2026-05-17 | Horizon/relative target v1 | safe (baseline 60) | Baseline 60 PASS, Final 65 FAIL |

## Boundary

- No trading-readiness, profitability, or live deployment claim.
