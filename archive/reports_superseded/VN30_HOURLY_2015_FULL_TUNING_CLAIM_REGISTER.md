# VN30 Hourly 2015 - Full Tuning Claim Register

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

| Model | Horizon | Target | Policy | Accuracy | Coverage | Rows |
|-------|---------|--------|--------|----------|----------|------|
| Random Forest | 60 | Absolute | Per-ticker whitelist | 60.31% | 100.0% | 3,474 |

### Final 65 Safe Claims

**None.** No result reached >=65%.

### Conditional Claims

| Model | Horizon | Target | Policy | Accuracy | Coverage | Rows |
|-------|---------|--------|--------|----------|----------|------|
| Random Forest | 40 | Absolute | Confidence abstention | 61.48% | 31.5% | 1,285 |
| Random Forest | 60 | Absolute | Per-ticker whitelist | 60.31% | 100.0% | 3,474 |
| LightGBM | 40 | Absolute | Confidence abstention | 59.31% | 31.9% | 1,300 |
| LightGBM | 40 | Absolute | Confidence abstention | 59.27% | 30.2% | 1,230 |
| LightGBM | 40 | Absolute | Confidence abstention | 59.25% | 30.9% | 1,259 |

### Exploratory Claims

- Multiple confidence abstention policies reached 65-100% accuracy but with coverage <30% and rows <1000
- Relative targets showed severe overfitting: 99%+ validation accuracy but 50% eval accuracy
- XGBoost h=8 with confidence abstention reached 59.10% at 30.4% coverage

### Unsafe Claims

**None permitted.**

## Unsafe Always

- Profitability claims
- Trading readiness claims
- Live deployment claims
- Global 65 if only conditional candidate passes
- Any claim from final-label-selected policy
- Hiding coverage
- Comparing changed target to original target without disclosure

## Register History

| Date | Claim | Class | Status |
|------|-------|-------|--------|
| 2026-05-17 | Full tuning sweep | safe (baseline60) | Baseline60 PASS, Final65 FAIL |

## Boundary

- No trading-readiness, profitability, or live deployment claim.
