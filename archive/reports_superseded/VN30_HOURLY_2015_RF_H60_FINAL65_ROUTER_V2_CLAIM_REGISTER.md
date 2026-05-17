# VN30 Hourly 2015 - RF h=60 Final65 Router v2 Claim Register

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

| Policy Type | Threshold | Val Accuracy | Final Accuracy | Coverage | Rows |
|-------------|-----------|--------------|----------------|----------|------|
| Per-ticker whitelist | - | 49.24% | 59.87% | 100.0% | 3,474 |
| Confidence abstention | Various | ~49% | <60% | Varies | Varies |
| Market regime abstention | Various | ~49% | <60% | Varies | Varies |
| Ticker + confidence | Various | ~49% | <60% | Varies | Varies |
| Ticker + regime + confidence | Various | ~49% | <60% | Varies | Varies |

### Unsafe Claims

**None permitted.**

## Unsafe Always

- Global 65 if only conditional policy passes
- Trading readiness
- Profitability
- Live deployment
- Final65 claim if coverage <30% or rows <1000
- Policy selected using final labels

## Register History

| Date | Claim | Class | Status |
|------|-------|-------|--------|
| 2026-05-17 | Router v2 | exploratory | Final65 FAIL |

## Boundary

- No trading-readiness, profitability, or live deployment claim.
