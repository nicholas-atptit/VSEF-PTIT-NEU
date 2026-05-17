# VN30 Hourly 2015 - Final65 Focus v3 Claim Register

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
- RF h=60 absolute direction: 60.31%, 100% coverage, 3,474 rows

### Final 65 Safe Claims

**None.** No result reached >=65%.

### Conditional Claims

**None.** No result reached >=65% with coverage qualification.

### Exploratory Claims

| Policy Type | Final Accuracy | Coverage | Rows | Active Tickers |
|-------------|----------------|----------|------|----------------|
| Meta-label abstention | 61.28% | 25.23% | 1,028 | 30 |
| Market-regime abstention | 58.32% | 100.0% | 4,074 | 30 |
| Refined confidence | 58.32% | 100.0% | 4,074 | 30 |

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
| 2026-05-17 | Final65 focus v3 | exploratory | Final65 FAIL |

## Boundary

- No trading-readiness, profitability, or live deployment claim.
