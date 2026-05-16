# VN30 Hourly 2015 - Target 60/65 Claim Register

## Classification Framework

| Class | Definition |
|-------|-----------|
| baseline_60_safe | Full-universe >=60%, no filtering |
| final_65_safe | Full-universe >=65%, no filtering |
| conditional | >=60% or >=65% with coverage/rows disclosure |
| exploratory | Post-hoc observations, low coverage |
| unsafe | Violates leakage, benchmark, or data rules |

## Registered Claims

### Baseline 60 Safe Claims

**None.** No full-universe model/horizon reached >=60%.

### Final 65 Safe Claims

**None.** No full-universe result reached >=65%.

### Conditional Claims

| Model | Horizon | Feature Set | Threshold | Accuracy | Coverage | Rows | Target |
|-------|---------|-------------|-----------|----------|----------|------|--------|
| XGBoost | 8 | C | 0.625 | 56.76% | 66.39% | 3,342 | final_65 (failed) |
| XGBoost | 8 | B | 0.725 | 56.29% | 49.76% | 2,505 | final_65 (failed) |
| XGBoost | 8 | A | 0.55 | 55.86% | 86.00% | 4,329 | final_65 (failed) |
| LightGBM | 8 | A | 0.675 | 55.46% | 61.86% | 3,114 | final_65 (failed) |
| LightGBM | 8 | C | 0.675 | 55.46% | 61.86% | 3,114 | final_65 (failed) |

### Exploratory Claims

- Random Forest h=1 C, threshold 0.65: 56.42%, 15.45% coverage, 810 rows
- Random Forest h=8 D, threshold 0.675: 55.12%, 12.79% coverage, 644 rows

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
| 2026-05-17 | Target 60/65 phase | - | All failed |

## Boundary

- No trading-readiness, profitability, or live deployment claim.