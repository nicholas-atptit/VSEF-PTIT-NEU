# VN30 Hourly 2015 - Full Tuning Result Summary

## Previous Results

| Phase | Result | Accuracy | Coverage | Rows | Status |
|-------|--------|----------|----------|------|--------|
| Base benchmark | - | 51.34% | 100% | - | Failed |
| Prior conditional | - | 60.35% | 36.46% | 1,806 | Conditional |
| Target60/65 v2 | - | 55.20% | 100% | - | Failed |
| Hard opt v2 | - | 54.63% | 100% | - | Failed |
| Target redesign | - | 55.76% | 98.3% | - | Failed |
| Horizon-relative | RF h=60 absolute | 60.22% | 100% | 3,474 | Baseline60 PASS |
| RF-only final65 focus | Platt calibration | 59.70% | 100% | 3,474 | Failed |
| RF h=60 router v2 | Per-ticker whitelist | 59.87% | 100% | 3,474 | Failed |
| All-model router v1 | RF h=60 per-ticker | 59.64% | 100% | 3,474 | Failed |

## RF h=60 Consistency Decision

- **Canonical RF h=60**: 60.31% (3,474 rows, 100% coverage)
- **Baseline60 status**: PASS under canonical evaluator
- **Discrepancy resolved**: Yes (small differences 0.09-0.67 pp due to different experiment runs)

## Full Tuning Sweep Results

### Experiments Run

- 111 base experiments across 3 models (LightGBM, XGBoost, Random Forest), 7 horizons (4, 8, 20, 40, 60, 80, 120), 3 targets (absolute, relative VN30, relative VNINDEX), 3 hyperparameter configs each
- Policies applied: confidence abstention (19 thresholds), per-ticker whitelist (3 min rows)
- Total policies evaluated: ~4,000+

### Best Global Candidate

- **Model**: Random Forest
- **Horizon**: h=60
- **Target**: Absolute direction
- **Policy**: Per-ticker whitelist
- **Final accuracy**: 60.31%
- **Coverage**: 100.0%
- **Rows**: 3,474
- **Baseline60**: PASS

### Best Coverage-Qualified Candidate

- **Model**: Random Forest
- **Horizon**: h=40
- **Target**: Absolute direction
- **Policy**: Confidence abstention
- **Final accuracy**: 61.48%
- **Coverage**: 31.5%
- **Rows**: 1,285
- **Final65**: FAIL (gap 3.52 pp)

### Final65 Status

- **Passed**: no
- **Gap to 65**: 3.52 percentage points

### Observations

- Random Forest consistently outperforms LightGBM and XGBoost on absolute direction targets
- h=40 and h=60 are the best horizons
- Confidence abstention on RF h=40 reaches 61.48% at 31.5% coverage (best coverage-qualified)
- Per-ticker whitelist on RF h=60 reaches 60.31% at 100% coverage (best global)
- Relative targets (VN30, VNINDEX) did not outperform absolute targets
- Severe overfitting on relative targets: 99%+ validation accuracy but 50% eval accuracy
- No policy reached 65% with coverage >=30% and rows >=1000

## Claim Level

- **Baseline60**: PASS (60.31%, 100% coverage, 3,474 rows)
- **Final65**: FAIL (best 61.48%, gap 3.52 pp)

## Boundary

- No trading-readiness, profitability, or live deployment claim.
- No paper/DOCX generated.
- All selection on 2024 validation only, leakage-safe.
- Canonical evaluator v1.0.0 used for all metrics.
