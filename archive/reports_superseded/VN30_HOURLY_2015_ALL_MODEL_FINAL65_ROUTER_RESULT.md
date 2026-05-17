# VN30 Hourly 2015 - All-Model Final65 Router Result

## Baseline60 Locked Result

- **Model**: Random Forest
- **Horizon**: h=60
- **Target**: Absolute direction
- **Final accuracy**: 60.22%
- **Coverage**: 100%
- **Rows**: 3,474
- **Claim level**: global_full_universe

## Previous Attempts

- **RF-only final65 focus v1**: Failed (59.70%, gap 5.30 pp)
- **RF h=60 router v2**: Failed (59.87%, gap 5.13 pp)

## All-Model Final65 Router Results

### Experiments Run

- 34 base experiments across 3 models (LightGBM, XGBoost, Random Forest), 6 horizons (8, 20, 40, 60, 80, 120), and 3 target types (absolute, relative VN30, relative VNINDEX)
- Policies applied: confidence abstention (19 thresholds), per-ticker whitelist (3 min rows)
- Total policies evaluated: ~1,000+

### Best Selected Policy

- **Policy type**: Per-ticker whitelist
- **Model**: Random Forest
- **Horizon**: h=60
- **Target**: Absolute direction
- **Validation accuracy**: 59.64%
- **Validation coverage**: 100.0%
- **Validation rows**: 30,030
- **Final accuracy**: 59.64%
- **Final coverage**: 100.0%
- **Final rows**: 3,474
- **Active tickers**: 30 (all)

### Top Qualified Results (coverage >=30%, rows >=1000)

| Policy | Model | Horizon | Target | Final Accuracy | Coverage | Rows |
|--------|-------|---------|--------|----------------|----------|------|
| Per-ticker whitelist | Random Forest | 60 | Absolute | 59.64% | 100.0% | 3,474 |
| Per-ticker whitelist | LightGBM | 60 | Absolute | 59.12% | 100.0% | 3,474 |
| Per-ticker whitelist | Random Forest | 40 | Absolute | 58.17% | 100.0% | 4,074 |
| Confidence abstention | XGBoost | 8 | Absolute | 57.99% | 31.8% | 1,602 |
| Confidence abstention | XGBoost | 8 | Absolute | 57.70% | 34.4% | 1,733 |
| Per-ticker whitelist | XGBoost | 60 | Absolute | 57.63% | 100.0% | 3,474 |

### Exploratory Results (accuracy >=65% but coverage <30% or rows <1000)

- Multiple confidence abstention policies reached 65-100% accuracy but with coverage <30% and rows <1000
- These are exploratory only and do not qualify for final65 claim

### Final65 Status

- **Passed**: no
- **Gap to 65**: 5.36 percentage points

### Observations

- Random Forest h=60 remains the best model/horizon combination
- XGBoost h=8 with confidence filtering shows promise (57.99% at 31.8% coverage)
- Relative targets (VN30, VNINDEX) did not outperform absolute targets
- Per-ticker whitelist selected all 30 tickers in most cases (no filtering helped)
- Severe overfitting observed: some policies reached 99%+ validation accuracy but only 50% eval accuracy
- The gap between validation and eval accuracy suggests regime shift between 2024 and 2025-2026
- No policy reached 65% with coverage >=30% and rows >=1000

## Claim Level

- **Failed**: No candidate reaches final65 target
- Best qualified: 59.64% (RF h=60 per-ticker whitelist)
- Best exploratory: 100% accuracy at 0.0% coverage (1 row) - not meaningful

## Boundary

- No trading-readiness, profitability, or live deployment claim.
- No paper/DOCX generated.
- All selection on 2024 validation only, leakage-safe.
