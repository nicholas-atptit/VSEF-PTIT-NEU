# VN30 Hourly 2015 - Final65 Focus v3 Result

## Baseline60 Locked Result

- **Model**: Random Forest
- **Horizon**: h=60
- **Target**: Absolute direction
- **Final accuracy**: 60.31%
- **Coverage**: 100%
- **Rows**: 3,474
- **Claim level**: global_full_universe

## Previous Best Coverage-Qualified Result

- **Model**: Random Forest
- **Horizon**: h=40
- **Target**: Absolute direction
- **Policy**: Confidence abstention
- **Final accuracy**: 61.48%
- **Coverage**: 31.5%
- **Rows**: 1,285

## Final65 Focus v3 Results

### Policies Tested

1. **Refined confidence threshold** (48 thresholds): No improvement over base
2. **Per-ticker confidence threshold**: No valid combination meeting coverage >=30% and rows >=1000
3. **Ticker whitelist + confidence**: No improvement
4. **Market-regime abstention**: 58.32%, 100% coverage, 4,074 rows
5. **Meta-label abstention**: 61.28% accuracy but 25.23% coverage (below 30% threshold)
6. **Hybrid policy**: No improvement

### Selected Policy (Validation)

- **Policy type**: Meta-label abstention
- **Validation accuracy**: 56.92%
- **Validation coverage**: 35.17%
- **Validation rows**: 10,563
- **Final accuracy**: 61.28%
- **Final coverage**: 25.23%
- **Final rows**: 1,028
- **Active tickers**: 30

### Best Coverage-Qualified Result

- **Policy type**: Market-regime abstention
- **Final accuracy**: 58.32%
- **Coverage**: 100.0%
- **Rows**: 4,074
- **Active tickers**: 30

### Final65 Status

- **Passed**: no
- **Gap to 65**: 6.68 percentage points

### Observations

- Meta-label abstention achieved 61.28% accuracy but only 25.23% coverage (below 30% threshold)
- Market-regime abstention maintained 100% coverage but only 58.32% accuracy
- Refined confidence thresholds did not improve over the previous 61.48% result
- The gap between validation and eval accuracy suggests regime shift
- No policy reached 65% with coverage >=30% and rows >=1000

## Claim Level

- **Failed**: No candidate reaches final65 target
- Best coverage-qualified: 58.32% (market-regime abstention)
- Best exploratory: 61.28% (meta-label abstention, 25.23% coverage)

## Boundary

- No trading-readiness, profitability, or live deployment claim.
- No paper/DOCX generated.
- All selection on 2024 validation only, leakage-safe.
- Canonical evaluator v1.0.0 used for all metrics.
