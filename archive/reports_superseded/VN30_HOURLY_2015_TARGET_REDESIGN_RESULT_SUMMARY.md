# VN30 Hourly 2015 - Target Redesign Result Summary

## Previous Results

- Original binary benchmark: 51.34%
- Hard optimization v2 best global: 54.63%
- Hard optimization v2 best conditional: 56.13%

## Targets

- Baseline target: 60% full-universe
- Final target: 65% coverage-qualified (coverage >=30%, rows >=1000)

## New Target Redesign Results

### Best Global Result

- **55.76%** (Noise-band 0.05%/0.10%, Three-class 0.05%/0.10%, XGBoost h=8, feature set C)
- Baseline 60 target: **FAIL**
- Gap to 60: **4.24 percentage points**

### Best Coverage-Qualified Result

- **58.22%** (Quantile top/bottom 40%, XGBoost h=8, feature set C)
- Coverage: 74.3%, Rows: 3,741
- Final 65 target: **FAIL**
- Gap to 65: **6.78 percentage points**

### Strategy Results

| Target Type | Threshold | Model | Horizon | Accuracy | Coverage | Rows | Full Universe | Qualified |
|-------------|-----------|-------|---------|----------|----------|------|---------------|-----------|
| Quantile | 0.40 | XGBoost | 8 | 58.22% | 74.3% | 3,741 | No | Yes |
| Quantile | 0.35 | XGBoost | 8 | 57.75% | 71.1% | 3,581 | No | Yes |
| Quantile | 0.30 | XGBoost | 8 | 57.25% | 67.9% | 3,420 | No | Yes |
| Noise-band | 0.50% | XGBoost | 8 | 56.62% | 91.6% | 4,610 | No | Yes |
| Vol-adjusted | 0.3 | XGBoost | 8 | 56.51% | 87.2% | 4,387 | No | Yes |
| Noise-band | 0.10% | XGBoost | 8 | 55.76% | 98.3% | 4,950 | Yes | Yes |
| Three-class | 0.10% | XGBoost | 8 | 55.76% | 98.3% | 4,950 | Yes | Yes |
| Binary | - | XGBoost | 8 | 54.67% | 100% | 5,034 | Yes | Yes |

### What Failed

- No strategy achieved >=60% full-universe accuracy
- No strategy achieved >=65% coverage-qualified accuracy
- Quantile targets improved accuracy but lost coverage (74.3% max)
- Noise-band targets provided modest improvement at low thresholds
- Volatility-adjusted targets showed marginal improvement
- Three-class targets performed similarly to noise-band at equivalent thresholds
- Meta-labeling did not outperform base models

### Observations

- XGBoost consistently outperformed LightGBM and Random Forest
- Horizon h=8 was the best across most target types
- Quantile targets (stronger events) showed highest accuracy but at coverage cost
- The accuracy-coverage tradeoff is clear: higher thresholds = higher accuracy but lower coverage
- Even at 74.3% coverage, quantile targets only reached 58.22%, still 6.78 pp below 65%

## Claim Level

- **Exploratory**: No baseline or final target passed
- Best global: 55.76% (gap 4.24 pp to 60%)
- Best coverage-qualified: 58.22% (gap 6.78 pp to 65%)

## Boundary

- No trading-readiness, profitability, or live deployment claim
- No paper/DOCX generated
- No prediction labels edited manually
- All target thresholds selected on 2024 validation only
- Final evaluation used only once for scoring
