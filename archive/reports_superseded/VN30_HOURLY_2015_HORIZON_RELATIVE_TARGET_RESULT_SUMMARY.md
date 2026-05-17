# VN30 Hourly 2015 - Horizon & Relative Target Result Summary

## Previous Results

- Original binary benchmark: 51.34%
- Hard optimization v2 best global: 54.63%
- Target redesign best global: 55.76%
- Target redesign best coverage-qualified: 58.22%

## Targets

- Baseline target: 60% full-universe
- Final target: 65% coverage-qualified (coverage >=30%, rows >=1000)

## New Horizon & Relative Target Results

### Best Global Result

- **60.22%** (Absolute direction, Random Forest, h=60, feature set C)
- Coverage: 100.0%, Rows: 3,474
- Baseline 60 target: **PASS**
- Margin above 60: **0.22 percentage points**

### Best Coverage-Qualified Result

- **62.73%** (Absolute direction, Random Forest, h=60, feature set C, confidence filtered)
- Coverage: 61.2%, Rows: 2,125
- Final 65 target: **FAIL**
- Gap to 65: **2.27 percentage points**

### Strategy Results

| Target Type | Market | Horizon | Model | Noise Band | Accuracy | Coverage | Rows | Full Universe | Qualified |
|-------------|--------|---------|-------|------------|----------|----------|------|---------------|-----------|
| Absolute | - | 60 | Random Forest | 0.0 | 60.22% | 100.0% | 3,474 | Yes | Yes |
| Absolute | - | 60 | LightGBM | 0.0 | 59.51% | 95.0% | 3,302 | Yes | Yes |
| Absolute | - | 60 | LightGBM | 0.0 | 59.28% | 96.6% | 3,357 | Yes | Yes |
| Absolute | - | 60 | Random Forest | 0.0 | 62.73% | 61.2% | 2,125 | No | Yes |
| Absolute | - | 60 | Random Forest | 0.0 | 62.67% | 52.0% | 1,808 | No | Yes |
| Absolute | - | 60 | Random Forest | 0.0 | 62.35% | 69.0% | 2,396 | No | Yes |
| Relative VNINDEX | VNINDEX | 40 | LightGBM | 0.0 | 59.89% | 100.0% | 4,074 | Yes | Yes |
| Relative VN30 | VN30 | 40 | LightGBM | 0.0 | 59.45% | 100.0% | 4,074 | Yes | Yes |

### Key Observations

- **Horizon h=60 (15 trading days) is optimal** for absolute direction
- **Random Forest outperforms** LightGBM and XGBoost at longer horizons
- **Longer horizons reduce noise** and expose trend structure
- **Relative targets did not outperform** absolute targets at long horizons
- **Confidence filtering improves accuracy** but reduces coverage
- **Best global result (60.22%) barely passes** the 60% threshold
- **Final 65 target still failed** by 2.27 percentage points

### What Failed

- Relative-to-VN30 and relative-to-VNINDEX targets did not outperform absolute
- Noise-band filtering on relative targets provided marginal improvement
- Shorter horizons (h=40) performed worse than h=60
- Longer horizons (h=80, h=120) showed diminishing returns
- No result reached 65% coverage-qualified accuracy

## Claim Level

- **Global full-universe**: Baseline 60 passed (60.22%, 100% coverage, 3,474 rows)
- **Coverage-qualified**: Final 65 failed (best 62.73%, gap 2.27 pp)

## Boundary

- No trading-readiness, profitability, or live deployment claim
- No paper/DOCX generated
- No prediction labels edited manually
- All target thresholds selected on 2024 validation only
- Final evaluation used only once for scoring
