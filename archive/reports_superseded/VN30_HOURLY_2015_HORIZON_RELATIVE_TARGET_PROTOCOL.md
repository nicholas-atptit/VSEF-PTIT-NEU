# VN30 Hourly 2015 - Horizon Expansion & Relative Target Protocol

## Why Previous Phases Failed

1. **Absolute binary hourly direction is noisy**
   - Hourly returns are often <0.1%, dominated by micro-structure noise
   - Forcing every tiny move into up/down creates a near-random target
   - Best global result across all phases remained below 60%
   - Best conditional result remained below 65%
   - Further shallow tuning on the same target is unlikely to solve the fundamental noise problem

2. **Target redesign results**
   - Noise-band filtering: modest improvement at coverage cost
   - Volatility-adjusted: marginal gains
   - Quantile targets: 58.22% at 74.3% coverage (still 6.78 pp below 65%)
   - Three-class: similar to noise-band
   - Meta-labeling: no significant improvement over base models

## Horizon Expansion

### Rationale
- Longer horizons may reduce micro-noise and expose trend structure
- Hourly noise averages out over longer periods
- Trend-following signals may be more learnable at longer horizons

### Horizons to Test
- h=40 (10 trading days)
- h=60 (15 trading days)
- h=80 (20 trading days)
- h=120 (30 trading days)

## Relative Target

### Definition
```
relative_return = stock_future_return_h - market_future_return_h
label = 1 if relative_return > 0 else 0
```

### Rationale
- Absolute direction is dominated by market beta
- Relative outperformance isolates stock-specific alpha
- Market-neutral targets may be more learnable
- Removes systematic market movement from the signal

### Market References
- **VN30 index** (preferred): direct universe benchmark
- **VNINDEX**: broader market reference
- Use only existing hourly index cache
- No future features used for prediction
- Future market return used only to define the target label (same as future stock return)

## Target Variants

### A. Absolute Direction, Long Horizons
- `label = 1` if `stock_future_return_h > 0` else `0`
- Horizons: h=40, 60, 80, 120

### B. Relative to VN30, Long Horizons
- `label = 1` if `stock_future_return_h - vn30_future_return_h > 0` else `0`
- Horizons: h=40, 60, 80, 120

### C. Relative to VNINDEX, Long Horizons
- `label = 1` if `stock_future_return_h - vnindex_future_return_h > 0` else `0`
- Horizons: h=40, 60, 80, 120

### D. Relative Target with Noise Band
- Ignore observations where `|relative_return| < threshold`
- Noise bands: 0.05%, 0.10%, 0.15%, 0.20%, 0.30%
- Remaining samples classified as outperform/underperform

### E. Relative Target with Confidence Filtering
- Model predicts probability of outperformance
- Apply confidence threshold to filter low-confidence predictions
- Thresholds: 0.50 to 0.95 by 0.025
- Primary coverage floor: >=30%
- Minimum rows: >=1000

## Success Criteria

### Baseline Target Pass
- Full active universe, no confidence filter
- Accuracy >= 60%

### Final Target Pass
- Coverage-qualified accuracy >= 65%
- Coverage >= 30%
- Rows >= 1000

### Exploratory
- Anything not meeting baseline or final criteria

## Leakage Control

- Model/tuning/threshold selection uses train + 2024 validation only
- Final 2025-2026 labels used only once for scoring
- No final-eval threshold selection
- All target definitions generated programmatically
- Rolling computations use past data only

## Boundary

- No trading-readiness, profitability, or live deployment claim
- No paper/DOCX generated
- No prediction labels edited manually
