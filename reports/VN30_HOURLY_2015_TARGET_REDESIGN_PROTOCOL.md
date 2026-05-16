# VN30 Hourly 2015 - Target Redesign Protocol

## Problem Statement

The current binary direction label (`future return > 0`) forces every hourly observation into up/down, regardless of move magnitude. Hourly returns in VN30 stocks are often small and close to zero, making the target inherently noisy.

### Why the Current Binary Label is Noisy

1. **Micro-moves dominate**: Many hourly returns are <0.1%, essentially flat noise
2. **Bid-ask bounce**: Tick-level noise propagates into hourly candles
3. **Forced classification**: Forcing tiny moves into up/down creates a near-random target
4. **Reduced learnability**: Signal-to-noise ratio is low when most labels represent economically meaningless moves
5. **Model confusion**: Models learn to predict noise rather than meaningful directional moves

### Design Principles

- **No final-eval leakage**: All target thresholds selected using 2024 validation only
- **Coverage transparency**: Always report coverage and row count
- **No unfair comparisons**: Three-class targets reported separately from binary
- **Leakage-safe**: Rolling computations use past data only

## Target Candidates

### A. Original Binary Direction (Baseline)
- `label = 1` if `future_return > 0` else `0`
- Current benchmark baseline
- Full coverage by definition

### B. Noise-Band Binary Direction
- Ignore observations where `|future_return| < threshold`
- Remaining samples classified as up/down
- Thresholds selected on 2024 validation only
- Candidate thresholds: `0.05%`, `0.10%`, `0.15%`, `0.20%`, `0.30%`, `0.50%`
- Coverage decreases as threshold increases

### C. Volatility-Adjusted Direction
- `label = 1` if `future_return > k * rolling_vol` else `0`
- `label = NaN` if `|future_return| <= k * rolling_vol`
- Rolling volatility computed using past data only (no lookahead)
- k candidates: `0.10`, `0.20`, `0.30`, `0.50`
- Adaptive to market regime

### D. Top/Bottom Quantile Event Target
- Classify only stronger directional events
- Quantile thresholds computed on train/validation returns
- Candidate quantiles: top/bottom `40%`, `35%`, `30%`, `25%`
- Middle zone abstained (NaN)
- Quantile thresholds selected using train/validation only

### E. Three-Class Target
- `label = 1` (up) if `future_return > upper_threshold`
- `label = 0` (down) if `future_return < lower_threshold`
- `label = NaN` (flat) otherwise
- Final directional accuracy measured only on up/down predicted events
- Report separate classification metrics
- Do not compare unfairly to binary full-coverage accuracy

### F. Meta-Label Target
- Base model predicts direction
- Meta-label predicts whether base forecast is likely correct
- Meta-label trained only on pre-2025 data
- Final evaluation used only once for scoring

## Target Selection Rule

1. Generate all target definitions using train/validation data only
2. Train/tune models on train subset (2015-2023)
3. Select target definition, model, horizon, and threshold on 2024 validation
4. Freeze selected policy
5. Score on final evaluation (2025-2026) once

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

## Leakage Prevention

- No future data in feature computation
- No final-eval labels in threshold selection
- Rolling volatility uses past data only
- Quantile thresholds computed on train/validation only
- All target definitions generated programmatically from price data

## Boundary

- No trading-readiness, profitability, or live deployment claim
- No paper/DOCX generated
- No prediction labels edited manually
