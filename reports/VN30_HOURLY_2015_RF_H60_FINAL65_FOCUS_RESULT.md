# VN30 Hourly 2015 - RF h=60 Final65 Focus Result

## Baseline60 Lock

- **Locked**: yes
- **Model**: Random Forest
- **Horizon**: h=60
- **Target**: Absolute direction
- **Final accuracy**: 60.22%
- **Coverage**: 100%
- **Rows**: 3,474
- **Claim level**: global_full_universe

## Final65 Focus Results

### Experiments Run

1. **Confidence threshold selection** (validation only): No valid result
2. **Probability calibration** (Platt scaling): 59.70%, 100% coverage, 3,474 rows
3. **Meta-label abstention** (pre-2025 data): No valid result
4. **Per-ticker confidence thresholds** (validation only): No valid result
5. **Regime-aware abstention** (lagged features): No valid result

### Best Final65 Candidate

- **Experiment**: Platt calibration
- **Accuracy**: 59.70%
- **Coverage**: 100.0%
- **Rows**: 3,474

### Final65 Status

- **Passed**: no
- **Gap to 65**: 5.30 percentage points

### Observations

- Base model validation accuracy: ~49% (near random)
- Base model eval accuracy: ~59%
- Confidence filtering did not improve accuracy on validation set
- Meta-label abstention did not find valid thresholds
- Per-ticker thresholds did not produce qualifying results
- Regime-aware abstention did not produce qualifying results
- Platt calibration slightly improved eval accuracy (59.70% vs 59.38% base)
- The model's validation performance (~49%) suggests poor learnability on 2024 data
- The gap between validation (~49%) and eval (~59%) suggests regime shift

## Boundary

- No trading-readiness, profitability, or live deployment claim.
- No paper/DOCX generated.
- All selection on 2024 validation only, leakage-safe.
