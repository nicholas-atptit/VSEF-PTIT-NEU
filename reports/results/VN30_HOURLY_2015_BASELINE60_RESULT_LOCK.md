# VN30 Hourly 2015 - Baseline 60 Result Lock

## Locked Result

- **Active universe**: VN30 January 2025 review
- **Model**: Random Forest
- **Horizon**: h=60 (15 trading days)
- **Target**: Absolute direction (label = 1 if future_close > current_close else 0)
- **Feature set**: Stock lagged + market-index context (99 features)
- **Final accuracy**: 60.22%
- **Coverage**: 100.0%
- **Rows**: 3,474
- **Claim level**: global_full_universe
- **Commit**: 60e65442
- **Tag**: nckh/vn30-hourly-2015-horizon-relative-target-v1

## Allowed Wording

- "The model achieves 60.22% directional accuracy on the full active universe."
- "Coverage is 100% across all 30 tickers in the VN30 January 2025 universe."
- "The result is validated on hourly data from 2025-01-01 to 2026-05-14."
- "All model selection and threshold tuning used 2024 validation data only."
- "No future data was leaked during training or selection."

## Forbidden Wording

- "The model is profitable."
- "The model is ready for live trading."
- "The model outperforms the market."
- "The model achieves 60.22% accuracy with confidence."
- Any claim of trading-readiness, profitability, or live deployment.
- Any comparison to other results without stating the exact target definition.

## Boundary

- No trading-readiness claim.
- No profitability claim.
- No live deployment claim.
- This result is a research benchmark only.
- The 60.22% accuracy is directional, not financial performance.

## Selection Protocol

- Model (Random Forest) and horizon (h=60) selected using 2024 validation only.
- Final evaluation (2025-2026) used only once for scoring.
- No threshold tuning on final evaluation labels.
- No leakage indicators detected.
