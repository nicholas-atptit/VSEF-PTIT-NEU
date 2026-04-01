# Prompt Run: ranked-predictions

## Status
- Completed

## Traceability
- Source of evidence used:
  - `src/reporting/ranked_predictions.py`
  - `docs/ranked_predictions.md`
  - `src/api/schemas_v2.py` (v5.1 update)
  - `src/ml/signal_generator.py`

## Objective
- Create a robust, automated reporting module that transforms batch inference outputs into actionable, ranked prediction tables for the VN100 universe.
- Prioritize assets by conviction, expected return, and risk.

## Files Created
- `src/reporting/ranked_predictions.py`
- `docs/ranked_predictions.md`

## Files Modified
- `src/api/schemas_v2.py` (Added `current_price` and `volatility_score` to payloads)
- `src/ml/signal_generator.py` (Signal embedding logic)
- `scripts/per_session_predict.py` (Extraction of volatility predictions)

## Key Changes
- **Payload Upgrade (v5.1)**: Embedded `current_price` and `volatility_score` into JSON outputs.
- **JSON Flattening**: Implemented high-performance ingestion of nested inference results into a `pandas.DataFrame`.
- **Ranking Modules**:
  - `top_long`: Sorted by directional `up` probability.
  - `expected_return`: Sorted by $(P_{median} / P_{current} - 1)$.
  - `high_risk`: Sorted by regression-based volatility scores.
- **Reporting**: Automated generation of CSV and Markdown reports in `reports/`.

## Implementation Details
- **`src/reporting/ranked_predictions.py`**:
  - `load_data()`: Reads from `data/latest_predictions.json` or batch archives.
  - `rank_tickers()`: Multi-criteria sorting logic.
  - `export_markdown()`: Generates professional-grade summary tables.

## Algorithms / Methods / Rules Applied
- Probability-based ranking for classification models.
- Median-return estimation from regression residuals.
- Volatility-threshold filtering for risk management.

## Data Flow Impact
- Input: `data/latest_predictions.json` (v5.1 schema).
- Output: `reports/ranked_predictions_<date>.md` and `.csv`.

## Backward Compatibility
- ✅ Logic detects schema version and falls back gracefully (though ranking may be limited without price/vol).

## Risks / Limitations
- Ranking is only as good as the underlying model's calibration (probability `up` vs. actual hit rate).
- Missing price data in old payloads will break expected return calculations.

## Verification
- `python -m src.reporting.ranked_predictions --latest`
- Reviewing `docs/ranked_predictions.md`.

## Open TODOs
- Sector-wise ranking (Top 3 by Industry).
- Integration with historical backtest hit rates (Dynamic confidence adjustment).
