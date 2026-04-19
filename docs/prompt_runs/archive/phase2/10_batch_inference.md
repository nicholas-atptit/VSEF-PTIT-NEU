# Prompt Run: batch-inference

## Status
- Completed

## Traceability
- Source of evidence used:
  - `scripts/per_session_predict.py`
  - `docs/batch_vn100_inference.md`
  - `data/latest_predictions.json`

## Objective
- Extend `scripts/per_session_predict.py` to support robust, production-grade batch inference for the VN100 universe.
- Generate structured JSON reports with multi-model prediction output.

## Files Created
- `docs/batch_vn100_inference.md`

## Files Modified
- `scripts/per_session_predict.py` (Extended CLI logic and batch loop)

## Key Changes
- **`--batch` flag**: Enables high-throughput processing across multiple tickers.
- **Dynamic Universe**: Uses `get_vn100_universe()` to target the VN100 cohort.
- **Parallel Tasks**: Implemented asynchronous prediction tasks with a concurrency semaphore.
- **Unified JSON Storage**: Saves the latest run to `data/latest_predictions.json` and archival reports to `data/processed/`.

## Implementation Details
- **Trend Classification**: Directional probabilities (Up, Sideways, Down).
- **Price Regression**: Quantile-based ranges (Bottom 10th, Median, Ceiling 90th).
- **Volatility Output**: Annualized realized volatility prediction (if trained).
- **Graceful Failures**: Skips tickers with `model_not_found` or `insufficient_data` errors.

## Algorithms / Methods / Rules Applied
- Asynchronous orchestration with `asyncio`.
- Quantile-based prediction ranges ($Q_{0.1}, Q_{0.5}, Q_{0.9}$).

## Data Flow Impact
- Input: `models/`, `data/daily_market_split_data/`.
- Output: `data/latest_predictions.json` (Overwrite).
- Output: `data/processed/batch_inference_<timestamp>.json` (Archive).

## Backward Compatibility
- ✅ Single-ticker prediction (`--ticker SSI`) remains fully functional.

## Risks / Limitations
- JSON reports can become large (several MBs) with 100+ detailed ticker predictions.
- Memory: Large batch processing requires monitoring if parallelized aggressively.

## Verification
- `python scripts/per_session_predict.py --batch --vn100 --limit 5`
- Reviewing `docs/batch_vn100_inference.md`.

## Open TODOs
- Integration with LLM and news sentiment at the batch level.
