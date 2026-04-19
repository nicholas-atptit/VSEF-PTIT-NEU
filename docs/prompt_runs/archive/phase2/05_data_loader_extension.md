# Prompt Run: data-loader-extension

## Status
- Completed

## Traceability
- Source of evidence used:
  - `src/ml/data_loader.py`
  - `docs/data_loader_vn100.md`
  - `tests/test_data_loader_vn100.py`

## Objective
- Extend `src/ml/data_loader.py` to support robust, production-grade daily dataset ingestion for the VN100 universe.
- Implement file-backed (CSV) and database-backed (TimescaleDB) loading with automatic fallback.

## Files Created
- `docs/data_loader_vn100.md`
- `tests/test_data_loader_vn100.py`

## Files Modified
- `src/ml/data_loader.py` (Added `VN100DataLoader` and helper functions)

## Key Changes
- Implemented `VN100DataLoader` class for batch loading across multiple tickers.
- Added `load_ohlcv_from_csv` for per-ticker CSV ingestion.
- Added `load_market_proxy` for joining VNINDEX returns.
- Added `load_vn100_daily_dataset` convenience wrapper.
- Implemented configurable data source priority (CSV vs DB).

## Implementation Details
- **`VN100DataLoader`**:
  - `build_dataset`: Orchestrates loading, index joining, and fundamental merging.
  - `build_inference_dataset`: Specifically for the latest data point prediction.
- **Fallback Chain**: `CSV` -> `TimescaleDB` -> `vnstock API`.

## Algorithms / Methods / Rules Applied
- Left-join logic for market benchmarks and fundamental data.
- Graceful error handling for missing ticker files (returns empty DataFrame).

## Data Flow Impact
- Input: `data/daily_market_split_data/`, `data/market_proxy.csv`, `data/fundamentals_latest.csv`.
- Output: Standardized Pandas DataFrame with `date` and `ticker` indices.

## Backward Compatibility
- ✅ Original `load_ohlcv_from_db` and `load_ohlcv_from_vnstock` APIs preserved.

## Risks / Limitations
- Reliance on `scripts/extract_daily_csv.py` for the CSV format.
- Performance: Large batch loads (100+ tickers) may be slow on single-threaded CSV reads.

## Verification
- `python -m pytest tests/test_data_loader_vn100.py -v`
- Reviewing `docs/data_loader_vn100.md`.

## Open TODOs
- Concurrent CSV loading for large universes.
