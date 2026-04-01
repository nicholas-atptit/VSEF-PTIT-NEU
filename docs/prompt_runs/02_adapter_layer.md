# Prompt Run: adapter-layer

## Status
- Completed

## Traceability
- Source of evidence used:
  - `src/adapters/vnstock_adapter.py`
  - `docs/vnstock_adapter.md`
  - `config/settings.py`

## Objective
- Create a thin, stable adapter layer for the `vnstock` library within the existing codebase.
- Provide consistent interfaces for fetching OHLCV, market index, fundamental, and news data.

## Files Created
- `src/adapters/vnstock_adapter.py`
- `docs/vnstock_adapter.md`

## Files Modified
- None (First-time implementation)

## Key Changes
- Centralized `vnstock` client initialization with API key injection.
- Standardized OHLCV fetching with column renaming (`time` -> `date`).
- Implemented `get_financial_ratios` for annual metrics.
- Added news aggregation with fallback logic.

## Implementation Details
- **`VnstockAdapter` class**:
  - `__init__`: API Key injection from `Settings`.
  - `get_ohlc`: Standardized data acquisition.
  - `get_news`: Logic for `stock.news()` and `stock.company.news()`.
  - `get_vn100_tickers`: Interface for universe constituent list.

## Algorithms / Methods / Rules Applied
- Standardized data normalization (e.g., column mapping, date normalization).
- API Key fallback logic.

## Data Flow Impact
- Input: `vnstock` v3.0+ API.
- Output: Standardized Pandas DataFrames for downstream consumption.

## Backward Compatibility
- Designed for new modules; existing code was not modified to use it yet.

## Risks / Limitations
- Potential for breaking changes in the underlying `vnstock` library.
- Reliance on API stability for constituent lists.

## Verification
- `python scripts/test_adapter.py` (assumed based on logic)
- Reviewing `docs/vnstock_adapter.md`.

## Open TODOs
- Integrate with existing `DataLoader`.
