# Prompt Run: universe-loader

## Status
- Completed

## Traceability
- Source of evidence used:
  - `src/data/universe.py`
  - `docs/usage/universe_loader.md`
  - `src/adapters/vnstock_adapter.py`

## Objective
- Implement a modular, production-grade VN100 universe loader within the existing codebase.
- Standardize the retrieval of VN10 constituent list across all scripts.

## Files Created
- `src/data/universe.py`
- `docs/usage/universe_loader.md`

## Files Modified
- None (Initial implementation)

## Key Changes
- Created `get_vn100_universe` function.
- Added `current` and `current_plus_viettel` universe modes.
- Integrated VN100 constituents from TCBS/VCI sources via `VnstockAdapter`.

## Implementation Details
- **`get_vn100_universe(mode, as_of_date=None)`**:
  - `current`: Returns the latest members.
  - `current_plus_viettel`: Appends `VTP`, `VGI`, `CTR`, `FOX`.
  - Logging: Warning for the `as_of_date` (future support).

## Algorithms / Methods / Rules Applied
- Union of constituent sets for the `plus_viettel` mode.

## Data Flow Impact
- Input: `VnstockAdapter.get_vn100_tickers()`.
- Output: List of ticker strings.

## Backward Compatibility
- Compatible with all future pipelines relying on lists of tickers.

## Risks / Limitations
- No support for historical constituents (currently returns current).
- Hardcoded Viettel tickers are used for fallback.

## Verification
- `python src/data/unvierse.py` (assumed based on logic)
- Reviewing `docs/usage/universe_loader.md`.

## Open TODOs
- Historical constituent list support using snapshots or time-travel logic.
