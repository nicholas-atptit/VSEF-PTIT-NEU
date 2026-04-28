# Prompt Run: daily-sync

## Status
- Completed

## Traceability
- Source of evidence used:
  - `scripts/sync_all_data.py`
  - `docs/usage/daily_sync_vn100.md`
  - `src/historical/backdate.py`

## Objective
- Extend `scripts/sync_all_data.py` script to support a robust, daily ingestion flow specifically for the VN100 universe.
- Implement configurable parameters: `universe_mode`, `start_date`, `end_date`, and `force_refresh`.

## Files Created
- `docs/usage/daily_sync_vn100.md`

## Files Modified
- `scripts/sync_all_data.py` (Extended CLI logic)

## Key Changes
- Integrated `get_vn100_universe()` with the `sync_all_data.py` entry point.
- Support for `all` (legacy VIP) or `current_vn100` (dynamic) universes.
- Added `--force_refresh` and `--save_raw_copy` features.
- Ingests `VNINDEX` as a benchmark when requested.

## Implementation Details
- **CLI Arguments**: Added `--universe_mode`, `--start_date`, `--end_date`, `--force`, etc.
- **`sync_all_data.py`**:
  - `batch_vn100`: Logic to loop through constituents.
  - Logging: Structured summary at the end of the execution.

## Algorithms / Methods / Rules Applied
- Sequential ticker synchronization with 1s cooldown.
- Timezone handling: VN Time (UTC+7).

## Data Flow Impact
- Input: `vnstock` v3.0+ API.
- Storage: TimescaleDB (as primary).
- Storage: `data/raw/` (as secondary for backups).

## Backward Compatibility
- Preserved support for the old list of tickers (`--vn100` legacy flag).

## Risks / Limitations
- Rate limits on the `vnstock` source.
- DB connection reliability.

## Verification
- `python scripts/sync_all_data.py --universe_mode current_vn100 --start_date 2024-03-01`
- Reviewing `docs/usage/daily_sync_vn100.md`.

## Open TODOs
- Concurrent sync (currently sequential for safety).
