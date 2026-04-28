# Prompt Run: feature-engineering-extension

## Status
- Completed

## Traceability
- Source of evidence used:
  - `src/ml/feature_engineering.py`
  - `docs/usage/feature_engineering_vn100.md`
  - `tests/test_feature_engineering_vn100.py`

## Objective
- Enhance `FeatureEngineer` class with 19 standardised VN100-oriented daily prediction features.
- Ensure time-safety and compatibility with existing ML pipelines.

## Files Created
- `docs/usage/feature_engineering_vn100.md`
- `tests/test_feature_engineering_vn100.py`

## Files Modified
- `src/ml/feature_engineering.py` (Added `VN100_DAILY_FEATURES` and `_add_vn100_daily_features`)

## Key Changes
- **New Price Features**: `overnight_return_1d`, `open_to_close_return_1d`, `high_low_range_pct`.
- **New Rolling Returns**: `return_3d`, `return_10d`.
- **New Volatility/Volume**: `rolling_volatility_5`, `rolling_volatility_20`, `value_ratio_5`, `value_ratio_20`.
- **Logic**: Implemented `_add_vn100_daily_features()` to group and compute these.
- **Standards**: Used `VN100_DAILY_FEATURES` constant for canonical referencing.

## Implementation Details
- **`FeatureEngineer._add_vn100_daily_features()`**:
  - Computed per-ticker using vectorized Pandas operations.
  - Used `.pct_change()` and `.rolling()` for time-safety.
  - Turnover Proxy: `close * volume`.

## Algorithms / Methods / Rules Applied
- **Overnight Return**: `(open - prev_close) / prev_close`.
- **Intraday Return**: `(close - open) / open`.
- **Volatility**: Moving standard deviation of `pct_return`.
- **Value Ratios**: Turnover relative to its 5-day/20-day trailing mean.

## Data Flow Impact
- Input: Standardized OHLCV DataFrame from `DataLoader`.
- Output: Enhanced DataFrame with 19 additional columns.

## Backward Compatibility
- ✅ Guarded with `if "feature_name" not in df.columns` to avoid duplication.
- ✅ Uses aliases for legacy features (e.g., `return_5d` -> `return_roll_5`).

## Risks / Limitations
- `value_ratio` is a proxy (not true VWAP turnover).
- Warmup period required (minimum 20 sessions for 20-day features).

## Verification
- `python -m pytest tests/test_feature_engineering_vn100.py -v`
- Reviewing `docs/usage/feature_engineering_vn100.md`.

## Open TODOs
- Add support for sector-relative features.
