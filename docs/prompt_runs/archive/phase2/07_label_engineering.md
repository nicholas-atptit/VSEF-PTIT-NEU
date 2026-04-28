# Prompt Run: label-engineering

## Status
- Completed

## Traceability
- Source of evidence used:
  - `src/ml/labels/` package
  - `docs/usage/label_engineering.md`
  - `tests/test_label_engineering.py`

## Objective
- Implement a modular, production-grade label engineering layer for the VN100 stock prediction system.
- Support binary classification, ternary classification, regression, and volatility targets with strict time-safety.

## Files Created
- `src/ml/labels/__init__.py` (Registry + API)
- `src/ml/labels/base.py` (Abstract Base Class)
- `src/ml/labels/classification.py` (Binary/Ternary)
- `src/ml/labels/regression.py` (Returns)
- `src/ml/labels/volatility.py` (Realized Vol)
- `docs/usage/label_engineering.md`
- `tests/test_label_engineering.py`

## Files Modified
- `config/settings.py` (Added label thresholds)

## Key Changes
- **Registry Pattern**: Implemented `LABEL_REGISTRY` to map keys to generator classes.
- **Time Safety**: Forced use of `shift(-horizon)` in all generators.
- **Configurable Thresholds**: Integrated `LABEL_CLS_1D_THRESHOLD` and `LABEL_CLS_5D_THRESHOLD` settings.
- **Factory Function**: Added `get_generator(name)` for easy instantiation.

## Implementation Details
- **`BaseLabelGenerator`**: Defines the interface (`name`, `label_columns`, `generate`).
- **Binary Targets**: `cls_1d_updown`, `cls_5d_updown`, `cls_20d_updown`.
- **Ternary Targets**: `cls_1d_3class`, `cls_5d_3class` (Up, Sideways, Down).
- **Regression Targets**: `reg_next_close_return`, `reg_5d_return`.
- **Volatility Targets**: `future_realized_vol_5d`.

## Algorithms / Methods / Rules Applied
- **Shift-based labeling**: Labels computed from future sessions.
- **Ternary thresholding**: Sideways class defined by `abs(return) < threshold`.
- **Realized Volatility**: Annualized standard deviation of forward returns.

## Data Flow Impact
- Input: Standardized OHLCV DataFrame.
- Output: DataFrame with additional target columns (e.g., `label_cls_1d_updown`).

## Backward Compatibility
- ✅ Self-contained package; does not interfere with old `trend_threshold_pct` logic unless requested.

## Risks / Limitations
- Last `horizon` rows correctly contain NaNs (must be dropped).
- Thresholds are global but can be overridden per ticker (future).

## Verification
- `python -m pytest tests/test_label_engineering.py -v`
- Reviewing `docs/usage/label_engineering.md`.

## Open TODOs
- Triple-barrier labeling implementation.
- Metalabeling support.
