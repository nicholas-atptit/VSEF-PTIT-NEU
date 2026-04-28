# Prompt Run: data-quality

## Status
- Completed

## Traceability
- Source of evidence used:
  - `src/validators/data_quality.py`
  - `docs/governance/data_quality.md`
  - `scripts/sync_all_data.py`
  - `tests/test_data_quality.py`

## Objective
- Develop a robust data quality validation layer for the VN100 stock prediction system.
- Enforce data integrity during ingestion, feature engineering, and model training.

## Files Created
- `src/validators/data_quality.py`
- `docs/governance/data_quality.md`
- `tests/test_data_quality.py`

## Files Modified
- `scripts/sync_all_data.py` (Integrated `DataQualityValidator`)
- `scripts/train_ml_tickers.py` (Integrated `DataQualityValidator`)

## Key Changes
- **Validation Rules**:
  - `ohlcv`: Checks for missing columns, negative values, and High < Low errors.
  - `feature_matrix`: Detects excessive nulls (20%) and infinite values.
- **Fail-Fast Policy**: Raises errors for critical price discrepancies (Negative prices, price inversions).
- **Graceful Warnings**: Warns for zero-volume days or sparse feature coverage.

## Implementation Details
- **`DataQualityValidator`**:
  - `validate_ohlcv(df, raise_on_error=True)`: Primary OHLCV integrity check.
  - `validate_features(df)`: Post-transformation feature matrix check.
- **Integration**:
  - `BackdateIngestor`: Validates live API data before DB commit.
  - `ML Data Loading`: Validates historical CSVs before training.

## Algorithms / Methods / Rules Applied
- OHLC price inversion detection.
- Statistical NaN-thresholding (20% limit).
- Duplicate session detection.

## Data Flow Impact
- Input: Raw `vnstock` results or historical CSVs.
- Protection: Blocks corrupt data from entering the database or training pipeline.

## Backward Compatibility
- ✅ Optional validation; can be disabled with `raise_on_error=False`.

## Risks / Limitations
- 20% NaN threshold might be too loose for certain technical indicators.
- No cross-ticker consistency validation (Index vs. Constituents syncing) yet.

## Verification
- `pytest tests/test_data_quality.py -v`
- Reviewing `docs/governance/data_quality.md`.

## Open TODOs
- Dynamic thresholding per indicator type.
- Drift detection for feature distributions.
