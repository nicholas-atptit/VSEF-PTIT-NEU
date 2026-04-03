# 17. Batch 2 Patch Cleanup - Alignment and Metrics

## Prompt Intent
Fix signal/return alignment issues (purge_end, current_split undefined), demote accuracy to diagnostic-only, and add a deterministic alignment test.

## Actual Outcome
Successfully fixed the logic errors in `scripts/train_ml_tickers.py` and added a verification test suite.

## Files Created
- `tests/ml/test_train_ml_alignment.py`

## Files Modified
- `scripts/train_ml_tickers.py`

## Key Code Changes
### scripts/train_ml_tickers.py
- Fixed `current_split` -> `train_end` on line 145.
- Fixed `purge_end` -> `split_idx` on line 167 to align `test_returns` with `X_test`.
- Updated `_train_custom_label` to calculate and return `sharpe` and `cagr` as primary metrics.
- Updated `train_ticker` to log and include horizon-specific `sharpe` and `cagr` metrics.
- Demoted accuracy to a diagnostic field in the return dictionary.

## Functions / Classes / Scripts Added or Updated
- `_train_custom_label`: Updated return structure and fixed variables.
- `train_ticker`: Updated metric collection.

## Algorithms / Methods / Logic Introduced
- Implemented: Sharpe Ratio and CAGR prioritized over accuracy in training metrics.

## Config / CLI / Environment Changes
- No new predictive algorithm introduced in this task.

## Storage / Persistence Impact
- `models/evaluation_report.csv` (if generated) and `ExperimentTracker` logs now contain `sharpe` and `cagr`.

## Compatibility Notes
- Backward compatible with existing data and model formats.

## Tests / Validation
- Created `tests/ml/test_train_ml_alignment.py`.
- Verified alignment of `test_preds` and `test_returns` for the `MetricsEvaluator`.
- Proved that transaction-cost-adjusted metrics compute successfully.

## Remaining Gaps
- None identified for this specific patch.
