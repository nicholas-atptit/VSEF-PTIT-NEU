# Prompt Run: training-pipeline-integration

## Status
- Completed

## Traceability
- Source of evidence used:
  - `scripts/train_ml_tickers.py`
  - `src/ml/labels/training_adapter.py`
  - `docs/train_pipeline_labels.md`

## Objective
- Integrate the `src/ml/labels` package into the existing `scripts/train_ml_tickers.py` training flow.
- Support multiple label types (binary, ternary, regression, volatility) via CLI arguments.

## Files Created
- `src/ml/labels/training_adapter.py`
- `docs/train_pipeline_labels.md`

## Files Modified
- `scripts/train_ml_tickers.py` (Extended CLI logic)

## Key Changes
- **`LabelTrainingAdapter`**: Created to bridge the CLI modes to the label registry.
- **`mode` Suffixing**: Implemented dynamic logic to save models with mode-specific suffixes (e.g., `_binary_1d`).
- **`XGBoost` / `LightGBM` Handling**: Ensured regressor vs. classifier selection based on target type.

## Implementation Details
- **CLI Arg**: `--label-mode` (default: `None`).
- **`_train_custom_label` function**:
  - Handles classification/regression split.
  - Automatically filters NaN-rows from target and features.
  - Log: Detailed performance metrics (Accuracy, F1, RMSE, MAE).

## Algorithms / Methods / Rules Applied
- Automatic selection of objective: `binary:logistic`, `multi:softmax`, or `reg:squarederror`.
- Dynamic sample weight generation based on session recency.

## Data Flow Impact
- Input: Ticker OHLCV and selected label mode.
- Storage: `models/<TICKER>/trend_classifier_<mode>.joblib`.
- Storage: `models/<TICKER>/feature_cols_<mode>.joblib`.

## Backward Compatibility
- ✅ Legacy code continues to use triple-horizon (5d, 20d, 120d) if `--label-mode` is omitted.

## Risks / Limitations
- High-horizon labels (e.g. 120d) will return many NaNs and reduce the training dataset size significantly.

## Verification
- `python scripts/train_ml_tickers.py --tickers HPG --label-mode ternary_5d`
- Reviewing `docs/train_pipeline_labels.md`.

## Open TODOs
- Support ensemble of multiple label modes during training.
