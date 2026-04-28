# Prompt Run: experiment-tracking

## Status
- Completed

## Traceability
- Source of evidence used:
  - `src/ml/experiment_tracker.py`
  - `docs/governance/experiment_tracking.md`
  - `reports/experiments.jsonl`
  - `scripts/train_ml_tickers.py`

## Objective
- Implement a lightweight experiment tracking module to log training metadata, hyperparameters, and performance metrics.
- Support cross-model comparison for different feature/label versions.

## Files Created
- `src/ml/experiment_tracker.py`
- `docs/governance/experiment_tracking.md`

## Files Modified
- `scripts/train_ml_tickers.py` (Integrated `ExperimentTracker`)

## Key Changes
- **Metadata Logging**: Implemented `run_id`, `ticker`, `label_type`, and `feature_count` tracking.
- **JSONL Storage**: Standardized storage in `reports/experiments.jsonl` for easy parsing.
- **Performance Snapshots**: Automatically captures `accuracy`, `MAE`, and `elite_signal` metrics.

## Implementation Details
- **`ExperimentTracker`**:
  - `start_run()`: Generates unique session IDs.
  - `log_metrics(metrics)`: Appends to the JSONL log.
  - `end_run()`: Finalizes the session with duration.

## Algorithms / Methods / Rules Applied
- Unique ID generation: 8-character hex IDs.
- Time-series experiment tracking.

## Data Flow Impact
- Input: Hyperparameters and evaluation scores from the Trainer.
- Storage: `reports/experiments.jsonl`.

## Backward Compatibility
- ✅ Optional logging; does not break training if disabled.

## Risks / Limitations
- JSONL file can grow indefinitely over time.
- No UI for comparing experiments (CLI `tail` and `jq` only).

## Verification
- `tail -n 10 reports/experiments.jsonl | jq`
- Reviewing `docs/governance/experiment_tracking.md`.

## Open TODOs
- Integration with Weights & Biases or MLflow.
- Automatic feature-importance logging.
