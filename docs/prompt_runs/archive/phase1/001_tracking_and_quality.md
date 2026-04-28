# Implementation Document: Experiment Tracking & Data Quality Validation
**Date**: 2026-04-02
**Time**: 03:22 AM (System Time)
**Status**: Completed

## 1. Objective
Extend the VN100-centric prediction system with:
1.  **Lightweight Experiment Tracking**: Capture and version model metrics, feature counts, and label configurations.
2.  **Data Quality Validation Layer**: Implement automated checks for data ingestion and training datasets.

## 2. Tasks Carried Out

### Task 1: Module Creation (Experiment Tracking)
- **Module**: `src/ml/experiment_tracker.py`
- **Action**: Developed a lightweight tracker class that logs to `reports/experiments.jsonl`.
- **Reasoning**: JSONL (JSON Lines) was chosen for minimal overhead and easy append-only logging without needing structured database migrations.
- **Commit Details**: Added `log_experiment` method that captures `run_id`, `ticker`, `metrics`, `model_path`, etc.

### Task 2: Module Creation (Data Quality)
- **Module**: `src/validators/data_quality.py`
- **Action**: Implemented `DataQualityValidator` class with `validate_ohlcv` and `validate_features` methods.
- **Reasoning**: Centralizing validation ensures consistency between data ingestion (sync) and model training.
- **Commit Details**: Added checks for negative prices, High < Low violations, duplicate dates, and excessive null rates.

### Task 3: Ingestion Integration
- **File**: `src/historical/backdate.py`
- **Action**: Integrated `DataQualityValidator` into `BackdateIngestor._sync_one`.
- **Reasoning**: Catching data issues immediately after fetching from `vnstock` prevents corrupted data from entering the database.
- **Commit Details**: Added validation call before database batch insertion.

### Task 4: Training Integration
- **File**: `scripts/train_ml_tickers.py`
- **Action**: Integrated both `ExperimentTracker` and `DataQualityValidator`.
- **Reasoning**: Ensures that models are only trained on valid data and that every training attempt is audit-logged.
- **Commit Details**:
    - Initialized `ExperimentTracker` in `main`.
    - Added data validation at the start of `train_ticker`.
    - Added experiment logging at the end of `train_ticker` (and in custom label paths).
    - Removed a duplicate return statement found in the original script.

### Task 5: Verification & Tests
- **Files**: `tests/test_data_quality.py`, `tests/smoke_test_validation.py`
- **Action**: Created unit tests for the validator and a full-pipeline smoke test.
- **Verification**: Ran `python tests/smoke_test_validation.py` and confirmed all checks passed.

## 3. Implementation Summary

| Component | File Path | Integrated Into |
| :--- | :--- | :--- |
| **Tracker** | `src/ml/experiment_tracker.py` | `scripts/train_ml_tickers.py` |
| **Validator** | `src/validators/data_quality.py` | `src/historical/backdate.py`, `scripts/train_ml_tickers.py` |
| **Tests** | `tests/test_data_quality.py` | N/A |
| **Smoke Test** | `tests/smoke_test_validation.py` | N/A |
| **Docs** | `docs/governance/data_quality.md`, `docs/governance/experiment_tracking.md` | N/A |

## 4. Commands to Run/Test

**Run Smoke Tests:**
```bash
python tests/smoke_test_validation.py
```

**Run Unit Tests:**
```bash
pytest tests/test_data_quality.py
```

**View Experiment Logs:**
```bash
cat reports/experiments.jsonl
```
