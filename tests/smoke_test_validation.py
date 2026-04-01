"""Smoke test for Experiment Tracking and Data Quality Validation.
"""

import os
import pandas as pd
import json
from pathlib import Path
from src.ml.experiment_tracker import ExperimentTracker
from src.validators.data_quality import DataQualityValidator

def test_experiment_tracking():
    print("Testing Experiment Tracking...")
    ticker = "SMOKE_TEST"
    tracker = ExperimentTracker(storage_path="reports/smoke_experiments.jsonl")
    
    run_id = tracker.log_experiment(
        ticker=ticker,
        label_type="binary",
        model_type="SmokeModel",
        feature_count=10,
        metrics={"acc": 0.5, "mae": 1.2},
        train_start="2024-04-01T00:00:00",
        train_end="2024-04-01T00:01:00",
        model_path="models/SMOKE_TEST"
    )
    
    assert run_id is not None
    assert Path("reports/smoke_experiments.jsonl").exists()
    
    with open("reports/smoke_experiments.jsonl", "r") as f:
        last_run = json.loads(f.readlines()[-1])
        assert last_run["ticker"] == ticker
        assert last_run["run_id"] == run_id

    # Clean up
    os.remove("reports/smoke_experiments.jsonl")
    print("✅ Experiment Tracking OK.")

def test_data_quality_validation():
    print("Testing Data Quality Validation...")
    df = pd.DataFrame({
        "date": ["2024-01-01"],
        "open": [100], "high": [105], "low": [95], "close": [102],
        "volume": [1000]
    })
    
    validator = DataQualityValidator(ticker="SMOKE")
    success, errors = validator.validate_ohlcv(df, raise_on_error=False)
    assert success is True
    print("✅ Data Quality Validation OK.")

if __name__ == "__main__":
    try:
        test_experiment_tracking()
        test_data_quality_validation()
        print("\n🚀 ALL SMOKE TESTS PASSED!")
    except Exception as e:
        print(f"\n❌ SMOKE TEST FAILED: {str(e)}")
        exit(1)
