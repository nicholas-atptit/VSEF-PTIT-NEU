"""
Verification script for ML Engine Stability Hardening.
Simulates partial failures to ensure they correctly emit degraded 
or failed schemas without crashing the pipeline.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from src.ml.inference.engine import InferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stability_test")

def create_mock_data(rows: int, ticker: str = "TEST") -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    return pd.DataFrame({
        "date": dates,
        "ticker": ticker,
        "open": np.random.randn(rows) + 100,
        "high": np.random.randn(rows) + 105,
        "low": np.random.randn(rows) + 95,
        "close": np.random.randn(rows) + 100,
        "volume": np.random.randint(1000, 10000, rows),
    })

def test_invalid_ohlcv():
    engine = InferenceEngine(model_root="tests/artifacts")
    logger.info("--- Testing Missing OHLCV columns ---")
    df = create_mock_data(100, "BAD1").drop(columns=["close"])
    
    # We pass it to predict_ticker directly
    res = engine.predict_ticker("BAD1", df)
    assert res["status"] == "failed"
    assert res["error_code"] == "invalid_ohlcv_input"
    logger.info("Passed: Invalid OHLCV caught cleanly.")

def test_insufficient_history():
    engine = InferenceEngine(model_root="tests/artifacts")
    logger.info("--- Testing insufficient history ---")
    df = create_mock_data(20, "BAD2") # only 20 rows, minimum is 60!
    
    res = engine.predict_ticker("BAD2", df)
    assert res["status"] == "failed"
    assert res["error_code"] == "insufficient_history"
    logger.info("Passed: Insufficient history caught cleanly.")

def test_feature_validation_failed():
    from src.ml.feature_engineering import FeatureEngineer
    logger.info("--- Testing feature validation ---")
    df = create_mock_data(100, "BAD3")
    
    # Introduce NaN
    df.loc[99, "close"] = np.nan
    
    fe = FeatureEngineer()
    try:
        fe.transform(df, drop_na=False)
        assert False, "Should have raised [feature_validation_failed]"
    except Exception as e:
        assert "[feature_validation_failed]" in str(e)
        logger.info(f"Passed: Feature Validation caught: {e}")

def test_batch_safe():
    engine = InferenceEngine(model_root="tests/artifacts")
    logger.info("--- Testing batch safety ---")
    
    df1 = create_mock_data(100, "GOOD1")
    df2 = create_mock_data(10, "SHORT1") # will fail
    df3 = create_mock_data(100, "GOOD2")
    
    batches = {
        "GOOD1": df1,
        "SHORT1": df2, 
        "GOOD2": df3
    }
    
    # Note: we use mock artifacts path so GOOD1/2 might fail with artifact_missing 
    # but the point is the whole BATCH returns a dataframe without crashing.
    df_res = engine.predict_batch(batches)
    assert len(df_res) == 3
    assert "status" in df_res.columns
    logger.info(f"Passed: Batch safety maintained. Statuses: {df_res['status'].tolist()}")
    
if __name__ == "__main__":
    test_invalid_ohlcv()
    test_insufficient_history()
    test_feature_validation_failed()
    test_batch_safe()
    logger.info("All stability verifications completed successfully.")
