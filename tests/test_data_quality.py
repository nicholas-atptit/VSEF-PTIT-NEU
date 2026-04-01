import pytest
import pandas as pd
import numpy as np
from src.validators.data_quality import DataQualityValidator

@pytest.fixture
def sample_ohlcv():
    return pd.DataFrame({
        "date": pd.date_range(start="2024-01-01", periods=5),
        "open": [100, 101, 102, 103, 104],
        "high": [105, 106, 107, 108, 109],
        "low": [95, 96, 97, 98, 99],
        "close": [102, 103, 104, 105, 106],
        "volume": [1000, 1100, 1200, 1300, 1400]
    })

def test_valid_ohlcv(sample_ohlcv):
    validator = DataQualityValidator(ticker="TEST")
    success, errors = validator.validate_ohlcv(sample_ohlcv, raise_on_error=False)
    assert success is True
    assert len(errors) == 0

def test_missing_column(sample_ohlcv):
    df = sample_ohlcv.drop(columns=["high"])
    validator = DataQualityValidator(ticker="TEST")
    success, errors = validator.validate_ohlcv(df, raise_on_error=False)
    assert success is False
    assert any("Missing columns" in e for e in errors)

def test_negative_price(sample_ohlcv):
    df = sample_ohlcv.copy()
    df.loc[0, "close"] = -10
    validator = DataQualityValidator(ticker="TEST")
    success, errors = validator.validate_ohlcv(df, raise_on_error=False)
    assert success is False
    assert any("Negative values found in 'close'" in e for e in errors)

def test_high_low_violation(sample_ohlcv):
    df = sample_ohlcv.copy()
    df.loc[0, "high"] = 90
    df.loc[0, "low"] = 100
    validator = DataQualityValidator(ticker="TEST")
    success, errors = validator.validate_ohlcv(df, raise_on_error=False)
    assert success is False
    assert any("High < Low" in e for e in errors)

def test_feature_validation():
    df = pd.DataFrame({
        "feat1": [1, 2, np.nan, 4, 5],
        "feat2": [1, 1, 1, 1, 1]
    })
    validator = DataQualityValidator(ticker="TEST")
    # 1/5 = 20% null, threshold is 0.2, should be at edge
    success, warnings = validator.validate_features(df, ["feat1", "feat2"], null_threshold=0.1)
    assert success is True
    assert any("High null rate for feature 'feat1'" in w for w in warnings)
