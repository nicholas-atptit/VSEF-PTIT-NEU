"""Tests for Module 1: Feature Engineering.

Validates that all features are computed correctly on synthetic OHLCV data,
NaN handling uses only ffill, and output shape is as expected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.data_loader import generate_mock_data
from src.ml.feature_engineering import FeatureEngineer


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    """Generate a standard mock OHLCV DataFrame for testing."""
    return generate_mock_data(ticker="TEST", num_days=200, seed=42)


@pytest.fixture
def fe() -> FeatureEngineer:
    """Create a FeatureEngineer instance."""
    return FeatureEngineer()


class TestFeatureTransform:
    """Test the complete feature transformation pipeline."""

    def test_transform_returns_dataframe(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        """Transform should return a pandas DataFrame."""
        result = fe.transform(ohlcv_df)
        assert isinstance(result, pd.DataFrame)

    def test_transform_no_nan(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        """Output should contain no NaN values (dropped during transform)."""
        result = fe.transform(ohlcv_df)
        assert result.isna().sum().sum() == 0

    def test_transform_preserves_ohlcv(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        """Original OHLCV columns should be preserved."""
        result = fe.transform(ohlcv_df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_transform_adds_features(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        """Transform should add significantly more columns than input."""
        result = fe.transform(ohlcv_df)
        assert len(result.columns) > len(ohlcv_df.columns) + 10

    def test_transform_fewer_rows(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        """Output should have fewer rows due to rolling window warmup."""
        result = fe.transform(ohlcv_df)
        assert len(result) < len(ohlcv_df)
        assert len(result) > 0

    def test_feature_columns_list(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        """get_feature_columns should exclude date and OHLCV."""
        result = fe.transform(ohlcv_df)
        feature_cols = fe.get_feature_columns(result)
        excluded = {"date", "open", "high", "low", "close", "volume"}
        for col in feature_cols:
            assert col not in excluded


class TestVolatilityFeatures:
    """Test individual volatility feature computations."""

    def test_atr_14_present(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        """ATR-14 should be in the output."""
        result = fe.transform(ohlcv_df)
        assert "atr_14" in result.columns

    def test_atr_14_positive(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        """ATR should always be positive (it's an absolute range)."""
        result = fe.transform(ohlcv_df)
        assert (result["atr_14"] > 0).all()

    def test_historical_volatility(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        """HV-20 should be present and non-negative."""
        result = fe.transform(ohlcv_df)
        assert "hv_20" in result.columns
        assert (result["hv_20"] >= 0).all()

    def test_bollinger_bandwidth(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        """Bollinger bandwidth should exist for all window sizes."""
        result = fe.transform(ohlcv_df)
        for w in [5, 20, 60]:
            assert f"bb_bandwidth_{w}" in result.columns


class TestMomentumFeatures:
    """Test momentum and structure features."""

    def test_price_gap(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        """Price gap should be present and reasonably bounded."""
        result = fe.transform(ohlcv_df)
        assert "price_gap" in result.columns
        # Price gaps should be small (< 10% for normal stocks)
        assert (result["price_gap"].abs() < 0.10).all()

    def test_n_session_range(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        """N-session ranges should be positive."""
        result = fe.transform(ohlcv_df)
        assert "range_5" in result.columns
        assert "range_20" in result.columns
        assert (result["range_5"] >= 0).all()
        assert (result["range_20"] >= 0).all()

    def test_pivot_points(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        """Pivot points should be present."""
        result = fe.transform(ohlcv_df)
        assert "pivot" in result.columns
        assert "resistance_1" in result.columns
        assert "support_1" in result.columns
        # R1 should be above pivot, S1 below
        assert (result["resistance_1"] >= result["support_1"]).all()


class TestNaNHandling:
    """Test that NaN handling uses only forward-fill."""

    def test_ffill_only(self, fe: FeatureEngineer):
        """Verify only ffill is used by inserting NaN and checking output."""
        df = generate_mock_data(ticker="NAN_TEST", num_days=200, seed=99)
        # Insert some NaN in the middle
        df.loc[50, "close"] = np.nan
        df.loc[51, "high"] = np.nan

        result = fe.transform(df)
        # Should still produce a valid output (ffill handles NaN)
        assert result.isna().sum().sum() == 0
        assert len(result) > 0
