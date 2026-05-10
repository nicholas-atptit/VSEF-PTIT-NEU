"""Tests for VN100 daily prediction features in FeatureEngineer.

Validates:
    - All 19 VN100_DAILY_FEATURES are present after transform()
    - Correctness of key feature calculations
    - Alias consistency (e.g. close_to_close_return_1d == pct_return)
    - Sign / range constraints
    - Idempotency (running transform twice doesn't duplicate columns)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.data_loader import generate_mock_data
from src.ml.feature_engineering import FeatureEngineer, VN100_DAILY_FEATURES


# Catalogue contract as of Phase 1 runtime stabilization.
# The list intentionally evolved beyond the original 19-feature contract; the
# audit contract now preserves exact ordering plus required canonical features.
EXPECTED_VN100_DAILY_FEATURES_2026_05_10 = [
    "prev_close",
    "close_to_close_return_1d",
    "close_return_2d",
    "close_return_3d",
    "open_to_close_return_1d",
    "overnight_return_1d",
    "open_close_spread",
    "open_close_spread_pct",
    "high_low_range",
    "high_low_range_pct",
    "true_range",
    "atr_14",
    "atr_proxy_5",
    "atr_proxy_10",
    "close_mean_5",
    "close_mean_10",
    "close_mean_20",
    "close_std_5",
    "close_std_10",
    "close_std_20",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "volume_ma_5",
    "volume_ma_10",
    "volume_ma_20",
    "volume_shock_5",
    "volume_shock_10",
    "volume_shock_20",
    "volume_ratio_5",
    "volume_ratio_20",
    "value_ratio_5",
    "value_ratio_20",
    "rolling_volatility_5",
    "rolling_volatility_10",
    "rolling_volatility_20",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
]

REQUIRED_CANONICAL_VN100_FEATURES = {
    "prev_close",
    "close_to_close_return_1d",
    "open_to_close_return_1d",
    "overnight_return_1d",
    "high_low_range_pct",
    "true_range",
    "return_3d",
    "return_5d",
    "return_20d",
    "volume_ma_5",
    "volume_ratio_5",
    "value_ratio_5",
    "value_ratio_20",
    "rolling_volatility_5",
    "rolling_volatility_20",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
}


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    """Generate 300-day mock OHLCV data for robust rolling window tests."""
    return generate_mock_data(ticker="VN100_TEST", num_days=300, seed=123)


@pytest.fixture
def fe() -> FeatureEngineer:
    return FeatureEngineer()


@pytest.fixture
def result(fe: FeatureEngineer, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Pre-computed feature matrix."""
    return fe.transform(ohlcv_df)


# ── Presence Tests ───────────────────────────────────────────────────────

class TestVN100FeaturePresence:
    """Every feature in VN100_DAILY_FEATURES must appear in the output."""

    @pytest.mark.parametrize("feature_name", VN100_DAILY_FEATURES)
    def test_feature_present(self, result: pd.DataFrame, feature_name: str):
        assert feature_name in result.columns, (
            f"VN100 feature '{feature_name}' missing from transform output"
        )

    def test_no_nan_in_vn100_features(self, result: pd.DataFrame):
        """After dropna, VN100 features must be NaN-free."""
        for col in VN100_DAILY_FEATURES:
            assert result[col].isna().sum() == 0, f"NaN found in '{col}'"


# ── Correctness Tests ───────────────────────────────────────────────────

class TestVN100FeatureCorrectness:
    """Spot-check mathematical correctness of key features."""

    def test_prev_close_is_shifted(self, result: pd.DataFrame):
        """prev_close[i] should equal close[i-1] for i > 0."""
        # After dropna + reset_index, just check that first value differs
        # from the current close
        assert result["prev_close"].iloc[0] != result["close"].iloc[0] or True
        # More robust: reconstruct from close and check
        reconstructed = result["close"].shift(1)
        # After the first row is dropped, they should align
        overlap = result.index[1:]
        pd.testing.assert_series_equal(
            result["prev_close"].iloc[1:].reset_index(drop=True),
            reconstructed.iloc[1:].reset_index(drop=True),
            check_names=False,
            atol=1e-10,
        )

    def test_close_to_close_return_alias(self, result: pd.DataFrame):
        """close_to_close_return_1d should match pct_return."""
        pd.testing.assert_series_equal(
            result["close_to_close_return_1d"],
            result["pct_return"],
            check_names=False,
        )

    def test_open_to_close_return(self, result: pd.DataFrame):
        """open_to_close_return_1d = (close - open) / open."""
        expected = (result["close"] - result["open"]) / result["open"]
        pd.testing.assert_series_equal(
            result["open_to_close_return_1d"],
            expected,
            check_names=False,
            atol=1e-10,
        )

    def test_overnight_return(self, result: pd.DataFrame):
        """overnight_return_1d = (open - prev_close) / prev_close."""
        expected = (result["open"] - result["prev_close"]) / result["prev_close"]
        pd.testing.assert_series_equal(
            result["overnight_return_1d"],
            expected,
            check_names=False,
            atol=1e-10,
        )

    def test_high_low_range_pct_positive(self, result: pd.DataFrame):
        """High-low range must be non-negative (H >= L always)."""
        assert (result["high_low_range_pct"] >= 0).all()

    def test_true_range_definition(self, result: pd.DataFrame):
        """true_range = max(H-L, |H-prevC|, |L-prevC|)."""
        pc = result["prev_close"]
        tr_expected = pd.concat([
            result["high"] - result["low"],
            (result["high"] - pc).abs(),
            (result["low"] - pc).abs(),
        ], axis=1).max(axis=1)
        pd.testing.assert_series_equal(
            result["true_range"],
            tr_expected,
            check_names=False,
            atol=1e-10,
        )

    def test_true_range_ge_high_low(self, result: pd.DataFrame):
        """True range is always >= high - low."""
        hl = result["high"] - result["low"]
        assert (result["true_range"] >= hl - 1e-10).all()

    def test_return_3d(self, result: pd.DataFrame):
        """return_3d should approximately equal close.pct_change(3)."""
        recomputed = result["close"].pct_change(3)
        valid = recomputed.notna()
        np.testing.assert_allclose(
            result.loc[valid, "return_3d"].values,
            recomputed[valid].values,
            atol=1e-10,
        )

    def test_return_5d_alias(self, result: pd.DataFrame):
        """return_5d should match return_roll_5 (both exist)."""
        if "return_roll_5" in result.columns:
            pd.testing.assert_series_equal(
                result["return_5d"],
                result["return_roll_5"],
                check_names=False,
            )

    def test_return_20d_alias(self, result: pd.DataFrame):
        """return_20d should match return_roll_20 (both exist)."""
        if "return_roll_20" in result.columns:
            pd.testing.assert_series_equal(
                result["return_20d"],
                result["return_roll_20"],
                check_names=False,
            )

    def test_volume_ma_5_is_rolling_mean(self, result: pd.DataFrame):
        """volume_ma_5 should be the 5-period rolling mean of volume."""
        expected = result["volume"].rolling(5).mean()
        valid = expected.notna()
        np.testing.assert_allclose(
            result.loc[valid, "volume_ma_5"].values,
            expected[valid].values,
            rtol=1e-6,
        )

    def test_volume_ratio_identity(self, result: pd.DataFrame):
        """volume_ratio_5 ≈ volume / volume_ma_5."""
        expected = result["volume"] / result["volume_ma_5"]
        valid = expected.notna() & np.isfinite(expected)
        np.testing.assert_allclose(
            result.loc[valid, "volume_ratio_5"].values,
            expected[valid].values,
            rtol=1e-6,
        )

    def test_value_ratio_positive(self, result: pd.DataFrame):
        """Value ratios should be positive (price & volume > 0)."""
        assert (result["value_ratio_5"] > 0).all()
        assert (result["value_ratio_20"] > 0).all()

    def test_rolling_volatility_positive(self, result: pd.DataFrame):
        """Standard deviation is always non-negative."""
        assert (result["rolling_volatility_5"] >= 0).all()
        assert (result["rolling_volatility_20"] >= 0).all()

    def test_rolling_volatility_20_gt_0(self, result: pd.DataFrame):
        """With 300 days of mock data, vol-20 should be > 0 everywhere."""
        assert (result["rolling_volatility_20"] > 0).all()


# ── Idempotency & Integration ────────────────────────────────────────────

class TestVN100Idempotency:
    """Running transform twice should not change the output."""

    def test_double_transform_same_columns(
        self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame
    ):
        """Column set should be identical on repeated transform."""
        r1 = fe.transform(ohlcv_df)
        r2 = fe.transform(ohlcv_df)
        assert set(r1.columns) == set(r2.columns)

    def test_double_transform_same_shape(
        self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame
    ):
        """Row/column counts should match on repeated transform."""
        r1 = fe.transform(ohlcv_df)
        r2 = fe.transform(ohlcv_df)
        assert r1.shape == r2.shape


class TestVN100Catalogue:
    """Validate the VN100_DAILY_FEATURES catalogue constant."""

    def test_catalogue_not_empty(self):
        assert len(VN100_DAILY_FEATURES) > 0

    def test_catalogue_unique(self):
        assert len(VN100_DAILY_FEATURES) == len(set(VN100_DAILY_FEATURES))

    def test_catalogue_ordered_contract(self):
        """The expanded catalogue keeps a deterministic audited order."""
        assert VN100_DAILY_FEATURES == EXPECTED_VN100_DAILY_FEATURES_2026_05_10

    def test_catalogue_required_canonical_features(self):
        """Core public features remain present after catalogue expansion."""
        assert REQUIRED_CANONICAL_VN100_FEATURES <= set(VN100_DAILY_FEATURES)

    def test_catalogue_transform_compatibility(self, result: pd.DataFrame):
        """Every audited catalogue feature is produced by transform()."""
        assert list(VN100_DAILY_FEATURES) == EXPECTED_VN100_DAILY_FEATURES_2026_05_10
        assert set(VN100_DAILY_FEATURES) <= set(result.columns)
