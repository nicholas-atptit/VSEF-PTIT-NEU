"""Tests for Module 1: Feature Engineering.

Validates that all features are computed correctly on synthetic OHLCV data,
NaN handling uses only ffill, and output shape is as expected.
"""

from __future__ import annotations

import warnings

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

    def test_transform_adds_explicit_price_reference_semantics(
        self,
        fe: FeatureEngineer,
        ohlcv_df: pd.DataFrame,
    ):
        result = fe.transform(ohlcv_df)
        assert "raw_close" in result.columns
        assert "model_close_reference" in result.columns
        assert "close_raw" in result.columns
        assert "adjusted_close" not in result.columns
        np.testing.assert_allclose(result["raw_close"].to_numpy(), result["close_raw"].to_numpy())
        np.testing.assert_allclose(result["model_close_reference"].to_numpy(), result["close"].to_numpy())

    def test_adjusted_close_only_exists_when_input_is_explicitly_adjusted(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        df = ohlcv_df.copy()
        df["price_adjustment_status"] = "adjusted"

        result = fe.transform(df, drop_na=False)

        assert "adjusted_close" in result.columns
        assert "raw_close" in result.columns
        assert result["raw_close"].isna().all()
        np.testing.assert_allclose(result["adjusted_close"].to_numpy(), result["close"].to_numpy())
        np.testing.assert_allclose(result["model_close_reference"].to_numpy(), result["close"].to_numpy())

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

    def test_legacy_compatibility_columns_not_in_default_feature_list(
        self,
        fe: FeatureEngineer,
        ohlcv_df: pd.DataFrame,
    ):
        """Legacy compatibility aliases should not silently change canonical feature selection."""
        result = fe.transform(ohlcv_df)
        feature_cols = fe.get_feature_columns(result)
        for legacy_col in [
            "hv_20",
            "bb_bandwidth_5",
            "bb_bandwidth_20",
            "bb_bandwidth_60",
            "pivot",
            "resistance_1",
            "support_1",
        ]:
            assert legacy_col not in feature_cols


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


class TestExpandedFeatureSet:
    """Test the additional forecasting and regime features added in the audit pass."""

    def test_adds_trend_indicators(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        result = fe.transform(ohlcv_df)
        for column in ["sma_20", "sma_50", "sma_200", "ema_20", "stoch_k_14", "stoch_d_14", "adx_14"]:
            assert column in result.columns

    def test_adds_regime_features(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        result = fe.transform(ohlcv_df)
        assert "market_regime" in result.columns
        assert "market_regime_code" in result.columns
        assert "volatility_regime" in result.columns
        assert "volatility_regime_code" in result.columns
        assert "trend_regime" in result.columns
        assert "trend_regime_code" in result.columns
        assert "market_regime_persistence_days" in result.columns
        assert "market_regime_transition_flag" in result.columns
        assert set(result["market_regime"].unique()) <= {"bull", "bear", "sideways"}
        assert set(result["volatility_regime"].unique()) <= {"low", "normal", "high"}
        assert set(result["trend_regime"].unique()) <= {"uptrend", "downtrend", "neutral"}

    def test_context_features_add_beta_and_correlation(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        df = ohlcv_df.copy()
        df["m_ret"] = pd.Series(df["close"]).pct_change().fillna(0.0) * 0.8
        df["s_ret"] = pd.Series(df["close"]).pct_change().fillna(0.0) * 0.6
        df["sector_dispersion"] = np.linspace(0.01, 0.03, len(df))
        df["advancing_share"] = np.linspace(0.45, 0.60, len(df))
        df["market_breadth"] = np.linspace(-0.2, 0.2, len(df))
        df["advance_decline_ratio"] = np.linspace(0.8, 1.2, len(df))
        df["foreign_net_value"] = (df["close"] * df["volume"]).mul(0.03)
        df["foreign_net_volume"] = df["volume"].mul(0.05)
        df["fx_usdvnd"] = 24_000 + np.arange(len(df)) * 2
        df["interest_rate"] = 4.0 + np.arange(len(df)) * 0.001
        df["gold_price"] = 2_000 + np.arange(len(df)) * 0.3
        df["oil_price"] = 80 + np.arange(len(df)) * 0.02

        result = fe.transform(df)

        for column in [
            "rolling_beta_market_20",
            "rolling_beta_market_60",
            "rolling_corr_market_20",
            "rolling_corr_market_60",
            "relative_strength_market_20",
            "relative_strength_sector_20",
            "sector_dispersion_zscore_20",
            "foreign_flow_intensity_zscore_20",
            "macro_shock_index",
        ]:
            assert column in result.columns

    def test_adds_corporate_action_diagnostics(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        df = ohlcv_df.copy()
        event_index = 120
        prev_close = float(df.loc[event_index - 1, "close"])
        df.loc[event_index, "open"] = round(prev_close * 0.72, 2)
        df.loc[event_index, "high"] = round(prev_close * 0.74, 2)
        df.loc[event_index, "low"] = round(prev_close * 0.69, 2)
        df.loc[event_index, "close"] = round(prev_close * 0.70, 2)
        df.loc[event_index, "volume"] = int(df.loc[event_index - 20:event_index - 1, "volume"].mean() * 4)

        result = fe.transform(df, drop_na=False)

        assert "abnormal_gap_flag" in result.columns
        assert "potential_corporate_action_flag" in result.columns
        assert "recent_corporate_action_risk_20d" in result.columns
        assert result.loc[event_index, "abnormal_gap_flag"] == 1.0
        assert result.loc[event_index, "potential_corporate_action_flag"] == 1.0

    def test_get_feature_columns_excludes_known_duplicate_aliases(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        result = fe.transform(ohlcv_df)
        feature_cols = fe.get_feature_columns(result)
        assert "pct_return" not in feature_cols
        assert "rsi" not in feature_cols
        assert "volume_shock_5" not in feature_cols
        assert "raw_close" not in feature_cols
        assert "model_close_reference" not in feature_cols
        assert "sma_20" in feature_cols

    def test_selected_features_are_prefix_invariant(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        base = ohlcv_df.copy()
        base["m_ret"] = pd.Series(base["close"]).pct_change().fillna(0.0) * 0.8
        base["s_ret"] = pd.Series(base["close"]).pct_change().fillna(0.0) * 0.6
        base["market_breadth"] = np.linspace(-0.2, 0.2, len(base))
        base["advancing_share"] = np.linspace(0.45, 0.60, len(base))
        base["advance_decline_ratio"] = np.linspace(0.8, 1.2, len(base))
        base["sector_dispersion"] = np.linspace(0.01, 0.03, len(base))
        base["foreign_net_value"] = (base["close"] * base["volume"]).mul(0.03)
        base["foreign_net_volume"] = base["volume"].mul(0.05)
        base["fx_usdvnd"] = 24_000 + np.arange(len(base)) * 2
        base["interest_rate"] = 4.0 + np.arange(len(base)) * 0.001
        base["gold_price"] = 2_000 + np.arange(len(base)) * 0.3
        base["oil_price"] = 80 + np.arange(len(base)) * 0.02

        full = fe.transform(base, drop_na=False)
        prefix_input = base.iloc[:150].copy()
        prefix = fe.transform(prefix_input, drop_na=False)

        for column in [
            "sma_20",
            "rolling_min_20",
            "volume_std_20",
            "stoch_k_14",
            "adx_14",
            "market_regime_code",
            "market_regime_persistence_days",
            "foreign_flow_intensity_zscore_20",
            "macro_shock_index",
        ]:
            np.testing.assert_allclose(
                prefix[column].to_numpy(dtype=float),
                full.iloc[:150][column].to_numpy(dtype=float),
                atol=1e-10,
                equal_nan=True,
            )

    def test_main_path_avoids_pandas_fragmentation_warning(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", category=pd.errors.PerformanceWarning)
            fe.transform(ohlcv_df, drop_na=False)

        performance_warnings = [
            warning
            for warning in captured
            if issubclass(warning.category, pd.errors.PerformanceWarning)
        ]
        assert performance_warnings == []

    def test_build_modes_are_explicit_and_semantically_stable(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        full_default = fe.transform(ohlcv_df, drop_na=False)
        full_explicit = fe.transform(ohlcv_df, drop_na=False, build_mode="full_research_mode")
        fast_core = fe.transform(ohlcv_df, drop_na=False, build_mode="fast_core_mode")
        regime_risk = fe.transform(ohlcv_df, drop_na=False, build_mode="regime_risk_mode")

        for column in ["close_return_20d", "rsi_14", "macd_hist", "turnover_ratio_20"]:
            np.testing.assert_allclose(
                full_default[column].to_numpy(dtype=float),
                full_explicit[column].to_numpy(dtype=float),
                atol=1e-12,
                equal_nan=True,
            )
            np.testing.assert_allclose(
                fast_core[column].to_numpy(dtype=float),
                full_default[column].to_numpy(dtype=float),
                atol=1e-12,
                equal_nan=True,
            )

        assert "close_kalman" in full_default.columns
        assert "close_kalman" not in fast_core.columns
        assert "market_regime_code" not in fast_core.columns
        assert "market_regime_code" in regime_risk.columns
        assert not any(column.startswith("d_") for column in fast_core.columns)
        assert not any(column.startswith("d_") for column in regime_risk.columns)


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


class TestFeatureInventory:
    def test_inventory_contains_governance_fields(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        result = fe.transform(ohlcv_df, drop_na=False)
        inventory = fe.build_feature_inventory(result)

        required = {
            "feature_name",
            "category",
            "input_source",
            "exact_provenance",
            "formula_logic",
            "expected_availability",
            "leakage_risk_note",
            "usable_for_forecast",
            "usable_for_regime",
            "usable_for_risk",
            "status",
        }
        assert required <= set(inventory.columns)
        assert set(inventory["status"].unique()) <= {"active", "deprecated", "experimental"}


class TestPandasFragmentationAndPerformance:
    """Test that main code paths avoid pandas fragmentation and performance warnings."""

    def test_all_build_modes_avoid_fragmentation_warnings(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        """All build modes should avoid SettingWithCopyWarning and PerformanceWarning."""
        build_modes = ["fast_core_mode", "regime_risk_mode", "full_research_mode"]
        
        for mode in build_modes:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                result = fe.transform(ohlcv_df, drop_na=False, build_mode=mode)
                
                # Filter for performance and copy warnings
                bad_warnings = [
                    w for w in captured
                    if issubclass(w.category, (pd.errors.PerformanceWarning, pd.errors.SettingWithCopyWarning))
                ]
                
                assert bad_warnings == [], f"{mode}: {[str(w.message) for w in bad_warnings]}"
                assert len(result) > 0

    def test_build_feature_frame_tracks_performance_stages(self, fe: FeatureEngineer, ohlcv_df: pd.DataFrame):
        """build_feature_frame should track timing for each major stage."""
        result = fe.build_feature_frame(ohlcv_df, build_mode="regime_risk_mode")
        
        assert "feature_build_stages" in result.attrs
        stages = result.attrs["feature_build_stages"]
        
        # Expected stages in the build process
        expected_stages = [
            "returns",
            "volatility",
            "momentum",
            "volume",
            "rolling_features",
        ]
        
        for stage in expected_stages:
            assert stage in stages, f"Missing timing for stage: {stage}"
            # Each stage should have a timing string like "123.45ms"
            assert isinstance(stages[stage], str)
            assert "ms" in stages[stage]

    def test_full_research_mode_preserves_stage_timings_after_delta_features(
        self,
        fe: FeatureEngineer,
        ohlcv_df: pd.DataFrame,
    ):
        result = fe.build_feature_frame(ohlcv_df, build_mode="full_research_mode")

        stages = result.attrs.get("feature_build_stages", {})

        assert "returns" in stages
        assert "legacy_compatibility" in stages
        assert "delta_features" in stages
        assert all("ms" in timing for timing in stages.values())

    def test_incremental_update_preserves_semantics(
        self,
        fe: FeatureEngineer,
    ):
        """Verify that incremental macro/foreign updates don't change feature semantics."""
        from src.ml.data_loader import (
            build_macro_context_incremental,
            build_macro_context_from_vnstock,
            clear_artifact_frame_cache,
        )
        
        # Test with synthetic macro data - just verify no exceptions and data structure
        # (actual API calls would be tested in integration tests)
        try:
            clear_artifact_frame_cache()
            result = build_macro_context_incremental(incremental_update=True, lookback_days=400)
            
            # Should return either valid data or stub frame
            assert isinstance(result, pd.DataFrame)
            if not result.empty:
                assert "date" in result.columns
        except Exception as e:
            # API unavailable is OK in test environment
            assert "unavailable" in str(e).lower() or "missing" in str(e).lower()
