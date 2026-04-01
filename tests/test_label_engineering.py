"""Comprehensive tests for the label engineering package.

Validates:
    - All 8 label generators produce correct output
    - Time-safety (labels use future data via shift, last N rows are NaN)
    - Registry discovery and batch application
    - Settings-driven threshold overrides
    - Edge cases and mathematical correctness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.data_loader import generate_mock_data
from src.ml.labels import (
    LABEL_NAMES,
    LABEL_REGISTRY,
    apply_all_labels,
    get_generator,
)
from src.ml.labels.base import BaseLabelGenerator
from src.ml.labels.classification import (
    Cls1d3Class,
    Cls1dUpDown,
    Cls20dUpDown,
    Cls5d3Class,
    Cls5dUpDown,
)
from src.ml.labels.regression import Reg5dReturn, RegNextCloseReturn
from src.ml.labels.volatility import FutureRealizedVol5d


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    """Generate 300-day mock OHLCV data for label tests."""
    return generate_mock_data(ticker="LABEL_TEST", num_days=300, seed=999)


# ═══════════════════════════════════════════════════════════════════════════
# REGISTRY TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestRegistry:
    """Verify the label registry is complete and functional."""

    def test_registry_has_8_entries(self):
        assert len(LABEL_REGISTRY) == 8

    def test_all_names_match_keys(self):
        assert set(LABEL_NAMES) == set(LABEL_REGISTRY.keys())

    @pytest.mark.parametrize("name", LABEL_NAMES)
    def test_get_generator_returns_instance(self, name: str):
        gen = get_generator(name, use_settings=False)
        assert isinstance(gen, BaseLabelGenerator)
        assert gen.name == name

    def test_get_generator_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown label"):
            get_generator("nonexistent_label")

    @pytest.mark.parametrize("name", LABEL_NAMES)
    def test_generator_has_label_columns(self, name: str):
        gen = get_generator(name, use_settings=False)
        assert len(gen.label_columns) >= 1
        assert all(isinstance(c, str) for c in gen.label_columns)


# ═══════════════════════════════════════════════════════════════════════════
# BINARY CLASSIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestCls1dUpDown:
    """Tests for 1-day binary up/down classifier."""

    def test_column_present(self, ohlcv_df: pd.DataFrame):
        gen = Cls1dUpDown()
        result = gen.generate(ohlcv_df)
        assert "label_cls_1d_updown" in result.columns

    def test_values_are_0_or_1_or_na(self, ohlcv_df: pd.DataFrame):
        gen = Cls1dUpDown()
        result = gen.generate(ohlcv_df)
        valid = result["label_cls_1d_updown"].dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_last_row_is_nan(self, ohlcv_df: pd.DataFrame):
        gen = Cls1dUpDown()
        result = gen.generate(ohlcv_df)
        assert pd.isna(result["label_cls_1d_updown"].iloc[-1])

    def test_time_safety_correct_direction(self, ohlcv_df: pd.DataFrame):
        """If close[t+1] > close[t], label should be 1."""
        gen = Cls1dUpDown()
        result = gen.generate(ohlcv_df)
        for i in range(len(result) - 1):
            label = result["label_cls_1d_updown"].iloc[i]
            current_close = result["close"].iloc[i]
            next_close = result["close"].iloc[i + 1]
            if next_close > current_close:
                assert label == 1
            else:
                assert label == 0


class TestCls5dUpDown:
    """Tests for 5-day binary up/down classifier."""

    def test_last_5_rows_nan(self, ohlcv_df: pd.DataFrame):
        gen = Cls5dUpDown()
        result = gen.generate(ohlcv_df)
        assert result["label_cls_5d_updown"].iloc[-5:].isna().all()

    def test_valid_labels_count(self, ohlcv_df: pd.DataFrame):
        gen = Cls5dUpDown()
        result = gen.generate(ohlcv_df)
        valid = result["label_cls_5d_updown"].dropna()
        assert len(valid) == len(ohlcv_df) - 5


class TestCls20dUpDown:
    """Tests for 20-day binary up/down classifier."""

    def test_last_20_rows_nan(self, ohlcv_df: pd.DataFrame):
        gen = Cls20dUpDown()
        result = gen.generate(ohlcv_df)
        assert result["label_cls_20d_updown"].iloc[-20:].isna().all()

    def test_valid_labels_count(self, ohlcv_df: pd.DataFrame):
        gen = Cls20dUpDown()
        result = gen.generate(ohlcv_df)
        valid = result["label_cls_20d_updown"].dropna()
        assert len(valid) == len(ohlcv_df) - 20


# ═══════════════════════════════════════════════════════════════════════════
# TERNARY CLASSIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestCls1d3Class:
    """Tests for 1-day ternary (up/sideways/down) classifier."""

    def test_column_present(self, ohlcv_df: pd.DataFrame):
        gen = Cls1d3Class(threshold=0.01)
        result = gen.generate(ohlcv_df)
        assert "label_cls_1d_3class" in result.columns

    def test_values_are_0_1_2_or_na(self, ohlcv_df: pd.DataFrame):
        gen = Cls1d3Class(threshold=0.01)
        result = gen.generate(ohlcv_df)
        valid = result["label_cls_1d_3class"].dropna()
        assert set(valid.unique()).issubset({0, 1, 2})

    def test_threshold_effects(self, ohlcv_df: pd.DataFrame):
        """Wider threshold → more 'sideways' labels."""
        narrow = Cls1d3Class(threshold=0.001).generate(ohlcv_df)
        wide = Cls1d3Class(threshold=0.05).generate(ohlcv_df)
        n_sideways_narrow = (narrow["label_cls_1d_3class"].dropna() == 1).sum()
        n_sideways_wide = (wide["label_cls_1d_3class"].dropna() == 1).sum()
        assert n_sideways_wide >= n_sideways_narrow

    def test_up_label_when_return_exceeds_threshold(self):
        """Explicit check: large positive return → label 0 (Up)."""
        df = pd.DataFrame({
            "close": [100.0, 110.0],  # 10% return
            "open": [100.0, 100.0],
            "high": [100.0, 110.0],
            "low": [100.0, 100.0],
            "volume": [1000, 1000],
        })
        gen = Cls1d3Class(threshold=0.01)
        result = gen.generate(df)
        assert result["label_cls_1d_3class"].iloc[0] == 0  # Up

    def test_down_label_when_return_below_threshold(self):
        """Explicit check: large negative return → label 2 (Down)."""
        df = pd.DataFrame({
            "close": [100.0, 90.0],  # -10% return
            "open": [100.0, 100.0],
            "high": [100.0, 100.0],
            "low": [90.0, 90.0],
            "volume": [1000, 1000],
        })
        gen = Cls1d3Class(threshold=0.01)
        result = gen.generate(df)
        assert result["label_cls_1d_3class"].iloc[0] == 2  # Down


class TestCls5d3Class:
    """Tests for 5-day ternary classifier."""

    def test_last_5_rows_nan(self, ohlcv_df: pd.DataFrame):
        gen = Cls5d3Class(threshold=0.02)
        result = gen.generate(ohlcv_df)
        assert result["label_cls_5d_3class"].iloc[-5:].isna().all()

    def test_has_all_three_classes(self, ohlcv_df: pd.DataFrame):
        """With 300 days of data and 2% threshold, expect all 3 classes."""
        gen = Cls5d3Class(threshold=0.02)
        result = gen.generate(ohlcv_df)
        valid = result["label_cls_5d_3class"].dropna()
        assert len(valid.unique()) == 3


# ═══════════════════════════════════════════════════════════════════════════
# REGRESSION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestRegNextCloseReturn:
    """Tests for 1-day forward return regression target."""

    def test_column_present(self, ohlcv_df: pd.DataFrame):
        gen = RegNextCloseReturn()
        result = gen.generate(ohlcv_df)
        assert "target_reg_next_close_return" in result.columns

    def test_last_row_nan(self, ohlcv_df: pd.DataFrame):
        gen = RegNextCloseReturn()
        result = gen.generate(ohlcv_df)
        assert pd.isna(result["target_reg_next_close_return"].iloc[-1])

    def test_return_matches_manual_calc(self, ohlcv_df: pd.DataFrame):
        gen = RegNextCloseReturn()
        result = gen.generate(ohlcv_df)
        for i in range(min(10, len(result) - 1)):
            expected = result["close"].iloc[i + 1] / result["close"].iloc[i] - 1
            actual = result["target_reg_next_close_return"].iloc[i]
            np.testing.assert_almost_equal(actual, expected, decimal=10)

    def test_dtype_is_float(self, ohlcv_df: pd.DataFrame):
        gen = RegNextCloseReturn()
        result = gen.generate(ohlcv_df)
        assert result["target_reg_next_close_return"].dtype == np.float64


class TestReg5dReturn:
    """Tests for 5-day forward return regression target."""

    def test_last_5_rows_nan(self, ohlcv_df: pd.DataFrame):
        gen = Reg5dReturn()
        result = gen.generate(ohlcv_df)
        assert result["target_reg_5d_return"].iloc[-5:].isna().all()

    def test_return_matches_manual_calc(self, ohlcv_df: pd.DataFrame):
        gen = Reg5dReturn()
        result = gen.generate(ohlcv_df)
        for i in range(min(10, len(result) - 5)):
            expected = result["close"].iloc[i + 5] / result["close"].iloc[i] - 1
            actual = result["target_reg_5d_return"].iloc[i]
            np.testing.assert_almost_equal(actual, expected, decimal=10)


# ═══════════════════════════════════════════════════════════════════════════
# VOLATILITY TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestFutureRealizedVol5d:
    """Tests for future realised volatility target."""

    def test_column_present(self, ohlcv_df: pd.DataFrame):
        gen = FutureRealizedVol5d()
        result = gen.generate(ohlcv_df)
        assert "target_future_realized_vol_5d" in result.columns

    def test_last_5_rows_nan(self, ohlcv_df: pd.DataFrame):
        gen = FutureRealizedVol5d()
        result = gen.generate(ohlcv_df)
        assert result["target_future_realized_vol_5d"].iloc[-5:].isna().all()

    def test_volatility_is_non_negative(self, ohlcv_df: pd.DataFrame):
        gen = FutureRealizedVol5d()
        result = gen.generate(ohlcv_df)
        valid = result["target_future_realized_vol_5d"].dropna()
        assert (valid >= 0).all()

    def test_annualisation_increases_values(self, ohlcv_df: pd.DataFrame):
        """Annualised vol should be larger than raw daily vol."""
        annualised = FutureRealizedVol5d(annualise=True).generate(ohlcv_df)
        raw = FutureRealizedVol5d(annualise=False).generate(ohlcv_df)
        a_vals = annualised["target_future_realized_vol_5d"].dropna()
        r_vals = raw["target_future_realized_vol_5d"].dropna()
        # Annualised = raw * sqrt(252) ≈ raw * 15.87 — should be larger
        assert a_vals.mean() > r_vals.mean()


# ═══════════════════════════════════════════════════════════════════════════
# BATCH APPLICATION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestApplyAllLabels:
    """Tests for the batch label application helper."""

    def test_all_columns_present(self, ohlcv_df: pd.DataFrame):
        result = apply_all_labels(ohlcv_df, use_settings=False)
        expected_cols = [
            "label_cls_1d_updown",
            "label_cls_5d_updown",
            "label_cls_20d_updown",
            "label_cls_1d_3class",
            "label_cls_5d_3class",
            "target_reg_next_close_return",
            "target_reg_5d_return",
            "target_future_realized_vol_5d",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing: {col}"

    def test_original_columns_preserved(self, ohlcv_df: pd.DataFrame):
        result = apply_all_labels(ohlcv_df, use_settings=False)
        for col in ["date", "open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_selective_application(self, ohlcv_df: pd.DataFrame):
        """Apply only a subset of labels."""
        result = apply_all_labels(
            ohlcv_df,
            names=["cls_1d_updown", "reg_next_close_return"],
            use_settings=False,
        )
        assert "label_cls_1d_updown" in result.columns
        assert "target_reg_next_close_return" in result.columns
        assert "label_cls_5d_updown" not in result.columns

    def test_row_count_unchanged(self, ohlcv_df: pd.DataFrame):
        """apply_all_labels should not drop any rows."""
        result = apply_all_labels(ohlcv_df, use_settings=False)
        assert len(result) == len(ohlcv_df)


# ═══════════════════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_empty_df_raises(self):
        df = pd.DataFrame(columns=["close", "open", "high", "low", "volume"])
        gen = Cls1dUpDown()
        with pytest.raises(ValueError, match="empty"):
            gen.generate(df)

    def test_missing_close_raises(self):
        df = pd.DataFrame({"open": [1, 2, 3], "volume": [100, 200, 300]})
        gen = Cls1dUpDown()
        with pytest.raises(ValueError, match="Missing required columns"):
            gen.generate(df)

    def test_does_not_mutate_input(self, ohlcv_df: pd.DataFrame):
        """generate() should return a copy, not modify the input."""
        original_cols = set(ohlcv_df.columns)
        gen = Cls1dUpDown()
        _ = gen.generate(ohlcv_df)
        assert set(ohlcv_df.columns) == original_cols
