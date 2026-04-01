"""Smoke tests for VN100 data loader extensions.

These tests exercise the new batch-loading functions added to
``src.ml.data_loader`` without requiring a live database or API connection.
They rely entirely on the CSV files already present in
``data/daily_market_split_data/``.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

# ── Ensure project root is on sys.path ──────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.data_loader import (
    generate_mock_data,
    load_ohlcv_from_csv,
    load_market_proxy,
    load_fundamentals,
    load_sentiment,
    VN100DataLoader,
    load_vn100_daily_dataset,
)

# ── Constants ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DAILY_CSV_DIR = PROJECT_ROOT / "data" / "daily_market_split_data"

# Pick tickers that we know exist in the CSV data directory
SAMPLE_TICKERS = ["FPT", "HPG", "VNM"]
# Sanity: at least one should exist
_TICKERS_AVAILABLE = [t for t in SAMPLE_TICKERS if (DAILY_CSV_DIR / f"{t}.csv").exists()]


# ═══════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY — original interfaces still work
# ═══════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    """Ensure the original public API is unchanged."""

    def test_generate_mock_data_still_works(self):
        df = generate_mock_data(ticker="MOCK", num_days=100)
        assert len(df) == 100
        assert set(["date", "open", "high", "low", "close", "volume"]).issubset(df.columns)

    def test_load_ohlcv_from_csv_exists(self):
        """The new CSV loader is importable."""
        assert callable(load_ohlcv_from_csv)

    def test_vn100_data_loader_class_exists(self):
        """The VN100DataLoader class is importable."""
        assert callable(VN100DataLoader)


# ═══════════════════════════════════════════════════════════════════
# load_ohlcv_from_csv
# ═══════════════════════════════════════════════════════════════════


class TestLoadOhlcvFromCsv:
    """Test file-backed CSV loading."""

    @pytest.mark.skipif(not _TICKERS_AVAILABLE, reason="No sample CSV files found")
    def test_load_known_ticker(self):
        ticker = _TICKERS_AVAILABLE[0]
        df = load_ohlcv_from_csv(ticker, csv_dir=DAILY_CSV_DIR)
        assert not df.empty
        assert "date" in df.columns
        assert "close" in df.columns
        assert "ticker" in df.columns
        assert df["ticker"].iloc[0] == ticker.upper()

    @pytest.mark.skipif(not _TICKERS_AVAILABLE, reason="No sample CSV files found")
    def test_date_filtering(self):
        ticker = _TICKERS_AVAILABLE[0]
        start = dt.date(2023, 1, 1)
        end = dt.date(2023, 12, 31)
        df = load_ohlcv_from_csv(ticker, csv_dir=DAILY_CSV_DIR, start_date=start, end_date=end)
        if not df.empty:
            assert df["date"].min() >= pd.Timestamp(start)
            assert df["date"].max() <= pd.Timestamp(end)

    def test_missing_ticker_returns_empty(self):
        df = load_ohlcv_from_csv("ZZZZNOTEXIST", csv_dir=DAILY_CSV_DIR)
        assert df.empty


# ═══════════════════════════════════════════════════════════════════
# Auxiliary loaders
# ═══════════════════════════════════════════════════════════════════


class TestAuxiliaryLoaders:
    """Test market proxy, fundamentals, and sentiment loaders."""

    def test_load_market_proxy(self):
        market_path = PROJECT_ROOT / "data" / "market_proxy.csv"
        if market_path.exists():
            df = load_market_proxy()
            assert not df.empty
            assert "date" in df.columns
            assert "m_ret" in df.columns
        else:
            # Should return empty gracefully
            df = load_market_proxy(path="/nonexistent/path.csv")
            assert df.empty

    def test_load_fundamentals_no_crash(self):
        """Fundamentals loader never crashes even if file missing."""
        df = load_fundamentals(path="/nonexistent/fund.csv")
        assert isinstance(df, pd.DataFrame)

    def test_load_sentiment_no_crash(self):
        """Sentiment loader never crashes even if file missing."""
        df = load_sentiment(path="/nonexistent/sent.csv")
        assert isinstance(df, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════
# VN100DataLoader class
# ═══════════════════════════════════════════════════════════════════


class TestVN100DataLoader:
    """Test the batch data loader."""

    @pytest.mark.skipif(not _TICKERS_AVAILABLE, reason="No sample CSV files found")
    def test_build_dataset_basic(self):
        loader = VN100DataLoader(csv_dir=DAILY_CSV_DIR, prefer_source="csv")
        df = loader.build_dataset(
            tickers=_TICKERS_AVAILABLE[:2],
            join_market=False,
        )
        assert not df.empty
        assert "ticker" in df.columns
        assert "date" in df.columns
        assert "close" in df.columns
        # Should have data for at least one ticker
        assert df["ticker"].nunique() >= 1

    @pytest.mark.skipif(not _TICKERS_AVAILABLE, reason="No sample CSV files found")
    def test_build_dataset_with_market_join(self):
        loader = VN100DataLoader(csv_dir=DAILY_CSV_DIR, prefer_source="csv")
        df = loader.build_dataset(
            tickers=_TICKERS_AVAILABLE[:1],
            join_market=True,
        )
        market_path = PROJECT_ROOT / "data" / "market_proxy.csv"
        if market_path.exists() and not df.empty:
            assert "m_ret" in df.columns

    @pytest.mark.skipif(not _TICKERS_AVAILABLE, reason="No sample CSV files found")
    def test_build_dataset_with_date_filter(self):
        loader = VN100DataLoader(csv_dir=DAILY_CSV_DIR, prefer_source="csv")
        start = dt.date(2023, 6, 1)
        end = dt.date(2023, 12, 31)
        df = loader.build_dataset(
            tickers=_TICKERS_AVAILABLE[:1],
            start_date=start,
            end_date=end,
            join_market=False,
        )
        if not df.empty:
            assert df["date"].min() >= pd.Timestamp(start)
            assert df["date"].max() <= pd.Timestamp(end)

    def test_build_dataset_empty_tickers(self):
        loader = VN100DataLoader(csv_dir=DAILY_CSV_DIR)
        df = loader.build_dataset(tickers=[])
        assert df.empty

    def test_build_dataset_unknown_tickers(self):
        loader = VN100DataLoader(csv_dir=DAILY_CSV_DIR)
        df = loader.build_dataset(tickers=["ZZZZNOTEXIST1", "ZZZZNOTEXIST2"])
        assert df.empty

    @pytest.mark.skipif(not _TICKERS_AVAILABLE, reason="No sample CSV files found")
    def test_build_inference_dataset(self):
        loader = VN100DataLoader(csv_dir=DAILY_CSV_DIR, prefer_source="csv")
        df = loader.build_inference_dataset(
            tickers=_TICKERS_AVAILABLE[:1],
            lookback_days=365,
        )
        assert not df.empty
        assert "ticker" in df.columns

    @pytest.mark.skipif(not _TICKERS_AVAILABLE, reason="No sample CSV files found")
    def test_dataset_sorted_by_ticker_date(self):
        loader = VN100DataLoader(csv_dir=DAILY_CSV_DIR, prefer_source="csv")
        df = loader.build_dataset(
            tickers=_TICKERS_AVAILABLE[:2],
            join_market=False,
        )
        if len(df) > 1:
            # Verify sorted by [ticker, date]
            is_sorted = (
                df["ticker"].is_monotonic_increasing
                or all(
                    df.groupby("ticker")["date"]
                    .apply(lambda s: s.is_monotonic_increasing)
                )
            )
            assert is_sorted


# ═══════════════════════════════════════════════════════════════════
# Convenience function
# ═══════════════════════════════════════════════════════════════════


class TestConvenienceFunction:
    """Test the load_vn100_daily_dataset one-liner."""

    @pytest.mark.skipif(not _TICKERS_AVAILABLE, reason="No sample CSV files found")
    def test_explicit_tickers(self):
        df = load_vn100_daily_dataset(
            tickers=_TICKERS_AVAILABLE[:1],
            start_date=dt.date(2023, 1, 1),
            join_market=False,
            prefer_source="csv",
        )
        assert not df.empty


# ═══════════════════════════════════════════════════════════════════
# DataFrame schema validation
# ═══════════════════════════════════════════════════════════════════


class TestDataFrameSchema:
    """Verify the output DataFrame matches the ML pipeline expectations."""

    @pytest.mark.skipif(not _TICKERS_AVAILABLE, reason="No sample CSV files found")
    def test_output_columns_match_pipeline(self):
        loader = VN100DataLoader(csv_dir=DAILY_CSV_DIR, prefer_source="csv")
        df = loader.build_dataset(
            tickers=_TICKERS_AVAILABLE[:1],
            join_market=False,
        )
        if not df.empty:
            required = {"date", "open", "high", "low", "close", "volume", "ticker"}
            assert required.issubset(set(df.columns)), (
                f"Missing columns: {required - set(df.columns)}"
            )

    @pytest.mark.skipif(not _TICKERS_AVAILABLE, reason="No sample CSV files found")
    def test_numeric_types(self):
        loader = VN100DataLoader(csv_dir=DAILY_CSV_DIR, prefer_source="csv")
        df = loader.build_dataset(
            tickers=_TICKERS_AVAILABLE[:1],
            join_market=False,
        )
        if not df.empty:
            assert pd.api.types.is_float_dtype(df["close"])
            assert pd.api.types.is_float_dtype(df["open"])
            assert pd.api.types.is_integer_dtype(df["volume"])
