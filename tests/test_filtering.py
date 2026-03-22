"""Tests for Module 2: Filter Engine & Data Standardization."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from src.filtering.blacklist import BlacklistFilter
from src.filtering.noise import NoiseFilter
from src.filtering.standardizer import DataStandardizer
from src.filtering.watchlist import IndicatorBuffer, WatchlistComputer
from src.utils.time_utils import VN_TZ


class TestBlacklistFilter:
    """Test blacklist filtering logic."""

    def test_initial_empty(self):
        """Blacklist should be empty initially."""
        bf = BlacklistFilter()
        assert bf.count == 0

    def test_is_blacklisted_empty(self):
        """Nothing is blacklisted when list is empty."""
        bf = BlacklistFilter()
        assert not bf.is_blacklisted("HPG")

    def test_filter_message_keeps_clean(self):
        """Clean messages should pass through."""
        bf = BlacklistFilter()
        assert bf.filter_message({"ticker": "HPG"}) is True

    def test_filter_message_drops_blacklisted(self):
        """Blacklisted messages should be dropped."""
        bf = BlacklistFilter()
        bf._blacklist = {"FLC", "ROS", "HAI"}
        assert bf.filter_message({"ticker": "FLC"}) is False
        assert bf.filter_message({"ticker": "HPG"}) is True

    def test_case_insensitive(self):
        """Blacklist check should be case-insensitive."""
        bf = BlacklistFilter()
        bf._blacklist = {"FLC"}
        assert bf.is_blacklisted("flc") is True
        assert bf.is_blacklisted("FLC") is True


class TestNoiseFilter:
    """Test put-through trade filtering."""

    def test_order_matching_passes(self):
        """Normal order-matching trades should pass."""
        nf = NoiseFilter()
        assert nf.filter({"ticker": "HPG", "board_id": "ROUND_LOT"}) is True

    def test_put_through_dropped(self):
        """Put-through trades should be filtered out."""
        nf = NoiseFilter()
        assert nf.filter({"ticker": "HPG", "board_id": "PT"}) is False
        assert nf.filter({"ticker": "HPG", "trade_type": "TT"}) is False

    def test_is_put_through_flag(self):
        """Explicit is_put_through flag should filter."""
        nf = NoiseFilter()
        assert nf.filter({"ticker": "HPG", "is_put_through": True}) is False
        assert nf.filter({"ticker": "HPG", "is_put_through": False}) is True

    def test_stats_tracking(self):
        """Filter should track pass/filter counts."""
        nf = NoiseFilter()
        nf.filter({"ticker": "HPG"})
        nf.filter({"ticker": "VIC", "board_id": "PT"})
        nf.filter({"ticker": "VNM"})

        stats = nf.stats
        assert stats["total_processed"] == 3
        assert stats["passed"] == 2
        assert stats["filtered"] == 1

    def test_stats_reset(self):
        """Reset should clear all stats."""
        nf = NoiseFilter()
        nf.filter({"ticker": "HPG"})
        nf.reset_stats()
        assert nf.stats["total_processed"] == 0

    def test_stream_message_object(self):
        """Should handle DNSE stream message objects."""
        nf = NoiseFilter()
        msg = MagicMock()
        msg.board_id = "ROUND_LOT"
        msg.trade_type = None
        msg.match_type = None
        assert nf.is_order_matching(msg) is True

        msg2 = MagicMock()
        msg2.board_id = "PUT_THROUGH"
        msg2.trade_type = None
        msg2.match_type = None
        assert nf.is_order_matching(msg2) is False


class TestDataStandardizer:
    """Test DNSE message standardization."""

    def test_standardize_trade(self):
        """Trade message should map to canonical schema."""
        std = DataStandardizer()
        msg = MagicMock()
        msg.symbol = "HPG"
        msg.price = 25500
        msg.volume = 1000
        msg.timestamp = None
        msg.exchange = "HOSE"

        result = std.standardize_trade(msg)

        assert result is not None
        assert result["ticker"] == "HPG"
        assert result["close"] == Decimal("25500")
        assert result["volume"] == 1000
        assert result["timeframe"] == "tick"
        assert result["message_type"] == "trade"

    def test_standardize_ohlc(self):
        """OHLC message should map to canonical schema."""
        std = DataStandardizer()
        msg = MagicMock()
        msg.symbol = "VIC"
        msg.open = 45000
        msg.high = 46000
        msg.low = 44500
        msg.close = 45800
        msg.volume = 5000
        msg.timestamp = None
        msg.exchange = "HOSE"

        result = std.standardize_ohlc(msg)

        assert result is not None
        assert result["ticker"] == "VIC"
        assert result["open"] == Decimal("45000")
        assert result["high"] == Decimal("46000")
        assert result["timeframe"] == "1m"

    def test_standardize_dict(self):
        """Dict data should be standardized."""
        std = DataStandardizer()
        result = std.standardize_dict({
            "ticker": "hpg",
            "open": 25000,
            "high": 25500,
            "low": 24800,
            "close": 25200,
            "volume": 10000,
            "exchange": "HOSE",
        })

        assert result["ticker"] == "HPG"  # Uppercased
        assert result["exchange"] == "HOSE"
        assert result["source"] == "rest_api"

    def test_standardize_none_symbol(self):
        """Messages without symbol should return None."""
        std = DataStandardizer()
        msg = MagicMock()
        msg.symbol = None
        assert std.standardize_trade(msg) is None


class TestIndicatorBuffer:
    """Test rolling indicator buffer."""

    def test_add_candle(self):
        """Buffer should track candle count."""
        buf = IndicatorBuffer("HPG", buffer_size=10)
        now = dt.datetime.now(VN_TZ)
        buf.add_candle(now, 25000, 25500, 24800, 1000)
        assert buf.candle_count == 1

    def test_buffer_trimming(self):
        """Buffer should trim to max size."""
        buf = IndicatorBuffer("HPG", buffer_size=5)
        for i in range(10):
            buf.add_candle(
                dt.datetime.now(VN_TZ),
                close=25000 + i * 100,
            )
        assert buf.candle_count == 5

    def test_empty_indicators(self):
        """Empty buffer should return None indicators."""
        result = IndicatorBuffer._empty_indicators()
        assert result["rsi"] is None
        assert result["macd"] is None
        assert result["sma_20"] is None

    def test_compute_with_few_candles(self):
        """Should handle computation with too few candles."""
        buf = IndicatorBuffer("HPG", buffer_size=200)
        buf.add_candle(dt.datetime.now(VN_TZ), 25000)
        indicators = buf.compute_indicators()
        assert indicators["rsi"] is None
