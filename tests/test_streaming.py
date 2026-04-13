"""Tests for Module 1: Session Streaming Manager."""

from __future__ import annotations

import asyncio
from importlib import import_module
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.streaming.session_manager import ConnectionState, SessionStreamingManager


class TestSessionStreamingManager:
    """Test WebSocket lifecycle management."""

    def test_legacy_import_path_alias(self):
        """Legacy streaming import path should resolve to the canonical module."""
        legacy_module = import_module("src.streaming.session_manager")
        assert legacy_module.SessionStreamingManager is SessionStreamingManager

    def test_initial_state(self):
        """Manager starts in DISCONNECTED state."""
        mgr = SessionStreamingManager()
        assert mgr.state == ConnectionState.DISCONNECTED
        assert not mgr.is_connected

    @patch("src.streaming.session_manager.Redis")
    def test_set_watchlist(self, mock_redis_class):
        """Setting watchlist updates internal symbols list (mocked Redis)."""
        mgr = SessionStreamingManager()
        
        # Setup AsyncMock for Redis instance methods
        mock_redis_instance = AsyncMock()
        mock_redis_class.from_url.return_value = mock_redis_instance

        async def _test():
            await mgr.set_watchlist(["HPG", "VIC", "VNM"])
            assert mgr._watchlist_symbols == ["HPG", "VIC", "VNM"]
            mock_redis_instance.sadd.assert_called_once()
            
        asyncio.run(_test())

    def test_empty_watchlist_prevents_open(self):
        """Cannot open session without watchlist symbols."""
        mgr = SessionStreamingManager()

        async def _test():
            await mgr.open_session()
            assert mgr.state == ConnectionState.DISCONNECTED

        asyncio.run(_test())

    def test_last_message_age_initial(self):
        """Initial message age should be infinity."""
        mgr = SessionStreamingManager()
        assert mgr.last_message_age_seconds == float("inf")

    def test_gap_info_initial(self):
        """Initial gap info should have None timestamps."""
        mgr = SessionStreamingManager()
        gap = mgr.get_gap_info()
        assert gap["disconnect_time"] is None
        assert gap["reconnect_time"] is None
        assert gap["state"] == "disconnected"

    def test_callbacks_registered(self):
        """Callbacks should be stored correctly."""
        on_trade = AsyncMock()
        on_quote = AsyncMock()
        mgr = SessionStreamingManager(on_trade=on_trade, on_quote=on_quote)
        assert mgr._on_trade is on_trade
        assert mgr._on_quote is on_quote


class TestConnectionState:
    """Test connection state enum."""

    def test_states_exist(self):
        assert ConnectionState.DISCONNECTED == "disconnected"
        assert ConnectionState.CONNECTING == "connecting"
        assert ConnectionState.CONNECTED == "connected"
        assert ConnectionState.RECONNECTING == "reconnecting"
        assert ConnectionState.CLOSING == "closing"


class TestSchedulerIntegration:
    """Test scheduler integration."""

    def test_scheduler_creation(self):
        """Scheduler should be creatable with stream manager."""
        from src.api.streaming.scheduler import TradingSessionScheduler

        mgr = SessionStreamingManager()
        scheduler = TradingSessionScheduler(mgr)
        assert not scheduler.is_running

    def test_scheduler_stops_cleanly(self):
        """Scheduler should stop without errors."""
        from src.api.streaming.scheduler import TradingSessionScheduler

        mgr = SessionStreamingManager()
        scheduler = TradingSessionScheduler(mgr)
        scheduler.stop()  # Should not raise even if not started


class TestFallbackFiller:
    """Test REST API fallback gap-filling."""

    def test_initial_no_gaps(self):
        """Should start with no pending gaps."""
        from src.api.streaming.fallback import FallbackFiller

        filler = FallbackFiller()
        assert len(filler.pending_gaps) == 0

    def test_on_disconnect_records_gap(self):
        """Disconnect should record gap timestamp."""
        from src.api.streaming.fallback import FallbackFiller

        filler = FallbackFiller()

        async def _test():
            await filler.on_disconnect(time.time())
            assert len(filler.pending_gaps) == 1
            assert filler.pending_gaps[0]["reconnect_time"] is None

        asyncio.run(_test())

    def test_clear_gaps(self):
        """Clear should remove all pending gaps."""
        from src.api.streaming.fallback import FallbackFiller

        filler = FallbackFiller()

        async def _test():
            await filler.on_disconnect(time.time())
            filler.clear_gaps()
            assert len(filler.pending_gaps) == 0

        asyncio.run(_test())
