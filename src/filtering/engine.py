"""Module 2: Filter Engine — Main pipeline orchestrator.

Pipeline: raw_message → blacklist_check → noise_filter → standardize →
          watchlist_compute → [Fork 1: real-time indicators] AND [Fork 2: DB insert]

Uses asyncio queues for backpressure handling between stages.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine

from src.filtering.blacklist import BlacklistFilter
from src.filtering.noise import NoiseFilter
from src.filtering.standardizer import DataStandardizer
from src.filtering.watchlist import WatchlistComputer
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FilterEngine:
    """Gatekeeper pipeline that processes raw market data before storage.

    Stages:
    1. Blacklist check — drop banned tickers instantly
    2. Noise filter — drop put-through trades
    3. Standardize — map to canonical schema
    4. Watchlist compute — enrich with indicators (only for watched symbols)
    5. Fork — send to both real-time output and DB insertion queues
    """

    def __init__(
        self,
        on_processed: Callable[[dict], Coroutine] | None = None,
        on_indicator: Callable[[dict], Coroutine] | None = None,
        queue_maxsize: int = 10000,
    ) -> None:
        self._blacklist = BlacklistFilter()
        self._noise = NoiseFilter()
        self._standardizer = DataStandardizer()
        self._watchlist = WatchlistComputer()

        # Output callbacks
        self._on_processed = on_processed  # → DB insert
        self._on_indicator = on_indicator  # → real-time indicator output

        # Internal processing queue
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
        self._processing_task: asyncio.Task | None = None

        # Stats
        self._stats = {
            "received": 0,
            "blacklisted": 0,
            "noise_filtered": 0,
            "standardize_failed": 0,
            "processed": 0,
            "errors": 0,
        }

    async def start(self) -> None:
        """Initialize filters and start the processing loop."""
        logger.info("filter_engine_starting")

        # Load blacklist and watchlist from DB
        await self._blacklist.start_auto_refresh()
        await self._watchlist.load_watchlist()

        # Start processing loop
        self._processing_task = asyncio.create_task(self._processing_loop())

        logger.info(
            "filter_engine_started",
            blacklist_count=self._blacklist.count,
            watchlist_count=len(self._watchlist.watchlist),
        )

    async def stop(self) -> None:
        """Stop the filter engine gracefully."""
        await self._blacklist.stop_auto_refresh()

        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        logger.info("filter_engine_stopped", stats=self._stats)

    # ── Input handlers (called by SessionStreamingManager) ───

    async def handle_trade(self, msg: Any) -> None:
        """Handle incoming trade message from WebSocket."""
        self._stats["received"] += 1
        await self._enqueue("trade", msg)

    async def handle_ohlc(self, msg: Any) -> None:
        """Handle incoming OHLC candle message from WebSocket."""
        self._stats["received"] += 1
        await self._enqueue("ohlc", msg)

    async def handle_quote(self, msg: Any) -> None:
        """Handle incoming quote message from WebSocket."""
        self._stats["received"] += 1
        await self._enqueue("quote", msg)

    # ── Pipeline ─────────────────────────────────────────────

    async def _enqueue(self, msg_type: str, msg: Any) -> None:
        """Add message to processing queue with backpressure."""
        try:
            self._queue.put_nowait((msg_type, msg))
        except asyncio.QueueFull:
            logger.warning("queue_full", queue_size=self._queue.qsize())
            # Drop oldest message to make room
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait((msg_type, msg))

    async def _processing_loop(self) -> None:
        """Main processing loop — dequeues and processes messages."""
        while True:
            try:
                msg_type, msg = await self._queue.get()
                await self._process_message(msg_type, msg)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._stats["errors"] += 1
                logger.error("processing_loop_error", error=str(e))

    async def _process_message(self, msg_type: str, msg: Any) -> None:
        """Run a single message through the entire pipeline."""

        # Stage 1: Quick blacklist check on raw message
        ticker = self._extract_ticker(msg)
        if ticker and self._blacklist.is_blacklisted(ticker):
            self._stats["blacklisted"] += 1
            return

        # Stage 2: Noise filter (for trade messages)
        if msg_type == "trade" and not self._noise.filter(msg):
            self._stats["noise_filtered"] += 1
            return

        # Stage 3: Standardize
        standardized = self._standardize(msg_type, msg)
        if standardized is None:
            self._stats["standardize_failed"] += 1
            return

        # Stage 4: Watchlist indicator computation
        enriched = self._watchlist.process_candle(standardized)

        # Stage 5: Fork — output to both channels
        self._stats["processed"] += 1

        # Fork 1: DB insertion
        if self._on_processed:
            try:
                await self._on_processed(enriched)
            except Exception as e:
                logger.error("on_processed_error", error=str(e))

        # Fork 2: Real-time indicator output (only if has indicators)
        if self._on_indicator and enriched.get("indicators"):
            try:
                await self._on_indicator(enriched)
            except Exception as e:
                logger.error("on_indicator_error", error=str(e))

    def _standardize(self, msg_type: str, msg: Any) -> dict | None:
        """Route to appropriate standardizer method."""
        if msg_type == "trade":
            return self._standardizer.standardize_trade(msg)
        elif msg_type == "ohlc":
            return self._standardizer.standardize_ohlc(msg)
        elif msg_type == "quote":
            return self._standardizer.standardize_quote(msg)
        return None

    @staticmethod
    def _extract_ticker(msg: Any) -> str | None:
        """Extract ticker symbol from any message type."""
        if isinstance(msg, dict):
            return msg.get("ticker") or msg.get("symbol")
        return getattr(msg, "symbol", None) or getattr(msg, "ticker", None)

    # ── Public API ───────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Return pipeline statistics."""
        return {
            **self._stats,
            "queue_size": self._queue.qsize(),
            "noise_stats": self._noise.stats,
        }

    def get_watchlist_symbols(self) -> list[str]:
        """Return current watchlist symbols."""
        return self._watchlist.get_symbols()

    async def refresh_lists(self) -> None:
        """Force refresh of blacklist and watchlist."""
        await self._blacklist.load()
        await self._watchlist.load_watchlist()
        logger.info("lists_refreshed")
