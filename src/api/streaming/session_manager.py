"""Module 1: Session Streaming Manager.

Manages WebSocket lifecycle for real-time stock data from DNSE API.
Handles connection state, auto-ping, reconnection, and message forwarding.
"""

from __future__ import annotations

import asyncio
import time
import json
import asyncio
import time
from enum import Enum
from typing import Any, Callable, Coroutine

import msgpack
from redis.asyncio import Redis

from config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ConnectionState(str, Enum):
    """WebSocket connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"


class SessionStreamingManager:
    """Manages DNSE WebSocket streaming lifecycle.

    Responsibilities:
    - Open/close WebSocket connections on schedule
    - Subscribe to trades, quotes, OHLC for watchlist symbols
    - Auto-ping to keep connection alive
    - Detection of disconnection and triggering fallback
    - Forward received messages to the callback (FilterEngine)
    """

    def __init__(
        self,
        on_trade: Callable[..., Coroutine] | None = None,
        on_quote: Callable[..., Coroutine] | None = None,
        on_ohlc: Callable[..., Coroutine] | None = None,
        on_disconnect: Callable[..., Coroutine] | None = None,
        on_reconnect: Callable[..., Coroutine] | None = None,
    ) -> None:
        self._settings = get_settings()
        self._state = ConnectionState.DISCONNECTED
        self._stream = None
        self._ping_task: asyncio.Task | None = None
        self._reconnect_attempts = 0
        self._last_message_time: float = 0
        self._watchlist_symbols: list[str] = []

        # Disconnect tracking for gap-filling
        self._disconnect_time: float | None = None
        self._disconnect_time: float | None = None
        self._reconnect_time: float | None = None

        # Redis Stream configuration
        self._redis: Redis | None = None
        self._stream_name = "market_stream_raw"
        self._watchlist_key = "system:watchlist"

        # Legacy Callbacks (Left for backward compatibility or debugging, but main flow is Redis)
        self._on_trade = on_trade
        self._on_quote = on_quote
        self._on_ohlc = on_ohlc
        self._on_disconnect = on_disconnect
        self._on_reconnect = on_reconnect

    @property
    def state(self) -> ConnectionState:
        """Current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._state == ConnectionState.CONNECTED

    @property
    def last_message_age_seconds(self) -> float:
        """Seconds since last received message."""
        if self._last_message_time == 0:
            return float("inf")
        return time.time() - self._last_message_time

    async def set_watchlist(self, symbols: list[str]) -> None:
        """Update the watchlist symbols in Redis (O(1) lookups)."""
        if not self._redis:
            self._redis = Redis.from_url(self._settings.redis_url)
        
        # Clear existing set and rebuild
        await self._redis.delete(self._watchlist_key)
        if symbols:
            await self._redis.sadd(self._watchlist_key, *symbols)
            
        self._watchlist_symbols = symbols
        logger.info("watchlist_updated_in_redis", symbols=symbols, count=len(symbols))

    async def open_session(self) -> None:
        """Open WebSocket connection and subscribe to watchlist symbols."""
        if self._state != ConnectionState.DISCONNECTED:
            logger.warning("session_already_active", state=self._state)
            return

        if not self._watchlist_symbols:
            logger.warning("no_watchlist_symbols", msg="Cannot open session without watchlist")
            return

        self._state = ConnectionState.CONNECTING
        logger.info("session_opening", symbols_count=len(self._watchlist_symbols))

        try:
            # 1. Initialize Redis connection for fast streams
            if not self._redis:
                self._redis = Redis.from_url(self._settings.redis_url)

            # 2. Start the DNSE WebSocket Client
            from dnse import DnseMarketStream

            self._stream = DnseMarketStream(
                api_key=self._settings.dnse_api_key,
                api_secret=self._settings.dnse_api_secret,
            )

            # Subscribe to data channels
            self._stream.subscribe_trades(
                self._watchlist_symbols,
                self._handle_trade,
            )
            self._stream.subscribe_quotes(
                self._watchlist_symbols,
                self._handle_quote,
            )
            self._stream.subscribe_ohlc(
                self._watchlist_symbols,
                self._handle_ohlc,
                timeframe="1m",
            )

            # Start the stream in a background thread (stream.run() is blocking)
            asyncio.get_event_loop().run_in_executor(None, self._stream.run)

            self._state = ConnectionState.CONNECTED
            self._reconnect_attempts = 0
            self._last_message_time = time.time()

            # Start auto-ping task
            self._ping_task = asyncio.create_task(self._auto_ping_loop())

            logger.info(
                "session_opened",
                symbols=self._watchlist_symbols,
                state=self._state,
            )

        except Exception as e:
            logger.error("session_open_failed", error=str(e))
            self._state = ConnectionState.DISCONNECTED
            if self._redis:
                await self._redis.close()
                self._redis = None
            await self._attempt_reconnect()

    async def close_session(self) -> None:
        """Gracefully close WebSocket connection."""
        if self._state == ConnectionState.DISCONNECTED:
            return

        self._state = ConnectionState.CLOSING
        logger.info("session_closing")

        # Cancel ping task
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass

        # Close the stream (implementation depends on dnse SDK)
        self._stream = None
        
        # Close Redis connection
        if self._redis:
            await self._redis.close()
            self._redis = None
            
        self._state = ConnectionState.DISCONNECTED

        logger.info("session_closed")

    async def _push_to_redis(self, stream_type: str, msg: Any) -> None:
        """Serialize payload via MsgPack to cut memory bandwidth by 50% & push to Stream."""
        if not self._redis:
            return
            
        # Convert object dict/properties to dictionary
        payload_dict = msg if isinstance(msg, dict) else msg.__dict__
        
        # Add metadata
        payload_dict["_type"] = stream_type
        payload_dict["_recv_ts"] = time.time()
        
        # Serialize with msgpack
        try:
            packed_data = msgpack.packb(payload_dict)
            await self._redis.xadd(self._stream_name, {"payload": packed_data})
            
            # PHASE 23 FAST-PATH: Update O(1) price cache for TUI
            if stream_type in ["trade", "ohlc"]:
                ticker = payload_dict.get("symbol") or payload_dict.get("ticker")
                if ticker:
                    price = payload_dict.get("price") or payload_dict.get("close")
                    if price:
                        cache_key = f"live_price:{ticker}"
                        await self._redis.set(cache_key, json.dumps({
                            "price": float(price),
                            "ts": payload_dict.get("_recv_ts") or time.time(),
                            "type": stream_type
                        }))
        except Exception as e:
            logger.error("redis_stream_push_error", error=str(e), type=stream_type)

    async def _handle_trade(self, msg: Any) -> None:
        """Handle incoming trade message."""
        self._last_message_time = time.time()
        
        # Send raw data straight to Redis memory log instantly
        await self._push_to_redis("trade", msg)
        
        if self._on_trade:
            try:
                await self._on_trade(msg)
            except Exception as e:
                logger.error("trade_callback_error", error=str(e), symbol=getattr(msg, "symbol", "?"))

    async def _handle_quote(self, msg: Any) -> None:
        """Handle incoming quote message."""
        self._last_message_time = time.time()
        await self._push_to_redis("quote", msg)
        
        if self._on_quote:
            try:
                await self._on_quote(msg)
            except Exception as e:
                logger.error("quote_callback_error", error=str(e))

    async def _handle_ohlc(self, msg: Any) -> None:
        """Handle incoming OHLC candle message."""
        self._last_message_time = time.time()
        await self._push_to_redis("ohlc", msg)
        
        if self._on_ohlc:
            try:
                await self._on_ohlc(msg)
            except Exception as e:
                logger.error("ohlc_callback_error", error=str(e))

    async def _auto_ping_loop(self) -> None:
        """Auto-ping loop to detect stale connections.

        Runs every PING_INTERVAL_SECONDS. If no message received for
        3x ping interval, trigger reconnection.
        """
        ping_interval = self._settings.ping_interval_seconds
        stale_threshold = ping_interval * 3

        while self._state == ConnectionState.CONNECTED:
            await asyncio.sleep(ping_interval)

            if self.last_message_age_seconds > stale_threshold:
                logger.warning(
                    "connection_stale",
                    last_msg_age=self.last_message_age_seconds,
                    threshold=stale_threshold,
                )
                await self._handle_disconnect()
                return

    async def _handle_disconnect(self) -> None:
        """Handle unexpected disconnection."""
        self._disconnect_time = time.time()
        self._state = ConnectionState.RECONNECTING

        logger.warning("connection_lost", reconnect_attempt=self._reconnect_attempts + 1)

        # Notify fallback filler
        if self._on_disconnect:
            await self._on_disconnect(self._disconnect_time)

        await self._attempt_reconnect()

    async def _attempt_reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        max_attempts = self._settings.max_reconnect_attempts
        base_delay = self._settings.reconnect_delay_seconds

        while self._reconnect_attempts < max_attempts:
            self._reconnect_attempts += 1
            delay = min(base_delay * (2 ** (self._reconnect_attempts - 1)), 60)

            logger.info(
                "reconnect_attempt",
                attempt=self._reconnect_attempts,
                max_attempts=max_attempts,
                delay=delay,
            )

            await asyncio.sleep(delay)

            try:
                # Close existing broken stream
                self._stream = None
                self._state = ConnectionState.DISCONNECTED

                # Re-open session
                await self.open_session()

                if self._state == ConnectionState.CONNECTED:
                    self._reconnect_time = time.time()
                    logger.info(
                        "reconnect_success",
                        attempt=self._reconnect_attempts,
                        gap_seconds=(
                            self._reconnect_time - self._disconnect_time
                            if self._disconnect_time
                            else None
                        ),
                    )

                    # Notify reconnect callback (for gap-filling)
                    if self._on_reconnect and self._disconnect_time:
                        await self._on_reconnect(
                            self._disconnect_time,
                            self._reconnect_time,
                        )

                    self._disconnect_time = None
                    self._reconnect_time = None
                    return

            except Exception as e:
                logger.error(
                    "reconnect_failed",
                    attempt=self._reconnect_attempts,
                    error=str(e),
                )

        logger.critical(
            "reconnect_exhausted",
            max_attempts=max_attempts,
            msg="All reconnection attempts failed",
        )

    def get_gap_info(self) -> dict[str, float | None]:
        """Get disconnect/reconnect timestamps for gap-filling."""
        return {
            "disconnect_time": self._disconnect_time,
            "reconnect_time": self._reconnect_time,
            "state": self._state.value,
        }
