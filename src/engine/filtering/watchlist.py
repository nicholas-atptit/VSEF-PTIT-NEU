"""Watchlist computing — real-time technical indicator calculation.

Only allocates buffers and computes indicators for symbols in the active watchlist.
Uses pandas-ta for RSI, MACD, SMA calculations.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from config.settings import get_settings
from src.data.database.connection import get_session
from src.ml.models.watchlist import WatchlistItem
from src.utils.logging import get_logger

logger = get_logger(__name__)


class IndicatorBuffer:
    """Rolling price buffer for a single ticker with indicator computation."""

    def __init__(self, ticker: str, buffer_size: int = 200) -> None:
        self.ticker = ticker
        self.buffer_size = buffer_size
        self._closes: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._volumes: list[int] = []
        self._timestamps: list[dt.datetime] = []

    def add_candle(
        self,
        timestamp: dt.datetime,
        close: float,
        high: float | None = None,
        low: float | None = None,
        volume: int = 0,
    ) -> None:
        """Add a new candle to the buffer."""
        self._timestamps.append(timestamp)
        self._closes.append(close)
        self._highs.append(high or close)
        self._lows.append(low or close)
        self._volumes.append(volume)

        # Trim to buffer size
        if len(self._closes) > self.buffer_size:
            self._timestamps = self._timestamps[-self.buffer_size:]
            self._closes = self._closes[-self.buffer_size:]
            self._highs = self._highs[-self.buffer_size:]
            self._lows = self._lows[-self.buffer_size:]
            self._volumes = self._volumes[-self.buffer_size:]

    def compute_indicators(self, settings: Any = None) -> dict[str, float | None]:
        """Compute technical indicators from the current buffer.

        Returns dict with keys: rsi, macd, macd_signal, macd_hist,
        sma_20, sma_50, sma_200
        """
        if len(self._closes) < 2:
            return self._empty_indicators()

        s = settings or get_settings()
        try:
            import pandas_ta as ta

            df = pd.DataFrame({
                "close": self._closes,
                "high": self._highs,
                "low": self._lows,
                "volume": self._volumes,
            })

            result: dict[str, float | None] = {}

            # RSI
            rsi_series = ta.rsi(df["close"], length=s.rsi_period)
            result["rsi"] = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.empty and not pd.isna(rsi_series.iloc[-1]) else None

            # MACD
            macd_df = ta.macd(df["close"], fast=s.macd_fast, slow=s.macd_slow, signal=s.macd_signal)
            if macd_df is not None and not macd_df.empty:
                cols = macd_df.columns.tolist()
                result["macd"] = float(macd_df[cols[0]].iloc[-1]) if not pd.isna(macd_df[cols[0]].iloc[-1]) else None
                result["macd_signal"] = float(macd_df[cols[1]].iloc[-1]) if len(cols) > 1 and not pd.isna(macd_df[cols[1]].iloc[-1]) else None
                result["macd_hist"] = float(macd_df[cols[2]].iloc[-1]) if len(cols) > 2 and not pd.isna(macd_df[cols[2]].iloc[-1]) else None
            else:
                result["macd"] = None
                result["macd_signal"] = None
                result["macd_hist"] = None

            # SMAs
            for period in s.sma_periods:
                sma = ta.sma(df["close"], length=period)
                result[f"sma_{period}"] = float(sma.iloc[-1]) if sma is not None and not sma.empty and not pd.isna(sma.iloc[-1]) else None

            return result

        except ImportError:
            logger.warning("pandas_ta_not_installed", msg="Falling back to manual RSI")
            return self._compute_basic_rsi(s.rsi_period)

    def _compute_basic_rsi(self, period: int = 14) -> dict[str, float | None]:
        """Fallback RSI calculation without pandas-ta."""
        if len(self._closes) < period + 1:
            return self._empty_indicators()

        deltas = np.diff(self._closes[-period - 1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        result = self._empty_indicators()
        result["rsi"] = float(rsi)
        return result

    @staticmethod
    def _empty_indicators() -> dict[str, float | None]:
        """Return empty indicator dict."""
        return {
            "rsi": None,
            "macd": None,
            "macd_signal": None,
            "macd_hist": None,
            "sma_20": None,
            "sma_50": None,
            "sma_200": None,
        }

    @property
    def candle_count(self) -> int:
        return len(self._closes)


class WatchlistComputer:
    """Manages indicator computation for watchlist symbols only.

    Allocates IndicatorBuffers only for symbols in the active watchlist.
    Non-watchlist symbols are passed through without computation.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._watchlist: set[str] = set()
        self._buffers: dict[str, IndicatorBuffer] = {}

    @property
    def watchlist(self) -> set[str]:
        return self._watchlist.copy()

    async def load_watchlist(self) -> None:
        """Load watchlist from database."""
        try:
            from sqlalchemy import select

            async with get_session() as session:
                result = await session.execute(
                    select(WatchlistItem.ticker)
                )
                tickers = {row[0] for row in result.all()}

            # Allocate buffers for new tickers
            for ticker in tickers:
                if ticker not in self._buffers:
                    self._buffers[ticker] = IndicatorBuffer(
                        ticker, self._settings.indicator_buffer_size
                    )

            # Remove buffers for removed tickers
            removed = set(self._buffers.keys()) - tickers
            for ticker in removed:
                del self._buffers[ticker]

            self._watchlist = tickers
            logger.info("watchlist_loaded", count=len(tickers), tickers=list(tickers))

        except Exception as e:
            logger.error("watchlist_load_failed", error=str(e))

    def is_in_watchlist(self, ticker: str) -> bool:
        """Check if a ticker is in the watchlist."""
        return ticker.upper() in self._watchlist

    def process_candle(self, standardized: dict) -> dict:
        """Process a standardized candle and compute indicators if watched.

        Args:
            standardized: Dict with keys: timestamp, ticker, open, high, low, close, volume

        Returns:
            Original dict enriched with 'indicators' key if in watchlist.
        """
        ticker = standardized.get("ticker", "").upper()

        if ticker not in self._watchlist:
            standardized["indicators"] = None
            return standardized

        buffer = self._buffers.get(ticker)
        if buffer is None:
            buffer = IndicatorBuffer(ticker, self._settings.indicator_buffer_size)
            self._buffers[ticker] = buffer

        buffer.add_candle(
            timestamp=standardized["timestamp"],
            close=float(standardized["close"]),
            high=float(standardized.get("high", standardized["close"])),
            low=float(standardized.get("low", standardized["close"])),
            volume=int(standardized.get("volume", 0)),
        )

        standardized["indicators"] = buffer.compute_indicators()
        return standardized

    def get_symbols(self) -> list[str]:
        """Return list of watchlist symbols (for WebSocket subscriptions)."""
        return list(self._watchlist)
