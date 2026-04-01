"""Data standardization — maps DNSE stream messages to canonical schema.

Canonical schema: [timestamp, ticker, exchange, open, high, low, close, volume]
All timestamps are normalized to Asia/Ho_Chi_Minh timezone.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

from src.utils.logging import get_logger
from src.utils.time_utils import VN_TZ

logger = get_logger(__name__)


# DNSE exchange mapping
EXCHANGE_MAP = {
    "HOSE": "HOSE",
    "HNX": "HNX",
    "UPCOM": "UPCOM",
    "HSX": "HOSE",   # Alternative name
}


class DataStandardizer:
    """Maps DNSE WebSocket messages to the canonical price schema.

    Input: DnseMarketStream messages (StreamTrade, StreamOhlc, StreamQuote)
    Output: Standardized dict with keys:
        timestamp, ticker, exchange, open, high, low, close, volume,
        timeframe, source, message_type
    """

    def standardize_trade(self, msg: Any) -> dict | None:
        """Standardize a StreamTrade message.

        Maps: symbol → ticker, price → close (OHLC all = price for tick data)
        """
        try:
            symbol = getattr(msg, "symbol", None)
            if not symbol:
                return None

            price = getattr(msg, "price", 0)
            volume = getattr(msg, "volume", 0)
            timestamp = getattr(msg, "timestamp", None)

            if timestamp is None:
                timestamp = dt.datetime.now(VN_TZ)
            elif isinstance(timestamp, (int, float)):
                timestamp = dt.datetime.fromtimestamp(timestamp, tz=VN_TZ)
            elif timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=VN_TZ)

            return {
                "timestamp": timestamp,
                "ticker": symbol.upper(),
                "exchange": self._detect_exchange(msg),
                "open": Decimal(str(price)),
                "high": Decimal(str(price)),
                "low": Decimal(str(price)),
                "close": Decimal(str(price)),
                "volume": int(volume),
                "timeframe": "tick",
                "source": "websocket",
                "message_type": "trade",
            }

        except Exception as e:
            logger.error("standardize_trade_error", error=str(e))
            return None

    def standardize_ohlc(self, msg: Any) -> dict | None:
        """Standardize a StreamOhlc (candle) message.

        Maps: open/high/low/close directly.
        """
        try:
            symbol = getattr(msg, "symbol", None)
            if not symbol:
                return None

            timestamp = getattr(msg, "timestamp", None)
            if timestamp is None:
                timestamp = dt.datetime.now(VN_TZ)
            elif isinstance(timestamp, (int, float)):
                timestamp = dt.datetime.fromtimestamp(timestamp, tz=VN_TZ)
            elif timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=VN_TZ)

            return {
                "timestamp": timestamp,
                "ticker": symbol.upper(),
                "exchange": self._detect_exchange(msg),
                "open": Decimal(str(getattr(msg, "open", 0))),
                "high": Decimal(str(getattr(msg, "high", 0))),
                "low": Decimal(str(getattr(msg, "low", 0))),
                "close": Decimal(str(getattr(msg, "close", 0))),
                "volume": int(getattr(msg, "volume", 0)),
                "timeframe": "1m",
                "source": "websocket",
                "message_type": "ohlc",
            }

        except Exception as e:
            logger.error("standardize_ohlc_error", error=str(e))
            return None

    def standardize_quote(self, msg: Any) -> dict | None:
        """Standardize a StreamQuote message (bid/ask data).

        Note: Quotes don't have OHLC — stored as reference data only.
        """
        try:
            symbol = getattr(msg, "symbol", None)
            if not symbol:
                return None

            timestamp = dt.datetime.now(VN_TZ)
            bid = getattr(msg, "bid_price", 0)
            ask = getattr(msg, "ask_price", 0)
            mid = (float(bid) + float(ask)) / 2 if bid and ask else 0

            return {
                "timestamp": timestamp,
                "ticker": symbol.upper(),
                "exchange": self._detect_exchange(msg),
                "bid_price": Decimal(str(bid)) if bid else None,
                "ask_price": Decimal(str(ask)) if ask else None,
                "bid_volume": int(getattr(msg, "bid_volume", 0)),
                "ask_volume": int(getattr(msg, "ask_volume", 0)),
                "mid_price": Decimal(str(mid)) if mid else None,
                "message_type": "quote",
                "source": "websocket",
            }

        except Exception as e:
            logger.error("standardize_quote_error", error=str(e))
            return None

    def standardize_dict(self, data: dict) -> dict:
        """Standardize a raw dict (e.g., from REST API or backdate scripts).

        Ensures all required fields are present with correct types.
        """
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = dt.datetime.fromisoformat(timestamp)
        if timestamp and timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=VN_TZ)

        return {
            "timestamp": timestamp or dt.datetime.now(VN_TZ),
            "ticker": str(data.get("ticker", "")).upper(),
            "exchange": EXCHANGE_MAP.get(
                str(data.get("exchange", "HOSE")).upper(), "HOSE"
            ),
            "open": Decimal(str(data.get("open", 0))),
            "high": Decimal(str(data.get("high", 0))),
            "low": Decimal(str(data.get("low", 0))),
            "close": Decimal(str(data.get("close", 0))),
            "volume": int(data.get("volume", 0)),
            "timeframe": data.get("timeframe", "1d"),
            "source": data.get("source", "rest_api"),
            "message_type": data.get("message_type", "candle"),
        }

    @staticmethod
    def _detect_exchange(msg: Any) -> str:
        """Try to detect exchange from message attributes."""
        # Check various possible attribute names
        for attr in ("exchange", "market", "board", "board_id"):
            val = getattr(msg, attr, None)
            if val:
                val_str = str(val).upper()
                return EXCHANGE_MAP.get(val_str, val_str)
        return "HOSE"  # Default to HOSE
