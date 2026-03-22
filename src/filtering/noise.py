"""Noise reduction — filter put-through (thỏa thuận) trades.

Only keeps order-matching (khớp lệnh liên tục) trades.
Drops put-through/negotiated/block trades from the pipeline.
"""

from __future__ import annotations

from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

# DNSE BoardId values that represent put-through / negotiated trades
# These are filtered out to keep only continuous order-matching data
PUT_THROUGH_BOARD_IDS = {
    "PT",          # Put-through
    "PUT_THROUGH",
    "NEGOTIATED",
    "BLOCK",
    "ODD_LOT",    # Odd lot trades (lô lẻ)
}

# Trade type flags that indicate put-through trades
PUT_THROUGH_TRADE_TYPES = {
    "PT",
    "TT",   # Thỏa thuận
    "DEAL",
    "BLOCK_TRADE",
}


class NoiseFilter:
    """Filters out put-through (negotiated) trades from the data pipeline.

    Only allows continuous order-matching trades through:
    - Board: ROUND_LOT / normal order-matching
    - Trade type: regular matching, not negotiated/block

    This ensures technical analysis is based on real market liquidity,
    not pre-arranged deals that distort volume/price patterns.
    """

    def __init__(self) -> None:
        self._filtered_count = 0
        self._passed_count = 0

    def is_order_matching(self, message: dict | Any) -> bool:
        """Check if a trade is from order-matching (not put-through).

        Args:
            message: Either a dict with relevant keys or a DNSE stream message.

        Returns:
            True if this is an order-matching trade (should be KEPT).
            False if this is a put-through/negotiated trade (should be DROPPED).
        """
        # Handle dict messages
        if isinstance(message, dict):
            return self._check_dict(message)

        # Handle DNSE stream message objects
        return self._check_stream_msg(message)

    def filter(self, message: dict | Any) -> bool:
        """Filter a message: return True if it should be KEPT.

        Args:
            message: Trade message to check.

        Returns:
            True to keep (order-matching), False to drop (put-through).
        """
        if self.is_order_matching(message):
            self._passed_count += 1
            return True
        else:
            self._filtered_count += 1
            ticker = (
                message.get("ticker", "?")
                if isinstance(message, dict)
                else getattr(message, "symbol", "?")
            )
            logger.debug("noise_filtered", ticker=ticker, reason="put_through_trade")
            return False

    def _check_dict(self, msg: dict) -> bool:
        """Check dict-based message for put-through indicators."""
        # Check board_id
        board_id = str(msg.get("board_id", "")).upper()
        if board_id in PUT_THROUGH_BOARD_IDS:
            return False

        # Check trade_type
        trade_type = str(msg.get("trade_type", "")).upper()
        if trade_type in PUT_THROUGH_TRADE_TYPES:
            return False

        # Check is_put_through flag
        if msg.get("is_put_through", False):
            return False

        return True

    def _check_stream_msg(self, msg: Any) -> bool:
        """Check DNSE stream message object for put-through indicators."""
        # Check board_id attribute
        board_id = getattr(msg, "board_id", None)
        if board_id is not None:
            board_str = str(board_id).upper()
            if board_str in PUT_THROUGH_BOARD_IDS:
                return False

        # Check trade_type attribute
        trade_type = getattr(msg, "trade_type", None)
        if trade_type is not None:
            type_str = str(trade_type).upper()
            if type_str in PUT_THROUGH_TRADE_TYPES:
                return False

        # Check side / match type indicators
        match_type = getattr(msg, "match_type", None)
        if match_type is not None:
            if str(match_type).upper() in ("PT", "TT", "DEAL"):
                return False

        return True

    @property
    def stats(self) -> dict[str, int]:
        """Return filtering statistics."""
        total = self._passed_count + self._filtered_count
        return {
            "total_processed": total,
            "passed": self._passed_count,
            "filtered": self._filtered_count,
            "filter_rate_pct": (
                round(self._filtered_count / total * 100, 2) if total > 0 else 0
            ),
        }

    def reset_stats(self) -> None:
        """Reset filtering statistics."""
        self._filtered_count = 0
        self._passed_count = 0
