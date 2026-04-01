"""Watchlist and Blacklist models.

Watchlist: Active symbols to track and compute indicators for.
Blacklist: Banned symbols to drop immediately from the pipeline.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from src.ml.models.base import Base, TimestampMixin


class WatchlistItem(Base, TimestampMixin):
    """A stock symbol on the active watchlist.

    Only symbols in the watchlist get real-time indicator computation
    and memory allocation in the Filter Engine.
    """

    __tablename__ = "watchlist"

    ticker: Mapped[str] = mapped_column(
        String(20),
        primary_key=True,
        comment="Stock symbol (e.g., HPG)",
    )
    exchange: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="HOSE",
    )
    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Reason for adding to watchlist",
    )
    added_by: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="manual",
        comment="Who/what added this: manual, scanner, signal",
    )

    def __repr__(self) -> str:
        return f"<WatchlistItem(ticker={self.ticker}, exchange={self.exchange})>"


class BlacklistItem(Base, TimestampMixin):
    """A stock symbol on the blacklist — all data is dropped immediately.

    Reasons: market manipulation, fake liquidity, regulatory warnings, etc.
    """

    __tablename__ = "blacklist"

    ticker: Mapped[str] = mapped_column(
        String(20),
        primary_key=True,
        comment="Banned stock symbol",
    )
    reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Reason for blacklisting",
    )
    source: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="manual",
        comment="Source: manual, regulatory, auto_detect",
    )
    banned_until: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Optional: auto-remove from blacklist after this date",
    )

    @property
    def is_active(self) -> bool:
        """Check if this blacklist entry is still active."""
        if self.banned_until is None:
            return True
        return dt.datetime.now(dt.timezone.utc) < self.banned_until

    def __repr__(self) -> str:
        return f"<BlacklistItem(ticker={self.ticker}, reason={self.reason[:30]})>"
