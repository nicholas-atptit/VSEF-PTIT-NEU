"""Price models — RawPrice and AdjustedPrice hypertables for TimescaleDB.

These tables store tick-level and M1 candle data.
TimescaleDB hypertables are partitioned by timestamp for efficient time-range queries.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base


class RawPrice(Base):
    """Raw price data — actual matching prices on trading day.

    Schema: [timestamp, ticker, exchange, open, high, low, close, volume]
    This table will be converted to a TimescaleDB hypertable partitioned by timestamp.
    """

    __tablename__ = "raw_prices"

    # Primary key: composite of timestamp + ticker for hypertable
    timestamp: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        primary_key=True,
        nullable=False,
        comment="Candle/tick timestamp in VN timezone",
    )
    ticker: Mapped[str] = mapped_column(
        String(20),
        primary_key=True,
        nullable=False,
        comment="Stock symbol (e.g., HPG, VIC, VNM)",
    )

    # Market metadata
    exchange: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="HOSE",
        comment="Exchange: HOSE, HNX, UPCOM",
    )
    timeframe: Mapped[str] = mapped_column(
        String(5),
        nullable=False,
        default="1m",
        comment="Candle timeframe: tick, 1m, 5m, 15m, 1h, 1d",
    )

    # OHLCV data
    open: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
        comment="Opening price",
    )
    high: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
        comment="Highest price",
    )
    low: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
        comment="Lowest price",
    )
    close: Mapped[Decimal] = mapped_column(
        Numeric(12, 2),
        nullable=False,
        comment="Closing price",
    )
    volume: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        comment="Trading volume",
    )

    # Source tracking
    source: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="websocket",
        comment="Data source: websocket, rest_api, backdate",
    )

    __table_args__ = (
        Index("ix_raw_prices_ticker_ts", "ticker", "timestamp"),
        Index("ix_raw_prices_exchange", "exchange"),
        {
            "comment": "Raw price data — actual matching prices (TimescaleDB hypertable)",
        },
    )

    def __repr__(self) -> str:
        return (
            f"<RawPrice(ticker={self.ticker}, ts={self.timestamp}, "
            f"O={self.open} H={self.high} L={self.low} C={self.close} V={self.volume})>"
        )


class AdjustedPrice(Base):
    """Adjusted price data — prices adjusted for corporate actions.

    Adjusted for: dividends, stock splits, rights issues, bonus shares.
    Used for Machine Learning training to avoid discontinuities.
    """

    __tablename__ = "adjusted_prices"

    timestamp: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        primary_key=True,
        nullable=False,
        comment="Candle timestamp in VN timezone",
    )
    ticker: Mapped[str] = mapped_column(
        String(20),
        primary_key=True,
        nullable=False,
        comment="Stock symbol",
    )

    exchange: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="HOSE",
    )
    timeframe: Mapped[str] = mapped_column(
        String(5),
        nullable=False,
        default="1m",
    )

    # Adjusted OHLCV
    open: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    # Adjustment metadata
    adjustment_factor: Mapped[Decimal] = mapped_column(
        Numeric(12, 8),
        nullable=False,
        default=1.0,
        comment="Cumulative adjustment factor applied to raw prices",
    )
    adjustment_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Description of corporate action (e.g., 'cash dividend 1500 VND')",
    )

    source: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="computed",
    )

    __table_args__ = (
        Index("ix_adj_prices_ticker_ts", "ticker", "timestamp"),
        Index("ix_adj_prices_exchange", "exchange"),
        {
            "comment": "Adjusted price data for ML — corrected for corporate actions",
        },
    )

    def __repr__(self) -> str:
        return (
            f"<AdjustedPrice(ticker={self.ticker}, ts={self.timestamp}, "
            f"adj_factor={self.adjustment_factor}, C={self.close})>"
        )


class CorporateAction(Base):
    """Corporate action events used for price adjustment calculations."""

    __tablename__ = "corporate_actions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    event_date: Mapped[dt.date] = mapped_column(nullable=False)
    action_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Type: cash_dividend, stock_dividend, stock_split, rights_issue, bonus_share",
    )
    factor: Mapped[Decimal] = mapped_column(
        Numeric(12, 8),
        nullable=False,
        comment="Adjustment factor for this event",
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_corp_actions_ticker_date", "ticker", "event_date"),
    )
