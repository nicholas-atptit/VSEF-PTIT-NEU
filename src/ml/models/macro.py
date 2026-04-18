"""Macro economic indicator models.

Stores time-series data for market indices, FX rates, commodities,
and interbank rates used as context for LLM risk assessment.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from sqlalchemy import DateTime, Index, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.ml.models.base import Base


class MacroIndicator(Base):
    """Time-series macro economic indicator.

    Stores: VN-Index, VN30, USD/VND, interbank rates, gold price, oil price.
    Designed as a TimescaleDB hypertable partitioned by timestamp.
    """

    __tablename__ = "macro_indicators"

    timestamp: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        primary_key=True,
        nullable=False,
    )
    indicator_name: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
        nullable=False,
        comment="Indicator: vnindex, vn30, usd_vnd, interbank_rate, gold, oil",
    )

    value: Mapped[Decimal] = mapped_column(
        Numeric(18, 4),
        nullable=False,
        comment="Indicator value",
    )

    # Optional OHLCV for index-type indicators
    open: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    high: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    low: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    close: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    volume: Mapped[int | None] = mapped_column(nullable=True)

    # Metadata
    source: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="api",
        comment="Data source provenance, with vnstock_data as the canonical provider for supported series",
    )
    unit: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        comment="Unit: VND, USD, %, points",
    )

    __table_args__ = (
        Index("ix_macro_indicator_name_ts", "indicator_name", "timestamp"),
        {
            "comment": "Macro economic indicators (TimescaleDB hypertable)",
        },
    )

    def __repr__(self) -> str:
        return f"<MacroIndicator({self.indicator_name}={self.value} @ {self.timestamp})>"


# Predefined indicator names for consistency
INDICATOR_VNINDEX = "vnindex"
INDICATOR_VN30 = "vn30"
INDICATOR_USD_VND = "usd_vnd"
INDICATOR_INTERBANK_RATE = "interbank_rate"
INDICATOR_GOLD = "gold"
INDICATOR_OIL = "oil"

ALL_INDICATORS = [
    INDICATOR_VNINDEX,
    INDICATOR_VN30,
    INDICATOR_USD_VND,
    INDICATOR_INTERBANK_RATE,
    INDICATOR_GOLD,
    INDICATOR_OIL,
]
