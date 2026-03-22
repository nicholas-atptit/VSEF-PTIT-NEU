"""Signal event models — Black Swan and market explosion event tagging.

Maps significant market events to dates and affected tickers for
contextual analysis by LLM risk assessment.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import Date, Index, String, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin


class SignalEvent(Base, TimestampMixin):
    """Significant market event (Black Swan, boom, regulatory shock, etc.).

    Examples:
        - 2020-03: COVID-19 pandemic panic sell-off
        - 2022-04: FLC/Tan Hoang Minh executive arrests
        - 2023-10: Van Thinh Phat real estate scandal
    """

    __tablename__ = "signal_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    event_date: Mapped[dt.date] = mapped_column(
        Date,
        nullable=False,
        comment="Date the event occurred or was announced",
    )
    event_end_date: Mapped[dt.date | None] = mapped_column(
        Date,
        nullable=True,
        comment="Optional end date for prolonged events",
    )

    event_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Type: black_swan, boom, regulatory, macro_shock, sector_rotation",
    )
    severity: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="medium",
        comment="Severity: low, medium, high, critical",
    )

    title: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        comment="Short event title",
    )
    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Detailed event description",
    )

    affected_tickers: Mapped[list[str] | None] = mapped_column(
        ARRAY(String(20)),
        nullable=True,
        comment="List of directly affected stock symbols",
    )
    affected_sectors: Mapped[list[str] | None] = mapped_column(
        ARRAY(String(50)),
        nullable=True,
        comment="List of affected sectors",
    )

    market_impact_pct: Mapped[float | None] = mapped_column(
        nullable=True,
        comment="Estimated VN-Index impact in percentage",
    )

    source: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
        comment="Source URL or reference",
    )

    __table_args__ = (
        Index("ix_signal_event_date", "event_date"),
        Index("ix_signal_event_type", "event_type"),
    )

    def __repr__(self) -> str:
        return f"<SignalEvent({self.event_type}: {self.title} @ {self.event_date})>"


# Predefined event types
EVENT_BLACK_SWAN = "black_swan"
EVENT_BOOM = "boom"
EVENT_REGULATORY = "regulatory"
EVENT_MACRO_SHOCK = "macro_shock"
EVENT_SECTOR_ROTATION = "sector_rotation"

# Severity levels
SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"
SEVERITY_CRITICAL = "critical"
