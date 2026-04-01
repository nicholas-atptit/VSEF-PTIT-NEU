"""Signal event tagging — Black Swan and market event management.

CRUD operations for significant market events used as context
for LLM risk assessment.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import select, and_

from src.data.database.connection import get_session
from src.ml.models.signal import (
    EVENT_BLACK_SWAN,
    EVENT_BOOM,
    EVENT_MACRO_SHOCK,
    EVENT_REGULATORY,
    SignalEvent,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SignalTagger:
    """Manages signal events (Black Swan, boom, regulatory shocks, etc.)."""

    async def add_event(
        self,
        event_date: dt.date,
        event_type: str,
        title: str,
        description: str,
        severity: str = "medium",
        affected_tickers: list[str] | None = None,
        affected_sectors: list[str] | None = None,
        market_impact_pct: float | None = None,
        event_end_date: dt.date | None = None,
        source: str | None = None,
    ) -> int:
        """Add a new signal event.

        Returns:
            ID of the created event.
        """
        async with get_session() as session:
            event = SignalEvent(
                event_date=event_date,
                event_end_date=event_end_date,
                event_type=event_type,
                severity=severity,
                title=title,
                description=description,
                affected_tickers=affected_tickers,
                affected_sectors=affected_sectors,
                market_impact_pct=market_impact_pct,
                source=source,
            )
            session.add(event)
            await session.flush()
            event_id = event.id

        logger.info("signal_event_added", id=event_id, title=title, type=event_type)
        return event_id

    async def get_events(
        self,
        event_type: str | None = None,
        start_date: dt.date | None = None,
        end_date: dt.date | None = None,
        ticker: str | None = None,
    ) -> list[dict]:
        """Query signal events with optional filters.

        Returns:
            List of event dicts.
        """
        async with get_session() as session:
            query = select(SignalEvent)

            conditions = []
            if event_type:
                conditions.append(SignalEvent.event_type == event_type)
            if start_date:
                conditions.append(SignalEvent.event_date >= start_date)
            if end_date:
                conditions.append(SignalEvent.event_date <= end_date)
            if ticker:
                conditions.append(
                    SignalEvent.affected_tickers.any(ticker.upper())
                )

            if conditions:
                query = query.where(and_(*conditions))

            query = query.order_by(SignalEvent.event_date.desc())
            result = await session.execute(query)
            events = result.scalars().all()

            return [
                {
                    "id": e.id,
                    "event_date": e.event_date.isoformat(),
                    "event_end_date": e.event_end_date.isoformat() if e.event_end_date else None,
                    "event_type": e.event_type,
                    "severity": e.severity,
                    "title": e.title,
                    "description": e.description,
                    "affected_tickers": e.affected_tickers,
                    "affected_sectors": e.affected_sectors,
                    "market_impact_pct": e.market_impact_pct,
                    "source": e.source,
                }
                for e in events
            ]

    async def seed_historical_events(self) -> int:
        """Seed the database with known historical Black Swan events.

        Returns:
            Number of events seeded.
        """
        events = [
            {
                "event_date": dt.date(2020, 3, 9),
                "event_end_date": dt.date(2020, 4, 1),
                "event_type": EVENT_BLACK_SWAN,
                "severity": "critical",
                "title": "COVID-19 Pandemic Panic Sell-off",
                "description": (
                    "Global COVID-19 pandemic triggered massive sell-off. "
                    "VN-Index dropped from ~930 to ~650 points (-30%). "
                    "Circuit breakers triggered multiple times. "
                    "Market closed for sanitization on several days."
                ),
                "affected_sectors": ["all"],
                "market_impact_pct": -30.0,
                "source": "historical",
            },
            {
                "event_date": dt.date(2022, 4, 5),
                "event_end_date": dt.date(2022, 11, 15),
                "event_type": EVENT_REGULATORY,
                "severity": "critical",
                "title": "Arrest of Tan Hoang Minh & FLC Executives",
                "description": (
                    "Chairman of FLC Group Trinh Van Quyet arrested for stock manipulation. "
                    "Followed by Tan Hoang Minh bond fraud scandal. "
                    "Triggered massive confidence crisis in real estate and bond markets. "
                    "VN-Index declined from ~1,500 to ~900 (-40%)."
                ),
                "affected_tickers": ["FLC", "ROS", "HAI", "GAB", "ART"],
                "affected_sectors": ["real_estate", "bonds", "banking"],
                "market_impact_pct": -40.0,
                "source": "historical",
            },
            {
                "event_date": dt.date(2022, 10, 7),
                "event_type": EVENT_BLACK_SWAN,
                "severity": "critical",
                "title": "Van Thinh Phat & SCB Banking Scandal",
                "description": (
                    "Truong My Lan (Van Thinh Phat) arrested for fraud involving "
                    "SCB bank. Largest financial fraud case in Vietnam's history. "
                    "Triggered bank run on SCB and panic in banking sector."
                ),
                "affected_tickers": ["SCB"],
                "affected_sectors": ["banking", "real_estate"],
                "market_impact_pct": -15.0,
                "source": "historical",
            },
            {
                "event_date": dt.date(2021, 1, 4),
                "event_end_date": dt.date(2022, 1, 10),
                "event_type": EVENT_BOOM,
                "severity": "high",
                "title": "Post-COVID Bull Run (New Retail Investors)",
                "description": (
                    "Massive influx of retail investors opened new trading accounts. "
                    "VN-Index surged from ~1,100 to ~1,530 all-time high. "
                    "Record trading volumes driven by new F0 (first-time) investors."
                ),
                "affected_sectors": ["all"],
                "market_impact_pct": 39.0,
                "source": "historical",
            },
            {
                "event_date": dt.date(2018, 4, 9),
                "event_end_date": dt.date(2018, 7, 6),
                "event_type": EVENT_MACRO_SHOCK,
                "severity": "high",
                "title": "US-China Trade War Impact",
                "description": (
                    "US-China trade war escalation triggered sell-off in emerging markets. "
                    "VN-Index dropped from all-time high of ~1,200 to ~900. "
                    "Foreign investors net-sold heavily."
                ),
                "affected_sectors": ["export", "manufacturing", "seafood"],
                "market_impact_pct": -25.0,
                "source": "historical",
            },
            {
                "event_date": dt.date(2024, 8, 5),
                "event_type": EVENT_MACRO_SHOCK,
                "severity": "high",
                "title": "Global Carry Trade Unwind (Japan Yen Crisis)",
                "description": (
                    "Bank of Japan rate hike triggered massive unwinding of yen carry trades. "
                    "Global equity markets crashed. VN-Index dropped ~6% in single session."
                ),
                "affected_sectors": ["all"],
                "market_impact_pct": -6.0,
                "source": "historical",
            },
        ]

        count = 0
        for event_data in events:
            try:
                await self.add_event(**event_data)
                count += 1
            except Exception as e:
                logger.warning("seed_event_error", title=event_data["title"], error=str(e))

        logger.info("historical_events_seeded", count=count)
        return count
