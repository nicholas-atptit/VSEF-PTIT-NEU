"""Price adjustment calculator for corporate actions.

Computes adjusted prices by applying cumulative adjustment factors
for events like dividends, stock splits, rights issues, etc.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

from sqlalchemy import select, and_
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config.settings import get_settings
from src.database.connection import get_session
from src.models.price import AdjustedPrice, CorporateAction, RawPrice
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PriceAdjuster:
    """Calculates and applies price adjustments for corporate actions.

    Adjustment types:
    - Cash dividend: factor = (price - dividend) / price
    - Stock split: factor = 1 / split_ratio (e.g., 2:1 → factor = 0.5)
    - Stock dividend (bonus): factor = 1 / (1 + bonus_ratio)
    - Rights issue: factor = (price + rights_price * ratio) / (price * (1 + ratio))
    """

    async def recalculate(self, ticker: str) -> int:
        """Recalculate all adjusted prices for a ticker.

        Fetches all corporate actions, computes cumulative factors,
        and updates the adjusted_prices table.

        Returns:
            Number of records updated.
        """
        logger.info("price_adjustment_start", ticker=ticker)

        # Fetch all corporate actions for this ticker, ordered by date
        async with get_session() as session:
            result = await session.execute(
                select(CorporateAction)
                .where(CorporateAction.ticker == ticker.upper())
                .order_by(CorporateAction.event_date.asc())
            )
            actions = result.scalars().all()

        if not actions:
            logger.info("no_corporate_actions", ticker=ticker)
            return 0

        # Build cumulative adjustment factor timeline
        factor_timeline = self._build_factor_timeline(actions)

        # Fetch raw prices
        async with get_session() as session:
            result = await session.execute(
                select(RawPrice)
                .where(RawPrice.ticker == ticker.upper())
                .order_by(RawPrice.timestamp.asc())
            )
            raw_prices = result.scalars().all()

        if not raw_prices:
            return 0

        # Calculate adjusted prices
        adjusted_records = []
        for raw in raw_prices:
            factor = self._get_factor_for_date(
                factor_timeline, raw.timestamp.date()
            )

            adjusted_records.append({
                "timestamp": raw.timestamp,
                "ticker": raw.ticker,
                "exchange": raw.exchange,
                "timeframe": raw.timeframe,
                "open": raw.open * factor,
                "high": raw.high * factor,
                "low": raw.low * factor,
                "close": raw.close * factor,
                "volume": self._adjust_volume(raw.volume, factor),
                "adjustment_factor": factor,
                "adjustment_reason": self._get_reason(actions, raw.timestamp.date()),
                "source": "computed",
            })

        # Upsert adjusted prices
        count = await self._upsert_adjusted(adjusted_records)

        logger.info(
            "price_adjustment_done",
            ticker=ticker,
            actions_count=len(actions),
            records_updated=count,
        )
        return count

    def _build_factor_timeline(
        self,
        actions: list[CorporateAction],
    ) -> list[tuple[dt.date, Decimal]]:
        """Build cumulative adjustment factor timeline.

        Returns list of (date, cumulative_factor) tuples.
        The factor applies to all data BEFORE the event date.
        """
        # Work backwards: most recent action first
        cumulative = Decimal("1.0")
        timeline = []

        for action in reversed(actions):
            cumulative *= action.factor
            timeline.append((action.event_date, cumulative))

        timeline.reverse()
        return timeline

    def _get_factor_for_date(
        self,
        timeline: list[tuple[dt.date, Decimal]],
        date: dt.date,
    ) -> Decimal:
        """Get the cumulative adjustment factor for a specific date.

        All data before the earliest corporate action gets the full
        cumulative factor. Data after the last action gets factor 1.0.
        """
        if not timeline:
            return Decimal("1.0")

        for event_date, factor in timeline:
            if date < event_date:
                return factor

        return Decimal("1.0")

    @staticmethod
    def _adjust_volume(volume: int, factor: Decimal) -> int:
        """Adjust volume inversely to price adjustment."""
        if factor == 0 or factor == Decimal("1.0"):
            return volume
        return int(volume / factor)

    @staticmethod
    def _get_reason(actions: list[CorporateAction], date: dt.date) -> str | None:
        """Get the most relevant corporate action description for a date."""
        for action in actions:
            if date < action.event_date:
                return action.description
        return None

    async def _upsert_adjusted(self, records: list[dict]) -> int:
        """Upsert adjusted price records in batches."""
        if not records:
            return 0

        batch_size = 500
        total = 0

        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]

            async with get_session() as session:
                stmt = pg_insert(AdjustedPrice).values(batch)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["timestamp", "ticker"],
                    set_={
                        "open": stmt.excluded.open,
                        "high": stmt.excluded.high,
                        "low": stmt.excluded.low,
                        "close": stmt.excluded.close,
                        "volume": stmt.excluded.volume,
                        "adjustment_factor": stmt.excluded.adjustment_factor,
                        "adjustment_reason": stmt.excluded.adjustment_reason,
                    },
                )
                await session.execute(stmt)
                total += len(batch)

        return total

    # ── Corporate Action Management ──────────────────────────

    async def add_action(
        self,
        ticker: str,
        event_date: dt.date,
        action_type: str,
        factor: Decimal,
        description: str | None = None,
    ) -> None:
        """Add a new corporate action and recalculate adjusted prices."""
        async with get_session() as session:
            action = CorporateAction(
                ticker=ticker.upper(),
                event_date=event_date,
                action_type=action_type,
                factor=factor,
                description=description,
            )
            session.add(action)

        logger.info(
            "corporate_action_added",
            ticker=ticker,
            type=action_type,
            date=event_date.isoformat(),
            factor=str(factor),
        )

        # Recalculate adjusted prices
        await self.recalculate(ticker)

    @staticmethod
    def compute_dividend_factor(
        price_before_ex: Decimal,
        dividend_per_share: Decimal,
    ) -> Decimal:
        """Compute adjustment factor for a cash dividend."""
        if price_before_ex == 0:
            return Decimal("1.0")
        return (price_before_ex - dividend_per_share) / price_before_ex

    @staticmethod
    def compute_split_factor(split_ratio: Decimal) -> Decimal:
        """Compute adjustment factor for a stock split.

        Args:
            split_ratio: e.g., 2 for a 2:1 split (each share becomes 2)
        """
        return Decimal("1.0") / split_ratio

    @staticmethod
    def compute_bonus_factor(bonus_ratio: Decimal) -> Decimal:
        """Compute adjustment factor for a stock dividend/bonus share.

        Args:
            bonus_ratio: e.g., 0.5 for 50% bonus (5 shares per 10)
        """
        return Decimal("1.0") / (Decimal("1.0") + bonus_ratio)
