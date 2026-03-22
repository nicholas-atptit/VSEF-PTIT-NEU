"""Blacklist filtering — drop banned stock symbols immediately.

Maintains an in-memory set of blacklisted tickers loaded from the database.
Refreshes periodically to pick up changes.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Set

from sqlalchemy import select

from config.settings import get_settings
from src.database.connection import get_session
from src.models.watchlist import BlacklistItem
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BlacklistFilter:
    """O(1) blacklist lookup filter.

    Loads banned tickers into an in-memory set for instant checks.
    Refreshes from database at configurable intervals.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._blacklist: Set[str] = set()
        self._last_refresh: dt.datetime | None = None
        self._refresh_task: asyncio.Task | None = None

    @property
    def blacklist(self) -> Set[str]:
        """Current blacklisted tickers."""
        return self._blacklist.copy()

    @property
    def count(self) -> int:
        """Number of blacklisted tickers."""
        return len(self._blacklist)

    async def load(self) -> None:
        """Load blacklist from database into memory."""
        try:
            async with get_session() as session:
                result = await session.execute(
                    select(BlacklistItem.ticker, BlacklistItem.banned_until)
                )
                rows = result.all()

                now = dt.datetime.now(dt.timezone.utc)
                active_tickers = set()

                for ticker, banned_until in rows:
                    # Only include active bans
                    if banned_until is None or now < banned_until:
                        active_tickers.add(ticker)

                old_count = len(self._blacklist)
                self._blacklist = active_tickers
                self._last_refresh = dt.datetime.now(dt.timezone.utc)

                if old_count != len(active_tickers):
                    logger.info(
                        "blacklist_refreshed",
                        count=len(active_tickers),
                        previous_count=old_count,
                    )
                else:
                    logger.debug("blacklist_refreshed", count=len(active_tickers))

        except Exception as e:
            logger.error("blacklist_load_failed", error=str(e))

    def is_blacklisted(self, ticker: str) -> bool:
        """Check if a ticker is on the blacklist. O(1) lookup."""
        return ticker.upper() in self._blacklist

    def filter_message(self, message: dict) -> bool:
        """Check if a message should be dropped.

        Args:
            message: Dict with at least a 'ticker' key.

        Returns:
            True if message should be KEPT (not blacklisted).
            False if message should be DROPPED (blacklisted).
        """
        ticker = message.get("ticker", "").upper()
        if self.is_blacklisted(ticker):
            logger.debug("blacklist_dropped", ticker=ticker)
            return False
        return True

    async def start_auto_refresh(self) -> None:
        """Start periodic auto-refresh of the blacklist."""
        await self.load()  # Initial load
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def _refresh_loop(self) -> None:
        """Periodically refresh blacklist from database."""
        interval = self._settings.blacklist_refresh_interval_minutes * 60
        while True:
            await asyncio.sleep(interval)
            await self.load()

    async def stop_auto_refresh(self) -> None:
        """Stop the auto-refresh task."""
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass

    # ── Manual management methods ────────────────────────────

    async def add_to_blacklist(
        self,
        ticker: str,
        reason: str,
        source: str = "manual",
        banned_until: dt.datetime | None = None,
    ) -> None:
        """Add a ticker to the blacklist."""
        async with get_session() as session:
            item = BlacklistItem(
                ticker=ticker.upper(),
                reason=reason,
                source=source,
                banned_until=banned_until,
            )
            await session.merge(item)

        self._blacklist.add(ticker.upper())
        logger.info("blacklist_added", ticker=ticker, reason=reason)

    async def remove_from_blacklist(self, ticker: str) -> None:
        """Remove a ticker from the blacklist."""
        async with get_session() as session:
            result = await session.execute(
                select(BlacklistItem).where(BlacklistItem.ticker == ticker.upper())
            )
            item = result.scalar_one_or_none()
            if item:
                await session.delete(item)

        self._blacklist.discard(ticker.upper())
        logger.info("blacklist_removed", ticker=ticker)
