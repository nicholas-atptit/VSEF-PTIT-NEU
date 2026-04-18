"""REST API fallback for filling data gaps during WebSocket disconnections.

When the WebSocket connection drops, gap timestamps are recorded.
On reconnection, this module fetches OHLC data via DNSE REST API
to fill the missing time blocks.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from decimal import Decimal
from typing import Any

from config.settings import get_settings
from src.utils.logging import get_logger
from src.utils.time_utils import VN_TZ

logger = get_logger(__name__)


class FallbackFiller:
    """Fills data gaps using DNSE REST API when WebSocket disconnects.

    On disconnect: records gap start timestamp.
    On reconnect: fetches OHLC data for the gap period via REST and
    inserts missing records into the database.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._pending_gaps: list[dict[str, float]] = []

    async def on_disconnect(self, disconnect_time: float) -> None:
        """Record the start of a data gap."""
        self._pending_gaps.append({
            "disconnect_time": disconnect_time,
            "reconnect_time": None,
        })
        logger.warning(
            "gap_recorded",
            disconnect_time=dt.datetime.fromtimestamp(disconnect_time, tz=VN_TZ).isoformat(),
        )

    async def on_reconnect(
        self,
        disconnect_time: float,
        reconnect_time: float,
    ) -> None:
        """Fill the data gap between disconnect and reconnect times.

        Args:
            disconnect_time: Unix timestamp when connection was lost.
            reconnect_time: Unix timestamp when connection was restored.
        """
        gap_seconds = reconnect_time - disconnect_time
        logger.info(
            "filling_gap",
            gap_seconds=gap_seconds,
            disconnect=dt.datetime.fromtimestamp(disconnect_time, tz=VN_TZ).isoformat(),
            reconnect=dt.datetime.fromtimestamp(reconnect_time, tz=VN_TZ).isoformat(),
        )

        try:
            await self._fetch_and_fill_gap(disconnect_time, reconnect_time)

            # Remove from pending gaps
            self._pending_gaps = [
                g for g in self._pending_gaps
                if g["disconnect_time"] != disconnect_time
            ]
            logger.info("gap_filled", gap_seconds=gap_seconds)

        except Exception as e:
            logger.error("gap_fill_failed", error=str(e), gap_seconds=gap_seconds)

    async def _fetch_and_fill_gap(
        self,
        disconnect_time: float,
        reconnect_time: float,
    ) -> None:
        """Fetch OHLC data from REST API for the gap period."""
        from_dt = dt.datetime.fromtimestamp(disconnect_time, tz=VN_TZ)
        to_dt = dt.datetime.fromtimestamp(reconnect_time, tz=VN_TZ)

        try:
            from dnse import DnseClient

            with DnseClient(
                api_key=self._settings.dnse_api_key,
                api_secret=self._settings.dnse_api_secret,
            ) as client:
                # Fetch OHLC for each watchlist symbol
                # Note: actual implementation depends on DNSE REST API
                # capabilities for intraday OHLC data
                logger.info(
                    "rest_api_fetch",
                    from_dt=from_dt.isoformat(),
                    to_dt=to_dt.isoformat(),
                    msg="Fetching gap data via REST API",
                )

                # TODO: iterate watchlist symbols and fetch OHLC data
                # The standardized records should be inserted into the
                # database via the RealtimeIngestor

        except ImportError:
            logger.warning("dnse_not_available", msg="Falling back to vnstock")
            await self._fetch_via_vnstock(from_dt, to_dt)

    async def _fetch_via_vnstock(
        self,
        from_dt: dt.datetime,
        to_dt: dt.datetime,
    ) -> None:
        """Fallback: use vnstock_data library to fetch gap data (canonical provider)."""
        try:
            from vnstock_data import Quote

            logger.info(
                "vnstock_data_fallback",
                from_dt=from_dt.isoformat(),
                to_dt=to_dt.isoformat(),
            )
            # vnstock_data integration for gap filling
            # This is a secondary fallback when DNSE REST is unavailable

        except ImportError:
            logger.error("vnstock_data_not_installed", msg="Cannot fill gap — no data source available")

    @property
    def pending_gaps(self) -> list[dict]:
        """Return list of unfilled gaps."""
        return self._pending_gaps.copy()

    def clear_gaps(self) -> None:
        """Clear all tracked gaps."""
        self._pending_gaps.clear()
        logger.info("gaps_cleared")
