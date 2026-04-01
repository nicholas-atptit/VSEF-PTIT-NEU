"""Trading session scheduler for Vietnamese stock market.

Manages automatic open/close of WebSocket sessions based on
VN trading hours (Asia/Ho_Chi_Minh timezone).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import get_settings
from src.utils.logging import get_logger
from src.utils.time_utils import VN_TZ, is_trading_day

if TYPE_CHECKING:
    from src.api.streaming.session_manager import SessionStreamingManager

logger = get_logger(__name__)


class TradingSessionScheduler:
    """Schedules WebSocket session open/close based on VN trading hours.

    Schedule (Vietnam time, Mon-Fri only):
    - 08:45 → Open morning session (data recording 09:00–11:30)
    - 11:30 → Close morning session
    - 12:45 → Open afternoon session (data recording 13:00–15:00 incl. ATC)
    - 15:00 → Close afternoon session
    """

    def __init__(self, stream_manager: SessionStreamingManager) -> None:
        self._stream_manager = stream_manager
        self._settings = get_settings()
        self._scheduler = AsyncIOScheduler(timezone=VN_TZ)
        self._is_running = False

    def start(self) -> None:
        """Start the trading session scheduler."""
        if self._is_running:
            logger.warning("scheduler_already_running")
            return

        # Parse session times from settings
        mo_h, mo_m = map(int, self._settings.morning_open.split(":"))
        mc_h, mc_m = map(int, self._settings.morning_close.split(":"))
        ao_h, ao_m = map(int, self._settings.afternoon_open.split(":"))
        ac_h, ac_m = map(int, self._settings.afternoon_close.split(":"))

        # ── Morning Session ───────────────────────────────────
        self._scheduler.add_job(
            self._open_morning_session,
            CronTrigger(
                hour=mo_h, minute=mo_m,
                day_of_week="mon-fri",
                timezone=VN_TZ,
            ),
            id="morning_open",
            name="Open Morning Session",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._close_morning_session,
            CronTrigger(
                hour=mc_h, minute=mc_m,
                day_of_week="mon-fri",
                timezone=VN_TZ,
            ),
            id="morning_close",
            name="Close Morning Session",
            replace_existing=True,
        )

        # ── Afternoon Session ────────────────────────────────
        self._scheduler.add_job(
            self._open_afternoon_session,
            CronTrigger(
                hour=ao_h, minute=ao_m,
                day_of_week="mon-fri",
                timezone=VN_TZ,
            ),
            id="afternoon_open",
            name="Open Afternoon Session",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._close_afternoon_session,
            CronTrigger(
                hour=ac_h, minute=ac_m,
                day_of_week="mon-fri",
                timezone=VN_TZ,
            ),
            id="afternoon_close",
            name="Close Afternoon Session",
            replace_existing=True,
        )

        # ── Market Data Sync (AM & PM) ─────────────
        self._scheduler.add_job(
            self._run_market_data_sync,
            CronTrigger(
                hour=11, minute=35,
                day_of_week="mon-fri",
                timezone=VN_TZ,
            ),
            id="am_market_sync",
            name="AM Market Data Sync",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._run_market_data_sync,
            CronTrigger(
                hour=15, minute=15,
                day_of_week="mon-fri",
                timezone=VN_TZ,
            ),
            id="pm_market_sync",
            name="PM Market Data Sync",
            replace_existing=True,
        )

        self._scheduler.start()
        self._is_running = True

        logger.info(
            "scheduler_started",
            morning=f"{self._settings.morning_open}–{self._settings.morning_close}",
            afternoon=f"{self._settings.afternoon_open}–{self._settings.afternoon_close}",
        )

    def stop(self) -> None:
        """Stop the scheduler and close any active session."""
        if self._is_running:
            self._scheduler.shutdown(wait=False)
            self._is_running = False
            logger.info("scheduler_stopped")

    async def _open_morning_session(self) -> None:
        """Open morning trading session if today is a trading day."""
        if not is_trading_day():
            logger.info("skip_morning_session", reason="not_trading_day")
            return

        logger.info("opening_morning_session")
        await self._stream_manager.open_session()

    async def _close_morning_session(self) -> None:
        """Close morning trading session."""
        if not self._stream_manager.is_connected:
            return

        logger.info("closing_morning_session")
        await self._stream_manager.close_session()

    async def _open_afternoon_session(self) -> None:
        """Open afternoon trading session if today is a trading day."""
        if not is_trading_day():
            logger.info("skip_afternoon_session", reason="not_trading_day")
            return

        logger.info("opening_afternoon_session")
        await self._stream_manager.open_session()

    async def _close_afternoon_session(self) -> None:
        """Close afternoon trading session (end of trading day)."""
        if not self._stream_manager.is_connected:
            return

        logger.info("closing_afternoon_session")
        await self._stream_manager.close_session()

    async def _run_market_data_sync(self) -> None:
        """Fetch AM/PM 1D candles & News for the entire market & publish to Kafka."""
        if not is_trading_day():
            return
            
        logger.info("starting_market_data_sync_and_news_harvest")
        try:
            from src.api.streaming.producers.market_data_producer import MarketDataProducer
            from src.api.streaming.producers.news_producer import NewsProducer
            
            # VNStock Price Data
            market_producer = MarketDataProducer()
            await market_producer.publish_all_tickers()
            
            # News Harvesting for key tickers (e.g. VN30 or Watchlist)
            news_producer = NewsProducer()
            # For demonstration, limit to some core tickers, or could pull from DB
            core_watchlist = ["SSI", "HPG", "FPT", "VNM", "MWG", "TCB", "MBB", "ACB"]
            await news_producer.fetch_and_publish(core_watchlist)
            
            logger.info("market_data_sync_complete")
        except Exception as e:
            logger.error("market_data_sync_error", error=str(e))

    @property
    def is_running(self) -> bool:
        """Check if scheduler is currently running."""
        return self._is_running

    def get_next_jobs(self) -> list[dict]:
        """Get upcoming scheduled jobs for monitoring."""
        jobs = []
        for job in self._scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": str(job.next_run_time) if job.next_run_time else None,
            })
        return jobs
