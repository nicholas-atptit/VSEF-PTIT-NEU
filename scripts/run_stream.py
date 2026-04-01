"""Main entry point: Starts the streaming pipeline.

Initializes: SessionStreamingManager -> FilterEngine -> RealtimeIngestor
Sets up scheduler and signal handlers for graceful shutdown.
"""

from __future__ import annotations

import asyncio
import signal
import sys

# Add project root to path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from src.engine.filtering.engine import FilterEngine
from src.data.historical.ingestor import RealtimeIngestor
from src.api.streaming.fallback import FallbackFiller
from src.api.streaming.scheduler import TradingSessionScheduler
from src.api.streaming.session_manager import SessionStreamingManager
from src.utils.logging import setup_logging, get_logger


async def main() -> None:
    """Main streaming pipeline entry point."""
    setup_logging()
    logger = get_logger("run_stream")
    settings = get_settings()

    logger.info("=== Algo Trading Data Pipeline Starting ===")

    # ── Initialize components ────────────────────────────────

    # 1. Real-time ingestor (DB writer)
    ingestor = RealtimeIngestor(
        batch_size=settings.backdate_batch_size,
        flush_interval=5.0,
    )

    # 2. Filter engine (data pipeline)
    filter_engine = FilterEngine(
        on_processed=ingestor.ingest,
    )

    # 3. Fallback filler (gap recovery)
    fallback = FallbackFiller()

    # 4. Session streaming manager (WebSocket)
    stream_manager = SessionStreamingManager(
        on_trade=filter_engine.handle_trade,
        on_ohlc=filter_engine.handle_ohlc,
        on_quote=filter_engine.handle_quote,
        on_disconnect=fallback.on_disconnect,
        on_reconnect=fallback.on_reconnect,
    )

    # 5. Scheduler
    scheduler = TradingSessionScheduler(stream_manager)

    # ── Start components ─────────────────────────────────────

    await filter_engine.start()
    await ingestor.start()

    # Load watchlist and set on stream manager
    watchlist_symbols = filter_engine.get_watchlist_symbols()
    stream_manager.set_watchlist(watchlist_symbols)

    # Start scheduler (auto open/close sessions)
    scheduler.start()

    logger.info(
        "pipeline_ready",
        watchlist_count=len(watchlist_symbols),
        next_jobs=scheduler.get_next_jobs(),
    )

    # ── Graceful shutdown handler ────────────────────────────

    shutdown_event = asyncio.Event()

    def handle_signal(sig: int, frame: object) -> None:
        logger.info("shutdown_signal_received", signal=sig)
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Wait for shutdown signal
    await shutdown_event.wait()

    # ── Shutdown ─────────────────────────────────────────────
    logger.info("shutting_down")
    scheduler.stop()
    await stream_manager.close_session()
    await filter_engine.stop()
    await ingestor.stop()

    logger.info("=== Pipeline shutdown complete ===")
    logger.info("final_stats", filter=filter_engine.stats, ingestor=ingestor.stats)


if __name__ == "__main__":
    asyncio.run(main())
