"""Real-time data ingestor — batch-inserts standardized data into TimescaleDB.

Receives processed records from the FilterEngine and efficiently
batch-inserts them into the raw_prices and adjusted_prices tables.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config.settings import get_settings
from src.database.connection import get_session
from src.models.price import RawPrice
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RealtimeIngestor:
    """Batch-inserts standardized price data into TimescaleDB.

    Buffers records and flushes either when batch size is reached
    or when flush interval elapses — whichever comes first.
    """

    def __init__(
        self,
        batch_size: int = 100,
        flush_interval: float = 5.0,
    ) -> None:
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._buffer: list[dict] = []
        self._flush_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

        # Stats
        self._total_inserted = 0
        self._total_errors = 0
        self._total_flushes = 0

    async def start(self) -> None:
        """Start the periodic flush loop."""
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info(
            "ingestor_started",
            batch_size=self._batch_size,
            flush_interval=self._flush_interval,
        )

    async def stop(self) -> None:
        """Stop the ingestor and flush remaining records."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush()
        logger.info(
            "ingestor_stopped",
            total_inserted=self._total_inserted,
            total_errors=self._total_errors,
        )

    async def ingest(self, record: dict) -> None:
        """Add a standardized record to the buffer.

        Records are flushed to DB when batch size is reached.
        NOTE_PHASE_1_5: Intraday storage is disabled by user constraint.
        Only '1d' (End of Day) records are permitted to be inserted.
        """
        # Block quotes and intraday timeframes (tick, 1m, etc.)
        if getattr(self, "_store_intraday", False) is False:
            if record.get("timeframe") != "1d":
                return
                
        msg_type = record.get("message_type")
        if msg_type == "quote":
            return

        async with self._lock:
            self._buffer.append(record)

            if len(self._buffer) >= self._batch_size:
                await self._flush()

    async def _flush_loop(self) -> None:
        """Periodically flush buffered records."""
        while True:
            await asyncio.sleep(self._flush_interval)
            async with self._lock:
                if self._buffer:
                    await self._flush()

    async def _flush(self) -> None:
        """Flush buffered records to TimescaleDB."""
        if not self._buffer:
            return

        records = self._buffer.copy()
        self._buffer.clear()

        try:
            await self._batch_insert(records)
            self._total_inserted += len(records)
            self._total_flushes += 1
            logger.info(
                "batch_flushed",
                count=len(records),
                total_inserted=self._total_inserted,
            )
        except Exception as e:
            self._total_errors += len(records)
            logger.error(
                "batch_flush_error",
                error=str(e),
                lost_records=len(records),
            )
            # Re-add failed records for retry (limited)
            if len(records) <= self._batch_size * 2:
                self._buffer.extend(records)

    async def _batch_insert(self, records: list[dict]) -> None:
        """Insert a batch of records into raw_prices using upsert."""
        if not records:
            return

        # Build values for bulk insert
        values = []
        for r in records:
            values.append({
                "timestamp": r["timestamp"],
                "ticker": r["ticker"],
                "exchange": r.get("exchange", "HOSE"),
                "timeframe": r.get("timeframe", "1m"),
                "open": r["open"],
                "high": r["high"],
                "low": r["low"],
                "close": r["close"],
                "volume": r.get("volume", 0),
                "source": r.get("source", "websocket"),
            })

        async with get_session() as session:
            # Use PostgreSQL INSERT ... ON CONFLICT for upsert
            stmt = pg_insert(RawPrice).values(values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["timestamp", "ticker"],
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                    "source": stmt.excluded.source,
                },
            )
            await session.execute(stmt)

    @property
    def stats(self) -> dict:
        """Return ingestor statistics."""
        return {
            "buffer_size": len(self._buffer),
            "total_inserted": self._total_inserted,
            "total_errors": self._total_errors,
            "total_flushes": self._total_flushes,
        }
