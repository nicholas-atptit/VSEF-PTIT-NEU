"""Kafka Consumer for saving raw market events into TimescaleDB.

Listens to the `market.data.raw` topic and upserts rows into the
PostgreSQL / Timescale DB `historical_price` table.
"""

import asyncio
import logging
from datetime import datetime

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.database.session import get_db_session
from src.ml.models.historical import HistoricalPrice
from src.api.streaming.kafka_client import KafkaSubscriber

logger = logging.getLogger(__name__)


class DatabaseWriterConsumer:
    """Consumes market data and writes safely to TimescaleDB."""

    TOPIC = "market.data.raw"
    GROUP_ID = "timescaledb_writer_group"

    def __init__(self) -> None:
        self.subscriber = KafkaSubscriber(self.TOPIC, self.GROUP_ID)

    async def run_forever(self) -> None:
        """Run the consumer loop continuously."""
        logger.info("db_writer_consumer_starting", topic=self.TOPIC)
        try:
            async for payload in self.subscriber.stream():
                await self._process_message(payload)
        except asyncio.CancelledError:
            logger.info("db_writer_consumer_cancelled")
        except Exception as e:
            logger.error("db_writer_consumer_error", error=str(e))
        finally:
            await self.subscriber.stop()

    async def _process_message(self, data: dict) -> None:
        """Process a single JSON message and save to DB."""
        try:
            ticker = data["ticker"]
            timestamp_str = data["timestamp"]
            
            # vnstock usually outputs "2024-03-24" or datetime string
            if len(timestamp_str) == 10:
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d")
            else:
                timestamp = datetime.fromisoformat(timestamp_str)

            stmt = insert(HistoricalPrice).values(
                ticker=ticker,
                timestamp=timestamp,
                resolution="11D",  # Store as 1-day by default from AM/PM crawls
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                volume=data["volume"],
            )

            # Upsert logic to avoid crashing on duplicate AM/PM pulls for the same day
            stmt = stmt.on_conflict_do_update(
                index_elements=["ticker", "timestamp", "resolution"],
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                },
            )

            async for session in get_db_session():
                await session.execute(stmt)
                await session.commit()
                # Break after one successful session execute
                break
                
            logger.debug("db_writer_saved", ticker=ticker, date=timestamp_str)
            
        except Exception as e:
            logger.error("db_writer_upsert_failed", data=data, error=str(e))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    consumer = DatabaseWriterConsumer()
    try:
        asyncio.run(consumer.run_forever())
    except KeyboardInterrupt:
        pass
