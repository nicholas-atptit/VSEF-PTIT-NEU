"""Kafka Consumer that caches the final LLM analysis results.

Listens to `llm.analysis` topic. For each event, it stores the latest
prediction + LLM analysis per ticker in a local JSON cache file.
API endpoints can read from this cache for instant sub-millisecond responses.
"""

import asyncio
import json
import os
from pathlib import Path

from src.api.streaming.kafka_client import KafkaSubscriber
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Cache directory
CACHE_DIR = Path("data/prediction_cache")
CACHE_FILE = CACHE_DIR / "latest_predictions.json"


class CacheWriterConsumer:
    """Consumes LLM analysis results and writes them to a JSON cache."""

    TOPIC = "llm.analysis"
    GROUP_ID = "cache_writer_group"

    def __init__(self) -> None:
        self.subscriber = KafkaSubscriber(self.TOPIC, self.GROUP_ID)
        self._cache: dict = {}
        self._load_existing_cache()

    def _load_existing_cache(self) -> None:
        """Load existing cache from disk if available."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info("cache_loaded", tickers=len(self._cache))
            except Exception:
                self._cache = {}

    def _flush_cache(self) -> None:
        """Write cache to disk."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("cache_flush_error", error=str(e))

    async def run_forever(self) -> None:
        """Run the consumer loop continuously."""
        logger.info("cache_writer_consumer_starting")
        try:
            async for event in self.subscriber.stream():
                await self._process_event(event)
        except asyncio.CancelledError:
            logger.info("cache_writer_consumer_cancelled")
        except Exception as e:
            logger.error("cache_writer_consumer_error", error=str(e))
        finally:
            self._flush_cache()
            await self.subscriber.stop()

    async def _process_event(self, event: dict) -> None:
        """Store the latest analysis for a ticker."""
        ticker = event.get("ticker", "UNKNOWN")
        
        self._cache[ticker] = {
            "ticker": ticker,
            "timestamp": event.get("timestamp", ""),
            "ml_prediction": event.get("ml_prediction", {}),
            "llm_analysis": event.get("llm_analysis", {}),
        }

        # Flush every 10 updates for durability
        if len(self._cache) % 10 == 0:
            self._flush_cache()

        logger.info("cache_updated", ticker=ticker, total_cached=len(self._cache))

    @classmethod
    def read_cache(cls, ticker: str | None = None) -> dict:
        """Static method for API endpoints to read from the cache.
        
        Args:
            ticker: If provided, return only that ticker's data.
                    If None, return the entire cache.
        """
        if not CACHE_FILE.exists():
            return {}
        
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            return {}

        if ticker:
            return cache.get(ticker.upper(), {})
        return cache


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    consumer = CacheWriterConsumer()
    try:
        asyncio.run(consumer.run_forever())
    except KeyboardInterrupt:
        pass
