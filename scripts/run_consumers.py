"""Kafka Consumer Supervisor — runs ALL consumer daemons in one process.

This is the single entry point for running all 5 Kafka consumers concurrently.
Each consumer runs as an independent asyncio task. If one crashes, it is
automatically restarted after a short delay.

Usage:
    python scripts/run_consumers.py
"""

import asyncio
import logging
import os
import sys

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logging import get_logger

logger = get_logger(__name__)


async def _run_with_restart(name: str, consumer_factory, max_retries: int = -1):
    """Run a consumer with automatic restart on failure.
    
    Args:
        name: Human-readable consumer name for logging.
        consumer_factory: Callable that returns a new consumer instance.
        max_retries: Max restart attempts (-1 = infinite).
    """
    attempt = 0
    while max_retries == -1 or attempt < max_retries:
        attempt += 1
        try:
            logger.info("consumer_starting", consumer=name, attempt=attempt)
            consumer = consumer_factory()
            await consumer.run_forever()
        except asyncio.CancelledError:
            logger.info("consumer_cancelled", consumer=name)
            break
        except Exception as e:
            logger.error("consumer_crashed", consumer=name, error=str(e), attempt=attempt)
            # Exponential backoff: 2s, 4s, 8s, 16s, max 30s
            delay = min(2 ** attempt, 30)
            logger.info("consumer_restarting_in", consumer=name, seconds=delay)
            await asyncio.sleep(delay)


async def main():
    """Start all 5 consumer daemons concurrently."""
    from src.streaming.consumers.db_writer_consumer import DatabaseWriterConsumer
    from src.streaming.consumers.ml_prediction_consumer import MLPredictionConsumer
    from src.streaming.consumers.llm_analysis_consumer import LLMAnalysisConsumer
    from src.streaming.consumers.news_embedder_consumer import NewsEmbedderConsumer
    from src.streaming.consumers.cache_writer_consumer import CacheWriterConsumer

    consumers = [
        ("DB-Writer",         lambda: DatabaseWriterConsumer()),
        ("ML-Prediction",     lambda: MLPredictionConsumer()),
        ("LLM-Analysis",      lambda: LLMAnalysisConsumer()),
        ("News-Embedder",     lambda: NewsEmbedderConsumer()),
        ("Cache-Writer",      lambda: CacheWriterConsumer()),
    ]

    logger.info("supervisor_starting", consumer_count=len(consumers))
    print("=" * 55)
    print("  🚀 KAFKA CONSUMER SUPERVISOR")
    print(f"  Starting {len(consumers)} consumer daemons...")
    print("=" * 55)
    for name, _ in consumers:
        print(f"   ├── {name}")
    print()

    tasks = [
        asyncio.create_task(_run_with_restart(name, factory))
        for name, factory in consumers
    ]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("supervisor_shutting_down")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Supervisor stopped.")
