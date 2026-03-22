"""Entry point: Seed historical Black Swan events into the database."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.context.signal_tagger import SignalTagger
from src.utils.logging import setup_logging, get_logger


async def main() -> None:
    """Seed historical signal events."""
    setup_logging()
    logger = get_logger("seed_signals")

    logger.info("Seeding historical signal events...")

    tagger = SignalTagger()
    count = await tagger.seed_historical_events()

    logger.info(f"Seeded {count} historical events")


if __name__ == "__main__":
    asyncio.run(main())
