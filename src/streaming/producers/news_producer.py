"""Kafka Producer for pushing scraped news to the message broker.

Extracts financial news AM and PM using NewsCrawler and publishes
them as JSON events to the `market.news.raw` topic.
"""

import logging

from src.context.news_crawler import NewsCrawler
from src.streaming.kafka_client import KafkaPublisher

logger = logging.getLogger(__name__)


class NewsProducer:
    """Producer for raw financial news."""

    TOPIC = "market.news.raw"

    def __init__(self) -> None:
        self.publisher = KafkaPublisher()
        self.crawler = NewsCrawler()

    async def fetch_and_publish(self, tickers: list[str]) -> None:
        """Fetch news for tickers and publish to Kafka."""
        try:
            logger.info("news_producer_starting", ticker_count=len(tickers))
            await self.publisher.start()
            
            # Use crawler directly
            docs = await self.crawler.crawl_watchlist(tickers, max_pages_per_ticker=2)
            
            published_count = 0
            for doc in docs:
                event = doc.metadata
                # Include content for consumer to process (e.g. embed)
                event["content"] = doc.content
                await self.publisher.publish(self.TOPIC, event)
                published_count += 1
                
            logger.info("news_producer_finished", published=published_count)
            
        except Exception as e:
            logger.error("news_producer_error", error=str(e))
        finally:
            await self.publisher.stop()
