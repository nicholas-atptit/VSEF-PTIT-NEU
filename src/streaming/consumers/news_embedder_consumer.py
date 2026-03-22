"""Kafka Consumer that embeds crawled news articles into ChromaDB.

Listens to `market.news.raw` topic. For each news article event,
it formats the document and inserts it into the ChromaDB vector DB
for RAG retrieval.
"""

import asyncio

from src.streaming.kafka_client import KafkaSubscriber
from src.utils.logging import get_logger

logger = get_logger(__name__)


class NewsEmbedderConsumer:
    """Consumes raw news events and embeds them into ChromaDB."""

    TOPIC = "market.news.raw"
    GROUP_ID = "news_embedder_group"

    def __init__(self) -> None:
        self.subscriber = KafkaSubscriber(self.TOPIC, self.GROUP_ID)

    async def run_forever(self) -> None:
        """Run the consumer loop continuously."""
        logger.info("news_embedder_consumer_starting")
        try:
            async for article in self.subscriber.stream():
                await self._process_article(article)
        except asyncio.CancelledError:
            logger.info("news_embedder_consumer_cancelled")
        except Exception as e:
            logger.error("news_embedder_consumer_error", error=str(e))
        finally:
            await self.subscriber.stop()

    async def _process_article(self, article: dict) -> None:
        """Embed a single news article into ChromaDB."""
        title = article.get("title", "")
        content = article.get("content", "")
        ticker = article.get("primary_ticker", "")
        url = article.get("url", "")

        if not content or len(content) < 30:
            return

        try:
            from src.context.rag_service import ZonedRAGService

            rag_service = ZonedRAGService()

            doc_payload = {
                "doc_id": article.get("doc_id", url),
                "title": title,
                "content": content,
                "source": article.get("source", "kafka_news"),
                "tickers": article.get("tickers", ticker),
                "primary_ticker": ticker,
                "url": url,
                "doc_type": article.get("doc_type", "news"),
                "zone": "zone_3",  # News zone
                "published_date": article.get("published_date", ""),
            }

            # Use run_in_executor for sync ChromaDB calls
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, rag_service.upsert_document, doc_payload)
            logger.info("news_embedded", ticker=ticker, title=title[:50])

        except Exception as e:
            logger.error("news_embed_failed", ticker=ticker, error=str(e))


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    consumer = NewsEmbedderConsumer()
    try:
        asyncio.run(consumer.run_forever())
    except KeyboardInterrupt:
        pass
