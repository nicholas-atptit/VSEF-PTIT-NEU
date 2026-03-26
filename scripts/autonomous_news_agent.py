"""
Autonomous News Agent (Phase 3).

Periodically crawls financial news, embeds into ChromaDB, 
and generates multi-horizon sentiment reports.
"""

import asyncio
import datetime as dt
from typing import List

from src.context.news_crawler import NewsCrawler, CrawledDocument
from src.context.embedder import DocumentEmbedder
from src.llm.news_intel import NewsIntelEngine
from src.utils.logging import get_logger
from config.settings import get_settings

logger = get_logger(__name__)

class AutonomousNewsAgent:
    def __init__(self):
        self.settings = get_settings()
        self.crawler = NewsCrawler()
        self.embedder = DocumentEmbedder()
        self.intel_engine = NewsIntelEngine()
        self.watchlist = ["SSI", "VIC", "VHM", "VNM", "TCB", "HPG", "FPT"] # Sample watchlist

    async def run_market_sync(self):
        """Full market news sync and embedding."""
        logger.info("market_sync_started")
        
        # 1. Batch crawl for watchlist
        docs = await self.crawler.crawl_watchlist(self.watchlist)
        
        # 2. Embed into ChromaDB (Zone 3)
        for doc in docs:
            try:
                embedding_doc = doc.as_embedding_document(zone="zone_3")
                self.embedder.add_documents([embedding_doc])
            except Exception as e:
                logger.error("embedding_failed", url=doc.url, error=str(e))

        # 3. Generate Multi-Horizon Sentiment per Ticker
        for ticker in self.watchlist:
            # Fetch recent news from Chroma/Crawler for intelligence
            ticker_docs = [d for d in docs if d.primary_ticker == ticker]
            if ticker_docs:
                # Short-term Sentiment
                await self.intel_engine.analyze_ticker_news(ticker, ticker_docs[:10], horizon="short")
                # Long-term Sentiment (would normally query more historical RAG)
                # await self.intel_engine.analyze_ticker_news(ticker, ticker_docs, horizon="long")

        logger.info("market_sync_completed", docs_processed=len(docs))

    async def run_forever(self, interval_seconds: int = 3600):
        """Main loop for the autonomous agent."""
        while True:
            try:
                await self.run_market_sync()
            except Exception as e:
                logger.error("agent_loop_error", error=str(e))
            
            logger.info("agent_sleeping", seconds=interval_seconds)
            await asyncio.sleep(interval_seconds)

if __name__ == "__main__":
    agent = AutonomousNewsAgent()
    asyncio.run(agent.run_forever())
