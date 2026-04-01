"""News and financial report crawler — Vnstock Edition.

Uses the standard vnstock library for financial news extraction.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
from typing import Any

import pandas as pd
from vnstock import Vnstock

from config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

class CrawledDocument:
    """A single crawled document with metadata."""
    def __init__(
        self,
        url: str,
        title: str,
        content: str,
        source: str = "Unknown",
        published_at: dt.datetime | None = None,
        tickers: list[str] | None = None,
        doc_type: str = "news"
    ):
        self.url = url
        self.title = title
        self.content = content
        self.source = source
        self.published_at = published_at or dt.datetime.now()
        self.tickers = tickers or []
        self.primary_ticker = self.tickers[0] if self.tickers else ""
        self.doc_type = doc_type
        self.doc_id = hashlib.sha256(url.encode()).hexdigest()[:16]

    @property
    def published_date(self) -> dt.datetime:
        """Alias for backward compatibility."""
        return self.published_at

    @property
    def metadata(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "source": self.source,
            "doc_type": self.doc_type,
            "tickers": ",".join(self.tickers),
            "published_at": self.published_at.isoformat()
        }

    def as_embedding_document(self, zone: str = "zone_3") -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": f"{self.title}\n\n{self.content}",
            "metadata": self.metadata,
            "zone": zone
        }

class Crawler:
    """Crawler using standard vnstock library."""
    def __init__(self):
        self.vn = Vnstock()

    async def get_news(self, ticker: str, count: int = 10) -> list[CrawledDocument]:
        """Fetch news via vnstock stock.news()"""
        try:
            stock = self.vn.stock(symbol=ticker.upper(), source='VCI')
            news_df = stock.news()
            
            docs = []
            if news_df is not None and not news_df.empty:
                for _, row in news_df.head(count).iterrows():
                    docs.append(CrawledDocument(
                        url=row.get('link', ''),
                        title=row.get('title', ''),
                        content=row.get('description', ''),
                        source=row.get('source', 'Vnstock'),
                        published_at=dt.datetime.now(),
                        tickers=[ticker.upper()]
                    ))
            return docs
        except Exception as e:
            logger.error("news_fetch_failed", ticker=ticker, error=str(e))
            return []

class BatchCrawler:
    def __init__(self):
        self.crawler = Crawler()
    async def crawl_tickers(self, tickers: list[str]) -> dict[str, list[CrawledDocument]]:
        return {t: await self.crawler.get_news(t) for t in tickers}

class NewsCrawler:
    """Async news crawler utilizing Vnstock API."""
    def __init__(self, concurrency: int = 5) -> None:
        self.vn = Vnstock()
        self._semaphore = asyncio.Semaphore(concurrency)

    async def crawl_ticker(self, ticker: str, count: int = 10, **kwargs) -> list[CrawledDocument]:
        """Crawl news for a specific ticker with strict timeout."""
        try:
            count = max(1, int(count))
            return await asyncio.wait_for(self._crawl_ticker_internal(ticker, count, **kwargs), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("news_crawl_timeout", ticker=ticker)
            return []
        except Exception as e:
            logger.error("news_crawl_fatal", ticker=ticker, error=str(e))
            return []

    async def _crawl_ticker_internal(self, ticker: str, count: int = 10, **kwargs) -> list[CrawledDocument]:
        if "max_pages" in kwargs:
            count = kwargs["max_pages"] * 5
        elif "max_pages_per_ticker" in kwargs:
            count = kwargs["max_pages_per_ticker"] * 5
            
        async with self._semaphore:
            try:
                loop = asyncio.get_running_loop()
                stock = await loop.run_in_executor(None, lambda: self.vn.stock(symbol=ticker.upper(), source='VCI'))

                # Prefer stock.news() (stable API), then fallback to stock.company.news().
                def _fetch_news_df():
                    try:
                        return stock.news()
                    except Exception:
                        company = getattr(stock, "company", None)
                        if company is None:
                            raise
                        return company.news()

                news_df = await loop.run_in_executor(None, _fetch_news_df)
                
                docs = []
                if news_df is not None and not news_df.empty:
                    logger.info("news_fetch_success", ticker=ticker, rows=len(news_df))
                    for _, row in news_df.head(count).iterrows():
                        docs.append(CrawledDocument(
                            url=row.get('link', ''),
                            title=row.get('title', ''),
                            content=row.get('description', ''),
                            source=row.get('source', 'Vnstock'),
                            published_at=dt.datetime.now(),
                            tickers=[ticker.upper()]
                        ))
                return docs
            except Exception as e:
                logger.error("news_crawl_failed", ticker=ticker, error=str(e))
                return []

    async def crawl_watchlist(self, tickers: list[str], **kwargs) -> list[CrawledDocument]:
        """Backward compatibility for bulk crawling."""
        tasks = [self.crawl_ticker(t, **kwargs) for t in tickers]
        results = await asyncio.gather(*tasks)
        # Flatten list of lists
        return [doc for sublist in results for doc in sublist]
