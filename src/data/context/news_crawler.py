"""News and financial report crawler.

The vnstock_data library does not expose a news/company API.
This module fetches news via the VNStock web API if available,
or returns empty results gracefully.

Canonical data provider: vnstock_data (for OHLCV/listing).
News fetching is handled separately and does NOT use vnstock library.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
from typing import Any, Callable

import pandas as pd

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
    """Crawler — news is fetched via VNStock company API (over HTTP, no vnstock import)."""
    def __init__(self):
        # vnstock_data does not expose a news API; news is not available via this path.
        pass

    async def get_news(self, ticker: str, count: int = 10) -> list[CrawledDocument]:
        """Returns empty list — news fetching requires a separate news source."""
        logger.warning("news_runtime_not_enabled_via_vnstock_data", ticker=ticker)
        return []

class BatchCrawler:
    def __init__(self):
        self.crawler = Crawler()
    async def crawl_tickers(self, tickers: list[str]) -> dict[str, list[CrawledDocument]]:
        return {t: await self.crawler.get_news(t) for t in tickers}

class NewsCrawler:
    """Async news crawler with optional public provider injection."""

    def __init__(
        self,
        concurrency: int = 5,
        provider_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._semaphore = asyncio.Semaphore(concurrency)
        self._provider_factory = provider_factory

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
        """Fetch ticker news through an injected provider when available."""
        if self._provider_factory is None:
            logger.warning("news_crawl_skipped_no_provider", ticker=ticker)
            return []

        async with self._semaphore:
            provider = self._provider_factory()
            stock = provider.stock(symbol=ticker)
            news = stock.news()
            if not isinstance(news, pd.DataFrame) or news.empty:
                return []
            docs: list[CrawledDocument] = []
            for _, row in news.head(count).iterrows():
                docs.append(self._document_from_row(ticker, row))
            return docs

    @staticmethod
    def _document_from_row(ticker: str, row: pd.Series) -> CrawledDocument:
        title = str(row.get("title") or row.get("headline") or "")
        content = str(row.get("description") or row.get("summary") or row.get("content") or "")
        url = str(
            row.get("link")
            or row.get("url")
            or f"provider://news/{ticker}/{hashlib.sha256(title.encode()).hexdigest()[:12]}"
        )
        source = str(row.get("source") or row.get("publisher") or "provider")
        published_raw = row.get("published_at") or row.get("published") or row.get("date")
        published_at = pd.to_datetime(published_raw, errors="coerce")
        published_dt = published_at.to_pydatetime() if pd.notna(published_at) else dt.datetime.now()
        return CrawledDocument(
            url=url,
            title=title,
            content=content,
            source=source,
            published_at=published_dt,
            tickers=[ticker],
        )

    async def crawl_watchlist(self, tickers: list[str], **kwargs) -> list[CrawledDocument]:
        """Backward compatibility for bulk crawling."""
        tasks = [self.crawl_ticker(t, **kwargs) for t in tickers]
        results = await asyncio.gather(*tasks)
        # Flatten list of lists
        return [doc for sublist in results for doc in sublist]
