"""News and financial report crawler — Vnstock News Edition.

Uses the vnstock_news library for high-performance extraction from 
Vietnamese financial sites with pre-configured selectors.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
from typing import Any

import pandas as pd
from vnstock_news.core.crawler import Crawler
from vnstock_news.core.batch import BatchCrawler

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
        published_date: dt.datetime | None = None,
        source: str = "",
        tickers: list[str] | None = None,
        primary_ticker: str = "",
        doc_type: str = "news",
    ) -> None:
        self.url = url
        self.title = title
        self.content = content
        self.published_date = published_date
        self.source = source
        self.tickers = tickers or []
        self.primary_ticker = primary_ticker.upper().strip()
        self.doc_type = doc_type
        self.doc_id = hashlib.sha256(url.encode()).hexdigest()[:16]

    @property
    def metadata(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "source": self.source,
            "doc_type": self.doc_type,
            "tickers": ",".join(self.tickers),
            "primary_ticker": self.primary_ticker,
            "published_date": (
                self.published_date.isoformat() if self.published_date else ""
            ),
        }

    def as_embedding_document(self, zone: str = "zone_3") -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "published_date": self.published_date.isoformat() if self.published_date else "",
            "source": self.source,
            "tickers": ",".join(self.tickers),
            "primary_ticker": self.primary_ticker or (self.tickers[0] if self.tickers else ""),
            "url": self.url,
            "doc_type": self.doc_type,
            "zone": zone,
        }

class NewsCrawler:
    """Async web crawler powered by vnstock_news.

    Supported Sites:
    - CafeF, Cafebiz, Vietstock, VnExpress, Tuoi Tre, 
      VnEconomy, Plo, TheSaigonTimes, DienDanDoanhNghiep, BaoDauTu
    """

    TARGET_SITES = [
        'cafef', 'cafebiz', 'vietstock', 'vnexpress', 'tuoitre',
        'vneconomy', 'plo', 'thesaigontimes', 'diendandoanhnghiep', 'baodautu'
    ]

    def __init__(self, concurrency: int = 5) -> None:
        self._settings = get_settings()
        self._semaphore = asyncio.Semaphore(concurrency)

    async def crawl_ticker(
        self,
        ticker: str,
        sources: list[str] | None = None,
        max_pages: int = 5,
    ) -> list[CrawledDocument]:
        """Crawl news for a specific ticker (best effort across sites)."""
        sites = sources or self.TARGET_SITES
        docs: list[CrawledDocument] = []
        
        # Note: vnstock_news primarily works by site, not by ticker tag.
        # We fetch general news from these sites and filter by ticker mentions.
        
        for site in sites:
            try:
                site_docs = await self._crawl_site(site, limit=max_pages, ticker_filter=ticker)
                docs.extend(site_docs)
            except Exception as e:
                logger.debug("site_crawl_error", site=site, ticker=ticker, error=str(e))
        
        # Fallback to vnstock institutional news if scraper found little
        if len(docs) < 2:
            try:
                vn_docs = await self._crawl_via_vnstock(ticker)
                docs.extend(vn_docs)
            except Exception: pass

        logger.info("crawl_ticker_done", ticker=ticker, total_docs=len(docs))
        return docs

    async def _crawl_via_vnstock(self, ticker: str) -> list[CrawledDocument]:
        """Fetch news directly from vnstock API."""
        docs = []
        try:
            from vnstock import Vnstock
            import asyncio
            loop = asyncio.get_running_loop()
            def _fetch():
                stock = Vnstock().stock(symbol=ticker, source="VCI")
                return stock.company.news()
            df = await loop.run_in_executor(None, _fetch)
            if df is not None and not df.empty:
                for _, row in df.head(5).iterrows():
                    title = str(row.get("news_title", ""))
                    docs.append(CrawledDocument(
                        url=f"https://vnstock.tech/{ticker}/{hash(title)}",
                        title=title, content=title, source="vnstock",
                        tickers=[ticker], primary_ticker=ticker
                    ))
        except Exception: pass
        return docs

    async def crawl_watchlist(
        self,
        tickers: list[str],
        max_pages_per_ticker: int = 2,
    ) -> list[CrawledDocument]:
        """Unified crawl for a list of tickers."""
        all_docs = []
        # To avoid over-crawling, we crawl each site once and then map to tickers
        for site in self.TARGET_SITES:
            try:
                # Get more articles per site when doing a batch
                site_docs = await self._crawl_site(site, limit=15)
                # Map to tickers
                for doc in site_docs:
                    matched = [t for t in tickers if t.lower() in doc.content.lower() or t.lower() in doc.title.lower()]
                    if matched:
                        doc.tickers = matched
                        doc.primary_ticker = matched[0]
                        all_docs.append(doc)
            except Exception as e:
                logger.warning("batch_site_error", site=site, error=str(e))
                
        return all_docs

    async def _crawl_site(self, site_name: str, limit: int = 10, ticker_filter: str = None) -> list[CrawledDocument]:
        """Internal helper using vnstock_news Crawler and BatchCrawler."""
        docs = []
        try:
            # 1. Get metadata/URLs using Crawler
            # Using run_in_executor because vnstock_news is primarily synchronous/thread-based
            loop = asyncio.get_running_loop()
            crawler = await loop.run_in_executor(None, lambda: Crawler(site_name=site_name))
            
            # This fetches article links (RSS/Sitemap)
            meta_list = await loop.run_in_executor(None, lambda: crawler.get_articles(limit=limit))
            if not meta_list:
                return []

            # 2. Extract full details using BatchCrawler
            batch = await loop.run_in_executor(None, lambda: BatchCrawler(site_name=site_name))
            df = await loop.run_in_executor(None, lambda: batch.fetch_details_for_urls(urls=[m['url'] for m in meta_list]))
            
            if df is None or df.empty:
                return []

            for _, row in df.iterrows():
                content = str(row.get('markdown_content', '')) or str(row.get('short_description', ''))
                title = str(row.get('title', ''))
                
                # Apply ticker filter if requested
                if ticker_filter and (ticker_filter.lower() not in content.lower() and ticker_filter.lower() not in title.lower()):
                    continue

                pub_time = row.get('publish_time')
                published_date = None
                if pub_time:
                    try:
                        published_date = pd.to_datetime(pub_time).to_pydatetime()
                    except: pass

                docs.append(CrawledDocument(
                    url=str(row.get('url', '')),
                    title=title,
                    content=content,
                    published_date=published_date,
                    source=site_name,
                    tickers=[ticker_filter] if ticker_filter else [],
                    primary_ticker=ticker_filter or "",
                    doc_type="news"
                ))
                
        except Exception as e:
            logger.debug("vnstock_news_core_error", site=site_name, error=str(e))
            
        return docs
