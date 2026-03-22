"""News and financial report crawler.

Async web crawler targeting Vietnamese financial news sources
for building the LLM context database.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import re
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

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
        """Convert the crawled document to the embedder's expected payload."""
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

    def __repr__(self) -> str:
        return f"<CrawledDocument({self.title[:50]}... from {self.source})>"


class NewsCrawler:
    """Async web crawler for Vietnamese financial news.

    Targets:
    - CafeF (cafef.vn)
    - VnExpress Kinh Doanh
    - DNSE financial reports
    """

    # Base URLs for news sources
    SOURCES = {
        "cafef": {
            "base_url": "https://cafef.vn",
            "search_pattern": "/tag/{ticker}.chn",
        },
        "vnexpress": {
            "base_url": "https://vnexpress.net",
            "search_pattern": "/tim-kiem?q={ticker}&cate_code=1003",
        },
    }

    def __init__(self, concurrency: int = 5) -> None:
        self._settings = get_settings()
        self._semaphore = asyncio.Semaphore(concurrency)
        self._visited: set[str] = set()
        self._documents: list[CrawledDocument] = []

    async def crawl_ticker(
        self,
        ticker: str,
        sources: list[str] | None = None,
        max_pages: int = 10,
    ) -> list[CrawledDocument]:
        """Crawl news for a specific ticker from all sources."""
        target_sources = sources or list(self.SOURCES.keys())
        docs: list[CrawledDocument] = []

        for source_name in target_sources:
            source_config = self.SOURCES.get(source_name)
            if not source_config:
                continue

            try:
                source_docs = await self._crawl_source(
                    ticker, source_name, source_config, max_pages
                )
                docs.extend(source_docs)
            except Exception as e:
                logger.error(
                    "crawl_source_error",
                    source=source_name,
                    ticker=ticker,
                    error=str(e),
                )

        # Also fetch institutional news using vnstock
        try:
            vnstock_docs = await self._crawl_via_vnstock(ticker)
            docs.extend(vnstock_docs)
        except Exception as e:
            logger.error("vnstock_news_error", ticker=ticker, error=str(e))

        logger.info("crawl_ticker_done", ticker=ticker, total_docs=len(docs))
        return docs

    async def crawl_watchlist(
        self,
        tickers: list[str],
        max_pages_per_ticker: int = 5,
    ) -> list[CrawledDocument]:
        """Crawl news for all tickers in the watchlist."""
        all_docs: list[CrawledDocument] = []

        for ticker in tickers:
            docs = await self.crawl_ticker(ticker, max_pages=max_pages_per_ticker)
            all_docs.extend(docs)
            # Rate limiting between tickers
            await asyncio.sleep(1.0)

        logger.info("crawl_watchlist_done", total_docs=len(all_docs))
        return all_docs

    async def _crawl_source(
        self,
        ticker: str,
        source_name: str,
        source_config: dict,
        max_pages: int,
    ) -> list[CrawledDocument]:
        """Crawl a single source for a ticker."""
        url = source_config["base_url"] + source_config["search_pattern"].format(
            ticker=ticker
        )

        docs = []
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; AlgoTrader/1.0)"},
        ) as client:
            try:
                async with self._semaphore:
                    response = await client.get(url)
                    if response.status_code != 200:
                        return docs

                    soup = BeautifulSoup(response.text, "lxml")
                    articles = self._extract_article_links(soup, source_config["base_url"])

                    for article_url in articles[:max_pages]:
                        if article_url in self._visited:
                            continue
                        self._visited.add(article_url)

                        doc = await self._fetch_article(
                            client, article_url, source_name, ticker
                        )
                        if doc:
                            docs.append(doc)
                        await asyncio.sleep(0.5)

            except Exception as e:
                logger.warning(
                    "source_crawl_error",
                    source=source_name,
                    url=url,
                    error=str(e),
                )

        return docs

    async def _fetch_article(
        self,
        client: httpx.AsyncClient,
        url: str,
        source: str,
        ticker: str,
    ) -> CrawledDocument | None:
        """Fetch and parse a single article."""
        try:
            async with self._semaphore:
                response = await client.get(url)
                if response.status_code != 200:
                    return None

            soup = BeautifulSoup(response.text, "lxml")

            # Extract title
            title = ""
            title_tag = soup.find("h1") or soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)

            # Extract content
            content = self._extract_content(soup)
            if not content or len(content) < 50:
                return None

            # Extract date (best-effort)
            pub_date = self._extract_date(soup)

            # Detect mentioned tickers
            tickers = self._detect_tickers(content, [ticker])

            return CrawledDocument(
                url=url,
                title=title,
                content=content,
                published_date=pub_date,
                source=source,
                tickers=tickers,
                primary_ticker=ticker,
                doc_type="news",
            )

        except Exception as e:
            logger.debug("article_fetch_error", url=url, error=str(e))
            return None

    @staticmethod
    def _extract_article_links(soup: BeautifulSoup, base_url: str) -> list[str]:
        """Extract article links from a listing page."""
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.startswith("http"):
                href = urljoin(base_url, href)
            # Filter for article-like URLs
            if any(
                ext in href
                for ext in [".chn", ".htm", ".html", "/article/", "/post/"]
            ):
                links.append(href)
        return list(dict.fromkeys(links))  # preserve order, deduplicate

    @staticmethod
    def _extract_content(soup: BeautifulSoup) -> str:
        """Extract main article content from HTML."""
        # Try common article content selectors
        content_selectors = [
            {"class": "detail-content"},
            {"class": "fck_detail"},
            {"class": "article-body"},
            {"class": "content-detail"},
            {"id": "main-detail"},
            "article",
        ]

        for selector in content_selectors:
            if isinstance(selector, dict):
                element = soup.find("div", selector)
            else:
                element = soup.find(selector)
            if element:
                # Remove scripts and styles
                for tag in element.find_all(["script", "style", "iframe"]):
                    tag.decompose()
                return element.get_text(separator="\n", strip=True)

        # Fallback: get all paragraph text
        paragraphs = soup.find_all("p")
        return "\n".join(p.get_text(strip=True) for p in paragraphs)

    @staticmethod
    def _extract_date(soup: BeautifulSoup) -> dt.datetime | None:
        """Try to extract publication date from the page."""
        # Check meta tags
        for meta_name in ["article:published_time", "pubdate", "date"]:
            meta = soup.find("meta", {"property": meta_name}) or soup.find(
                "meta", {"name": meta_name}
            )
            if meta and meta.get("content"):
                try:
                    return dt.datetime.fromisoformat(
                        meta["content"].replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass
        return None

    @staticmethod
    def _detect_tickers(content: str, known_tickers: list[str]) -> list[str]:
        """Detect stock ticker mentions in content."""
        tickers = set(known_tickers)
        # Match 3-letter uppercase patterns that could be stock tickers
        pattern = r"\b([A-Z]{3})\b"
        matches = re.findall(pattern, content)
        for match in matches:
            if match in {"THE", "AND", "FOR", "NOT", "ARE", "HAS", "WAS", "HAD"}:
                continue
            tickers.add(match)
        return list(tickers)

    async def _crawl_via_vnstock(self, ticker: str) -> list[CrawledDocument]:
        """Fetch institutional news directly from vnstock API."""
        docs = []
        try:
            from vnstock import Vnstock
            import asyncio
            
            # Use run_in_executor to avoid blocking the event loop 
            # if underlying requests are synchronous
            loop = asyncio.get_running_loop()
            
            def _fetch():
                stock = Vnstock().stock(symbol=ticker, source="VCI")
                return stock.company.news()
                
            df = await loop.run_in_executor(None, _fetch)
            
            if df is None or df.empty:
                return docs
                
            for _, row in df.iterrows():
                # Extract news title and fake a URL if none exists
                title = str(row.get("news_title", ""))
                news_id = str(row.get("id", ""))
                url = f"https://vnstock.tech/news/{ticker}/{news_id}" if news_id else f"https://vnstock.tech/news/{ticker}/{hash(title)}"
                
                # Check if it's already a known url (using ID)
                if url in self._visited:
                    continue
                self._visited.add(url)
                
                # We may not have full content, but we have the title/summary
                content = title
                pub_date = None  # Could parse row.get("publish_date") if exists
                
                docs.append(CrawledDocument(
                    url=url,
                    title=title,
                    content=content,
                    published_date=pub_date,
                    source="vnstock_news",
                    tickers=[ticker],
                    primary_ticker=ticker,
                    doc_type="institutional_news"
                ))
                
        except ImportError:
            logger.warning("vnstock_not_installed_news_skipped")
        except Exception as e:
            logger.debug("vnstock_news_fetch_failed", error=str(e))
            
        return docs
