"""Ingestion pipeline for crawling/loading documents and storing them in ChromaDB."""

from __future__ import annotations

import datetime as dt
from typing import Any

from config.settings import get_settings
from src.context.bctc_loader import BCTCLoader
from src.context.embedder import DocumentEmbedder
from src.context.news_crawler import CrawledDocument, NewsCrawler
from src.utils.logging import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """Orchestrates the Crawl -> Chunk -> Embed -> Insert flow."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._crawler = NewsCrawler(concurrency=3)
        self._bctc_loader = BCTCLoader(data_dir=self._settings.bctc_data_dir)
        self._embedder = DocumentEmbedder(chunk_size=512, chunk_overlap=50)

    async def ingest_news(
        self,
        tickers: list[str],
        max_pages_per_ticker: int = 5,
        lookback_days: int | None = None,
    ) -> dict[str, Any]:
        """Crawl news, filter to the recent window, and embed into ChromaDB."""
        normalized_tickers = [ticker.upper().strip() for ticker in tickers if ticker.strip()]
        logger.info("news_ingestion_started", tickers=normalized_tickers)

        crawled_docs = await self._crawler.crawl_watchlist(
            tickers=normalized_tickers,
            max_pages_per_ticker=max_pages_per_ticker,
        )
        if not crawled_docs:
            return {"status": "no_documents_crawled", "documents_crawled": 0, "chunks_embedded": 0}

        cutoff_days = lookback_days or self._settings.rag_news_lookback_days
        cutoff_dt = dt.datetime.now(dt.UTC) - dt.timedelta(days=cutoff_days)
        recent_docs, stale_count = self._filter_recent_news(crawled_docs, cutoff_dt)
        if not recent_docs:
            return {
                "status": "no_recent_documents_crawled",
                "documents_crawled": len(crawled_docs),
                "documents_embedded": 0,
                "documents_skipped_old": stale_count,
                "chunks_embedded": 0,
                "cutoff_date": cutoff_dt.date().isoformat(),
            }

        doc_dicts = [doc.as_embedding_document(zone="zone_3") for doc in recent_docs]
        total_chunks = self._embedder.embed_documents(doc_dicts, doc_type="news")

        result = {
            "status": "success",
            "tickers": normalized_tickers,
            "documents_crawled": len(crawled_docs),
            "documents_embedded": len(recent_docs),
            "documents_skipped_old": stale_count,
            "chunks_embedded": total_chunks,
            "cutoff_date": cutoff_dt.date().isoformat(),
        }
        logger.info("news_ingestion_done", **result)
        return result

    def ingest_bctc(
        self,
        ticker: str | None = None,
        max_files: int = 50,
    ) -> dict[str, Any]:
        """Load local BCTC reports and embed them as Zone 1 context."""
        normalized_ticker = ticker.upper().strip() if ticker else None
        logger.info("bctc_ingestion_started", ticker=normalized_ticker)

        documents = self._bctc_loader.load_directory(ticker=normalized_ticker, max_files=max_files)
        if not documents:
            return {"status": "no_bctc_files_found", "files_loaded": 0, "chunks_embedded": 0}

        total_chunks = self._embedder.embed_documents(documents, doc_type="report")
        result = {
            "status": "success",
            "ticker_filter": normalized_ticker,
            "files_loaded": len(documents),
            "chunks_embedded": total_chunks,
        }
        logger.info("bctc_ingestion_done", **result)
        return result

    async def ingest_all(
        self,
        tickers: list[str],
        max_news_pages: int = 5,
        max_bctc_files: int = 50,
    ) -> dict[str, Any]:
        """Run the combined news and BCTC ingestion pipeline."""
        news_result = await self.ingest_news(tickers=tickers, max_pages_per_ticker=max_news_pages)
        bctc_result = self.ingest_bctc(max_files=max_bctc_files)
        return {
            "news": news_result,
            "bctc": bctc_result,
            "total_chunks": news_result.get("chunks_embedded", 0) + bctc_result.get("chunks_embedded", 0),
        }

    @staticmethod
    def _filter_recent_news(
        documents: list[CrawledDocument],
        cutoff_dt: dt.datetime,
    ) -> tuple[list[CrawledDocument], int]:
        """Keep only documents inside the recent lookback window."""
        recent_docs: list[CrawledDocument] = []
        stale_count = 0

        for doc in documents:
            if doc.published_date is None:
                recent_docs.append(doc)
                continue

            published_dt = doc.published_date
            if published_dt.tzinfo is None:
                published_dt = published_dt.replace(tzinfo=dt.UTC)

            if published_dt >= cutoff_dt:
                recent_docs.append(doc)
            else:
                stale_count += 1

        return recent_docs, stale_count
