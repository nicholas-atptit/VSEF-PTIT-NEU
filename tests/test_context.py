"""Tests for Module 4: Market Context & Signal DB."""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock, patch

import pytest

from src.context.bctc_loader import BCTCLoader
from src.context.embedder import DocumentEmbedder
from src.context.ingestion_pipeline import IngestionPipeline
from src.context.news_crawler import CrawledDocument, NewsCrawler


class TestCrawledDocument:
    """Test document model."""

    def test_creation(self):
        """Document should be created with all fields."""
        doc = CrawledDocument(
            url="https://example.com/article-1",
            title="HPG Q4 Results",
            content="HPG reported strong Q4 earnings...",
            published_date=dt.datetime(2024, 1, 15),
            source="cafef",
            tickers=["HPG"],
        )
        assert doc.title == "HPG Q4 Results"
        assert doc.source == "cafef"
        assert "HPG" in doc.tickers

    def test_doc_id_generation(self):
        """Document ID should be generated from URL hash."""
        doc = CrawledDocument(
            url="https://example.com/unique-article",
            title="Test",
            content="Content",
        )
        assert len(doc.doc_id) == 16
        assert doc.doc_id.isalnum()

    def test_metadata(self):
        """Metadata should contain all required fields."""
        doc = CrawledDocument(
            url="https://example.com/test",
            title="Test Article",
            content="Test content",
            source="vnexpress",
            tickers=["VIC", "VNM"],
        )
        meta = doc.metadata
        assert meta["source"] == "vnexpress"
        assert "VIC" in meta["tickers"]
        assert "VNM" in meta["tickers"]


class TestNewsCrawler:
    """Test news crawler."""

    def test_sources_configured(self):
        """Crawler should have predefined sources."""
        assert "cafef" in NewsCrawler.SOURCES
        assert "vnexpress" in NewsCrawler.SOURCES

    def test_extract_article_links(self):
        """Should extract article-like links from HTML."""
        from bs4 import BeautifulSoup

        html = """
        <div>
            <a href="/stock/HPG-analysis.chn">Analysis</a>
            <a href="/news/markets.htm">Markets</a>
            <a href="/about">About</a>
        </div>
        """
        soup = BeautifulSoup(html, "lxml")
        links = NewsCrawler._extract_article_links(soup, "https://cafef.vn")

        assert len(links) == 2  # .chn and .htm links
        assert any("analysis.chn" in l for l in links)

    def test_extract_content(self):
        """Should extract main article content."""
        from bs4 import BeautifulSoup

        html = """
        <div class="detail-content">
            <p>This is the article content.</p>
            <p>Second paragraph.</p>
            <script>alert('skip');</script>
        </div>
        """
        soup = BeautifulSoup(html, "lxml")
        content = NewsCrawler._extract_content(soup)

        assert "article content" in content
        assert "Second paragraph" in content
        assert "alert" not in content  # Scripts removed

    def test_detect_tickers(self):
        """Should detect stock ticker mentions in text."""
        content = "HPG reported strong results. VIC and VNM also performed well."
        tickers = NewsCrawler._detect_tickers(content, ["HPG"])

        assert "HPG" in tickers
        assert "VIC" in tickers
        assert "VNM" in tickers

    def test_detect_tickers_excludes_common_words(self):
        """Should not detect common English words as tickers."""
        content = "THE market HAS been volatile FOR investors."
        tickers = NewsCrawler._detect_tickers(content, [])

        assert "THE" not in tickers
        assert "HAS" not in tickers
        assert "FOR" not in tickers


class TestDocumentEmbedder:
    """Test document chunking (embedding requires model)."""

    def test_chunk_small_document(self):
        """Small document should produce one chunk."""
        embedder = DocumentEmbedder(chunk_size=1000, chunk_overlap=50)
        doc = {
            "doc_id": "test123",
            "content": "Short document content.",
            "title": "Test",
            "url": "https://example.com/test",
        }
        chunks = embedder._chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0]["chunk_id"] == "test123_chunk_0"

    def test_chunk_large_document(self):
        """Large document should produce multiple chunks."""
        embedder = DocumentEmbedder(chunk_size=100, chunk_overlap=20)

        long_content = "\n".join([f"Paragraph {i} with some content." for i in range(50)])
        doc = {
            "doc_id": "large_doc",
            "content": long_content,
            "title": "Large Doc",
            "url": "https://example.com/large",
        }
        chunks = embedder._chunk_document(doc)
        assert len(chunks) > 1

    def test_chunk_empty_document(self):
        """Empty document should produce no chunks."""
        embedder = DocumentEmbedder()
        doc = {"doc_id": "empty", "content": ""}
        chunks = embedder._chunk_document(doc)
        assert len(chunks) == 0

    def test_chunk_metadata(self):
        """Chunks should carry document metadata."""
        embedder = DocumentEmbedder(chunk_size=1000)
        doc = {
            "doc_id": "meta_test",
            "content": "Some content here.",
            "title": "Meta Test",
            "source": "cafef",
            "url": "https://cafef.vn/article",
        }
        chunks = embedder._chunk_document(doc)
        assert chunks[0]["metadata"]["source"] == "cafef"
        assert chunks[0]["metadata"]["title"] == "Meta Test"

    def test_build_where_filter(self):
        """Metadata filter should include ticker, zone, and timestamp bounds."""
        embedder = DocumentEmbedder()
        where_filter = embedder._build_where_filter(
            ticker="hpg",
            zone="zone_1",
            published_before="2024-12-31",
            published_after="2024-01-01",
            extra_where=None,
        )

        assert where_filter is not None
        assert {"primary_ticker": {"$eq": "HPG"}} in where_filter["$and"]
        assert {"zone": {"$eq": "zone_1"}} in where_filter["$and"]
        assert any("published_timestamp" in clause for clause in where_filter["$and"])


class TestBCTCLoader:
    def test_load_directory_builds_zone_1_metadata(self, tmp_path):
        """CSV BCTC files should produce Zone 1 embedding documents."""
        csv_path = tmp_path / "HPG_Q3_2024_BCTC.csv"
        csv_path.write_text(
            "metric,value,notes\n"
            "Revenue,1000,Strong quarter with higher export demand\n"
            "Profit,200,Margin expansion after input costs eased\n"
            "Cash Flow,150,Operating cash flow remained positive\n"
            "Assets,5000,Total assets continued to grow\n",
            encoding="utf-8",
        )

        loader = BCTCLoader(data_dir=str(tmp_path))
        documents = loader.load_directory(ticker="HPG", max_files=5)

        assert len(documents) == 1
        document = documents[0]
        assert document["zone"] == "zone_1"
        assert document["primary_ticker"] == "HPG"
        assert document["report_quarter"] == "Q3"
        assert document["report_period"] == "Q3-2024"
        assert document["published_date"] == "2024-09-30"


class TestIngestionPipeline:
    def test_filter_recent_news(self):
        """Only documents inside the lookback window should be embedded."""
        cutoff = dt.datetime(2026, 3, 15, tzinfo=dt.UTC)
        recent_doc = CrawledDocument(
            url="https://example.com/recent",
            title="Recent",
            content="Recent content " * 10,
            published_date=cutoff - dt.timedelta(days=30),
            source="cafef",
            tickers=["HPG"],
            primary_ticker="HPG",
        )
        stale_doc = CrawledDocument(
            url="https://example.com/stale",
            title="Old",
            content="Old content " * 10,
            published_date=cutoff - dt.timedelta(days=1500),
            source="cafef",
            tickers=["HPG"],
            primary_ticker="HPG",
        )

        recent_docs, stale_count = IngestionPipeline._filter_recent_news(
            [recent_doc, stale_doc],
            cutoff,
        )

        assert len(recent_docs) == 1
        assert recent_docs[0].url == "https://example.com/recent"
        assert stale_count == 1


class TestTimeUtils:
    """Test time utility functions."""

    def test_is_weekday(self):
        """Monday should be a weekday."""
        from src.utils.time_utils import is_weekday
        monday = dt.date(2024, 1, 1)  # Monday
        assert is_weekday(monday) is True

    def test_is_weekend(self):
        """Saturday should not be a weekday."""
        from src.utils.time_utils import is_weekday
        saturday = dt.date(2024, 1, 6)  # Saturday
        assert is_weekday(saturday) is False

    def test_is_vn_holiday(self):
        """Known holidays should be detected."""
        from src.utils.time_utils import is_vn_holiday
        new_year = dt.date(2024, 1, 1)
        assert is_vn_holiday(new_year) is True

    def test_not_holiday(self):
        """Regular dates should not be holidays."""
        from src.utils.time_utils import is_vn_holiday
        regular = dt.date(2024, 3, 15)
        assert is_vn_holiday(regular) is False

    def test_trading_session_label(self):
        """Should identify trading session by time."""
        from src.utils.time_utils import trading_session_label
        morning = dt.time(10, 0)
        assert trading_session_label(morning) == "morning"

        afternoon = dt.time(14, 0)
        assert trading_session_label(afternoon) == "afternoon"

        off_hours = dt.time(20, 0)
        assert trading_session_label(off_hours) is None
