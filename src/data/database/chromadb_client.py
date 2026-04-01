"""ChromaDB connection manager for vector storage."""

from __future__ import annotations

import chromadb
from chromadb import Collection

from config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_client: chromadb.HttpClient | None = None

# Collection names
COLLECTION_FINANCIAL_REPORTS = "financial_reports"
COLLECTION_MARKET_NEWS = "market_news"


def get_chroma_client() -> chromadb.HttpClient:
    """Get or create the ChromaDB HTTP client (singleton)."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        logger.info(
            "chromadb_client_created",
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
    return _client


def get_or_create_collection(
    name: str,
    metadata: dict | None = None,
) -> Collection:
    """Get or create a ChromaDB collection.

    Args:
        name: Collection name.
        metadata: Optional metadata dict (e.g., distance function).

    Returns:
        ChromaDB Collection instance.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=name,
        metadata=metadata or {"hnsw:space": "cosine"},
    )
    logger.info("chromadb_collection_ready", collection=name)
    return collection


def get_financial_reports_collection() -> Collection:
    """Get the financial reports collection."""
    return get_or_create_collection(COLLECTION_FINANCIAL_REPORTS)


def get_market_news_collection() -> Collection:
    """Get the market news collection."""
    return get_or_create_collection(COLLECTION_MARKET_NEWS)
