"""Embedding pipeline for chunking, vectorizing, and querying documents."""

from __future__ import annotations

import datetime as dt
import hashlib
from typing import Any

from config.settings import get_settings
from src.data.database.chromadb_client import (
    get_financial_reports_collection,
    get_market_news_collection,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentEmbedder:
    """Chunks documents, stores embeddings in ChromaDB, and runs semantic search."""

    ZONE_BY_DOC_TYPE = {
        "news": "zone_3",
        "report": "zone_1",
        "analysis": "zone_2",
    }

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        self._settings = get_settings()
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._model = None

    def _get_model(self) -> Any:
        """Lazy-load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._settings.embedding_model)
            logger.info("embedding_model_loaded", model=self._settings.embedding_model)
        return self._model

    def embed_documents(
        self,
        documents: list[dict[str, Any]],
        doc_type: str = "news",
    ) -> int:
        """Embed and store a batch of documents."""
        collection = self._get_collection(doc_type)
        model = self._get_model()
        total_chunks = 0

        for raw_doc in documents:
            try:
                doc = self._normalize_document(raw_doc, doc_type=doc_type)
                chunks = self._chunk_document(doc)
                if not chunks:
                    continue

                texts = [chunk["text"] for chunk in chunks]
                embeddings = model.encode(texts).tolist()
                ids = [chunk["chunk_id"] for chunk in chunks]
                metadatas = [chunk["metadata"] for chunk in chunks]

                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )
                total_chunks += len(chunks)
                logger.debug(
                    "doc_embedded",
                    doc_id=doc.get("doc_id", "?"),
                    chunks=len(chunks),
                    zone=doc.get("zone"),
                    ticker=doc.get("primary_ticker"),
                )
            except Exception as exc:
                logger.error(
                    "embed_error",
                    doc_id=raw_doc.get("doc_id", "?"),
                    error=str(exc),
                )

        logger.info(
            "batch_embed_done",
            total_chunks=total_chunks,
            doc_count=len(documents),
            doc_type=doc_type,
        )
        return total_chunks

    def embed_single(self, doc: dict[str, Any], doc_type: str = "news") -> int:
        """Embed and store a single document."""
        return self.embed_documents([doc], doc_type=doc_type)

    def search(
        self,
        query: str,
        doc_type: str = "news",
        n_results: int = 10,
        ticker: str | None = None,
        zone: str | None = None,
        published_before: dt.datetime | dt.date | str | None = None,
        published_after: dt.datetime | dt.date | str | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run semantic similarity search with metadata filters."""
        collection = self._get_collection(doc_type)
        model = self._get_model()
        query_embedding = model.encode([query]).tolist()
        where_filter = self._build_where_filter(
            ticker=ticker,
            zone=zone,
            published_before=published_before,
            published_after=published_after,
            extra_where=where,
        )

        logger.info(
            "semantic_search_started",
            doc_type=doc_type,
            ticker=ticker,
            zone=zone,
            n_results=n_results,
        )
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter,
        )

        formatted: list[dict[str, Any]] = []
        documents = results.get("documents") or []
        if not documents:
            return formatted

        metadatas = results.get("metadatas") or []
        distances = results.get("distances") or []
        ids = results.get("ids") or []

        first_documents = documents[0]
        first_metadatas = metadatas[0] if metadatas else []
        first_distances = distances[0] if distances else []
        first_ids = ids[0] if ids else []

        for index, doc_text in enumerate(first_documents):
            formatted.append(
                {
                    "text": doc_text,
                    "metadata": first_metadatas[index] if index < len(first_metadatas) else {},
                    "distance": first_distances[index] if index < len(first_distances) else None,
                    "id": first_ids[index] if index < len(first_ids) else None,
                }
            )

        logger.info(
            "semantic_search_completed",
            doc_type=doc_type,
            ticker=ticker,
            zone=zone,
            matches=len(formatted),
        )
        return formatted

    def _chunk_document(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        """Split a document using RecursiveCharacterTextSplitter."""
        content = str(doc.get("content", "")).strip()
        if not content:
            return []

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        chunks: list[dict[str, Any]] = []
        for chunk_idx, text in enumerate(splitter.split_text(content)):
            if text.strip():
                chunks.append(self._make_chunk(doc, text, chunk_idx))
        return chunks

    def _normalize_document(self, doc: dict[str, Any], doc_type: str) -> dict[str, Any]:
        """Normalize a document before chunking so metadata is queryable in ChromaDB."""
        normalized = dict(doc)
        normalized_doc_type = str(doc.get("doc_type") or doc_type or "news")
        normalized_zone = str(
            doc.get("zone") or self.ZONE_BY_DOC_TYPE.get(normalized_doc_type, "zone_3")
        )

        tickers = self._normalize_tickers(doc)
        primary_ticker = str(doc.get("primary_ticker") or "").upper().strip()
        if not primary_ticker and tickers:
            primary_ticker = tickers.split(",")[0]

        published_date = self._normalize_datetime(doc.get("published_date"))

        normalized["title"] = str(doc.get("title", "")).strip()
        normalized["content"] = str(doc.get("content", "")).strip()
        normalized["source"] = str(doc.get("source", "")).strip()
        normalized["url"] = str(doc.get("url", "")).strip()
        normalized["doc_type"] = normalized_doc_type
        normalized["zone"] = normalized_zone
        normalized["tickers"] = tickers
        normalized["primary_ticker"] = primary_ticker
        normalized["published_date"] = published_date
        normalized["published_timestamp"] = self._coerce_timestamp(published_date)
        normalized["ingested_at"] = dt.datetime.now(dt.UTC).isoformat()
        return normalized

    @staticmethod
    def _normalize_tickers(doc: dict[str, Any]) -> str:
        """Normalize tickers into a comma-separated uppercase string."""
        tickers_raw = (
            doc.get("tickers")
            or doc.get("ticker")
            or doc.get("primary_ticker")
            or ""
        )

        if isinstance(tickers_raw, str):
            tickers = [item.strip().upper() for item in tickers_raw.split(",") if item.strip()]
        elif isinstance(tickers_raw, (list, tuple, set)):
            tickers = [str(item).strip().upper() for item in tickers_raw if str(item).strip()]
        else:
            tickers = []

        deduped = list(dict.fromkeys(tickers))
        return ",".join(deduped)

    @staticmethod
    def _normalize_datetime(value: dt.datetime | dt.date | str | None) -> str:
        """Normalize dates into ISO-8601 strings."""
        if value is None or value == "":
            return dt.datetime.now(dt.UTC).isoformat()

        if isinstance(value, dt.datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=dt.UTC)
            return value.astimezone(dt.UTC).isoformat()

        if isinstance(value, dt.date):
            normalized = dt.datetime.combine(value, dt.time.min, tzinfo=dt.UTC)
            return normalized.isoformat()

        raw = str(value).strip()
        if not raw:
            return dt.datetime.now(dt.UTC).isoformat()

        try:
            parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.UTC)
            return parsed.astimezone(dt.UTC).isoformat()
        except ValueError:
            try:
                parsed_date = dt.date.fromisoformat(raw[:10])
                return dt.datetime.combine(parsed_date, dt.time.min, tzinfo=dt.UTC).isoformat()
            except ValueError:
                return raw

    @classmethod
    def _coerce_timestamp(cls, value: dt.datetime | dt.date | str | None) -> int:
        """Coerce supported date values to a unix timestamp."""
        normalized = cls._normalize_datetime(value)
        try:
            parsed = dt.datetime.fromisoformat(normalized.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.UTC)
            return int(parsed.timestamp())
        except ValueError:
            return int(dt.datetime.now(dt.UTC).timestamp())

    def _build_where_filter(
        self,
        ticker: str | None,
        zone: str | None,
        published_before: dt.datetime | dt.date | str | None,
        published_after: dt.datetime | dt.date | str | None,
        extra_where: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Build a ChromaDB metadata filter."""
        clauses: list[dict[str, Any]] = []

        if ticker:
            clauses.append({"primary_ticker": {"$eq": ticker.upper().strip()}})
        if zone:
            clauses.append({"zone": {"$eq": zone}})
        if published_before is not None:
            clauses.append({"published_timestamp": {"$lte": self._coerce_timestamp(published_before)}})
        if published_after is not None:
            clauses.append({"published_timestamp": {"$gte": self._coerce_timestamp(published_after)}})
        if extra_where:
            clauses.append(extra_where)

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _make_chunk(doc: dict[str, Any], text: str, chunk_idx: int) -> dict[str, Any]:
        """Create a chunk dictionary with stable metadata."""
        doc_id = doc.get("doc_id") or hashlib.sha256(doc.get("url", "").encode()).hexdigest()[:16]
        chunk_id = f"{doc_id}_chunk_{chunk_idx}"

        metadata: dict[str, Any] = {
            "doc_id": doc_id,
            "title": str(doc.get("title", "")),
            "source": str(doc.get("source", "")),
            "url": str(doc.get("url", "")),
            "tickers": str(doc.get("tickers", "")),
            "primary_ticker": str(doc.get("primary_ticker", "")),
            "published_date": str(doc.get("published_date", "")),
            "published_timestamp": int(doc.get("published_timestamp", 0)),
            "doc_type": str(doc.get("doc_type", "news")),
            "zone": str(doc.get("zone", "")),
            "chunk_idx": int(chunk_idx),
        }

        optional_fields = [
            "report_period",
            "report_year",
            "report_quarter",
            "file_extension",
            "source_type",
        ]
        for field in optional_fields:
            value = doc.get(field)
            if value not in (None, ""):
                metadata[field] = value

        return {
            "chunk_id": chunk_id,
            "text": text,
            "metadata": metadata,
        }

    @staticmethod
    def _get_collection(doc_type: str) -> Any:
        """Return the matching ChromaDB collection for a document type."""
        if doc_type in ("report", "analysis"):
            return get_financial_reports_collection()
        return get_market_news_collection()
