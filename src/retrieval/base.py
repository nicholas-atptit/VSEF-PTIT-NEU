"""Abstract Base Class for Retrieval Adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.retrieval.types import RetrievalIndexRecord, RetrievalQueryResult
from src.retrieval.filters import RetrievalFilters


class RetrievalAdapter(ABC):
    """Base class for all retrieval backends (Vector DB, Local Store, etc.)."""

    @abstractmethod
    def ingest_documents(self, documents: List[RetrievalIndexRecord]) -> Dict[str, Any]:
        """Load documents into the store."""
        pass

    @abstractmethod
    def upsert_documents(self, documents: List[RetrievalIndexRecord]) -> Dict[str, Any]:
        """Insert or update documents."""
        pass

    @abstractmethod
    def delete_documents(self, retrieval_doc_ids: List[str]) -> Dict[str, Any]:
        """Remove documents by ID."""
        pass

    @abstractmethod
    def query(
        self, 
        text: str, 
        filters: Optional[RetrievalFilters] = None, 
        top_k: int = 10
    ) -> List[RetrievalQueryResult]:
        """Perform a search query."""
        pass

    @abstractmethod
    def fetch_by_id(self, retrieval_doc_ids: List[str]) -> List[RetrievalIndexRecord]:
        """Retrieve documents by primary ID."""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Verify backend connectivity and status."""
        pass
