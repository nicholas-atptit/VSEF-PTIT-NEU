"""Local file-backed retrieval adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.retrieval.base import RetrievalAdapter
from src.retrieval.types import RetrievalIndexRecord, RetrievalQueryResult
from src.retrieval.filters import RetrievalFilters


class FileRetrievalAdapter(RetrievalAdapter):
    """Reference implementation for local development using a JSONL file."""

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        self.store_path = self.index_path / "local_index_store.jsonl"
        self._data: Dict[str, RetrievalIndexRecord] = {}
        self._load()

    def _load(self):
        """Load document store from disk."""
        if not self.store_path.exists():
            return
        
        with open(self.store_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = RetrievalIndexRecord.model_validate_json(line)
                self._data[record.retrieval_doc_id] = record

    def _save(self):
        """Persist document store to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            for record in self._data.values():
                f.write(record.model_dump_json() + "\n")

    def ingest_documents(self, documents: List[RetrievalIndexRecord]) -> Dict[str, Any]:
        """Load documents into the store."""
        for doc in documents:
            if doc.retrieval_doc_id in self._data:
                continue
            self._data[doc.retrieval_doc_id] = doc
        self._save()
        return {"status": "success", "count": len(documents)}

    def upsert_documents(self, documents: List[RetrievalIndexRecord]) -> Dict[str, Any]:
        """Insert or update documents."""
        for doc in documents:
            self._data[doc.retrieval_doc_id] = doc
        self._save()
        return {"status": "success", "count": len(documents)}

    def delete_documents(self, retrieval_doc_ids: List[str]) -> Dict[str, Any]:
        """Remove documents by ID."""
        deleted = 0
        for rid in retrieval_doc_ids:
            if rid in self._data:
                del self._data[rid]
                deleted += 1
        self._save()
        return {"status": "success", "deleted": deleted}

    def query(
        self, 
        text: str, 
        filters: Optional[RetrievalFilters] = None, 
        top_k: int = 10
    ) -> List[RetrievalQueryResult]:
        """Simple lexical search (substring) and metadata filtering."""
        results: List[RetrievalQueryResult] = []
        
        filter_dict = filters.to_dict() if filters else {}

        for record in self._data.values():
            # 1. Metadata Filtering
            match = True
            for k, v in filter_dict.items():
                if record.metadata.get(k) != v and getattr(record, k, None) != v:
                    match = False
                    break
            if not match:
                continue

            # 2. Basic Text Matching (substring search for smoke validation)
            search_text = text.lower()
            if search_text and (search_text not in record.text.lower() and search_text not in record.summary.lower()):
                continue

            # 3. Create Result
            results.append(RetrievalQueryResult(
                retrieval_doc_id=record.retrieval_doc_id,
                score=1.0 if not text else 1.0, # Dummy score
                source_type=record.source_type,
                title=record.text.split("\n")[0][:100], # Guessing title from first line
                summary=record.summary,
                metadata=record.metadata,
                source_packet_id=record.metadata.get("source_packet_id"),
                source_case_id=record.metadata.get("source_case_id"),
                source_run_id=record.source_run_id
            ))

        # Sort by some deterministic criteria if no score
        results.sort(key=lambda x: x.retrieval_doc_id)
        return results[:top_k]

    def fetch_by_id(self, retrieval_doc_ids: List[str]) -> List[RetrievalIndexRecord]:
        """Retrieve documents by primary ID."""
        return [self._data[rid] for rid in retrieval_doc_ids if rid in self._data]

    def health_check(self) -> Dict[str, Any]:
        """Verify backend connectivity and status."""
        return {
            "status": "stable", 
            "backend": "local_file", 
            "document_count": len(self._data),
            "store_path": str(self.store_path)
        }
