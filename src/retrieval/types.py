"""Types and models for the Retrieval Adapter Layer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievalIndexRecord(BaseModel):
    """Represents one indexed unit in a vector store or search backend."""
    retrieval_doc_id: str
    source_type: str
    text: str
    summary: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Provider-specific statuses
    embedding_status: str = "pending"
    vector_id: Optional[str] = None
    
    # Provenance (Snapshot of sources)
    source_run_id: str
    source_manifest_path: str
    source_artifact_dir: str


class RetrievalQueryResult(BaseModel):
    """Represents a single retrieval result matched from a query."""
    retrieval_doc_id: str
    score: float
    source_type: str
    title: str
    summary: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Critical Linkage
    source_packet_id: Optional[str] = None
    source_case_id: Optional[str] = None
    
    # Provenance
    source_run_id: str
