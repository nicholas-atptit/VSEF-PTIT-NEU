"""Canonical Pydantic models for the RAG Preparation Layer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class RetrievalBase(BaseModel):
    """Common fields for all retrieval entities."""
    model_config = ConfigDict(populate_by_name=True)

    schema_version: str = "1.0.0"
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_run_id: str
    source_manifest_path: str
    source_artifact_dir: str


class RetrievalDocument(RetrievalBase):
    """The atomic unit of retrieval (merging text + metadata)."""
    retrieval_doc_id: str
    source_type: str # 'case', 'packet', 'memo'
    source_entity_id: str
    
    # Linked IDs
    source_packet_id: Optional[str] = None
    source_case_id: Optional[str] = None
    source_memo_id: Optional[str] = None
    
    # Content
    title: str
    summary: str
    text: str # Narrativized content for embedding
    
    # Metadata for indexing
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalChunk(RetrievalBase):
    """Sub-units of a RetrievalDocument (for longer docs)."""
    chunk_id: str
    retrieval_doc_id: str
    chunk_index: int
    chunk_text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parent_title: str
    source_type: str


class RetrievalFilterMetadata(BaseModel):
    """Canonical filterable fields for future RAG systems."""
    retrieval_doc_id: str
    source_type: str
    ticker: str
    ticker_group: Optional[str] = None
    sector: Optional[str] = None
    horizon: int
    target_type: str
    regime_label: Optional[str] = None
    volatility_bucket: str = "unknown"
    signal_strength_bucket: str = "unknown"
    agreement_bucket: str = "unknown"
    model_role_context: str = ""
    run_mode: str
    cost_mode: Optional[str] = None
    realized_outcome_label: Optional[str] = None
    date: str
    date_range_bucket: Optional[str] = None


class RetrievalManifest(RetrievalBase):
    """Manifest for a retrieval preparation batch."""
    retrieval_id: str
    source_analysis_feed_manifest: str
    document_count: int
    chunk_count: int
    primary_source_count: int
    secondary_source_count: int
    artifact_paths: Dict[str, str] = Field(default_factory=dict)
