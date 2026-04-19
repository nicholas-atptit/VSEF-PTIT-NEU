"""Canonical Pydantic models for the Analysis Feed Layer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class AnalysisFeedBase(BaseModel):
    """Common fields for all analysis feed entities."""
    model_config = ConfigDict(populate_by_name=True)

    schema_version: str = "1.0.0"
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_run_id: str
    source_manifest_path: str
    source_artifact_dir: str


class ForecastResearchPacket(AnalysisFeedBase):
    """Represents one ticker x timestamp x horizon x target_type record."""
    packet_id: str
    timestamp: str
    ticker: str
    ticker_group: Optional[str] = None
    horizon: int
    target_type: str
    run_mode: str
    cost_mode: Optional[str] = None

    # Quant Summary
    primary_model_name: str
    primary_model_role: str
    primary_prediction: Optional[float] = None
    model_by_model_predictions: List[Dict[str, Any]] = Field(default_factory=list)
    model_ranks: Dict[str, Any] = Field(default_factory=dict)
    ensemble_summary: Optional[Dict[str, Any]] = None
    
    # Consensus
    model_agreement_score: Optional[float] = None
    model_disagreement_score: Optional[float] = None
    dispersion_score: Optional[float] = None
    sign_conflict: bool = False
    rank_spread: Optional[float] = None
    agreement_bucket: str = "unknown"

    # Environmental Context
    risk_summary: Optional[Dict[str, Any]] = None
    regime_summary: Optional[Dict[str, Any]] = None
    policy_summary: Optional[Dict[str, Any]] = None
    policy_gate_disagreement_share: Optional[float] = None
    volatility_bucket: str = "unknown"
    signal_strength_bucket: str = "unknown"

    # Outcome (Post-validation/Enrichment)
    realized_return: Optional[float] = None
    realized_direction: Optional[int] = None
    realized_volatility: Optional[float] = None
    realized_outcome_label: Optional[str] = None


class HistoricalCaseRecord(AnalysisFeedBase):
    """Represents a retrieval-ready research case for RAG."""
    case_id: str
    source_packet_id: str
    ticker: str
    ticker_group: Optional[str] = None
    horizon: int
    target_type: str
    regime_label: Optional[str] = None
    volatility_bucket: str = "unknown"
    signal_strength_bucket: str = "unknown"
    agreement_bucket: str = "unknown"
    model_role_context: str = ""
    summary_text: str
    run_mode: str = "unknown"
    timestamp: str = ""
    realized_outcome_label: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class AnalystMemoDraft(AnalysisFeedBase):
    """Represents future LLM + human analyst handoff object."""
    memo_id: str
    related_packet_id: str
    ticker: str
    horizon: int
    
    # Deterministic population
    ticker_snapshot: Dict[str, Any] = Field(default_factory=dict)
    primary_signal_summary: str = ""
    model_consensus_summary: str = ""
    model_conflicts: List[str] = Field(default_factory=list)
    regime_summary: str = ""
    risk_summary: str = ""
    suggested_action_candidate: str = ""
    
    # Placeholders for future workflows
    comparable_cases: List[str] = Field(default_factory=list)
    bullish_points: List[str] = Field(default_factory=list)
    bearish_points: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    analyst_notes: str = ""
    final_decision: Optional[str] = None


class RetrievalMetadata(BaseModel):
    """Flat metadata optimized for vector-db / RAG filtering."""
    related_packet_id: str
    related_case_id: str
    related_memo_id: str
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
    date_range: Optional[str] = None
    source_type: str = "quant_core_forecast"


class AnalysisFeedManifest(AnalysisFeedBase):
    """Manifest for a generated analysis feed batch."""
    feed_id: str
    source_quant_core_manifest: str
    packet_count: int
    case_count: int
    memo_count: int
    artifact_paths: Dict[str, str] = Field(default_factory=dict)
