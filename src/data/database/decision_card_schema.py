from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal
import uuid

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class AuditMetadata(BaseModel):
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ticker: str
    timestamp: datetime = Field(default_factory=utc_now)
    provider: str
    model_version: str = "qwen2.5:7b"
    latency_sec: float = Field(default=0.0, ge=0.0)


class DecisionCard(BaseModel):
    """Audit trail payload for a single trading decision."""

    meta: AuditMetadata
    tech_summary: Dict[str, Any]
    news_summary: Dict[str, Any]
    evidence_ids: List[str] = Field(default_factory=list)
    consensus_score: float = Field(default=0.0, ge=0.0, le=1.0)
    regime_label: str = "sideways"
    dynamic_confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    bull_thesis: str
    bear_thesis: str
    risk_veto: bool
    risk_reason: str
    action: Literal["BUY", "SELL", "HOLD"] = Field(description="Final portfolio action")
    target_weight: float = Field(ge=0.0, le=1.0, description="Normalized portfolio allocation")
    execution_shares: int = Field(default=0, ge=0, description="Rounded lot shares for execution API")
    rationale: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
