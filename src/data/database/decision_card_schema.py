from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

class AuditMetadata(BaseModel):
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ticker: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provider: str
    model_version: str = "qwen2.5:7b"
    latency_sec: float = 0.0

class DecisionCard(BaseModel):
    """
    Schema lưu trữ toàn bộ lịch sử tư duy (Audit Trail) cho 1 quyết định.
    Gắn với Database (Timescale) và JSON Logs.
    """
    meta: AuditMetadata
    
    # Evidence Snapshots (Technical / News)
    tech_summary: Dict[str, Any]
    news_summary: Dict[str, Any]
    
    # Accuracy Module Info
    evidence_ids: List[str] = Field(default_factory=list)
    consensus_score: float = 0.0
    regime_label: str = "sideways"
    dynamic_confidence_threshold: float = 0.75
    
    # Theses
    bull_thesis: str
    bear_thesis: str
    
    # Risk
    risk_veto: bool
    risk_reason: str
    
    # Final Action
    action: str = Field(description="BUY | SELL | HOLD")
    target_weight: float = Field(description="% Portfolio Allocation")
    execution_shares: int = Field(default=0, description="Rounded lot shares for execution API")
    rationale: str
    confidence: float = 0.0

