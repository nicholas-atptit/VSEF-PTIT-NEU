"""Database models for the Agentic Architecture.

Stores:
- AgentRun: Audit trace for a fusion decision
- AgentPrediction: Multi-horizon technical forecasts
"""

from __future__ import annotations
import datetime as dt
import uuid
from sqlalchemy import Column, String, Float, Boolean, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.ml.models.base import Base, TimestampMixin

class AgentRun(Base, TimestampMixin):
    """Audit trace of a multi-agent fusion run."""
    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    status: Mapped[str] = mapped_column(String(20), default="success")
    
    # Fusion Results
    final_action: Mapped[str] = mapped_column(String(20))
    fusion_confidence: Mapped[float] = mapped_column(Float)
    
    # Risk Assessment
    veto_triggered: Mapped[bool] = mapped_column(Boolean, default=False)
    suggested_allocation: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Raw Payload (Full Audit Trace)
    payload: Mapped[dict] = mapped_column(JSONB)

    # Relationships
    predictions: Mapped[list["AgentPrediction"]] = relationship("AgentPrediction", back_populates="run", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<AgentRun({self.ticker}: {self.final_action} @ {self.created_at})>"

class AgentPrediction(Base, TimestampMixin):
    """Horizon-specific predictions linked to an AgentRun."""
    __tablename__ = "agent_predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    
    horizon: Mapped[str] = mapped_column(String(10)) # 1w, 1m, 6m
    trend: Mapped[str] = mapped_column(String(10)) # UP, DOWN, SIDE
    
    probability_up: Mapped[float] = mapped_column(Float)
    probability_down: Mapped[float] = mapped_column(Float)
    
    target_ceiling: Mapped[float] = mapped_column(Float)
    target_floor: Mapped[float] = mapped_column(Float)
    
    run: Mapped["AgentRun"] = relationship("AgentRun", back_populates="predictions")

    def __repr__(self) -> str:
        return f"<AgentPrediction({self.horizon}: {self.trend})>"
