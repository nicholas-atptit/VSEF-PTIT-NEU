"""Pydantic schemas v2 - diagnostic research API contract."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --- Branch 1: Technical Forecast ---


class TechnicalHorizon(BaseModel):
    horizon: str = Field(..., description="short, mid, long")
    trend_probs: Dict[str, float] = Field(..., description="up, sideways, down")
    expected_range: Dict[str, float] = Field(
        ...,
        description="bottom_10th, median_50th, ceiling_90th",
    )
    volatility_score: Optional[float] = Field(
        default=None,
        description="Model-predicted volatility for this horizon",
    )
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    indicators: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key technical analysis values like RSI, MACD",
    )


class TechnicalForecast(BaseModel):
    ticker: str
    timestamp: dt.datetime
    current_price: Optional[float] = Field(default=None, description="Observed price at forecast time")
    horizons: List[TechnicalHorizon]
    feature_set_version: str = "v4.0"
    data_provenance: Dict[str, Any] = Field(default_factory=dict)


# --- Branch 2: Sentiment Forecast ---


class SentimentSource(BaseModel):
    source: str
    headline: str
    sentiment: float
    score: float
    timestamp: Optional[dt.datetime] = None


class SentimentForecast(BaseModel):
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    sentiment_regime: str = Field(..., description="fear, greed, neutral, uncertain")
    sentiment_confidence: float = Field(..., ge=0.0, le=1.0)
    source_breakdown: List[SentimentSource]
    market_psychology_tags: List[str]
    narrative_risk_flags: List[str]
    data_provenance: Dict[str, Any] = Field(default_factory=dict)


# --- Branch 3: Fusion and Diagnostic Routing ---


class FusionDecision(BaseModel):
    diagnostic_signal: str = Field(..., description="Directional or range diagnostic label")
    route_decision: str = Field(..., description="Non-authoritative route classification")
    decision_lane: str = Field(..., description="Diagnostic lane used for research review")
    confidence: float = Field(..., ge=0.0, le=1.0)
    diagnostic_summary: str
    review_required: bool = Field(default=True)
    agent_weights: Dict[str, float] = Field(
        ...,
        description="weights used for technical vs sentiment diagnostics",
    )
    regime_detected: str = Field(default="trend", description="trend, range, event-driven")


# --- Branch 4: Risk Diagnostic Overlay ---


class RiskOverlay(BaseModel):
    allocation_candidate_weight: float = Field(
        default=0.0,
        ge=0.0,
        description="Research candidate weight only",
    )
    risk_flag: str = Field(default="review_required")
    review_required: bool = Field(default=True)
    constraints_hit: List[str] = Field(default_factory=list)
    risk_budget_consumed: float = Field(default=0.0)
    risk_control_note: str = Field(
        default="Research diagnostic only; human review required.",
    )
    model_accuracy_1w: Optional[float] = None


# --- Top Level: Terminal Payload ---


class TerminalPayload(BaseModel):
    """Diagnostic research payload exposed by governed API routes."""

    ticker: str
    timestamp: dt.datetime

    technical: TechnicalForecast
    sentiment: Optional[SentimentForecast] = None
    fusion: Optional[FusionDecision] = None
    risk: Optional[RiskOverlay] = None

    # Audit trace
    run_id: str
    status: str = "success"
    candidate_status: str = "diagnostic_only"
    review_required: bool = True
    diagnostic_plan: List[str] = Field(default_factory=list)
    non_authoritative_summary: str = (
        "Research diagnostics only; not financial advice or account-routing authority."
    )
    data_provenance: Dict[str, Any] = Field(default_factory=dict)
