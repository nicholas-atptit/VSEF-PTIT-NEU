"""Pydantic schemas v2 — The Unified Agent Contract.

Standardizes outputs for:
1. Technical Agent (Quantitative)
2. Sentiment Agent (Qualitative/LLM)
3. Fusion Agent (Decision Matrix)
4. Risk Agent (Overlay/Veto)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import datetime as dt

# --- Branch 1: Technical Forecast ---

class TechnicalHorizon(BaseModel):
    horizon: str = Field(..., description="short, mid, long")
    trend_probs: Dict[str, float] = Field(..., description="up, sideways, down")
    expected_range: Dict[str, float] = Field(..., description="bottom_10th, median_50th, ceiling_90th")
    confidence: float = Field(..., ge=0.0, le=1.0)
    indicators: Dict[str, Any] = Field(default_factory=dict, description="Key TA values like RSI, MACD")

class TechnicalForecast(BaseModel):
    ticker: str
    timestamp: dt.datetime
    horizons: List[TechnicalHorizon]
    feature_set_version: str = "v4.0"

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

# --- Branch 3: Fusion & Decision ---

class FusionDecision(BaseModel):
    action: str = Field(..., description="BUY, SELL, HOLD, RANGE_TRADE, STAND_ASIDE")
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    agent_weights: Dict[str, float] = Field(..., description="weights used for technical vs sentiment")
    regime_detected: str = Field(default="trend", description="trend, range, event-driven")

# --- Branch 4: Risk & Portfolio Overlay ---

class RiskOverlay(BaseModel):
    position_size_suggestion: float = Field(default=0.0, description="0.0 to 1.0 of portfolio")
    veto_flag: bool = Field(default=False)
    constraints_hit: List[str] = Field(default_factory=list)
    risk_budget_consumed: float = Field(default=0.0)

# --- Top Level: Terminal Payload ---

class TerminalPayload(BaseModel):
    """The master object consumed by TUI Dashboard v5.0."""
    ticker: str
    timestamp: dt.datetime
    
    technical: TechnicalForecast
    sentiment: Optional[SentimentForecast] = None
    fusion: Optional[FusionDecision] = None
    risk: Optional[RiskOverlay] = None
    
    # Audit Trace
    run_id: str
    status: str = "success"
