"""API v2 routes — The Agentic Domain.

Separates Technical, Sentiment, and Fusion logic into distinct domains.
"""

from __future__ import annotations
import datetime as dt
from fastapi import APIRouter, HTTPException, Query, Request
from src.api.schemas_v2 import (
    TechnicalForecast, TechnicalHorizon,
    SentimentForecast, SentimentSource,
    FusionDecision, RiskOverlay, TerminalPayload
)
from src.ml.trainer import DualModelTrainer
from src.ml.feature_engineering import FeatureEngineer
from src.ml.signal_generator import SignalGenerator
from src.llm.pipeline import run_qualitative_analysis
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v2", tags=["Phase 1 — Agentic Master Plan"])

# Shared instances
_trainer = DualModelTrainer()
_fe = FeatureEngineer()
_signal_gen = SignalGenerator()

@router.get("/predict/technical", response_model=TechnicalForecast)
async def predict_technical(ticker: str = Query(...)) -> TechnicalForecast:
    """Domain A: Quantitative Technical Forecasting."""
    ticker = ticker.upper().strip()
    # Mocking implementation for Phase 1 contract validation
    # In Phase 2, this will call the upgraded trainer with short/mid/long horizons
    horizons = []
    for h in ["short", "mid", "long"]:
        horizons.append(TechnicalHorizon(
            horizon=h,
            trend_probs={"up": 0.6, "sideways": 0.2, "down": 0.2},
            expected_range={"bottom_10th": 30.0, "median_50th": 32.0, "ceiling_90th": 35.0},
            confidence=0.85
        ))
    
    return TechnicalForecast(
        ticker=ticker,
        timestamp=dt.datetime.now(dt.UTC),
        horizons=horizons
    )

@router.get("/predict/sentiment", response_model=SentimentForecast)
async def predict_sentiment(ticker: str = Query(...)) -> SentimentForecast:
    """Domain B: qualitative/LLM Sentiment Analysis."""
    ticker = ticker.upper().strip()
    return SentimentForecast(
        sentiment_score=0.45,
        sentiment_regime="greed",
        sentiment_confidence=0.75,
        source_breakdown=[SentimentSource(source="Vnstock", headline="Positive growth expected", sentiment=0.8, score=0.8)],
        market_psychology_tags=["optimism", "momentum"],
        narrative_risk_flags=[]
    )

@router.get("/predict/fused", response_model=TerminalPayload)
async def predict_fused(ticker: str = Query(...)) -> TerminalPayload:
    """Domain C: Fused Decision Matrix + Risk Overlay."""
    ticker = ticker.upper().strip()
    
    tech = await predict_technical(ticker)
    sent = await predict_sentiment(ticker)
    
    fusion = FusionDecision(
        action="BUY",
        confidence=0.82,
        rationale="Strong technical trend confirmed by positive news sentiment.",
        agent_weights={"technical": 0.6, "sentiment": 0.4}
    )
    
    risk = RiskOverlay(
        position_size_suggestion=0.15,
        veto_flag=False,
        constraints_hit=[],
        risk_budget_consumed=0.15
    )
    
    return TerminalPayload(
        ticker=ticker,
        timestamp=dt.datetime.now(dt.UTC),
        technical=tech,
        sentiment=sent,
        fusion=fusion,
        risk=risk,
        run_id="AGENT-RUN-123"
    )
