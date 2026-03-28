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
from src.utils.logging import get_logger
from src.agents.orchestrator import TradingOrchestrator
from src.database.decision_repository import DecisionRepository
from src.database.decision_card_schema import DecisionCard
import os

logger = get_logger(__name__)

# Khởi tạo repository & orchestrator dùng chung
_decision_repo = DecisionRepository()
# Tự động lấy provider từ env hoặc default 'ollama'
_orchestrator = TradingOrchestrator(use_llm_provider=os.getenv('LLM_PROVIDER', 'ollama'))

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

@router.get("/debate", tags=["Multi-Agent Debate"])
async def run_debate(ticker: str = Query(...)):
    """
    Thực thi luồng Multi-Agent đầy đủ:
    Technical/News -> Bull/Bear Debate -> Risk Veto -> Portfolio Allocation.
    Trả về cấu trúc Decision Card và lưu lại Audit Trail.
    """
    ticker = ticker.upper().strip()
    
    try:
        # Gọi Orchestrator thực thi graph
        decision_dict = await _orchestrator.execute_debate(ticker)
        
        # Format lại data chuẩn bị parse bằng pydantic schema
        card_data = {
            "meta": {
                "decision_id": decision_dict["decision_id"],
                "ticker": decision_dict["ticker"],
                "provider": decision_dict["provider"],
                "latency_sec": decision_dict["latency_sec"]
            },
            "tech_summary": decision_dict["tech_summary"],
            "news_summary": decision_dict["news_summary"],
            "bull_thesis": decision_dict["bull_thesis"],
            "bear_thesis": decision_dict["bear_thesis"],
            "risk_veto": decision_dict["risk_veto"],
            "risk_reason": decision_dict["risk_reason"],
            "action": decision_dict["action"],
            "target_weight": decision_dict["target_weight"],
            "rationale": decision_dict["rationale"]
        }
        
        # Validate bằng Pydantic
        decision_card = DecisionCard(**card_data)
        
        # Save audit trail
        _decision_repo.save_decision(decision_card)
        
        return decision_card
        
    except Exception as e:
        logger.error("debate_error", error=str(e), ticker=ticker)
        raise HTTPException(status_code=500, detail=f"Lỗi khi chạy logic debate: {str(e)}")
