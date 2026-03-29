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
from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    ticker: Optional[str] = None
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
        
        # Portfolio Target to VN-Market Lot Execution Sizing (100 shares chunk)
        MOCK_PORTFOLIO_VALUE = 1_000_000_000 # 1 Tỷ VND
        import math
        tech_sum = decision_dict.get("tech_summary", {})
        # Dự phòng giá nếu tech_summary không có current_price
        current_price = tech_sum.get("price", 50000) 
        if current_price <= 0: current_price = 50000
        target_wt = decision_dict["target_weight"]
        execution_shares = math.floor(((MOCK_PORTFOLIO_VALUE * target_wt) / current_price) / 100) * 100

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
            "evidence_ids": decision_dict.get("evidence_ids", []),
            "consensus_score": decision_dict.get("consensus_score", 0.0),
            "regime_label": decision_dict.get("regime_label", "sideways"),
            "dynamic_confidence_threshold": decision_dict.get("dynamic_confidence_threshold", 0.75),
            "bull_thesis": decision_dict["bull_thesis"],
            "bear_thesis": decision_dict["bear_thesis"],
            "risk_veto": decision_dict["risk_veto"],
            "risk_reason": decision_dict["risk_reason"],
            "action": decision_dict["action"],
            "target_weight": target_wt,
            "execution_shares": execution_shares,
            "rationale": decision_dict["rationale"],
            "confidence": decision_dict.get("confidence", 0.0)
        }
        
        # Validate bằng Pydantic
        decision_card = DecisionCard(**card_data)
        
        # Save audit trail
        _decision_repo.save_decision(decision_card)
        
        return decision_card
        
    except Exception as e:
        logger.error("debate_error", error=str(e), ticker=ticker)
        raise HTTPException(status_code=500, detail=f"Lỗi khi chạy logic debate: {str(e)}")

@router.post("/chat", tags=["AI Interactive Chat"])
async def chat_interactive(req: ChatRequest):
    """
    Điểm giao tiếp trực tiếp với AI (LLM / Agent). 
    Sử dụng context từ mã chứng khoán (nếu có).
    """
    try:
        from src.llm.client import get_llm_client
        from config.settings import get_settings
        client = get_llm_client()
        conf = get_settings()
        
        system_prompt = "Bạn là trợ lý AI Trading chuyên sâu về chứng khoán Việt Nam (VN30/VN100). Đưa ra phân tích súc tích, chuyên nghiệp."
        
        # Tự động trích xuất Ticker từ tin nhắn người dùng (Regex 3 chữ cái viết hoa)
        import re
        detected_tickers = re.findall(r'\b[A-Z]{3}\b', req.message.upper())
        ticker_to_use = req.ticker or (detected_tickers[0] if detected_tickers else None)
        
        context_str = ""
        if ticker_to_use:
            ticker_to_use = ticker_to_use.upper().strip()
            context_str = f"Cảnh báo: Người dùng đang hỏi về mã {ticker_to_use}.\n"
            
            # Kéo dữ liệu thật từ DB ra cho AI đọc
            latest_cards = _decision_repo.get_decisions_by_ticker(ticker_to_use)
            if latest_cards:
                latest_card = latest_cards[-1]  # Lấy cái mới nhất
                news = latest_card.get('news_summary', 'Chưa có tin tức mới')
                tech = latest_card.get('tech_summary', {})
                price = tech.get('price', 'N/A')
                trend = tech.get('trend', 'N/A')
                
                context_str += f"[Dữ liệu Market Real-time cho {ticker_to_use}]:\n"
                context_str += f"- Giá hiện tại: {price}\n"
                context_str += f"- Xu hướng kỹ thuật: {trend}\n"
                context_str += f"- Tóm tắt Tin Tức gần nhất: {news}\n"
                context_str += "Hãy dựa vào đúng các dữ liệu thực tế này để trả lời, không được tự bịa ra thông tin. Tính toán theo [Dữ liệu Market].\n"
            else:
                context_str += f"Hệ thống hiện chưa có Decision Card cho mã {ticker_to_use} trong phiên hôm nay. Hãy phân tích dựa trên kiến thức thị trường chung của bạn.\n"

            
        messages = [{"role": "system", "content": system_prompt + context_str}]
        
        for msg in req.history:
            messages.append({"role": msg.role, "content": msg.content})
            
        messages.append({"role": "user", "content": req.message})

        # Bỏ qua Gemini trong POC này vì Gemini API deprecation, dùng OpenAI tương thích Ollama
        response = await client.chat.completions.create(
            model=conf.ollama_model_name, # or whatever default
            messages=messages,
            temperature=0.3
        )
        answer = response.choices[0].message.content
        return {"response": answer}
        
    except Exception as e:
        logger.error("chat_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
