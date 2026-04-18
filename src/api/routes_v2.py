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
from src.engine.agents.orchestrator import TradingOrchestrator
from src.data.database.decision_repository import DecisionRepository
from src.data.database.decision_card_schema import DecisionCard
import os

logger = get_logger(__name__)

# Khởi tạo repository & orchestrator dùng chung
_decision_repo = DecisionRepository()
# Tự động lấy provider từ env hoặc default 'ollama'



# In-memory price cache: { ticker: { price, change, source, ts } }
import time as _time
_price_cache: dict = {}
_PRICE_CACHE_TTL = 5  # seconds

# ── Lightweight Price Endpoint (for 100ms web dashboard polling) ──
@router.get("/price", tags=["Real-time Price"])
async def get_price(ticker: str = "FPT"):
    """Ultra-fast price lookup for the web dashboard (no ML pipeline).

    Uses a 5-second in-memory cache to avoid flooding the DB/vnstock_data
    with requests during 100ms polling from the frontend.
    """
    ticker = ticker.upper().strip()
    now = _time.time()

    # Check in-memory cache first (instant, <1ms)
    cached = _price_cache.get(ticker)
    if cached and (now - cached["ts"]) < _PRICE_CACHE_TTL:
        return {"ticker": ticker, "price": cached["price"], "change": cached["change"], "source": cached["source"]}

    price = 0.0
    change = 0.0
    source = "none"

    # 1. Try Redis (fastest)
    try:
        import json
        from redis.asyncio import Redis
        from config.settings import get_settings
        conf = get_settings()
        r = Redis.from_url(conf.redis_url)
        raw = await r.get(f"live_price:{ticker}")
        await r.close()
        if raw:
            data = json.loads(raw)
            price = float(data.get("price", 0))
            change = float(data.get("change", 0))
            source = "redis"
    except Exception:
        pass

    # 2. Try Database
    if price == 0:
        try:
            from sqlalchemy.ext.asyncio import create_async_engine
            from sqlalchemy import text
            from config.settings import get_settings
            conf = get_settings()
            engine = create_async_engine(conf.timescale_url)
            async with engine.connect() as conn:
                res = await conn.execute(
                    text("SELECT close FROM raw_prices WHERE ticker = :t ORDER BY timestamp DESC LIMIT 2"),
                    {"t": ticker}
                )
                rows = res.fetchall()
                if rows:
                    price = float(rows[0][0])
                    source = "db"
                    if len(rows) >= 2:
                        prev = float(rows[1][0])
                        change = ((price - prev) / prev * 100) if prev > 0 else 0.0
            await engine.dispose()
        except Exception:
            pass

    # 3. Try vnstock_data REST (canonical provider, slowest fallback)
    if price == 0:
        try:
            from src.data.adapters.vnstock_adapter import VnstockAdapter
            import datetime as _dt
            end_d = _dt.date.today()
            start_d = end_d - _dt.timedelta(days=10)
            df = VnstockAdapter().get_ohlcv(
                ticker,
                start_date=start_d.strftime("%Y-%m-%d"),
                end_date=end_d.strftime("%Y-%m-%d"),
                interval="1D",
            )
            if df is not None and not df.empty:
                price = float(df.iloc[-1]["close"])
                source = "vnstock_data"
                if len(df) >= 2:
                    prev = float(df.iloc[-2]["close"])
                    change = ((price - prev) / prev * 100) if prev > 0 else 0.0
        except Exception:
            pass

    # 4. JSON cache fallback
    if price == 0:
        try:
            import json
            from pathlib import Path
            cache_path = Path("data/latest_predictions.json")
            if cache_path.exists():
                with open(cache_path, "r", encoding="utf-8") as f:
                    file_cache = json.load(f)
                td = file_cache.get(ticker, {})
                if td:
                    tech = td.get("technical", {})
                    horizons = tech.get("horizons", [])
                    if horizons:
                        price = horizons[0].get("expected_range", {}).get("median_50th", 0)
                    source = "cache"
        except Exception:
            pass

    # Store in memory cache
    _price_cache[ticker] = {"price": price, "change": change, "source": source, "ts": now}

    return {"ticker": ticker, "price": price, "change": change, "source": source}


@router.get("/market/summary")
async def get_market_summary():
    """Domain D: Universal Market Intelligence Overview (104 Tickers)."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text
    from config.settings import get_settings
    conf = get_settings()
    
    try:
        engine = create_async_engine(conf.timescale_url)
        async with engine.connect() as conn:
            # 1. Prediction Stats
            pred_query = text("SELECT trend, COUNT(*) FROM agent_predictions GROUP BY trend")
            pred_res = await conn.execute(pred_query)
            predictions = {row[0]: row[1] for row in pred_res.fetchall()}
            
            # 2. Sentiment Average
            sent_query = text("SELECT AVG(sentiment_score) FROM news_intelligence WHERE timestamp > NOW() - INTERVAL '24 hours'")
            sent_res = await conn.execute(sent_query)
            avg_sentiment = sent_res.scalar() or 0.0
            
            # 3. Top Buzz Tickers
            buzz_query = text("SELECT ticker, COUNT(*) as news_count FROM news_intelligence WHERE timestamp > NOW() - INTERVAL '24 hours' GROUP BY ticker ORDER BY news_count DESC LIMIT 5")
            buzz_res = await conn.execute(buzz_query)
            top_buzz = [{"ticker": row[0], "count": row[1]} for row in buzz_res.fetchall()]
            
        await engine.dispose()
        
        return {
            "total_tracked": 104,
            "sentiment_24h": float(avg_sentiment),
            "prediction_distribution": predictions,
            "top_buzz": top_buzz,
            "status": "Hybrid Training v4 Active"
        }
    except Exception as e:
        logger.error("market_summary_error", error=str(e))
        return {"error": str(e), "status": "Initializing"}

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
        await _decision_repo.save_decision(decision_card)
        
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
        from src.ml.llm.client import get_llm_client
        from config.settings import get_settings
        client = get_llm_client()
        conf = get_settings()
        logger.info("chat_request_debug", provider=conf.llm_provider, model=conf.ollama_model_name)
        
        system_prompt = "Bạn là trợ lý AI Trading chuyên sâu về chứng khoán Việt Nam (VN30/VN100). Đưa ra phân tích súc tích, chuyên nghiệp."
        
        # Tự động trích xuất Ticker từ tin nhắn người dùng (Regex 3 chữ cái viết hoa)
        import re
        detected_tickers = re.findall(r'\b[A-Z]{3}\b', req.message.upper())
        ticker_to_use = req.ticker or (detected_tickers[0] if detected_tickers else None)
        
        context_str = ""
        if ticker_to_use:
            ticker_to_use = ticker_to_use.upper().strip()
            context_str = f"Cảnh báo: Người dùng đang hỏi về mã {ticker_to_use}.\n"
            
            # 1. Thử lấy từ Database (News Intelligence)
            try:
                from sqlalchemy.ext.asyncio import create_async_engine
                from sqlalchemy import text
                engine = create_async_engine(conf.timescale_url)
                async with engine.connect() as conn:
                    # Lấy tin tức đã phân tích (Đã sửa column name: summary, timestamp)
                    news_query = text("SELECT summary, sentiment_score, trend FROM news_intelligence WHERE ticker = :t ORDER BY timestamp DESC LIMIT 1")
                    news_res = await conn.execute(news_query, {"t": ticker_to_use})
                    news_row = news_res.fetchone()
                    
                    if news_row:
                        context_str += f"[Phân tích Tin tức cho {ticker_to_use}]:\n"
                        context_str += f"- Tóm tắt: {news_row[0]}\n"
                        context_str += f"- Điểm Sentiment: {news_row[1]}\n"
                        context_str += f"- Xu hướng tin tức: {news_row[2]}\n"
                    
                    # Lấy dự báo kỹ thuật (nếu có)
                    quant_query = text("""
                        SELECT
                            ap.trend,
                            GREATEST(ap.probability_up, ap.probability_down) AS confidence,
                            CASE
                                WHEN ap.trend = 'UP' THEN ap.target_ceiling
                                WHEN ap.trend = 'DOWN' THEN ap.target_floor
                                ELSE NULL
                            END AS target_price
                        FROM agent_predictions ap
                        JOIN agent_runs ar ON ar.id = ap.run_id
                        WHERE ar.ticker = :t
                        ORDER BY ap.created_at DESC
                        LIMIT 1
                    """)
                    quant_res = await conn.execute(quant_query, {"t": ticker_to_use})
                    quant_row = quant_res.fetchone()
                    
                    if quant_row:
                        context_str += f"[Phân tích Kỹ thuật cho {ticker_to_use}]:\n"
                        context_str += f"- Dự báo: {quant_row[0]}\n"
                        context_str += f"- Độ tin cậy: {quant_row[1]}\n"
                        context_str += f"- Giá mục tiêu: {quant_row[2]}\n"
                await engine.dispose()
            except Exception as db_err:
                logger.warning("db_context_error", error=str(db_err))
                # Fallback to file-based legacy
                latest_cards = _decision_repo.get_decisions_by_ticker_from_artifacts(ticker_to_use)
                if latest_cards:
                    latest_card = latest_cards[-1]
                    context_str += f"[Dữ liệu Replay]: {latest_card.get('news_summary')}\n"

            if context_str == f"Cảnh báo: Người dùng đang hỏi về mã {ticker_to_use}.\n":
                context_str += f"Hệ thống hiện chưa có dữ liệu phân tích chi tiết cho mã {ticker_to_use}. Hãy trả lời dựa trên kiến thức chung của bạn.\n"
            else:
                context_str += "Hãy dựa vào đúng các dữ liệu thực tế (Phân tích Tin tức/Kỹ thuật) phía trên để trả lời, không được tự bịa ra thông tin.\n"

            
        messages = [{"role": "system", "content": system_prompt + context_str}]
        
        for msg in req.history:
            messages.append({"role": msg.role, "content": msg.content})
            
        messages.append({"role": "user", "content": req.message})

        # Bỏ qua Gemini trong POC này vì Gemini API deprecation, dùng OpenAI tương thích Ollama
        response = await client.chat.completions.create(
            model="qwen3:8b", # User-requested model
            messages=messages,
            temperature=0.3
        )
        answer = response.choices[0].message.content
        return {"response": answer}
        
    except Exception as e:
        logger.error("chat_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
