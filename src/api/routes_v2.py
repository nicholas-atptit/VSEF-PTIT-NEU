"""API v2 routes â€” The Agentic Domain.

Separates Technical, Sentiment, and Fusion logic into distinct domains.
"""

from __future__ import annotations
import datetime as dt
from fastapi import APIRouter, HTTPException, Query, Request
from src.api.schemas import LegacyRouteDiagnosticResponse
from src.api.schemas_v2 import (
    TechnicalForecast, TechnicalHorizon,
    SentimentForecast, SentimentSource,
    FusionDecision, RiskOverlay, TerminalPayload
)
from src.core.runtime_mode import RuntimeMode, build_data_provenance, ensure_mock_allowed, normalize_runtime_mode
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

router = APIRouter()


def _resolve_runtime_mode(runtime_mode: str | RuntimeMode | None) -> RuntimeMode:
    try:
        return normalize_runtime_mode(runtime_mode)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _demo_output_provenance(
    runtime_mode: str | RuntimeMode | None,
    *,
    source: str,
    allow_mock_data: bool = False,
) -> dict:
    mode = _resolve_runtime_mode(runtime_mode)
    if mode is RuntimeMode.RESEARCH and not allow_mock_data:
        raise HTTPException(
            status_code=403,
            detail="Mock output requires allow_mock_data=true in research mode",
        )
    try:
        ensure_mock_allowed(mode, explicit_mock=True)
    except RuntimeError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return build_data_provenance(
        source=source,
        uses_mock_data=True,
        fallback_triggered=False,
        runtime_mode=mode,
    )

# Khá»Ÿi táº¡o repository & orchestrator dÃ¹ng chung
_decision_repo = DecisionRepository()
# Tá»± Ä‘á»™ng láº¥y provider tá»« env hoáº·c default 'ollama'



# In-memory price cache: { ticker: { price, change, source, ts } }
import time as _time
_price_cache: dict = {}
_PRICE_CACHE_TTL = 5  # seconds

# â”€â”€ Lightweight Price Endpoint for API clients that poll price data â”€â”€
@router.get("/price", tags=["Real-time Price"])
async def get_price(
    ticker: str = "FPT",
    runtime_mode: str = Query(RuntimeMode.RESEARCH.value, description="Runtime mode: demo, research, or audit"),
):
    """Ultra-fast price lookup for API clients (no ML pipeline).

    Uses a 5-second in-memory cache to avoid flooding the DB/vnstock_data
    with requests during frequent client polling.
    """
    ticker = ticker.upper().strip()
    mode = _resolve_runtime_mode(runtime_mode)
    now = _time.time()

    # Check in-memory cache first (instant, <1ms)
    cached = _price_cache.get(ticker)
    if cached and (now - cached["ts"]) < _PRICE_CACHE_TTL:
        return {
            "ticker": ticker,
            "price": cached["price"],
            "change": cached["change"],
            "source": cached["source"],
            "data_provenance": build_data_provenance(
                source=str(cached["source"]),
                uses_mock_data=False,
                fallback_triggered=False,
                runtime_mode=mode,
            ),
        }

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

    return {
        "ticker": ticker,
        "price": price,
        "change": change,
        "source": source,
        "data_provenance": build_data_provenance(
            source=source,
            uses_mock_data=False,
            fallback_triggered=False,
            runtime_mode=mode,
        ),
    }


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
async def predict_technical(
    ticker: str = Query(...),
    runtime_mode: str = Query(RuntimeMode.DEMO.value, description="Runtime mode: demo, research, or audit"),
    allow_mock_data: bool = Query(False, description="Allow explicit mock output outside demo mode"),
) -> TechnicalForecast:
    """Domain A: Quantitative Technical Forecasting."""
    ticker = ticker.upper().strip()
    provenance = _demo_output_provenance(
        runtime_mode,
        source="routes_v2.predict_technical.demo_static",
        allow_mock_data=allow_mock_data,
    )
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
        horizons=horizons,
        data_provenance=provenance,
    )

@router.get("/predict/sentiment", response_model=SentimentForecast)
async def predict_sentiment(
    ticker: str = Query(...),
    runtime_mode: str = Query(RuntimeMode.DEMO.value, description="Runtime mode: demo, research, or audit"),
    allow_mock_data: bool = Query(False, description="Allow explicit mock output outside demo mode"),
) -> SentimentForecast:
    """Domain B: qualitative/LLM Sentiment Analysis."""
    ticker = ticker.upper().strip()
    provenance = _demo_output_provenance(
        runtime_mode,
        source="routes_v2.predict_sentiment.demo_static",
        allow_mock_data=allow_mock_data,
    )
    return SentimentForecast(
        sentiment_score=0.45,
        sentiment_regime="greed",
        sentiment_confidence=0.75,
        source_breakdown=[SentimentSource(source="Vnstock", headline="Positive growth expected", sentiment=0.8, score=0.8)],
        market_psychology_tags=["optimism", "momentum"],
        narrative_risk_flags=[],
        data_provenance=provenance,
    )

@router.get("/predict/fused", response_model=TerminalPayload)
async def predict_fused(
    ticker: str = Query(...),
    runtime_mode: str = Query(RuntimeMode.DEMO.value, description="Runtime mode: demo, research, or audit"),
    allow_mock_data: bool = Query(False, description="Allow explicit mock output outside demo mode"),
) -> TerminalPayload:
    """Domain C: Fused Decision Matrix + Risk Overlay."""
    ticker = ticker.upper().strip()
    provenance = _demo_output_provenance(
        runtime_mode,
        source="routes_v2.predict_fused.demo_static",
        allow_mock_data=allow_mock_data,
    )
    
    tech = await predict_technical(ticker, runtime_mode=runtime_mode, allow_mock_data=allow_mock_data)
    sent = await predict_sentiment(ticker, runtime_mode=runtime_mode, allow_mock_data=allow_mock_data)
    
    fusion = FusionDecision(
        diagnostic_signal="upward_bias",
        route_decision="demo_route_review",
        decision_lane="fused_demo_diagnostic",
        confidence=0.82,
        diagnostic_summary="Demo fused diagnostic built from static technical and sentiment inputs.",
        review_required=True,
        agent_weights={"technical": 0.6, "sentiment": 0.4}
    )
    
    risk = RiskOverlay(
        allocation_candidate_weight=0.15,
        risk_flag="demo_review",
        review_required=True,
        constraints_hit=[],
        risk_budget_consumed=0.15,
        risk_control_note="Demo diagnostic only; human review required.",
    )
    
    return TerminalPayload(
        ticker=ticker,
        timestamp=dt.datetime.now(dt.UTC),
        technical=tech,
        sentiment=sent,
        fusion=fusion,
        risk=risk,
        run_id="AGENT-RUN-123",
        candidate_status="demo_diagnostic_only",
        review_required=True,
        diagnostic_plan=[
            "Inspect demo provenance before research use.",
            "Compare technical and sentiment diagnostics manually.",
        ],
        non_authoritative_summary="Demo diagnostics only; not financial advice or account-routing authority.",
        data_provenance=provenance,
    )

@router.get("/debate", response_model=LegacyRouteDiagnosticResponse, tags=["Multi-Agent Debate"], deprecated=True)
async def run_debate(
    ticker: str = Query(...),
    runtime_mode: str = Query(RuntimeMode.RESEARCH.value, description="Runtime mode: demo, research, or audit"),
):
    """Return a diagnostic gate response for the retained debate route."""
    ticker = ticker.upper().strip()
    mode = _resolve_runtime_mode(runtime_mode)
    return {
        "route_id": "v2_legacy_debate_route",
        "ticker": ticker,
        "status": "legacy_route_gated",
        "candidate_status": "legacy_diagnostic_only",
        "diagnostic_signal": "review_required",
        "route_decision": "manual_research_review",
        "decision_lane": "legacy_route_review",
        "risk_flag": "review_required",
        "review_required": True,
        "diagnostic_summary": (
            "Retained debate route is gated for research diagnostics only and returns "
            "no account-routing payload."
        ),
        "diagnostic_plan": [
            "Use /predict/fused for demo diagnostics.",
            "Review provenance before any downstream workflow.",
        ],
        "data_provenance": build_data_provenance(
            source="legacy_governance_gate",
            uses_mock_data=False,
            fallback_triggered=False,
            runtime_mode=mode,
        ),
    }

@router.post("/chat", tags=["AI Interactive Chat"])
async def chat_interactive(req: ChatRequest):
    """
    Äiá»ƒm giao tiáº¿p trá»±c tiáº¿p vá»›i AI (LLM / Agent). 
    Sá»­ dá»¥ng context tá»« mÃ£ chá»©ng khoÃ¡n (náº¿u cÃ³).
    """
    try:
        from src.ml.llm.client import get_llm_client
        from config.settings import get_settings
        client = get_llm_client()
        conf = get_settings()
        logger.info("chat_request_debug", provider=conf.llm_provider, model=conf.ollama_model_name)
        
        system_prompt = "Báº¡n lÃ  trá»£ lÃ½ AI Trading chuyÃªn sÃ¢u vá» chá»©ng khoÃ¡n Viá»‡t Nam (VN30/VN100). ÄÆ°a ra phÃ¢n tÃ­ch sÃºc tÃ­ch, chuyÃªn nghiá»‡p."
        
        # Tá»± Ä‘á»™ng trÃ­ch xuáº¥t Ticker tá»« tin nháº¯n ngÆ°á»i dÃ¹ng (Regex 3 chá»¯ cÃ¡i viáº¿t hoa)
        import re
        detected_tickers = re.findall(r'\b[A-Z]{3}\b', req.message.upper())
        ticker_to_use = req.ticker or (detected_tickers[0] if detected_tickers else None)
        
        context_str = ""
        if ticker_to_use:
            ticker_to_use = ticker_to_use.upper().strip()
            context_str = f"Cáº£nh bÃ¡o: NgÆ°á»i dÃ¹ng Ä‘ang há»i vá» mÃ£ {ticker_to_use}.\n"
            
            # 1. Thá»­ láº¥y tá»« Database (News Intelligence)
            try:
                from sqlalchemy.ext.asyncio import create_async_engine
                from sqlalchemy import text
                engine = create_async_engine(conf.timescale_url)
                async with engine.connect() as conn:
                    # Láº¥y tin tá»©c Ä‘Ã£ phÃ¢n tÃ­ch (ÄÃ£ sá»­a column name: summary, timestamp)
                    news_query = text("SELECT summary, sentiment_score, trend FROM news_intelligence WHERE ticker = :t ORDER BY timestamp DESC LIMIT 1")
                    news_res = await conn.execute(news_query, {"t": ticker_to_use})
                    news_row = news_res.fetchone()
                    
                    if news_row:
                        context_str += f"[PhÃ¢n tÃ­ch Tin tá»©c cho {ticker_to_use}]:\n"
                        context_str += f"- TÃ³m táº¯t: {news_row[0]}\n"
                        context_str += f"- Äiá»ƒm Sentiment: {news_row[1]}\n"
                        context_str += f"- Xu hÆ°á»›ng tin tá»©c: {news_row[2]}\n"
                    
                    # Láº¥y dá»± bÃ¡o ká»¹ thuáº­t (náº¿u cÃ³)
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
                        context_str += f"[PhÃ¢n tÃ­ch Ká»¹ thuáº­t cho {ticker_to_use}]:\n"
                        context_str += f"- Dá»± bÃ¡o: {quant_row[0]}\n"
                        context_str += f"- Äá»™ tin cáº­y: {quant_row[1]}\n"
                        context_str += f"- GiÃ¡ má»¥c tiÃªu: {quant_row[2]}\n"
                await engine.dispose()
            except Exception as db_err:
                logger.warning("db_context_error", error=str(db_err))
                # Fallback to file-based legacy
                latest_cards = _decision_repo.get_decisions_by_ticker_from_artifacts(ticker_to_use)
                if latest_cards:
                    latest_card = latest_cards[-1]
                    context_str += f"[Dá»¯ liá»‡u Replay]: {latest_card.get('news_summary')}\n"

            if context_str == f"Cáº£nh bÃ¡o: NgÆ°á»i dÃ¹ng Ä‘ang há»i vá» mÃ£ {ticker_to_use}.\n":
                context_str += f"Há»‡ thá»‘ng hiá»‡n chÆ°a cÃ³ dá»¯ liá»‡u phÃ¢n tÃ­ch chi tiáº¿t cho mÃ£ {ticker_to_use}. HÃ£y tráº£ lá»i dá»±a trÃªn kiáº¿n thá»©c chung cá»§a báº¡n.\n"
            else:
                context_str += "HÃ£y dá»±a vÃ o Ä‘Ãºng cÃ¡c dá»¯ liá»‡u thá»±c táº¿ (PhÃ¢n tÃ­ch Tin tá»©c/Ká»¹ thuáº­t) phÃ­a trÃªn Ä‘á»ƒ tráº£ lá»i, khÃ´ng Ä‘Æ°á»£c tá»± bá»‹a ra thÃ´ng tin.\n"

            
        system_prompt += (
            " Governance boundary: research diagnostics only; no financial advice; "
            "no BUY/SELL recommendation authority; no trade execution instructions; "
            "no broker authority; no order authority."
        )
        messages = [{"role": "system", "content": system_prompt + context_str}]
        
        for msg in req.history:
            messages.append({"role": msg.role, "content": msg.content})
            
        messages.append({"role": "user", "content": req.message})

        # Bá» qua Gemini trong POC nÃ y vÃ¬ Gemini API deprecation, dÃ¹ng OpenAI tÆ°Æ¡ng thÃ­ch Ollama
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

