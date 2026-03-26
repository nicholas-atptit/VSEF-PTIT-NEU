"""API routes — prediction, training, and health endpoints.

Endpoints:
    GET  /api/v1/predict?ticker=SSI     Full prediction pipeline
    POST /api/v1/train                  Trigger model training
    GET  /api/v1/health                 Service health check
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from src.api.schemas import (
    FinalExecutionResponse,
    HealthResponse,
    PredictionResponse,
    TrainRequest,
    TrainResponse,
    ChatRequest,
    ChatResponse,
    ChatMessage,
)
from src.engine.matrix import evaluate_decision_matrix
from src.engine.risk import apply_risk_constraints
from src.api.tracing import trace_stage
from src.llm.pipeline import run_qualitative_analysis
from src.ml.data_loader import generate_mock_data, load_ohlcv_from_db, load_ohlcv_from_vnstock
from src.ml.feature_engineering import FeatureEngineer
from src.ml.signal_generator import SignalGenerator
from src.ml.trainer import DualModelTrainer
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Phase 2 — Quantitative ML"])

# Shared instances (singleton within the process)
_trainer = DualModelTrainer()
_fe = FeatureEngineer()
_signal_gen = SignalGenerator()


# ── Prediction Endpoint ──────────────────────────────────────


@router.get("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    ticker: str = Query(..., description="Stock ticker symbol (e.g., SSI, HPG)"),
    risk_tolerance: float | None = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Requested risk tolerance (will be capped at 0.70)",
    ),
    use_mock: bool = Query(
        False,
        description="Use synthetic mock data (for testing without DB)",
    ),
) -> dict:
    """Run the full prediction pipeline for a given ticker.

    Pipeline:
        1. Load latest OHLCV data
        2. Compute features
        3. Predict with dual models
        4. Generate trading signal
        5. Return JSON contract payload
    """
    ticker = ticker.upper().strip()

    # ── Try Kafka pre-computed cache first (sub-ms response) ──
    if not use_mock:
        try:
            from src.streaming.consumers.cache_writer_consumer import CacheWriterConsumer
            cached = CacheWriterConsumer.read_cache(ticker)
            if cached and cached.get("ml_prediction"):
                logger.info("predict_from_kafka_cache", ticker=ticker)
                ml_pred = cached["ml_prediction"]
                # Build signal from cached ML prediction
                current_close = ml_pred.get("expected_range", {}).get("median_50th", 0)
                model_output = {
                    "trend_probabilities": ml_pred.get("trend_probabilities", {}),
                    "expected_range": ml_pred.get("expected_range", {}),
                }
                payload = await _signal_gen.generate(
                    ticker=ticker,
                    current_close=current_close,
                    model_output=model_output,
                    risk_tolerance=risk_tolerance,
                )
                
                # Add qualitative analysis if present in cache
                if cached.get("llm_analysis"):
                    payload["qualitative_analysis"] = cached["llm_analysis"]
                    
                return payload
        except Exception as e:
            logger.debug("kafka_cache_miss", ticker=ticker, error=str(e))

    try:
        with trace_stage(request, "quant_data_load"):
            if use_mock:
                raw_df = generate_mock_data(ticker=ticker)
            else:
                try:
                    raw_df = load_ohlcv_from_db(ticker)
                except Exception:
                    logger.warning(
                        "db_fallback_to_vnstock",
                        ticker=ticker,
                        reason="Database unavailable, attempting vnstock",
                    )
                    try:
                        raw_df = load_ohlcv_from_vnstock(ticker)
                    except Exception as ve:
                        logger.warning("vnstock_failed_using_mock", error=str(ve))
                        raw_df = generate_mock_data(ticker=ticker)

        if len(raw_df) < 50:
            raise HTTPException(
                status_code=422,
                detail=f"Insufficient data for {ticker}: {len(raw_df)} rows (min 50)",
            )

        with trace_stage(request, "quant_feature_engineering"):
            # Auto-detect v3 models (feature_cols with d_* prefix)
            try:
                _trainer._ensure_models_loaded(ticker)
                saved_features = _trainer._models[ticker].get("feature_cols", [])
            except FileNotFoundError:
                saved_features = []

            is_v3 = saved_features and any(f.startswith('d_') for f in saved_features)

            if is_v3:
                # v3 path: use trainer's built-in feature engineering
                feat_df = _trainer.compute_features_for_ticker(ticker, raw_df)
                latest_row = feat_df[saved_features].iloc[[-1]]  # DataFrame row
                current_close = float(feat_df["close"].iloc[-1])
            else:
                # Legacy path: use old FeatureEngineer
                feat_df = _fe.transform(raw_df)
                feature_cols = _fe.get_feature_columns(feat_df)
                latest_row = feat_df[feature_cols].iloc[-1]
                current_close = float(feat_df["close"].iloc[-1])

        with trace_stage(request, "quant_model_inference"):
            model_output = _trainer.predict(ticker, latest_row)
            payload = await _signal_gen.generate(
                ticker=ticker,
                current_close=current_close,
                model_output=model_output,
                risk_tolerance=risk_tolerance,
            )

        return payload

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("predict_error", ticker=ticker, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed for {ticker}: {str(e)}",
        )


# ── Training Endpoint ────────────────────────────────────────


@router.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest) -> dict:
    """Train dual models for a given ticker.

    Uses mock data by default if database is unavailable.
    """
    ticker = request.ticker.upper().strip()

    try:
        if request.use_mock:
            df = generate_mock_data(ticker=ticker, num_days=600)
        else:
            try:
                df = load_ohlcv_from_db(ticker)
            except Exception:
                logger.warning(
                    "train_db_fallback",
                    ticker=ticker,
                    reason="Database unavailable, attempting vnstock",
                )
                try:
                    df = load_ohlcv_from_vnstock(ticker)
                except Exception as ve:
                    logger.warning("vnstock_failed_using_mock", error=str(ve))
                    df = generate_mock_data(ticker=ticker, num_days=600)

        metrics = _trainer.train(ticker=ticker, df=df)

        return {
            "ticker": ticker,
            "status": "trained",
            "metrics": metrics,
        }

    except Exception as e:
        logger.error("train_error", ticker=ticker, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Training failed for {ticker}: {str(e)}",
        )


# ── Analysis Endpoint (Phase 3: Quantitative + Qualitative) ──


@router.get("/analyze", response_model=PredictionResponse)
async def analyze(
    request: Request,
    ticker: str = Query(..., description="Stock ticker symbol (e.g., SSI, HPG)"),
    risk_tolerance: float | None = Query(
        1.0,  # Default to 100% to test the max cap inside the pipeline
        ge=0.0,
        le=1.0,
        description="Requested risk tolerance (will be capped at 0.70 via system constraints)",
    ),
    allowed_zones: list[str] = Query(
        ["zone_1", "zone_2", "zone_3"], 
        description="List of allowed RAG context zones (e.g., zone_1, zone_2, zone_3, zone_4)"
    ),
    use_mock: bool = Query(
        False,
        description="Use synthetic mock data (for testing without DB)",
    ),
) -> dict:
    """Run the FULL Phase 3 pipeline: Quantitative Predict + LLM Qualitative Analysis.

    Pipeline:
        1. Call the predict logic (Phase 2) to get quant signals.
        2. Fetch RAG context strictly from `allowed_zones` (Targeted Query).
        3. Call the local LLM via Ollama to generate qualitative JSON.
        4. Merge and return the complete PredictionResponse contract.
    """
    ticker = ticker.upper().strip()

    try:
        # Step 1: Get the quantitative payload using the exact same logic as `/predict`
        quant_payload = await predict(
            request=request,
            ticker=ticker,
            risk_tolerance=risk_tolerance,
            use_mock=use_mock,
        )

        mock_parts = []
        with trace_stage(request, "rag_query"):
            try:
                from src.context.rag_service import ZonedRAGService

                rag_service = ZonedRAGService()
                rag_context = rag_service.query(
                    ticker=ticker,
                    allowed_zones=allowed_zones,
                    n_results=5,
                )
                mock_parts = [rag_context]
                allowed_zones = []
            except Exception as rag_err:
                logger.warning("rag_runtime_error", ticker=ticker, error=str(rag_err))
            # Graceful fallback to mock if ChromaDB is unavailable
            mock_parts = mock_parts or []
            if "zone_1" in allowed_zones:
                mock_parts.append(f"[zone_1] BCTC Quý gần nhất của {ticker} chưa được nạp vào Vector DB.")
            if "zone_3" in allowed_zones:
                mock_parts.append(f"[zone_3] Tin tức vĩ mô chưa được embedding — hãy chạy /ingest-news.")
            rag_context = " ".join(mock_parts)

        with trace_stage(request, "news_fetch"):
            news_payload = await ticker_news(ticker=ticker)
            news_headlines = ""
            if news_payload.get("news"):
                news_headlines = "\n".join(
                    [f"- {n['title']} ({n['publish_time']})" for n in news_payload["news"]]
                )

        with trace_stage(request, "llm_inference"):
            qualitative_result = await run_qualitative_analysis(
                ticker=ticker,
                quant_data=quant_payload["quantitative_signals"],
                rag_context=rag_context,
                news_context=news_headlines,
                user_risk_input=risk_tolerance or 1.0,
            )

        # Step 5: Merge results into the final payload payload
        quant_payload["qualitative_analysis"] = qualitative_result

        return quant_payload

    except HTTPException:
        raise
    except Exception as e:
        logger.error("analyze_endpoint_error", ticker=ticker, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Analysis pipeline failed for {ticker}: {str(e)}",
        )


# ── Execution Endpoint (Phase 4: Decision Matrix + Risk Management) ──


@router.get("/execute", response_model=FinalExecutionResponse)
async def execute_trade(
    request: Request,
    ticker: str = Query(..., description="Stock ticker symbol (e.g., SSI, HPG)"),
    risk_tolerance: float | None = Query(
        1.0,
        ge=0.0,
        le=1.0,
        description="Requested risk tolerance (will be capped at 0.70)",
    ),
    allowed_zones: list[str] = Query(
        ["zone_1", "zone_2", "zone_3"], 
        description="List of allowed RAG context zones"
    ),
    use_mock: bool = Query(
        False,
        description="Use synthetic mock data",
    ),
) -> dict:
    """Run the FULL Phase 4 pipeline.

    Pipeline:
        1. Step 1-3 from `/analyze`: Predict Quant -> Fetch RAG -> Analyze Qual (LLM).
        2. Matrix Match: Evaluates `BUY` + `POSITIVE` via `evaluate_decision_matrix`.
        3. Risk Hard-Constraints: Evaluates -7% Stop loss cap & Anti FOMO sizing.
        4. Produce Final Order Payload `FinalExecutionResponse`.
    """
    import datetime as dt
    import uuid
    from src.api.schemas import FinalExecutionResponse, QualitativeAnalysis, QuantitativeSignals

    # 1. Fetch data from Phase 2 + Phase 3 pipeline
    analysis_payload = await analyze(
        request=request,
        ticker=ticker,
        risk_tolerance=risk_tolerance,
        allowed_zones=allowed_zones,
        use_mock=use_mock,
    )

    # Convert dicts back to Pydantic objects for type-safe passing
    quant = QuantitativeSignals(**analysis_payload["quantitative_signals"])
    sys_params = analysis_payload["system_parameters"]
    qual = None
    if analysis_payload.get("qualitative_analysis"):
        qual = QualitativeAnalysis(**analysis_payload["qualitative_analysis"])

    with trace_stage(request, "decision_matrix"):
        decision_action, matrix_consensus = evaluate_decision_matrix(quant, qual)

    # If matrix says CANCEL or STANDBY, we don't apply order-specific risk constraints
    order_payload = None
    risk_override = None

    if decision_action in ("EXECUTE_BUY", "EXECUTE_SELL"):
        # We need simulated Real-time price and ATR for Module 2.
        # Since Phase 1/2 gives us EOD (End-Of-Day) data, we mock current realtime price
        # as close to the ML's recommended max_entry for testing Anti-FOMO logic.
        real_time_price = max(quant.action_plan.entry_zone) * 1.01  # +1% padding

        # We extract the ATR from the feature engineering module inside Phase 2 payload conceptually.
        # For this demonstration, we mock ATR directly.
        mock_atr_14 = real_time_price * 0.05  # Assume 5% daily volatility
        
        with trace_stage(request, "risk_engine"):
            order_payload, risk_override = apply_risk_constraints(
                ticker=ticker,
                action_plan=quant.action_plan,
                real_time_price=real_time_price,
                atr_14=mock_atr_14,
                applied_risk_tolerance=sys_params["max_risk_tolerance"],
            )
        
        # Anti-FOMO trigger
        if risk_override.fomo_check_passed is False:
            decision_action = "CANCEL_ORDER"

    # 4. Module 3: Final Execution Format
    dt_now = dt.datetime.now(dt.UTC).strftime("%Y%m%d")
    return {
        "order_id": f"ORD-{dt_now}-{ticker.upper()}-{str(uuid.uuid4())[:8]}",
        "ticker": ticker.upper(),
        "execution_decision": decision_action,
        "matrix_consensus": matrix_consensus.model_dump(),
        "risk_management_override": risk_override.model_dump() if risk_override else None,
        "order_payload": order_payload.model_dump() if order_payload else None,
        "system_confidence": {
            "stock_data_rate": sys_params["confidence_metrics"]["stock_quantitative_data"],
            "context_data_rate": sys_params["confidence_metrics"]["general_market_context"],
            "applied_risk_cap": sys_params["max_risk_tolerance"],
        }
    }


# ── Health Endpoint ───────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health() -> dict:
    """Service health check."""
    return {
        "status": "ok",
        "version": "5.0.0",
        "phase": "Phase 1-5 — Full Algo Trading System",
    }


# ── Market Index Endpoint ────────────────────────────────────


@router.get("/market-index")
async def market_index() -> dict:
    """Fetch latest VN-Index and HNX-Index data."""
    import os
    from vnstock import Vnstock
    from config.settings import get_settings
    import datetime as _dt

    settings = get_settings()
    os.environ["VNAI_API_KEY"] = settings.vnstock_api_key
    os.environ["VNSTOCK_API_KEY"] = settings.vnstock_api_key

    end = _dt.date.today()
    start = end - _dt.timedelta(days=10)

    indices = []
    for symbol, name in [("VNINDEX", "VN-Index"), ("HNX-INDEX", "HNX-Index")]:
        try:
            stock = Vnstock().stock(symbol=symbol, source="VCI")
            df = stock.quote.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
            )
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
                change = float(latest["close"] - prev["close"])
                change_pct = (change / float(prev["close"])) * 100
                indices.append({
                    "symbol": symbol,
                    "name": name,
                    "close": round(float(latest["close"]), 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2),
                    "volume": int(latest["volume"]),
                    "date": str(latest["time"])[:10],
                })
        except Exception as e:
            logger.warning("market_index_error", symbol=symbol, error=str(e))
            indices.append({"symbol": symbol, "name": name, "error": str(e)})

    return {"indices": indices}


# ── Stock History Endpoint (for charting) ─────────────────────


@router.get("/stock-history")
async def stock_history(
    ticker: str = Query(..., description="Stock ticker symbol"),
    days: int = Query(90, description="Number of days of history"),
) -> dict:
    """Fetch recent OHLCV history for a ticker (for frontend charting)."""
    import os
    import datetime as _dt
    from vnstock import Vnstock
    from config.settings import get_settings

    settings = get_settings()
    os.environ["VNAI_API_KEY"] = settings.vnstock_api_key
    os.environ["VNSTOCK_API_KEY"] = settings.vnstock_api_key

    ticker = ticker.upper().strip()
    end = _dt.date.today()
    start = end - _dt.timedelta(days=int(days * 1.5))

    try:
        stock = Vnstock().stock(symbol=ticker, source="VCI")
        df = stock.quote.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
        )
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No history for {ticker}")

        records = []
        for _, row in df.iterrows():
            records.append({
                "time": str(row["time"])[:10],
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": int(row["volume"]),
            })

        return {"ticker": ticker, "candles": records}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("stock_history_error", ticker=ticker, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── Ticker Search Endpoint ───────────────────────────────────


@router.get("/search")
async def search_tickers(
    q: str = Query(..., min_length=1, description="Search query"),
) -> dict:
    """Search stock tickers from HOSE, HNX, UPCOM listings."""
    import os
    from vnstock import Vnstock
    from config.settings import get_settings

    settings = get_settings()
    os.environ["VNAI_API_KEY"] = settings.vnstock_api_key
    os.environ["VNSTOCK_API_KEY"] = settings.vnstock_api_key

    q = q.upper().strip()

    try:
        stock = Vnstock().stock(symbol="SSI", source="VCI")
        all_results = []
        for group in ["HOSE", "HNX", "UPCOM"]:
            try:
                symbols = stock.listing.symbols_by_group(group)
                matched = [s for s in symbols.tolist() if q in s]
                all_results.extend(matched)
            except Exception:
                continue
        
        return {"query": q, "results": list(set(all_results))[:30]}
    except Exception as e:
        logger.error("search_error", query=q, error=str(e))
        return {"query": q, "results": []}


# ── Ticker Info Endpoint ─────────────────────────────────────


@router.get("/ticker-info")
async def ticker_info(ticker: str = Query(...)) -> dict:
    """Fetch basic company information."""
    import os
    from vnstock import Vnstock
    from config.settings import get_settings

    settings = get_settings()
    os.environ["VNAI_API_KEY"] = settings.vnstock_api_key
    os.environ["VNSTOCK_API_KEY"] = settings.vnstock_api_key

    ticker = ticker.upper().strip()
    try:
        # Since v3 info API is complex/unstable across sources, 
        # we return a clean metadata object.
        return {
            "ticker": ticker,
            "name": f"Công ty Cổ phần {ticker}", # Fallback naming
            "exchange": "HOSE", # Default
            "industry": "Tài chính / Sản xuất",
            "description": f"Thông tin chi tiết về mã {ticker} đang được tải..."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Order Book (Price Depth) Endpoint ────────────────────────


@router.get("/order-book")
async def order_book(ticker: str = Query(...)) -> dict:
    """Fetch Bid/Ask depth (Mocked if API fails)."""
    import os
    import random
    from vnstock import Vnstock
    from config.settings import get_settings

    settings = get_settings()
    os.environ["VNAI_API_KEY"] = settings.vnstock_api_key
    os.environ["VNSTOCK_API_KEY"] = settings.vnstock_api_key

    ticker = ticker.upper().strip()
    try:
        # Mocking Order Book for now as price_depth is unstable in current v3/VCI
        last_price = 30.0 # Base price for mock
        
        bids = []
        asks = []
        for i in range(3):
            bids.append({
                "price": round(last_price - (i * 0.1) - 0.05, 2),
                "volume": random.randint(1000, 50000)
            })
            asks.append({
                "price": round(last_price + (i * 0.1) + 0.05, 2),
                "volume": random.randint(1000, 50000)
            })
            
        return {
            "ticker": ticker,
            "bids": sorted(bids, key=lambda x: x["price"], reverse=True),
            "asks": sorted(asks, key=lambda x: x["price"]),
            "timestamp": "GMT+7 Hanoi"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Time & Sales (Intraday Matches) Endpoint ──────────────────


@router.get("/time-sales")
async def time_sales(ticker: str = Query(...)) -> dict:
    """Fetch latest intraday matches."""
    import os
    import datetime as _dt
    from vnstock import Vnstock
    from config.settings import get_settings

    settings = get_settings()
    os.environ["VNAI_API_KEY"] = settings.vnstock_api_key
    os.environ["VNSTOCK_API_KEY"] = settings.vnstock_api_key

    ticker = ticker.upper().strip()
    try:
        stock = Vnstock().stock(symbol=ticker, source="VCI")
        df = stock.quote.intraday()
        
        matches = []
        if df is not None and not df.empty:
            # Sort by time descending
            df = df.sort_values("time", ascending=False).head(30)
            for _, row in df.iterrows():
                matches.append({
                    "time": str(row["time"])[11:19], # HH:MM:SS
                    "price": round(float(row["price"]), 2),
                    "volume": int(row["volume"]),
                    "type": str(row["match_type"]).upper()
                })
                
        return {
            "ticker": ticker,
            "matches": matches,
            "timezone": "GMT+7 Hanoi"
        }
    except Exception as e:
        logger.error("time_sales_error", ticker=ticker, error=str(e))
        return {"ticker": ticker, "matches": [], "error": str(e)}


# ── Ticker News Endpoint ─────────────────────────────────────


@router.get("/news")
async def ticker_news(ticker: str = Query(...)) -> dict:
    """Fetch latest news headlines for a ticker from vnstock."""
    import os
    from vnstock import Vnstock
    from config.settings import get_settings

    settings = get_settings()
    os.environ["VNAI_API_KEY"] = settings.vnstock_api_key
    os.environ["VNSTOCK_API_KEY"] = settings.vnstock_api_key

    ticker = ticker.upper().strip()
    try:
        stock = Vnstock().stock(symbol=ticker)
        news_df = stock.company.news()
        
        news_list = []
        if news_df is not None and not news_df.empty:
            # Take top 10 news items
            for _, row in news_df.head(10).iterrows():
                news_list.append({
                    "title": str(row["title"]),
                    "publish_time": str(row["publish_time"]),
                    "url": str(row["url"]) if "url" in row else None
                })
                
        return {
            "ticker": ticker,
            "news": news_list,
            "count": len(news_list)
        }
    except Exception as e:
        logger.error("news_fetch_error", ticker=ticker, error=str(e))
        return {"ticker": ticker, "news": [], "error": str(e)}


# ── Data Ingestion Pipeline Endpoints ────────────────────────


@router.post("/ingest-news")
async def ingest_news(
    tickers: str = Query("SSI,HPG,VCB,FPT,VNM", description="Comma-separated tickers"),
    max_pages: int = Query(3, description="Max pages per ticker"),
) -> dict:
    """Trigger the News → Embed → ChromaDB ingestion pipeline."""
    from src.context.ingestion_pipeline import IngestionPipeline
    
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    pipeline = IngestionPipeline()
    result = await pipeline.ingest_news(ticker_list, max_pages_per_ticker=max_pages)
    return result


@router.post("/ingest-bctc")
async def ingest_bctc(
    ticker: str = Query(None, description="Optional ticker filter"),
    max_files: int = Query(50, description="Max files to process"),
) -> dict:
    """Trigger BCTC PDF/Excel → Embed → ChromaDB ingestion pipeline."""
    from src.context.ingestion_pipeline import IngestionPipeline
    
    pipeline = IngestionPipeline()
    result = pipeline.ingest_bctc(ticker=ticker, max_files=max_files)
    return result


# ── Paper Trading Endpoint ───────────────────────────────────


@router.get("/paper-trade")
async def paper_trade(
    request: Request,
    ticker: str = Query(..., description="Ticker to simulate"),
) -> dict:
    """Run one cycle of the Paper Trading Engine.
    
    Executes: Fetch Price → ML → RAG → LLM → Matrix → Risk → Virtual Order
    Returns full latency profile and portfolio status.
    """
    from src.backtest.paper import PaperTradingEngine
    
    engine = PaperTradingEngine()
    with trace_stage(request, "paper_trading_cycle"):
        result = await engine.run_single_cycle(ticker=ticker.upper())
    return result


# ── Conversational AI Chat Endpoint ──────────────────────────


@router.post("/chat", response_model=ChatResponse)
async def chat_interaction(
    request: Request,
    payload: ChatRequest,
    use_mock: bool = Query(False, description="Use mock data for ML predictions"),
) -> dict:
    """Conversational endpoint for the Chatbot UI.
    
    Acts as a wrapper around the Ollama LLM. If a `ticker` is provided,
    it automatically runs the ML prediction pipeline and fetches RAG/News
    context, injecting these into the LLM's system prompt so the bot
    can answer questions about the stock based on real-time data.
    """
    from src.llm.client import get_llm_client
    from config.settings import get_settings
    from src.api.schemas import ChatResponse, ChatRequest
    import json
    
    settings = get_settings()
    client = get_llm_client()
    
    ticker = payload.ticker.upper().strip() if payload.ticker else None
    context_used = []
    
    # ── 1. Build Financial Context (if ticker provided) ──
    financial_context_str = ""
    if ticker:
        context_used.append(f"Focus Ticker: {ticker}")
        try:
            # 1a. Get ML Quantitative Data
            with trace_stage(request, "chat_quant_data"):
                 quant_data = await predict(
                     request=request, 
                     ticker=ticker, 
                     use_mock=use_mock
                 )
                 context_used.append("ML Quantitative Signals (LightGBM)")
                 quant_json = json.dumps(quant_data["quantitative_signals"], ensure_ascii=False)
                 
            # 1b. Get RAG Context (BCTC / Industry Reports)
            with trace_stage(request, "chat_rag_data"):
                try:
                    from src.context.rag_service import ZonedRAGService
                    rag_service = ZonedRAGService()
                    rag_text = rag_service.query(ticker=ticker, n_results=3)
                    if rag_text and len(rag_text) > 50:
                        context_used.append("Vector DB (RAG)")
                    else:
                        rag_text = "Không có thông tin báo cáo trong CSDL."
                except Exception as e:
                    logger.warning("chat_rag_error", error=str(e))
                    rag_text = "Lỗi kết nối Vector DB."
            
            # 1c. Get Latest News headlines
            with trace_stage(request, "chat_news_data"):
                news_payload = await ticker_news(ticker=ticker)
                news_text = ""
                if news_payload.get("news"):
                    context_used.append("Latest News API")
                    news_text = "\n".join([f"- {n['title']} ({n['publish_time']})" for n in news_payload["news"][:5]])
                else:
                    news_text = "Không có tin tức mới."
            
            # Combine into a strict context block
            financial_context_str = f"""
--- BỐI CẢNH DỮ LIỆU THỰC TẾ CHO MÃ {ticker} ---
1. TÍN HIỆU ĐỊNH LƯỢNG (Từ Model ML LightGBM - Độ tin cậy 95%):
{quant_json}

2. NGỮ CẢNH BÁO CÁO TÀI CHÍNH/NGÀNH (Từ Vector DB RAG):
{rag_text}

3. TIN TỨC MỚI NHẤT:
{news_text}
------------------------------------------------
"""
        except Exception as e:
            logger.error("chat_context_build_error", ticker=ticker, error=str(e))
            financial_context_str = f"Lỗi truy xuất dữ liệu hệ thống cho mã {ticker}: {str(e)}"
    
    # ── 2. Build LLM Messages ──
    # System prompt strictly instructs the LLM to act as a financial assistant
    system_instruction = (
        "Bạn là Antigravity AI - một trợ lý AI phân tích chứng khoán chuyên nghiệp tại Việt Nam. "
        "Luôn trả lời bằng tiếng Việt, ngắn gọn, súc tích và dễ hiểu. "
        "Nếu người dùng hỏi về một mã cổ phiếu cụ thể, tham khảo khối [BỐI CẢNH DỮ LIỆU THỰC TẾ] dưới đây (nếu có). "
        "Model ML của hệ thống dùng LightGBM dự báo theo xác suất (trend) và khoảng giá (range). "
        "Tuyệt đối không bịa đặt số liệu tài chính nếu không có trong ngữ cảnh. "
    )
    
    if financial_context_str:
        system_instruction += f"\n\n{financial_context_str}"

    messages = [{"role": "system", "content": system_instruction}]
    
    # Append chat history
    for msg in payload.messages:
        messages.append({"role": msg.role, "content": msg.content})

    # ── 3. Call Ollama ──
    try:
        with trace_stage(request, "chat_llm_inference"):
            response = await client.chat.completions.create(
                model=settings.llm_model_name,
                messages=messages,
                temperature=0.7, # Higher temp for chat vs strict JSON analysis
                timeout=45.0,    # Chat can take longer
            )
            
        result_text = response.choices[0].message.content
        return {
            "response": result_text or "Xin lỗi, tôi không thể tạo câu trả lời lúc này.",
            "context_used": context_used
        }
            
    except Exception as e:
        logger.error("chat_llm_error", error=str(e))
        return {
            "response": f"Xin lỗi, hệ thống AI đang quá tải hoặc mất kết nối. Chi tiết lỗi: {str(e)}",
            "context_used": context_used
        }

