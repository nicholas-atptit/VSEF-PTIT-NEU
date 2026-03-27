"""Engine for generating structured intelligence from news articles.

Processes raw news text through local LLMs to extract trend, 
sentiment, and aggregated reports for tickers.
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import Any

from sqlalchemy import text
from config.settings import get_settings
from src.database.connection import get_session
from src.llm.client import get_llm_client
from src.llm.prompts import build_news_intelligence_prompt
from src.utils.logging import get_logger

logger = get_logger(__name__)

class NewsIntelEngine:
    """Orchestrator for transforming raw news into structured intelligence."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = get_llm_client()

    async def analyze_ticker_news(self, ticker: str, articles: list[Any], horizon: str = "short", _retry_count: int = 0) -> dict[str, Any] | None:
        """Run LLM analysis on a batch of articles for a ticker."""
        if not articles:
            return None

        ticker = ticker.upper()
        logger.info("news_intel_analyzing", ticker=ticker, article_count=len(articles))

        # 1. Build Prompt
        prompt = build_news_intelligence_prompt(ticker, articles, horizon=horizon)

        # 2. Call LLM
        try:
            provider = self.settings.llm_provider
            model = self.settings.gemini_model_name if provider == "gemini" else self.settings.openai_model_name
            
            if provider == "gemini":
                # --- Native Gemini Flow ---
                from src.llm.client import get_gemini_client
                genai = get_gemini_client()
                
                # Format model name (ensure models/ prefix for gemma/gemini)
                if not model.startswith("models/"): 
                    m_id = f"models/{model}"
                else:
                    m_id = model
                
                native_model = genai.GenerativeModel(
                    model_name=m_id,
                    generation_config={"response_mime_type": "application/json"}
                )
                
                response = await native_model.generate_content_async(prompt)
                result_text = response.text
                
            else:
                # --- OpenAI/Groq/Ollama Flow ---
                if provider == "groq": model = self.settings.groq_model_name
                elif provider == "ollama": model = self.settings.ollama_model_name
                
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    timeout=60.0,
                )
                result_text = response.choices[0].message.content

            if not result_text:
                return None
            
            # Clean possible markdown wrap from direct output
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            
            intel = json.loads(result_text)
            
            # 3. Store in Database
            await self._store_intelligence(ticker, intel, articles, horizon=horizon)
            
            return intel

        except Exception as e:
            if "429" in str(e) and _retry_count < 3:
                wait_time = 45 * (_retry_count + 1)
                logger.warning("rate_limit_hit", ticker=ticker, retry_delay=wait_time, attempt=_retry_count+1)
                await asyncio.sleep(wait_time)
                return await self.analyze_ticker_news(ticker, articles, horizon, _retry_count + 1)
            
            if "404" in str(e) or "not found" in str(e).lower():
                logger.error("model_not_found", ticker=ticker, model=model, error=str(e))
                # Fallback to 1.5-flash if 2.0 or 3.1 is not available
                if "gemini-1.5-flash" not in model:
                    logger.info("falling_back_to_gemini_1.5_flash", ticker=ticker)
                    old_model = self.settings.gemini_model_name
                    self.settings.gemini_model_name = "gemini-1.5-flash"
                    res = await self.analyze_ticker_news(ticker, articles, horizon, _retry_count)
                    self.settings.gemini_model_name = old_model
                    return res
            
            import traceback
            traceback.print_exc()
            logger.error("news_intel_failed", ticker=ticker, error=str(e))
            return None

    async def _store_intelligence(self, ticker: str, intel: dict, articles: list[Any], horizon: str = "short") -> None:
        """Persist analysis result to TimescaleDB."""
        try:
            article_ids = [getattr(a, 'url', '') for a in articles]
            source_sites = list(set([getattr(a, 'source', 'Unknown') for a in articles]))
            
            async with get_session() as session:
                query = text("""
                    INSERT INTO news_intelligence (
                        ticker, trend, sentiment_score, summary, full_report, article_ids, source_sites, horizon
                    ) VALUES (
                        :ticker, :trend, :sentiment, :summary, :report, :article_ids, :source_sites, :horizon
                    )
                """)
                await session.execute(query, {
                    "ticker": ticker,
                    "trend": intel.get("trend", "Neutral"),
                    "sentiment": float(intel.get("sentiment_score", 0.0)),
                    "summary": intel.get("summary", ""),
                    "report": intel.get("full_report", ""),
                    "article_ids": article_ids,
                    "source_sites": source_sites,
                    "horizon": horizon
                })
                await session.commit()
                logger.info("news_intel_stored", ticker=ticker)
        except Exception as e:
            logger.error("news_intel_db_error", ticker=ticker, error=str(e))

    async def get_latest_intelligence(self, ticker: str, horizon: str = "short") -> dict | None:
        """Retrieve the most recent analysis for a ticker."""
        try:
            async with get_session() as session:
                res = await session.execute(text("""
                    SELECT trend, sentiment_score, summary, full_report, timestamp
                    FROM news_intelligence
                    WHERE ticker = :t AND horizon = :h
                    ORDER BY timestamp DESC
                    LIMIT 1
                """), {"t": ticker.upper(), "h": horizon})
                row = res.fetchone()
                if row:
                    return {
                        "trend": row[0],
                        "sentiment_score": row[1],
                        "summary": row[2],
                        "full_report": row[3],
                        "timestamp": row[4]
                    }
        except Exception:
            return None
        return None
