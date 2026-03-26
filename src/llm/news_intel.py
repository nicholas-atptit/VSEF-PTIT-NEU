"""Engine for generating structured intelligence from news articles.

Processes raw news text through local LLMs to extract trend, 
sentiment, and aggregated reports for tickers.
"""

import json
import logging
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

    async def analyze_ticker_news(self, ticker: str, articles: list[Any]) -> dict[str, Any] | None:
        """Run LLM analysis on a batch of articles for a ticker."""
        if not articles:
            return None

        ticker = ticker.upper()
        logger.info("news_intel_analyzing", ticker=ticker, article_count=len(articles))

        # 1. Build Prompt
        prompt = build_news_intelligence_prompt(ticker, articles)

        # 2. Call LLM (Ollama)
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.ollama_model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
                timeout=45.0, # News synthesis can be slow
            )
            
            result_text = response.choices[0].message.content
            if not result_text:
                return None
            
            intel = json.loads(result_text)
            
            # 3. Store in Database
            await self._store_intelligence(ticker, intel, articles)
            
            return intel

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error("news_intel_failed", ticker=ticker, error=str(e))
            return None

    async def _store_intelligence(self, ticker: str, intel: dict, articles: list[Any]) -> None:
        """Persist analysis result to TimescaleDB."""
        try:
            article_ids = [getattr(a, 'url', '') for a in articles]
            source_sites = list(set([getattr(a, 'source', 'Unknown') for a in articles]))
            
            async with get_session() as session:
                query = text("""
                    INSERT INTO news_intelligence (
                        ticker, trend, sentiment_score, summary, full_report, article_ids, source_sites
                    ) VALUES (
                        :ticker, :trend, :sentiment, :summary, :report, :article_ids, :source_sites
                    )
                """)
                await session.execute(query, {
                    "ticker": ticker,
                    "trend": intel.get("trend", "Neutral"),
                    "sentiment": float(intel.get("sentiment_score", 0.0)),
                    "summary": intel.get("summary", ""),
                    "report": intel.get("full_report", ""),
                    "article_ids": article_ids,
                    "source_sites": source_sites
                })
                await session.commit()
                logger.info("news_intel_stored", ticker=ticker)
        except Exception as e:
            logger.error("news_intel_db_error", ticker=ticker, error=str(e))

    async def get_latest_intelligence(self, ticker: str) -> dict | None:
        """Retrieve the most recent analysis for a ticker."""
        try:
            async with get_session() as session:
                res = await session.execute(text("""
                    SELECT trend, sentiment_score, summary, full_report, timestamp
                    FROM news_intelligence
                    WHERE ticker = :t
                    ORDER BY timestamp DESC
                    LIMIT 1
                """), {"t": ticker.upper()})
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
