"""Kafka Consumer that triggers LLM qualitative analysis from ML predictions.

Listens to `ml.predictions` topic. For each prediction event, it fetches
RAG context and news, then calls the Ollama LLM to produce a qualitative
analysis. The result is published to `llm.analysis`.
"""

import asyncio
import json
import logging

from src.llm.pipeline import run_qualitative_analysis
from src.streaming.kafka_client import KafkaPublisher, KafkaSubscriber
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LLMAnalysisConsumer:
    """Consumes ML predictions and enriches them with LLM analysis."""

    INPUT_TOPIC = "ml.predictions"
    OUTPUT_TOPIC = "llm.analysis"
    GROUP_ID = "llm_analysis_group"

    def __init__(self) -> None:
        self.subscriber = KafkaSubscriber(self.INPUT_TOPIC, self.GROUP_ID)
        self.publisher = KafkaPublisher()

    async def run_forever(self) -> None:
        """Run the consumer loop continuously."""
        logger.info("llm_analysis_consumer_starting")
        await self.publisher.start()
        try:
            async for prediction in self.subscriber.stream():
                await self._process_prediction(prediction)
        except asyncio.CancelledError:
            logger.info("llm_analysis_consumer_cancelled")
        except Exception as e:
            logger.error("llm_analysis_consumer_error", error=str(e))
        finally:
            await self.subscriber.stop()
            await self.publisher.stop()

    async def _process_prediction(self, prediction: dict) -> None:
        """Take a prediction event and run LLM analysis on it."""
        ticker = prediction.get("ticker", "UNKNOWN")
        timestamp = prediction.get("timestamp", "")

        try:
            # 1. Build quant_data payload (same format as /predict response)
            quant_data = {
                "quantitative_signals": {
                    "trend_probabilities": prediction.get("trend_probabilities", {}),
                    "expected_range": prediction.get("expected_range", {}),
                }
            }

            # 2. Fetch RAG context
            rag_text = ""
            try:
                from src.context.rag_service import ZonedRAGService
                rag_service = ZonedRAGService()
                rag_text = rag_service.query(ticker=ticker, n_results=3) or ""
            except Exception as e:
                logger.warning("llm_consumer_rag_error", ticker=ticker, error=str(e))

            # 3. Fetch latest news
            news_text = ""
            try:
                from vnstock3 import Vnstock
                stock = Vnstock().stock(symbol=ticker, source="VCI")
                news_df = stock.company.news()
                if news_df is not None and not news_df.empty:
                    titles = news_df["news_title"].head(5).tolist()
                    news_text = "\n".join([f"- {t}" for t in titles])
            except Exception:
                pass

            # 4. Call LLM Pipeline
            llm_result = await run_qualitative_analysis(
                ticker=ticker,
                quant_data=quant_data,
                rag_context=rag_text,
                news_context=news_text,
                user_risk_input=0.70,
            )

            # 5. Publish combined result to llm.analysis topic
            full_event = {
                "ticker": ticker,
                "timestamp": timestamp,
                "ml_prediction": prediction,
                "llm_analysis": llm_result,
            }
            await self.publisher.publish(self.OUTPUT_TOPIC, full_event)
            logger.info("llm_analysis_published", ticker=ticker)

        except Exception as e:
            logger.error("llm_analysis_failed", ticker=ticker, error=str(e))


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    consumer = LLMAnalysisConsumer()
    try:
        asyncio.run(consumer.run_forever())
    except KeyboardInterrupt:
        pass
