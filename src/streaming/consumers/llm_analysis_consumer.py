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
            # 1. Build quant_data payload
            quant_data = {
                "trend_probabilities": prediction.get("trend_probabilities", {}),
                "expected_range": prediction.get("expected_range", {}),
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
            news_text = "No recent news found."
            news_headlines = []
            try:
                from vnstock3 import Vnstock
                stock = Vnstock().stock(symbol=ticker, source="VCI")
                news_df = stock.company.news()
                if news_df is not None and not news_df.empty:
                    news_headlines = news_df["news_title"].head(5).tolist()
                    news_text = "\n".join([f"- {t}" for t in news_headlines])
            except Exception:
                pass

            # 4. Fetch Phase 10 DL/RL Metadata (Simulated)
            dl_data = {
                "tft_sequence_forecast": {
                    "expected_trend": "UP",
                    "quantiles": {"q10": 0.0, "q50": 0.0, "q90": 0.0}
                },
                "cnn_order_book_microstructure": {
                    "imbalance_state": "NEUTRAL",
                    "slippage_optimization_flag": False
                }
            }
            rl_data = {
                "suggested_allocation_pct": 0.35,
                "rl_action_justification": "Stable market conditions; Sortino-based allocation."
            }

            # 5. Call LLM Pipeline
            llm_result = await run_qualitative_analysis(
                ticker=ticker,
                quant_data=quant_data,
                rag_context=rag_text,
                news_context=news_text,
                dl_data=dl_data,
                rl_data=rl_data,
            )

            # 6. Inject headlines for TUI display
            llm_result["news_headlines"] = news_text

            # 7. Publish combined result
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
    import asyncio as _asyncio
    consumer = LLMAnalysisConsumer()
    try:
        _asyncio.run(consumer.run_forever())
    except KeyboardInterrupt:
        pass
