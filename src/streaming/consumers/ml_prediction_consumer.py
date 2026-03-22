"""Kafka Consumer for triggering ML predictions in real-time.

Listens to the `market.data.raw` topic. When new AM/PM data arrives,
it triggers the saved LightGBM models to predict the trend and range,
then saves/publishes the output.
"""

import asyncio
import logging

import pandas as pd

from src.ml.trainer import DualModelTrainer
from src.streaming.kafka_client import KafkaSubscriber

logger = logging.getLogger(__name__)


class MLPredictionConsumer:
    """Consumes market data and triggers the ML pipeline."""

    TOPIC = "market.data.raw"
    GROUP_ID = "ml_prediction_group"
    OUTPUT_TOPIC = "ml.predictions"

    def __init__(self) -> None:
        self.subscriber = KafkaSubscriber(self.TOPIC, self.GROUP_ID)
        self.trainer = DualModelTrainer()
        # Publisher to announce predictions to the rest of the system
        from src.streaming.kafka_client import KafkaPublisher
        self.publisher = KafkaPublisher()

    async def run_forever(self) -> None:
        """Run the consumer loop continuously."""
        logger.info("ml_prediction_consumer_starting", topic=self.TOPIC)
        try:
            async for payload in self.subscriber.stream():
                await self._process_message(payload)
        except asyncio.CancelledError:
            logger.info("ml_prediction_consumer_cancelled")
        except Exception as e:
            logger.error("ml_prediction_consumer_error", error=str(e))
        finally:
            await self.subscriber.stop()
            await self.publisher.stop()

    async def _process_message(self, data: dict) -> None:
        """Run inference on the new data point."""
        try:
            ticker = data["ticker"]
            timestamp = data["timestamp"]

            # 1. Fetch recent data block for feature engineering (needs lookback)
            # Since ML needs rolling feats (SMA, RSI), one row isn't enough.
            import datetime as dt
            from src.ml.data_loader import load_ohlcv_from_vnstock
            
            # Simple synchronous fetch in a thread
            def _fetch_lookback():
                return load_ohlcv_from_vnstock(ticker, limit=400)
                
            df = await asyncio.to_thread(_fetch_lookback)
            
            if df is None or len(df) < 50:
                logger.warning("ml_consumer_insufficient_lookback", ticker=ticker)
                return

            # Append the incoming row if it's strictly newer
            # Convert incoming string to Pandas timestamp to match index
            # (Assuming the fetch didn't already get it due to caching lag)
            
            # 2. Run feature engineering (V3 features automatically detected by DualModelTrainer)
            feat_df = self.trainer.compute_features_for_ticker(ticker, df)
            
            # 3. Filter to just the columns the model expects
            self.trainer._ensure_models_loaded(ticker)
            saved_features = self.trainer._models[ticker].get("feature_cols", [])
            
            if not saved_features:
                logger.warning("ml_consumer_no_model", ticker=ticker)
                return
                
            latest_row = feat_df[saved_features].iloc[[-1]]
            
            # 4. Predict
            prediction = self.trainer.predict(ticker, latest_row)
            
            # 5. Publish Prediction Event
            event = {
                "ticker": ticker,
                "timestamp": timestamp,
                "trend_probabilities": prediction["trend_probabilities"],
                "expected_range": prediction["expected_range"]
            }
            
            await self.publisher.publish(self.OUTPUT_TOPIC, event)
            logger.info("ml_consumer_prediction_success", ticker=ticker)
            
        except Exception as e:
            logger.error("ml_consumer_inference_failed", ticker=data.get("ticker"), error=str(e))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    consumer = MLPredictionConsumer()
    try:
        asyncio.run(consumer.run_forever())
    except KeyboardInterrupt:
        pass
