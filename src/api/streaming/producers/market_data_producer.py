"""Kafka Producer for pushing real-time market data to the message broker.

Extracts EOD (End of Day) or Half-Day (11:30 AM) data using VNStock
and publishes it as JSON events to the `market.data.raw` topic.
"""

import asyncio
import datetime as dt
import logging

from config.settings import get_settings
from src.api.streaming.kafka_client import KafkaPublisher

logger = logging.getLogger(__name__)


class MarketDataProducer:
    """Producer for raw OHLCV market data."""

    TOPIC = "market.data.raw"

    def __init__(self) -> None:
        self.publisher = KafkaPublisher()
        self.settings = get_settings()

    async def publish_all_tickers(self) -> None:
        """Fetch daily data for all active tickers and publish to Kafka."""
        try:
            from vnstock_data import Listing
            
            # 1. Get List of Tickers
            df_listing = Listing(source="vnd").all_symbols()
            ticker_col = "ticker" if "ticker" in df_listing.columns else df_listing.columns[0]
            tickers = df_listing[ticker_col].tolist()
            
            logger.info("producer_fetching_all_tickers", count=len(tickers))
            await self.publisher.start()
            
            # 2. Iterate and publish (in small chunks to respect rate limits)
            today_str = dt.datetime.now().strftime("%Y-%m-%d")
            
            # Using asyncio.to_thread for synchronous vnstock calls
            # to prevent blocking the async event loop of the scheduler
            for i, ticker in enumerate(tickers):
                try:
                    df = await asyncio.to_thread(self._fetch_single_ticker, ticker, today_str)
                    
                    if df is not None and not df.empty:
                        for _, row in df.iterrows():
                            # Base format expected by DB Consumer
                            event = {
                                "ticker": ticker,
                                "timestamp": str(row["time"]),
                                "open": float(row["open"]),
                                "high": float(row["high"]),
                                "low": float(row["low"]),
                                "close": float(row["close"]),
                                "volume": int(row["volume"])
                            }
                            await self.publisher.publish(self.TOPIC, event)
                    
                    if i > 0 and i % 50 == 0:
                        logger.info("producer_published_batch", processed=i, total=len(tickers))
                        # Small delay to avoid overwhelming the VNStock API
                        await asyncio.sleep(0.5)
                        
                except Exception as ex:
                    logger.debug("producer_ticker_error", ticker=ticker, error=str(ex))
            
            logger.info("producer_finished_publish_all_tickers")
            
        except ImportError:
            logger.error("vnstock_missing_cannot_produce")
        except Exception as e:
            logger.error("market_data_producer_error", error=str(e))
        finally:
            await self.publisher.stop()

    def _fetch_single_ticker(self, ticker: str, today_str: str):
        """Synchronous fetch of a single ticker from vnstock_data."""
        from vnstock_data import Quote
        try:
            df = Quote(source="VCI", symbol=ticker).history(start=today_str, end=today_str, interval="1D")
            return df
        except Exception:
            return None
