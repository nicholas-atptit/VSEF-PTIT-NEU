"""Orchestrator for daily stock data ingestion."""

from __future__ import annotations

import datetime as dt
from typing import List
from src.data.adapters.vnstock_adapter import VnstockAdapter
from src.utils.logging import get_logger

logger = get_logger(__name__)

class IngestionPipeline:
    """Manages the daily flow of data from vnstock to storage."""

    def __init__(self, symbols: List[str]) -> None:
        self.adapter = VnstockAdapter(symbols)
        self.symbols = symbols

    def run_daily_sync(self, days_back: int = 3650) -> None:
        """Fetch latest daily OHLC and fundamental data for VN100 and store in DB.
        
        Args:
            days_back: Number of days to fetch (default: 10 years ~ 3650 days).
        """
        logger.info("starting_daily_sync", symbols_count=len(self.symbols), days_back=days_back)
        
        end_date = dt.date.today().strftime("%Y-%m-%d")
        start_date = (dt.date.today() - dt.timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        for symbol in self.symbols:
            try:
                # 1. Fetch OHLC
                ohlc_df = self.adapter.get_ohlc(symbol, start_date, end_date)
                if not ohlc_df.empty:
                    self._store_ohlc(symbol, ohlc_df)
                    
                # 2. Fetch Fundamental Ratios
                fund_df = self.adapter.get_financial_ratios(symbol)
                if not fund_df.empty:
                    self._store_fundamentals(symbol, fund_df)
                    
                logger.info("sync_completed_for_symbol", symbol=symbol, ohlc_rows=len(ohlc_df))
            except Exception as e:
                logger.error("sync_error_for_symbol", symbol=symbol, error=str(e))
                
        logger.info("daily_sync_completed")

    def _store_ohlc(self, symbol: str, df: pd.DataFrame) -> None:
        """Store OHLC data using the existing database layer or SQLAlchemy.
        
        TODO: Use src.database.connection and RawPrice model for bulk insert.
        """
        logger.debug("storing_ohlc", symbol=symbol, rows=len(df))
        # Placeholder for actual DB session logic
        pass

    def _store_fundamentals(self, symbol: str, df: pd.DataFrame) -> None:
        """Store fundamental data."""
        logger.debug("storing_fundamentals", symbol=symbol)
        # Placeholder
        pass
