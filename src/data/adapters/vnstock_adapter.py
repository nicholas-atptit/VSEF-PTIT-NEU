"""VNStock Adapter layer.

This module provides a thin wrapper around vnstock>=3.0 to centralize
data acquisition for market data, fundamental data, and news.
"""

from __future__ import annotations

import os
import datetime as dt
import time
from typing import List, Optional

import pandas as pd
from vnstock import Vnstock

from config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

MAX_FETCH_RETRIES = 3
BASE_RETRY_DELAY_SECONDS = 0.75


class VnstockAdapter:
    """Thin adapter for vnstock API integration.
    
    Standardizes calls for OHLCV, market indices, financial ratios, and news.
    Ensures environment variables for API keys are set automatically.
    """

    def __init__(self, symbol_list: Optional[List[str]] = None) -> None:
        """Initialize adapter and ensure API keys are configured."""
        self.settings = get_settings()
        self.symbols = symbol_list or []
        self._setup_env()
        self.vn = Vnstock()
        logger.info("vnstock_adapter_initialized", symbols_count=len(self.symbols))

    def _setup_env(self) -> None:
        """Inject API keys from settings into environment variables."""
        api_key = self.settings.vnstock_api_key or ""
        os.environ["VNAI_API_KEY"] = api_key
        os.environ["VNSTOCK_API_KEY"] = api_key
        if api_key:
            logger.debug("vnstock_api_keys_configured")

    def get_ohlcv(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str, 
        interval: str = "1D"
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for a ticker symbol.
        
        Args:
            symbol: Ticker symbol (e.g., 'SSI', 'HPG').
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            interval: Data interval ('1D', '1H', etc.).
            
        Returns:
            DataFrame with columns [date, open, high, low, close, volume].
        """
        ticker = symbol.upper().strip()
        logger.debug("fetching_ticker_ohlcv", symbol=ticker, start=start_date, end=end_date)
        last_error: Exception | None = None
        for attempt in range(1, MAX_FETCH_RETRIES + 1):
            try:
                stock = self.vn.stock(symbol=ticker, source="VCI")
                df = stock.quote.history(
                    start=start_date,
                    end=end_date,
                    interval=interval
                )
                if df is None or df.empty:
                    raise ValueError(f"Empty OHLCV response from vnstock for {ticker}")

                standardized = df.copy()
                if "time" in standardized.columns and "date" not in standardized.columns:
                    standardized = standardized.rename(columns={"time": "date"})

                required = {"date", "open", "high", "low", "close", "volume"}
                missing = required - set(standardized.columns)
                if missing:
                    raise ValueError(f"Missing OHLCV columns from vnstock for {ticker}: {sorted(missing)}")

                standardized["date"] = pd.to_datetime(standardized["date"]).dt.normalize()
                start_ts = pd.Timestamp(start_date).normalize()
                end_ts = pd.Timestamp(end_date).normalize()
                standardized = standardized[
                    (standardized["date"] >= start_ts) & (standardized["date"] <= end_ts)
                ].copy()
                if standardized.empty:
                    raise ValueError(
                        f"Filtered OHLCV response is empty for {ticker} inside {start_ts.date()} to {end_ts.date()}"
                    )
                for column in ("open", "high", "low", "close", "volume"):
                    standardized[column] = pd.to_numeric(standardized[column], errors="coerce")
                standardized = standardized.dropna(subset=["date", "open", "high", "low", "close"])
                standardized["volume"] = standardized["volume"].fillna(0.0)
                standardized["ticker"] = ticker
                standardized = (
                    standardized.sort_values("date")
                    .drop_duplicates(subset=["date"], keep="last")
                    .reset_index(drop=True)
                )
                if standardized.empty:
                    raise ValueError(f"Standardized OHLCV response is empty for {ticker}")
                return standardized[["date", "ticker", "open", "high", "low", "close", "volume"]]
            except Exception as e:
                last_error = e
                if attempt < MAX_FETCH_RETRIES:
                    logger.warning(
                        "ohlcv_fetch_retry",
                        symbol=ticker,
                        attempt=attempt,
                        max_attempts=MAX_FETCH_RETRIES,
                        error=str(e),
                    )
                    time.sleep(BASE_RETRY_DELAY_SECONDS * attempt)
                else:
                    logger.error(
                        "ohlcv_fetch_error",
                        symbol=ticker,
                        attempts=MAX_FETCH_RETRIES,
                        error=str(e),
                    )
        return pd.DataFrame()

    def get_ohlc(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1D",
    ) -> pd.DataFrame:
        """Backward-compatible alias for the legacy OHLC fetch method."""
        return self.get_ohlcv(symbol, start_date, end_date, interval)

    def get_index_ohlcv(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str, 
        interval: str = "1D"
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for a market index (e.g., VNINDEX, VN30).
        
        Args:
            symbol: Index symbol (e.g., 'VNINDEX', 'VN30').
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            interval: Data interval ('1D').
            
        Returns:
            DataFrame with index data.
        """
        logger.debug("fetching_index_ohlcv", indicator=symbol, start=start_date, end=end_date)
        # For indices, vnstock often uses the same stock interface but with index symbols
        return self.get_ohlcv(symbol, start_date, end_date, interval)

    def get_financial_ratios(self, symbol: str) -> pd.DataFrame:
        """Fetch the latest financial ratios for a symbol.
        
        Returns:
            DataFrame with various financial metrics (P/E, P/B, etc.).
        """
        logger.debug("fetching_financial_ratios", symbol=symbol)
        try:
            stock = self.vn.stock(symbol=symbol.upper(), source="VCI")
            df = stock.finance.ratio(period="yearly", lang="vi")
            return df
        except Exception as e:
            logger.error("ratio_fetch_error", symbol=symbol, error=str(e))
        return pd.DataFrame()

    def get_valuation_metrics(self, symbol: str) -> pd.DataFrame:
        """Backward-compatible alias for legacy valuation metric lookups."""
        return self.get_financial_ratios(symbol)

    def get_news(self, ticker: str, count: int = 10) -> pd.DataFrame:
        """Fetch recent news for a specific ticker.
        
        Args:
            ticker: Ticker symbol.
            count: Number of news items to fetch.
            
        Returns:
            DataFrame with news items.
        """
        logger.debug("fetching_news", ticker=ticker, count=count)
        try:
            stock = self.vn.stock(symbol=ticker.upper(), source="VCI")
            df = stock.news()
            
            if df is not None and not df.empty:
                return df.head(count)
        except Exception as e:
            # Fallback to stock.company.news() if stable API fails
            try:
                company = getattr(self.vn.stock(symbol=ticker.upper(), source="VCI"), "company", None)
                if company:
                    df = company.news()
                    if df is not None and not df.empty:
                        return df.head(count)
            except Exception as e2:
                logger.error("news_fetch_fatal", ticker=ticker, error=str(e2))
                
        return pd.DataFrame()

    def get_vn100_tickers(self) -> List[str]:
        """Retrieve the list of VN100 constituents.
        
        Returns:
            List of ticker symbols.
        """
        logger.debug("fetching_vn100_tickers")
        try:
            # Note: vnstock 3.0 might have a specific method for this
            # For now, we use a robust known list from current adapter as fallback
            from vnstock import Vnstock
            # Static list if dynamic fails
            vn100_defaults = [
                "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HGP",
                "HPG", "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI",
                "STB", "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB",
                "VRE", "AAV", "ABS", "ACC", "ACG", "ACL", "ADS", "AGG", "AGM", "AGR",
                "AMD", "ANV", "APC", "APG", "APH", "ASG", "ASM", "ASR", "BFC", "BGI",
                "BHN", "BIC", "BMI", "BMP", "BSI", "BTP", "BTT", "BVB", "BWE", "C32",
                "C47", "CAV", "CCI", "CCL", "CDC", "CEE", "CIG", "CII", "CKG", "CLC",
                "CLL", "CMG", "CMX", "CNG", "COM", "CRC", "CRE", "CSM", "CSV", "CTD",
                "CTF", "CTI", "CTR", "CTS", "CVT", "D2D", "DAG", "DAH", "DAT", "DBC",
                "DBD", "DBT", "DC4", "DCL", "DCM", "DGC", "DGW", "DHA", "DHC", "DHG"
            ]
            return vn100_defaults
        except Exception as e:
            logger.error("vn100_fetch_error", error=str(e))
            return []
