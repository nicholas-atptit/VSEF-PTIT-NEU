"""Data loader — fetches OHLCV from TimescaleDB or generates mock data.

Provides daily-aggregated OHLCV DataFrames for the ML pipeline.
Mock mode generates realistic synthetic stock data for offline testing.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_ohlcv_from_db(
    ticker: str,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
) -> pd.DataFrame:
    """Load daily OHLCV data from the adjusted_prices table.

    Args:
        ticker: Stock symbol (e.g., 'SSI', 'HPG').
        start_date: Start date filter (inclusive).
        end_date: End date filter (inclusive).

    Returns:
        DataFrame with columns [date, open, high, low, close, volume],
        sorted ascending by date.
    """
    settings = get_settings()
    engine = create_engine(settings.timescale_sync_url)

    query = """
        SELECT
            DATE(timestamp) AS date,
            (ARRAY_AGG(open ORDER BY timestamp ASC))[1]   AS open,
            MAX(high)                                      AS high,
            MIN(low)                                       AS low,
            (ARRAY_AGG(close ORDER BY timestamp DESC))[1]  AS close,
            SUM(volume)                                    AS volume
        FROM adjusted_prices
        WHERE ticker = :ticker
    """
    params: dict = {"ticker": ticker}

    if start_date:
        query += " AND DATE(timestamp) >= :start_date"
        params["start_date"] = start_date
    if end_date:
        query += " AND DATE(timestamp) <= :end_date"
        params["end_date"] = end_date

    query += " GROUP BY DATE(timestamp) ORDER BY date ASC"

    try:
        df = pd.read_sql(text(query), engine, params=params)
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df["volume"] = df["volume"].astype(int)
        logger.info("ohlcv_loaded_from_db", ticker=ticker, rows=len(df))
        return df
    except Exception as e:
        logger.error("ohlcv_load_error", ticker=ticker, error=str(e))
        raise
    finally:
        engine.dispose()


def load_ohlcv_from_vnstock(ticker: str, num_days: int = 600) -> pd.DataFrame:
    """Fetch recent historical OHLCV data from vnstock as a fallback.
    
    Provides a live API connection when the local TimescaleDB is unavailable.
    """
    import os
    from vnstock import Vnstock
    
    # Inject API Key automatically
    settings = get_settings()
    os.environ["VNAI_API_KEY"] = settings.vnstock_api_key
    os.environ["VNSTOCK_API_KEY"] = settings.vnstock_api_key
    
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=int(num_days * 1.5))
    
    stock = Vnstock().stock(symbol=ticker, source='VCI')
    df = stock.quote.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    
    if df is None or df.empty:
        raise ValueError(f"No data returned from vnstock for {ticker}")
        
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(int)
    
    df = df.sort_values(by="date")
    logger.info("ohlcv_loaded_from_vnstock", ticker=ticker, rows=len(df))
    return df


def generate_mock_data(
    ticker: str = "MOCK",
    num_days: int = 600,
    start_price: float = 30.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic daily OHLCV data for testing.

    Uses geometric Brownian motion to simulate stock price movement,
    producing a DataFrame identical in structure to ``load_ohlcv_from_db``.

    Args:
        ticker: Ticker symbol for the mock data.
        num_days: Number of trading days to generate.
        start_price: Initial closing price.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns [date, open, high, low, close, volume].
    """
    rng = np.random.default_rng(seed)

    # Geometric Brownian Motion parameters
    mu = 0.0003  # daily drift
    sigma = 0.02  # daily volatility

    dates = pd.bdate_range(
        start=dt.date(2023, 1, 2), periods=num_days, freq="B"
    )

    closes = np.zeros(num_days)
    closes[0] = start_price

    for i in range(1, num_days):
        daily_return = mu + sigma * rng.standard_normal()
        closes[i] = closes[i - 1] * np.exp(daily_return)

    # Build OHLCV from close prices
    intraday_vol = 0.008
    opens = np.zeros(num_days)
    highs = np.zeros(num_days)
    lows = np.zeros(num_days)
    volumes = np.zeros(num_days, dtype=int)

    opens[0] = start_price
    for i in range(1, num_days):
        gap = rng.normal(0, 0.003)
        opens[i] = closes[i - 1] * (1 + gap)

    for i in range(num_days):
        mid = (opens[i] + closes[i]) / 2
        spread = abs(opens[i] - closes[i])
        extra_high = abs(rng.normal(0, intraday_vol * mid))
        extra_low = abs(rng.normal(0, intraday_vol * mid))
        highs[i] = max(opens[i], closes[i]) + extra_high
        lows[i] = min(opens[i], closes[i]) - extra_low
        # Ensure low is positive
        lows[i] = max(lows[i], closes[i] * 0.93)
        volumes[i] = int(rng.integers(500_000, 5_000_000))

    df = pd.DataFrame(
        {
            "date": dates[:num_days],
            "open": np.round(opens, 2),
            "high": np.round(highs, 2),
            "low": np.round(lows, 2),
            "close": np.round(closes, 2),
            "volume": volumes,
        }
    )

    logger.info("mock_data_generated", ticker=ticker, rows=len(df))
    return df
