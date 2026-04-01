"""Data loader — fetches OHLCV from TimescaleDB or generates mock data.

Provides daily-aggregated OHLCV DataFrames for the ML pipeline.
Mock mode generates realistic synthetic stock data for offline testing.

VN100 Extensions (v2):
    - ``VN100DataLoader`` class for batch-loading across the VN100 universe
    - ``load_vn100_daily_dataset()`` convenience function
    - ``load_ohlcv_from_csv()``  file-backed fallback
    - Market index / fundamentals / news join helpers
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from config.settings import get_settings, PROJECT_ROOT
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Default paths for file-backed data  (relative to PROJECT_ROOT)
# ═══════════════════════════════════════════════════════════════════════════
_DEFAULT_DAILY_CSV_DIR = PROJECT_ROOT / "data" / "daily_market_split_data"
_DEFAULT_MARKET_PROXY_PATH = PROJECT_ROOT / "data" / "market_proxy.csv"
_DEFAULT_FUNDAMENTALS_PATH = PROJECT_ROOT / "data" / "fundamentals_latest.csv"
_DEFAULT_SENTIMENT_PATH = PROJECT_ROOT / "data" / "sentiment_features.csv"


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


# ═══════════════════════════════════════════════════════════════════════════
# FILE-BACKED LOADING  (CSV fallback)
# ═══════════════════════════════════════════════════════════════════════════


def load_ohlcv_from_csv(
    ticker: str,
    csv_dir: Path | str | None = None,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
) -> pd.DataFrame:
    """Load daily OHLCV from a per-ticker CSV file (``data/daily_market_split_data/<TICKER>.csv``).

    The CSV is expected to have columns ``time|date, open, high, low, close, volume``
    as written by the existing ``scripts/extract_daily_csv.py``.

    Args:
        ticker: Stock symbol (case-insensitive).
        csv_dir: Directory containing per-ticker CSVs.  Defaults to
                 ``data/daily_market_split_data/``.
        start_date: Optional inclusive lower-bound.
        end_date: Optional inclusive upper-bound.

    Returns:
        DataFrame with columns ``[date, open, high, low, close, volume, ticker]``.
    """
    csv_dir = Path(csv_dir) if csv_dir else _DEFAULT_DAILY_CSV_DIR
    csv_path = csv_dir / f"{ticker.upper()}.csv"

    if not csv_path.exists():
        logger.warning("csv_not_found", ticker=ticker, path=str(csv_path))
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)

        # Normalise column names
        if "time" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"time": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        # Ensure standard numeric types
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        if "volume" in df.columns:
            df["volume"] = df["volume"].astype(int)

        # Date filtering
        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["date"] <= pd.Timestamp(end_date)]

        # Ensure ticker column
        df["ticker"] = ticker.upper()
        df = df.sort_values("date").reset_index(drop=True)

        logger.debug("ohlcv_loaded_from_csv", ticker=ticker, rows=len(df))
        return df

    except Exception as exc:
        logger.error("csv_load_error", ticker=ticker, error=str(exc))
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# AUXILIARY DATA LOADERS  (market proxy, fundamentals, sentiment/news)
# ═══════════════════════════════════════════════════════════════════════════


def load_market_proxy(
    path: Path | str | None = None,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
) -> pd.DataFrame:
    """Load the market proxy (VNINDEX daily return) CSV.

    Args:
        path: Path to ``market_proxy.csv``.
        start_date: Optional inclusive lower-bound.
        end_date: Optional inclusive upper-bound.

    Returns:
        DataFrame with columns ``[date, m_ret]``.
    """
    path = Path(path) if path else _DEFAULT_MARKET_PROXY_PATH
    if not path.exists():
        logger.warning("market_proxy_not_found", path=str(path))
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    logger.debug("market_proxy_loaded", rows=len(df))
    return df


def load_fundamentals(
    path: Path | str | None = None,
    tickers: List[str] | None = None,
) -> pd.DataFrame:
    """Load the fundamentals CSV if it exists.

    Args:
        path: Path to fundamentals CSV.
        tickers: Optional ticker filter list.

    Returns:
        DataFrame with fundamental columns, or empty DataFrame.
    """
    path = Path(path) if path else _DEFAULT_FUNDAMENTALS_PATH
    if not path.exists():
        logger.debug("fundamentals_not_found", path=str(path))
        return pd.DataFrame()

    df = pd.read_csv(path)

    if tickers:
        upper = [t.upper() for t in tickers]
        if "ticker" in df.columns:
            df = df[df["ticker"].str.upper().isin(upper)]

    logger.debug("fundamentals_loaded", rows=len(df))
    return df


def load_sentiment(
    path: Path | str | None = None,
    tickers: List[str] | None = None,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
) -> pd.DataFrame:
    """Load the sentiment / news features CSV if it exists.

    Args:
        path: Path to sentiment_features CSV.
        tickers: Optional ticker filter.
        start_date: Optional date lower-bound.
        end_date: Optional date upper-bound.

    Returns:
        DataFrame with sentiment columns, or empty DataFrame.
    """
    path = Path(path) if path else _DEFAULT_SENTIMENT_PATH
    if not path.exists():
        logger.debug("sentiment_not_found", path=str(path))
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["date"] <= pd.Timestamp(end_date)]

    if tickers and "ticker" in df.columns:
        upper = [t.upper() for t in tickers]
        df = df[df["ticker"].str.upper().isin(upper)]

    logger.debug("sentiment_loaded", rows=len(df))
    return df


# ═══════════════════════════════════════════════════════════════════════════
# VN100 DATA LOADER  (batch dataset builder)
# ═══════════════════════════════════════════════════════════════════════════


class VN100DataLoader:
    """Build standardised daily datasets for batches of VN100 tickers.

    Typical usage::

        loader = VN100DataLoader()
        df = loader.build_dataset(
            tickers=["FPT", "HPG", "VNM"],
            start_date=dt.date(2022, 1, 1),
            end_date=dt.date(2024, 12, 31),
            join_market=True,
        )

    Data priority:
        1. DB-backed  (TimescaleDB ``adjusted_prices``)
        2. CSV fallback (``data/daily_market_split_data/<TICKER>.csv``)
        3. vnstock live API (last resort)

    The resulting DataFrame always has at least::

        [date, open, high, low, close, volume, ticker]

    with optional market / fundamental / sentiment columns merged.
    """

    def __init__(
        self,
        csv_dir: Path | str | None = None,
        prefer_source: str = "csv",
    ) -> None:
        """Initialise VN100DataLoader.

        Args:
            csv_dir: Root directory for per-ticker CSVs.
            prefer_source: Loading priority — ``"db"`` tries TimescaleDB first,
                           ``"csv"`` (default) tries local CSVs first.
        """
        self.csv_dir = Path(csv_dir) if csv_dir else _DEFAULT_DAILY_CSV_DIR
        self.prefer_source = prefer_source.lower()
        self._settings = get_settings()
        logger.info(
            "vn100_data_loader_init",
            csv_dir=str(self.csv_dir),
            prefer_source=self.prefer_source,
        )

    # ── Single-ticker loader with fallback chain ─────────────────────

    def _load_single(
        self,
        ticker: str,
        start_date: dt.date | None = None,
        end_date: dt.date | None = None,
    ) -> pd.DataFrame:
        """Load OHLCV for one ticker with automatic fallback.

        Tries sources in the order dictated by ``self.prefer_source``.
        """
        df = pd.DataFrame()

        if self.prefer_source == "db":
            sources = [
                ("db", lambda: load_ohlcv_from_db(ticker, start_date, end_date)),
                ("csv", lambda: load_ohlcv_from_csv(ticker, self.csv_dir, start_date, end_date)),
                ("vnstock", lambda: load_ohlcv_from_vnstock(ticker)),
            ]
        else:
            sources = [
                ("csv", lambda: load_ohlcv_from_csv(ticker, self.csv_dir, start_date, end_date)),
                ("db", lambda: load_ohlcv_from_db(ticker, start_date, end_date)),
                ("vnstock", lambda: load_ohlcv_from_vnstock(ticker)),
            ]

        for name, loader in sources:
            try:
                df = loader()
                if df is not None and not df.empty:
                    # Ensure ticker column is present
                    if "ticker" not in df.columns:
                        df["ticker"] = ticker.upper()
                    logger.debug(
                        "single_ticker_loaded",
                        ticker=ticker,
                        source=name,
                        rows=len(df),
                    )
                    return df
            except Exception as exc:
                logger.debug(
                    "source_fallback",
                    ticker=ticker,
                    source=name,
                    error=str(exc),
                )

        logger.warning("no_data_for_ticker", ticker=ticker)
        return pd.DataFrame()

    # ── Public: batch dataset builder ─────────────────────────────────

    def build_dataset(
        self,
        tickers: List[str],
        start_date: dt.date | None = None,
        end_date: dt.date | None = None,
        join_market: bool = True,
        join_fundamentals: bool = False,
        join_sentiment: bool = False,
        min_rows_per_ticker: int = 60,
    ) -> pd.DataFrame:
        """Build a multi-ticker daily dataset suitable for ML training.

        Args:
            tickers: List of ticker symbols.
            start_date: Inclusive start date.
            end_date: Inclusive end date.
            join_market: Merge market proxy (``m_ret``) column.
            join_fundamentals: Merge fundamentals if available.
            join_sentiment: Merge sentiment / news features if available.
            min_rows_per_ticker: Skip tickers with fewer rows.

        Returns:
            Concatenated DataFrame sorted ``[ticker, date]`` with a
            ``ticker`` column identifying each stock.
        """
        if not tickers:
            logger.warning("build_dataset_empty_tickers")
            return pd.DataFrame()

        frames: List[pd.DataFrame] = []
        skipped: List[str] = []

        for ticker in tickers:
            df = self._load_single(ticker, start_date, end_date)
            if df.empty or len(df) < min_rows_per_ticker:
                skipped.append(ticker)
                continue
            frames.append(df)

        if not frames:
            logger.warning(
                "build_dataset_no_data",
                requested=len(tickers),
                skipped=len(skipped),
            )
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)

        # ── Join auxiliary data ──────────────────────────────────
        if join_market:
            market_df = load_market_proxy(start_date=start_date, end_date=end_date)
            if not market_df.empty:
                result = result.merge(market_df, on="date", how="left")
                logger.debug("market_proxy_joined", market_rows=len(market_df))

        if join_fundamentals:
            fund_df = load_fundamentals(tickers=tickers)
            if not fund_df.empty and "date" in fund_df.columns:
                fund_df["date"] = pd.to_datetime(fund_df["date"]).dt.normalize()
                result = result.merge(fund_df, on=["ticker", "date"], how="left")
                logger.debug("fundamentals_joined")

        if join_sentiment:
            sent_df = load_sentiment(
                tickers=tickers, start_date=start_date, end_date=end_date
            )
            if not sent_df.empty:
                result = result.merge(sent_df, on=["ticker", "date"], how="left")
                logger.debug("sentiment_joined")

        result = result.sort_values(["ticker", "date"]).reset_index(drop=True)

        # ── Basic validation ────────────────────────────────────
        n_tickers = result["ticker"].nunique()
        logger.info(
            "vn100_dataset_built",
            total_rows=len(result),
            tickers_loaded=n_tickers,
            tickers_skipped=len(skipped),
            date_range=(
                f"{result['date'].min().date()} -> {result['date'].max().date()}"
                if len(result) > 0
                else "empty"
            ),
        )
        return result

    def build_inference_dataset(
        self,
        tickers: List[str],
        lookback_days: int = 120,
        join_market: bool = True,
    ) -> pd.DataFrame:
        """Build a dataset suitable for batch inference (latest N trading days).

        This is a convenience wrapper around :meth:`build_dataset` that
        automatically computes the date window from today.

        Args:
            tickers: List of ticker symbols.
            lookback_days: Calendar days to look back (default 120 ≈ 6 months).
            join_market: Merge market proxy column.

        Returns:
            DataFrame with the most recent ``lookback_days`` of data per ticker.
        """
        end = dt.date.today()
        start = end - dt.timedelta(days=lookback_days)
        return self.build_dataset(
            tickers=tickers,
            start_date=start,
            end_date=end,
            join_market=join_market,
            min_rows_per_ticker=10,
        )


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════


def load_vn100_daily_dataset(
    tickers: List[str] | None = None,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    join_market: bool = True,
    join_fundamentals: bool = False,
    join_sentiment: bool = False,
    prefer_source: str = "csv",
) -> pd.DataFrame:
    """One-call helper to build a VN100 daily dataset.

    If ``tickers`` is *None*, the full VN100 universe is loaded from
    :func:`src.data.universe.get_vn100_universe`.

    Args:
        tickers: Explicit ticker list, or None -> VN100 universe.
        start_date: Inclusive start date.
        end_date: Inclusive end date.
        join_market: Merge market proxy column.
        join_fundamentals: Merge fundamentals.
        join_sentiment: Merge sentiment / news.
        prefer_source: ``"csv"`` or ``"db"``.

    Returns:
        Multi-ticker DataFrame ready for feature engineering.
    """
    if tickers is None:
        from src.data.universe import get_vn100_universe
        tickers = get_vn100_universe()

    loader = VN100DataLoader(prefer_source=prefer_source)
    return loader.build_dataset(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        join_market=join_market,
        join_fundamentals=join_fundamentals,
        join_sentiment=join_sentiment,
    )

