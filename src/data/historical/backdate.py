"""One-time backdate script — fetch 10 years of historical OHLC data.

Fetches historical price data from 2014 to present and inserts into
both raw_prices and adjusted_prices tables.

Features:
- Resumable: tracks last-fetched date per ticker
- Rate-limited: configurable delay between API calls
- Multi-source: tries DNSE API first, falls back to vnstock
"""

from __future__ import annotations

import asyncio
import datetime as dt
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config.settings import get_settings
from src.data.database.connection import get_session
from src.ml.models.price import RawPrice, AdjustedPrice
from src.ml.models.company import CompanyProfile
from src.utils.logging import get_logger
from src.utils.time_utils import VN_TZ
from src.validators.data_quality import DataQualityValidator

logger = get_logger(__name__)


class BackdateIngestor:
    """Fetches and loads historical OHLC data into TimescaleDB.

    Supports resumable operation — tracks progress per ticker
    so it can be interrupted and continued safely.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._progress: dict[str, dt.date] = {}  # ticker → last fetched date

    async def run(
        self,
        tickers: list[str],
        start_date: dt.date | None = None,
        end_date: dt.date | None = None,
        force_refresh: bool = False,
        save_raw_copy: bool = False,
    ) -> int:
        """Run backdate ingestion for specified tickers.

        Args:
            tickers: List of stock symbols to backfill.
            start_date: Start date (default: 2014-01-01).
            end_date: End date (default: today).
            force_refresh: Ignore existing database progress.
            save_raw_copy: Save fetched data to local CSV files.
            
        Returns:
            int: Total rows ingested across all tickers.
        """
        start = start_date or dt.date(self._settings.backdate_start_year, 1, 1)
        end = end_date or dt.date.today()
        total_ingested = 0

        logger.info(
            "backdate_starting",
            tickers=tickers,
            start=start.isoformat(),
            end=end.isoformat(),
        )
        
        # TUI Priority: Reorder tickers to put active TUI ticker first
        tui_ticker_path = Path("data/.tui_ticker")
        if tui_ticker_path.exists():
            try:
                prio_ticker = tui_ticker_path.read_text().strip().upper()
                if prio_ticker in tickers:
                    tickers.remove(prio_ticker)
                    tickers.insert(0, prio_ticker)
                    logger.info("backdate_prioritizing", ticker=prio_ticker)
            except Exception: pass

        # Load progress from DB
        await self._load_progress(tickers)

        semaphore = asyncio.Semaphore(3)  # Balanced for speed vs rate limits

        async def _sync_one(ticker):
            nonlocal total_ingested
            # TUI Priority: Pause background sync if TUI is active
            lock_path = Path("data/.tui_lock")
            while lock_path.exists():
                logger.info("backdate_pause", reason="tui_active_priority")
                await asyncio.sleep(10)

            async with semaphore:
                try:
                    # If force_refresh is true, we ignore existing progress and start from start
                    ticker_start = start if force_refresh else self._progress.get(ticker, start)
                    
                    if ticker_start >= end:
                        logger.info("backdate_skip", ticker=ticker, reason="already_complete")
                        return
                    
                    logger.info("backdate_ticker", ticker=ticker, from_date=ticker_start.isoformat(), to_date=end.isoformat())
                    df = await self._fetch_ohlc(ticker, ticker_start, end)
                    if df is not None and not df.empty:
                        # Save raw copy if requested
                        if save_raw_copy:
                            raw_dir = Path("data/raw")
                            raw_dir.mkdir(parents=True, exist_ok=True)
                            file_path = raw_dir / f"{ticker}_{dt.date.today().isoformat()}.csv"
                            df.to_csv(file_path, index=False)
                            logger.info("backdate_raw_saved", ticker=ticker, path=str(file_path))

                        # Data Quality Validation
                        validator = DataQualityValidator(ticker=ticker)
                        validator.validate_ohlcv(df)
                        
                        await self._insert_batch(ticker, df)
                        rows_count = len(df)
                        total_ingested += rows_count
                        logger.info("backdate_ticker_done", ticker=ticker, rows=rows_count)
                    
                    # 1s cooldown per ticker to stay under 300 req/min safely
                    await asyncio.sleep(1.0)
                except Exception as e:
                    logger.error("backdate_ticker_error", ticker=ticker, error=str(e))

        tasks = [asyncio.create_task(_sync_one(t)) for t in tickers]
        await asyncio.gather(*tasks)
        logger.info("backdate_complete", ticker_count=len(tickers), total_ingested=total_ingested)
        return total_ingested

    async def _fetch_and_store(
        self,
        ticker: str,
        start: dt.date,
        end: dt.date,
    ) -> None:
        """Fetch historical data for a single ticker and store it."""
        df = await self._fetch_ohlc(ticker, start, end)
        if df is None or df.empty:
            logger.warning("backdate_no_data", ticker=ticker)
            return

        # Process in batches
        batch_size = self._settings.backdate_batch_size
        total_rows = len(df)

        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i:i + batch_size]
            await self._insert_batch(ticker, batch)

            # Rate limiting
            await asyncio.sleep(self._settings.backdate_rate_limit_delay)

        # Update progress
        latest_date = df.index.max().date() if hasattr(df.index.max(), "date") else end
        await self._save_progress(ticker, latest_date)

        logger.info(
            "backdate_ticker_done",
            ticker=ticker,
            rows=total_rows,
        )

    async def _fetch_ohlc(self, ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame | None:
        """Fetch with Robust Retry & Multi-Source Fallback."""
        for attempt in range(3):
            try:
                # Primary Source: Vnstock Improved
                df = await self._fetch_via_vnstock_basic(ticker, start, end)
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                wait = (attempt + 1) * 2
                logger.warning("fetch_retry", ticker=ticker, attempt=attempt+1, wait=wait, error=str(e))
                await asyncio.sleep(wait)
        return None

    async def _fetch_via_vnstock_basic(
        self,
        ticker: str,
        start: dt.date,
        end: dt.date,
    ) -> pd.DataFrame | None:
        """Fetch historical data using basic vnstock library."""
        try:
            from vnstock import Vnstock
            stock = Vnstock().stock(symbol=ticker, source="VCI")
            df = stock.quote.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1D"
            )
            if df is not None and not df.empty:
                # Rename to standard schema
                df = df.rename(columns={"time": "timestamp"})
                logger.info("vnstock_basic_fetch_ok", ticker=ticker, rows=len(df))
                return df
        except Exception as e:
            logger.debug("vnstock_basic_error", ticker=ticker, error=str(e))
        return None

    async def _fetch_via_vnstock(
        self,
        ticker: str,
        start: dt.date,
        end: dt.date,
    ) -> pd.DataFrame | None:
        """Fetch historical data using vnstock_data library in chunks."""
        try:
            from vnstock_data import QuoteHistory
            
            # Chunking loop (respecting constraints if necessary, though QuoteHistory 
            # often fetches long ranges without chunks)
            all_dfs = []
            current_start = start
            
            while current_start <= end:
                current_end = min(current_start + dt.timedelta(days=365), end) # 1 year chunks
                
                try:
                    df_chunk = QuoteHistory(source='vci', symbol=ticker).history(
                        start_date=current_start.strftime("%Y-%m-%d"),
                        end_date=current_end.strftime("%Y-%m-%d"),
                        timeframe="1D"
                    )
                except Exception as api_e:
                    logger.debug("quote_history_api_error", ticker=ticker, err=str(api_e))
                    df_chunk = None
                
                if df_chunk is not None and not df_chunk.empty:
                    all_dfs.append(df_chunk)
                
                current_start = current_end + dt.timedelta(days=1)
                await asyncio.sleep(0.2)  # Pause between chunks (300 req/min)

            if not all_dfs:
                return None
                
            # Combine chunks and drop duplicate dates if any overlap
            df = pd.concat(all_dfs).drop_duplicates(subset=["time"])

            # Rename columns to standard schema
            df = df.rename(columns={
                "time": "timestamp",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            })
            logger.info("vnstock_fetch_ok", ticker=ticker, rows=len(df))
            return df

        except ImportError:
            logger.warning("vnstock_not_installed")
        except Exception as e:
            logger.warning("vnstock_error", ticker=ticker, error=str(e))

        return None

    async def _fetch_via_dnse(
        self,
        ticker: str,
        start: dt.date,
        end: dt.date,
    ) -> pd.DataFrame | None:
        """Fetch historical data using DNSE REST API."""
        try:
            from dnse import DnseClient

            with DnseClient(
                api_key=self._settings.dnse_api_key,
                api_secret=self._settings.dnse_api_secret,
            ) as client:
                # DNSE client may have get_ohlc or similar method
                # This depends on the actual API endpoints available
                logger.info("dnse_historical_fetch", ticker=ticker)
                # TODO: implement when DNSE historical API is available
                return None

        except ImportError:
            logger.warning("dnse_not_installed")
        except Exception as e:
            logger.warning("dnse_error", ticker=ticker, error=str(e))

        return None

    async def _insert_batch(self, ticker: str, batch: pd.DataFrame) -> None:
        """Insert a batch of OHLC records into both raw and adjusted tables."""
        raw_values = []
        adj_values = []

        for _, row in batch.iterrows():
            timestamp = row.get("timestamp") or row.name
            if isinstance(timestamp, str):
                timestamp = pd.Timestamp(timestamp)
            if hasattr(timestamp, "tz") and timestamp.tz is None:
                timestamp = timestamp.tz_localize(VN_TZ)

            record = {
                "timestamp": timestamp,
                "ticker": ticker.upper(),
                "exchange": "HOSE",
                "timeframe": "1d",
                "open": Decimal(str(round(float(row["open"]), 2))),
                "high": Decimal(str(round(float(row["high"]), 2))),
                "low": Decimal(str(round(float(row["low"]), 2))),
                "close": Decimal(str(round(float(row["close"]), 2))),
                "volume": int(row.get("volume", 0)),
                "source": "backdate",
            }
            raw_values.append(record)

            adj_record = {**record, "adjustment_factor": Decimal("1.0"), "source": "computed"}
            adj_values.append(adj_record)

        async with get_session() as session:
            # Raw prices upsert
            if raw_values:
                stmt = pg_insert(RawPrice).values(raw_values)
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=["timestamp", "ticker"]
                )
                await session.execute(stmt)

            # Adjusted prices upsert
            if adj_values:
                stmt = pg_insert(AdjustedPrice).values(adj_values)
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=["timestamp", "ticker"]
                )
                await session.execute(stmt)

    async def _load_progress(self, tickers: list[str]) -> None:
        """Load last-fetched dates from database."""
        try:
            async with get_session() as session:
                for ticker in tickers:
                    result = await session.execute(
                        text(
                            "SELECT MAX(timestamp)::date FROM raw_prices "
                            "WHERE ticker = :ticker AND source = 'backdate'"
                        ),
                        {"ticker": ticker.upper()},
                    )
                    row = result.scalar()
                    if row:
                        self._progress[ticker] = row + dt.timedelta(days=1)
        except Exception as e:
            logger.warning("progress_load_failed", error=str(e))

    async def _save_progress(self, ticker: str, last_date: dt.date) -> None:
        """Save progress (tracked implicitly via MAX(timestamp) in raw_prices)."""
        self._progress[ticker] = last_date
        logger.info("progress_saved", ticker=ticker, last_date=last_date.isoformat())

    async def _fetch_and_store_company(self, ticker: str) -> None:
        """Fetch company context data from vnstock."""
        try:
            from vnstock import Vnstock
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            stock = Vnstock().stock(symbol=ticker, source="VCI")
            df = stock.company.overview()
            
            if df is None or df.empty:
                return
                
            # Extract basic data
            row = df.iloc[0]
            
            # Using defaults for fields that might not exist in VCI overview
            values = {
                "ticker": ticker.upper(),
                "exchange": str(row.get("exchange", "HOSE")),
                "industry": str(row.get("industry", "")),
                "company_name": str(row.get("company_name", "")),
                "short_name": str(row.get("short_name", "")),
                "issue_share": float(row.get("issue_share", 0)),
                "charter_capital": float(row.get("charter_capital", 0)),
                "market_cap": float(row.get("market_cap", 0)),
                "established_year": str(row.get("established_year", "")),
                "no_employees": int(row.get("no_employees", 0)),
                "no_shareholders": int(row.get("no_shareholders", 0)),
                "foreign_percent": float(row.get("foreign_percent", 0.0)),
                "website": str(row.get("website", "")),
            }
            
            async with get_session() as session:
                stmt = pg_insert(CompanyProfile).values([values])
                stmt = stmt.on_conflict_do_update(
                    index_elements=["ticker"],
                    set_={k: v for k, v in values.items() if k != "ticker"}
                )
                await session.execute(stmt)
                
        except Exception as e:
            logger.debug("company_info_fetch_error", ticker=ticker, error=str(e))
