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
from typing import Any

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config.settings import get_settings
from src.database.connection import get_session
from src.models.price import RawPrice, AdjustedPrice
from src.models.company import CompanyProfile
from src.utils.logging import get_logger
from src.utils.time_utils import VN_TZ

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
    ) -> None:
        """Run backdate ingestion for specified tickers.

        Args:
            tickers: List of stock symbols to backfill.
            start_date: Start date (default: 2014-01-01).
            end_date: End date (default: today).
        """
        start = start_date or dt.date(self._settings.backdate_start_year, 1, 1)
        end = end_date or dt.date.today()

        logger.info(
            "backdate_starting",
            tickers=tickers,
            start=start.isoformat(),
            end=end.isoformat(),
        )

        # Load progress from DB
        await self._load_progress(tickers)

        for ticker in tickers:
            ticker_start = self._progress.get(ticker, start)
            if ticker_start >= end:
                logger.info("backdate_skip", ticker=ticker, reason="already_complete")
                continue

            logger.info(
                "backdate_ticker",
                ticker=ticker,
                from_date=ticker_start.isoformat(),
                to_date=end.isoformat(),
            )

            try:
                # 1. Fetch price history (Company profiles are now fetched in bulk beforehand)
                if ticker_start < end:
                    await self._fetch_and_store(ticker, ticker_start, end)
                    
                # Strict rate limiting to avoid Vnstock Sponsor limits (300 req/min = 5 req/sec)
                await asyncio.sleep(0.2)
            except Exception as e:
                logger.error("backdate_ticker_error", ticker=ticker, error=str(e))
                # Even on error, sleep to cool down rate limits
                await asyncio.sleep(0.2)
                continue

        logger.info("backdate_complete")

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

    async def _fetch_ohlc(
        self,
        ticker: str,
        start: dt.date,
        end: dt.date,
    ) -> pd.DataFrame | None:
        """Fetch OHLC data from available data sources.

        Tries: 1) DNSE API  2) vnstock library
        """
        # Try vnstock first (more reliable for historical data)
        try:
            df = await self._fetch_via_vnstock(ticker, start, end)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning("vnstock_fetch_failed", ticker=ticker, error=str(e))

        # Try DNSE REST API
        try:
            df = await self._fetch_via_dnse(ticker, start, end)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning("dnse_fetch_failed", ticker=ticker, error=str(e))

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
