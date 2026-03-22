"""Macro economic data collector.

Collects VN-Index, VN30, USD/VND, interbank rates, gold, and oil prices
from various data sources and stores them in TimescaleDB.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from decimal import Decimal
from typing import Any

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config.settings import get_settings
from src.database.connection import get_session
from src.models.macro import (
    ALL_INDICATORS,
    INDICATOR_GOLD,
    INDICATOR_INTERBANK_RATE,
    INDICATOR_OIL,
    INDICATOR_USD_VND,
    INDICATOR_VN30,
    INDICATOR_VNINDEX,
    MacroIndicator,
)
from src.utils.logging import get_logger
from src.utils.time_utils import VN_TZ

logger = get_logger(__name__)


class MacroCollector:
    """Collects and stores macro economic indicators.

    Data sources:
    - VN-Index, VN30: vnstock or DNSE API
    - USD/VND, Gold, Oil: yfinance or manual
    - Interbank rates: manual CSV or web scraping
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    async def collect_all(
        self,
        start_date: dt.date | None = None,
        end_date: dt.date | None = None,
    ) -> dict[str, int]:
        """Collect all macro indicators for a date range.

        Returns:
            Dict mapping indicator name → number of records inserted.
        """
        start = start_date or dt.date(2014, 1, 1)
        end = end_date or dt.date.today()

        results = {}

        # VN-Index
        count = await self.collect_index(INDICATOR_VNINDEX, "VNINDEX", start, end)
        results[INDICATOR_VNINDEX] = count

        # VN30
        count = await self.collect_index(INDICATOR_VN30, "VN30", start, end)
        results[INDICATOR_VN30] = count

        # USD/VND
        count = await self.collect_fx(INDICATOR_USD_VND, "USDVND=X", start, end)
        results[INDICATOR_USD_VND] = count

        # Gold
        count = await self.collect_commodity(INDICATOR_GOLD, "GC=F", start, end)
        results[INDICATOR_GOLD] = count

        # Oil (Brent Crude)
        count = await self.collect_commodity(INDICATOR_OIL, "BZ=F", start, end)
        results[INDICATOR_OIL] = count

        logger.info("macro_collection_done", results=results)
        return results

    async def collect_index(
        self,
        indicator_name: str,
        symbol: str,
        start: dt.date,
        end: dt.date,
    ) -> int:
        """Collect VN-Index or VN30 data via vnstock."""
        try:
            from vnstock import Vnstock

            stock = Vnstock().stock(symbol=symbol, source="VCI")
            df = stock.quote.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1D",
            )

            if df is None or df.empty:
                return 0

            records = []
            for _, row in df.iterrows():
                timestamp = row.get("time", row.name)
                if isinstance(timestamp, str):
                    timestamp = pd.Timestamp(timestamp)
                if hasattr(timestamp, "tz_localize") and timestamp.tz is None:
                    timestamp = timestamp.tz_localize(VN_TZ)

                records.append({
                    "timestamp": timestamp,
                    "indicator_name": indicator_name,
                    "value": Decimal(str(round(float(row["close"]), 4))),
                    "open": Decimal(str(round(float(row.get("open", 0)), 4))),
                    "high": Decimal(str(round(float(row.get("high", 0)), 4))),
                    "low": Decimal(str(round(float(row.get("low", 0)), 4))),
                    "close": Decimal(str(round(float(row["close"]), 4))),
                    "volume": int(row.get("volume", 0)),
                    "source": "vnstock",
                    "unit": "points",
                })

            return await self._upsert_records(records)

        except ImportError:
            logger.warning("vnstock_not_installed")
            return 0
        except Exception as e:
            logger.error("index_collect_error", indicator=indicator_name, error=str(e))
            return 0

    async def collect_fx(
        self,
        indicator_name: str,
        yf_symbol: str,
        start: dt.date,
        end: dt.date,
    ) -> int:
        """Collect FX rate via yfinance."""
        return await self._collect_via_yfinance(
            indicator_name, yf_symbol, start, end, unit="VND"
        )

    async def collect_commodity(
        self,
        indicator_name: str,
        yf_symbol: str,
        start: dt.date,
        end: dt.date,
    ) -> int:
        """Collect commodity price via yfinance."""
        return await self._collect_via_yfinance(
            indicator_name, yf_symbol, start, end, unit="USD"
        )

    async def _collect_via_yfinance(
        self,
        indicator_name: str,
        symbol: str,
        start: dt.date,
        end: dt.date,
        unit: str = "USD",
    ) -> int:
        """Collect data from yfinance."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
            )

            if df is None or df.empty:
                return 0

            records = []
            for idx, row in df.iterrows():
                timestamp = idx
                if hasattr(timestamp, "tz_localize") and timestamp.tz is None:
                    timestamp = timestamp.tz_localize(VN_TZ)

                records.append({
                    "timestamp": timestamp,
                    "indicator_name": indicator_name,
                    "value": Decimal(str(round(float(row["Close"]), 4))),
                    "open": Decimal(str(round(float(row.get("Open", 0)), 4))),
                    "high": Decimal(str(round(float(row.get("High", 0)), 4))),
                    "low": Decimal(str(round(float(row.get("Low", 0)), 4))),
                    "close": Decimal(str(round(float(row["Close"]), 4))),
                    "volume": int(row.get("Volume", 0)),
                    "source": "yfinance",
                    "unit": unit,
                })

            return await self._upsert_records(records)

        except ImportError:
            logger.warning("yfinance_not_installed", indicator=indicator_name)
            return 0
        except Exception as e:
            logger.error("yfinance_error", indicator=indicator_name, error=str(e))
            return 0

    async def _upsert_records(self, records: list[dict]) -> int:
        """Upsert macro indicator records into TimescaleDB."""
        if not records:
            return 0

        batch_size = 500
        total = 0

        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            async with get_session() as session:
                stmt = pg_insert(MacroIndicator).values(batch)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["timestamp", "indicator_name"],
                    set_={
                        "value": stmt.excluded.value,
                        "open": stmt.excluded.open,
                        "high": stmt.excluded.high,
                        "low": stmt.excluded.low,
                        "close": stmt.excluded.close,
                        "volume": stmt.excluded.volume,
                        "source": stmt.excluded.source,
                    },
                )
                await session.execute(stmt)
                total += len(batch)

        logger.info("macro_records_upserted", count=total)
        return total
