"""Macro economic data collector.

This collector is intentionally ``vnstock_data``-first. If a required macro or
commodity series is not available from the installed provider/runtime, the
collector returns zero rows and logs an explicit gap instead of silently falling
back to another vendor.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config.settings import get_settings
from src.data.adapters.vnstock_adapter import VnstockAdapter
from src.data.database.connection import get_session
from src.ml.models.macro import (
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
    """Collect and store macro indicators using ``vnstock_data`` only."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._adapter = VnstockAdapter()

    async def collect_all(
        self,
        start_date: dt.date | None = None,
        end_date: dt.date | None = None,
    ) -> dict[str, int]:
        start = start_date or dt.date(2014, 1, 1)
        end = end_date or dt.date.today()

        results = {
            INDICATOR_VNINDEX: await self.collect_index(INDICATOR_VNINDEX, "VNINDEX", start, end),
            INDICATOR_VN30: await self.collect_index(INDICATOR_VN30, "VN30", start, end),
            INDICATOR_USD_VND: await self.collect_fx(INDICATOR_USD_VND, start, end),
            INDICATOR_INTERBANK_RATE: await self.collect_interest_rate(INDICATOR_INTERBANK_RATE, start, end),
            INDICATOR_GOLD: await self.collect_commodity(INDICATOR_GOLD, start, end),
            INDICATOR_OIL: await self.collect_commodity(INDICATOR_OIL, start, end),
        }
        logger.info("macro_collection_done", results=results)
        return results

    async def collect_index(
        self,
        indicator_name: str,
        symbol: str,
        start: dt.date,
        end: dt.date,
    ) -> int:
        df = self._adapter.get_index_ohlcv(
            symbol=symbol,
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            interval="1D",
        )
        if df is None or df.empty:
            logger.warning("macro_index_unavailable", indicator=indicator_name, symbol=symbol)
            return 0
        records = self._ohlcv_records(
            df,
            indicator_name=indicator_name,
            unit="points",
            source=df.attrs.get("source_name", "vnstock_data"),
        )
        return await self._upsert_records(records)

    async def collect_fx(
        self,
        indicator_name: str,
        start: dt.date,
        end: dt.date,
    ) -> int:
        df = self._adapter.get_macro_exchange_rate(
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )
        if df is None or df.empty:
            logger.warning("macro_fx_unavailable", indicator=indicator_name)
            return 0
        records = self._series_records(
            df,
            indicator_name=indicator_name,
            unit="VND",
            source=df.attrs.get("source_name", "vnstock_data"),
        )
        return await self._upsert_records(records)

    async def collect_interest_rate(
        self,
        indicator_name: str,
        start: dt.date,
        end: dt.date,
    ) -> int:
        df = self._adapter.get_macro_interest_rate(
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )
        if df is None or df.empty:
            logger.warning("macro_interest_rate_unavailable", indicator=indicator_name)
            return 0
        records = self._series_records(
            df,
            indicator_name=indicator_name,
            unit="%",
            source=df.attrs.get("source_name", "vnstock_data"),
        )
        return await self._upsert_records(records)

    async def collect_commodity(
        self,
        indicator_name: str,
        start: dt.date,
        end: dt.date,
    ) -> int:
        if indicator_name == INDICATOR_GOLD:
            df = self._adapter.get_commodity_gold(
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
            )
            unit = "VND"
        elif indicator_name == INDICATOR_OIL:
            df = self._adapter.get_commodity_oil(
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
            )
            unit = "USD"
        else:
            logger.warning("commodity_indicator_unsupported", indicator=indicator_name)
            return 0

        if df is None or df.empty:
            logger.warning("macro_commodity_unavailable", indicator=indicator_name)
            return 0
        records = self._series_records(
            df,
            indicator_name=indicator_name,
            unit=unit,
            source=df.attrs.get("source_name", "vnstock_data"),
        )
        return await self._upsert_records(records)

    @staticmethod
    def _normalize_timestamp(value: object) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(VN_TZ)
        return timestamp

    def _ohlcv_records(
        self,
        frame: pd.DataFrame,
        *,
        indicator_name: str,
        unit: str,
        source: str,
    ) -> list[dict]:
        records: list[dict] = []
        normalized = frame.copy()
        normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
        for _, row in normalized.iterrows():
            timestamp = self._normalize_timestamp(row["date"])
            records.append(
                {
                    "timestamp": timestamp,
                    "indicator_name": indicator_name,
                    "value": Decimal(str(round(float(row["close"]), 4))),
                    "open": Decimal(str(round(float(row.get("open", row["close"])), 4))),
                    "high": Decimal(str(round(float(row.get("high", row["close"])), 4))),
                    "low": Decimal(str(round(float(row.get("low", row["close"])), 4))),
                    "close": Decimal(str(round(float(row["close"]), 4))),
                    "volume": int(row.get("volume", 0) or 0),
                    "source": source,
                    "unit": unit,
                }
            )
        return records

    def _series_records(
        self,
        frame: pd.DataFrame,
        *,
        indicator_name: str,
        unit: str,
        source: str,
    ) -> list[dict]:
        normalized = frame.copy()
        if "date" not in normalized.columns:
            if isinstance(normalized.index, pd.DatetimeIndex):
                normalized = normalized.reset_index().rename(columns={normalized.index.name or "index": "date"})
            else:
                return []
        normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()

        value_series = None
        if {"buy", "sell"} <= set(normalized.columns):
            value_series = (pd.to_numeric(normalized["buy"], errors="coerce") + pd.to_numeric(normalized["sell"], errors="coerce")) / 2.0
        else:
            for candidate in ("close", "value", "rate", "mid", "price", "sell", "buy"):
                if candidate in normalized.columns:
                    value_series = pd.to_numeric(normalized[candidate], errors="coerce")
                    break
        if value_series is None:
            return []

        records: list[dict] = []
        for idx, row in normalized.iterrows():
            value = value_series.iloc[idx]
            if pd.isna(value):
                continue
            timestamp = self._normalize_timestamp(row["date"])
            records.append(
                {
                    "timestamp": timestamp,
                    "indicator_name": indicator_name,
                    "value": Decimal(str(round(float(value), 4))),
                    "open": Decimal(str(round(float(row.get("open", value)), 4))),
                    "high": Decimal(str(round(float(row.get("high", value)), 4))),
                    "low": Decimal(str(round(float(row.get("low", value)), 4))),
                    "close": Decimal(str(round(float(row.get("close", value)), 4))),
                    "volume": int(row.get("volume", 0) or 0),
                    "source": source,
                    "unit": unit,
                }
            )
        return records

    async def _upsert_records(self, records: list[dict]) -> int:
        if not records:
            return 0

        batch_size = 500
        total = 0
        for offset in range(0, len(records), batch_size):
            batch = records[offset:offset + batch_size]
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
