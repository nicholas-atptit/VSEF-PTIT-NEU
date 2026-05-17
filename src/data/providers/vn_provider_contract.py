"""Canonical contract for Vietnamese stock/index OHLCV provider requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd


class ProviderName(StrEnum):
    VNSTOCK_DATA = "vnstock_data"
    LEGACY_VNSTOCK = "legacy_vnstock"


class SourceName(StrEnum):
    KBS = "KBS"
    VCI = "VCI"


class AssetType(StrEnum):
    STOCK = "stock"
    INDEX = "index"


class Frequency(StrEnum):
    HOURLY = "1H"
    DAILY = "1D"


ALLOWED_INDEX_CODES = {"VNINDEX", "HNXINDEX", "UPCOMINDEX", "VN30", "HNX30", "VN100"}
DISALLOWED_INDEX_ALIASES = {"VN30INDEX": "VN30"}
UNSUPPORTED_INDEX_CODES = {"VNXALL"}
DEFAULT_HOURLY_INTERVAL = Frequency.HOURLY.value


@dataclass(frozen=True)
class FetchRequest:
    symbol: str
    asset_type: AssetType
    start: str
    end: str
    frequency: Frequency = Frequency.HOURLY
    preferred_sources: tuple[SourceName, ...] = (SourceName.KBS, SourceName.VCI)
    allow_legacy_fallback: bool = False
    allow_daily: bool = False
    allow_resample: bool = False


@dataclass(frozen=True)
class FetchResponse:
    symbol: str
    asset_type: AssetType
    frequency: Frequency
    provider: ProviderName | str
    source: SourceName | str
    rows: int
    first_datetime: pd.Timestamp | None
    last_datetime: pd.Timestamp | None
    data: pd.DataFrame = field(repr=False)
    attempts: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FetchFailure:
    symbol: str
    provider: ProviderName | str
    source: SourceName | str
    error_type: str
    error_message: str
