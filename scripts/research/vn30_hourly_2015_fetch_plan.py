"""Planning helpers for VN30/index hourly 2015 reverse fetches."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_START = date(2015, 1, 1)
TRAIN_END = pd.Timestamp("2024-12-31 23:59:59")
EVAL_START = pd.Timestamp("2025-01-01")
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
LISTING_DATES_PATH = REPO_ROOT / "configs" / "universes" / "vn30_listing_dates.csv"
STOCK_CACHE_ROOT = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_2015"
INDEX_CACHE_ROOT = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly_2015"
STATE_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015" / "fetch_state"
OHLCV = ["open", "high", "low", "close", "volume"]
STOCK_COLUMNS = ["datetime", "ticker", *OHLCV, "provider", "source", "frequency"]
INDEX_COLUMNS = ["datetime", "index_code", *OHLCV, "provider", "source", "frequency"]


@dataclass(frozen=True)
class FetchChunk:
    start: date
    end: date
    granularity: str

    @property
    def key(self) -> str:
        return f"{self.granularity}:{self.start:%Y%m%d}:{self.end:%Y%m%d}"


def parse_date(value: str) -> date | None:
    value = (value or "").strip()
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_universe() -> list[str]:
    return [row["ticker"].strip().upper() for row in read_csv_rows(UNIVERSE_PATH) if row.get("ticker")]


def load_effective_starts() -> dict[str, date]:
    listings = {
        row["ticker"].strip().upper(): parse_date(row.get("first_trading_date", ""))
        for row in read_csv_rows(LISTING_DATES_PATH)
        if row.get("ticker")
    }
    starts: dict[str, date] = {}
    for ticker in read_universe():
        first_trading_date = listings.get(ticker)
        starts[ticker] = max(BASE_START, first_trading_date or BASE_START)
    return starts


def get_provider_current_end() -> date:
    return date.today()


def _month_start(value: date) -> date:
    return date(value.year, value.month, 1)


def _previous_month_end(value: date) -> date:
    return _month_start(value) - timedelta(days=1)


def _quarter_start(value: date) -> date:
    month = ((value.month - 1) // 3) * 3 + 1
    return date(value.year, month, 1)


def _previous_quarter_end(value: date) -> date:
    return _quarter_start(value) - timedelta(days=1)


def build_reverse_chunks(start: date, end: date, granularity: str = "year") -> list[FetchChunk]:
    chunks: list[FetchChunk] = []
    current_end = min(end, get_provider_current_end())
    while current_end >= start:
        if granularity == "year":
            current_start = max(start, date(current_end.year, 1, 1))
            next_end = current_start - timedelta(days=1)
        elif granularity == "quarter":
            current_start = max(start, _quarter_start(current_end))
            next_end = _previous_quarter_end(current_end)
        elif granularity == "month":
            current_start = max(start, _month_start(current_end))
            next_end = _previous_month_end(current_end)
        elif granularity == "5day":
            current_start = max(start, current_end - timedelta(days=4))
            next_end = current_start - timedelta(days=1)
        elif granularity == "1day":
            current_start = current_end
            next_end = current_end - timedelta(days=1)
        else:
            raise ValueError(f"Unsupported granularity: {granularity}")
        chunks.append(FetchChunk(current_start, current_end, granularity))
        current_end = next_end
    return chunks


def fallback_granularity(granularity: str) -> str | None:
    return {
        "year": "quarter",
        "quarter": "month",
        "month": "5day",
        "5day": "1day",
        "1day": None,
    }[granularity]


def split_chunk(chunk: FetchChunk) -> list[FetchChunk]:
    fallback = fallback_granularity(chunk.granularity)
    if fallback is None:
        return []
    return build_reverse_chunks(chunk.start, chunk.end, fallback)


def read_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame(columns=columns)
    if any(column not in frame.columns for column in columns):
        return pd.DataFrame(columns=columns)
    frame = frame[columns].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    return frame.dropna(subset=["datetime"])


def write_frame(path: Path, frame: pd.DataFrame, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame[columns].copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)


def ohlcv_valid(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    frame = frame.copy()
    for column in OHLCV:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[OHLCV].isna().any().any():
        return False
    prices_ok = bool((frame[["open", "high", "low", "close"]] > 0).all().all())
    volume_ok = bool((frame["volume"] >= 0).all())
    ohlc_ok = bool(
        (frame["high"] >= frame["low"]).all()
        and (frame["high"] >= frame[["open", "close"]].max(axis=1)).all()
        and (frame["low"] <= frame[["open", "close"]].min(axis=1)).all()
    )
    return prices_ok and volume_ok and ohlc_ok


def cache_is_usable(symbol: str, asset_type: str) -> bool:
    symbol = symbol.upper()
    if asset_type == "stock":
        path = STOCK_CACHE_ROOT / f"{symbol}.csv"
        frame = read_frame(path, STOCK_COLUMNS)
        if frame.empty:
            return False
        frame = frame[frame["ticker"].astype(str).str.upper().eq(symbol)].copy()
        train_rows = int((frame["datetime"] <= TRAIN_END).sum()) if not frame.empty else 0
        eval_rows = int((frame["datetime"] >= EVAL_START).sum()) if not frame.empty else 0
        enough_rows = train_rows >= 1000 and eval_rows >= 100
    elif asset_type == "index":
        path = INDEX_CACHE_ROOT / f"{symbol}.csv"
        frame = read_frame(path, INDEX_COLUMNS)
        if frame.empty:
            return False
        frame = frame[frame["index_code"].astype(str).str.upper().eq(symbol)].copy()
        enough_rows = not frame.empty
    else:
        raise ValueError(f"Unsupported asset_type: {asset_type}")
    frequency_ok = bool(frame["frequency"].astype(str).eq("1H").all()) if not frame.empty else False
    return frequency_ok and enough_rows and ohlcv_valid(frame)


def checkpoint_path(symbol: str) -> Path:
    return STATE_ROOT / f"{symbol.upper()}_state.json"


def read_checkpoint(symbol: str) -> dict[str, Any]:
    path = checkpoint_path(symbol)
    if not path.exists():
        return {"symbol": symbol.upper(), "completed_chunks": [], "failed_chunks": [], "last_unfinished_chunk": ""}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"symbol": symbol.upper(), "completed_chunks": [], "failed_chunks": [], "last_unfinished_chunk": ""}


def write_checkpoint(symbol: str, state: dict[str, Any]) -> None:
    path = checkpoint_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["symbol"] = symbol.upper()
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def chunk_to_dict(chunk: FetchChunk) -> dict[str, str]:
    row = asdict(chunk)
    row["start"] = chunk.start.isoformat()
    row["end"] = chunk.end.isoformat()
    row["key"] = chunk.key
    return row
