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

from copy import deepcopy
import datetime as dt
import importlib.util
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
_DEFAULT_SECTOR_PROXIES_PATH = PROJECT_ROOT / "data" / "sector_proxies.csv"
_DEFAULT_TICKER_SECTORS_PATH = PROJECT_ROOT / "data" / "ticker_sectors.csv"
_DEFAULT_MARKET_BREADTH_PATH = PROJECT_ROOT / "data" / "market_breadth.csv"
_DEFAULT_MACRO_CONTEXT_PATH = PROJECT_ROOT / "data" / "macro_context.csv"
_DEFAULT_FOREIGN_FLOW_PATH = PROJECT_ROOT / "data" / "foreign_flow.csv"
_BREADTH_CONTEXT_METADATA_COLUMNS = [
    "breadth_context_available",
    "breadth_context_source_date",
    "breadth_context_missing",
]
_FOREIGN_FLOW_CONTEXT_METADATA_COLUMNS = [
    "foreign_flow_context_available",
    "foreign_flow_context_source_date",
    "foreign_flow_context_missing",
]
_FOREIGN_FLOW_PROVENANCE_COLUMNS = {
    "source",
    "source_date",
    "retrieved_at",
    "provider",
    "coverage_note",
}

DIRECT_VNSTOCK_PROVENANCE = "direct_vnstock_data"
DERIVED_VNSTOCK_PROVENANCE = "derived_from_vnstock_data"
LOCAL_COMPUTATION_PROVENANCE = "local_computation"
STUB_PROVENANCE = "stub_todo"
DATA_QUALITY_CONTRACT_VERSION = 1
DEFAULT_STALE_AFTER_DAYS = {
    "market_proxy.csv": 14,
    "fundamentals_latest.csv": 45,
    "sentiment_features.csv": 7,
    "sector_proxies.csv": 14,
    "ticker_sectors.csv": 30,
    "market_breadth.csv": 7,
    "macro_context.csv": 7,
    "foreign_flow.csv": 7,
}
ARTIFACT_CACHE_VERSION = 2
INCREMENTAL_CONTEXT_LOOKBACK_DAYS = 400
_ARTIFACT_FRAME_CACHE: dict[tuple[Any, ...], pd.DataFrame] = {}


def _clone_with_attrs(df: pd.DataFrame) -> pd.DataFrame:
    cloned = df.copy(deep=True)
    cloned.attrs = deepcopy(getattr(df, "attrs", {}))
    return cloned


def _concat_frame_block(
    df: pd.DataFrame,
    feature_map: dict[str, Any] | pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(feature_map, pd.DataFrame):
        if feature_map.empty and len(feature_map.columns) == 0:
            return df
        block = feature_map.copy()
        if not block.index.equals(df.index):
            block = block.reindex(df.index)
    else:
        if not feature_map:
            return df
        block = pd.DataFrame(feature_map, index=df.index)
    attrs = deepcopy(getattr(df, "attrs", {}))
    overlap = [column for column in block.columns if column in df.columns]
    base = df.drop(columns=overlap) if overlap else df
    result = pd.concat([base, block], axis=1)
    result.attrs = attrs
    return result


def _merge_incremental_frames(
    historical_prefix: pd.DataFrame | None,
    recent_frame: pd.DataFrame | None,
    *,
    sort_columns: tuple[str, ...],
    dedupe_columns: tuple[str, ...],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for candidate in (historical_prefix, recent_frame):
        if candidate is not None and not candidate.empty:
            frames.append(candidate.copy())
    if not frames:
        return pd.DataFrame()
    all_columns = list(dict.fromkeys(column for frame in frames for column in frame.columns))
    merged = pd.concat([frame.reindex(columns=all_columns) for frame in frames], ignore_index=True)
    merged = merged.sort_values(list(sort_columns)).drop_duplicates(subset=list(dedupe_columns), keep="last")
    return merged.reset_index(drop=True)


def _normalize_cache_part(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve()) if value.exists() else str(value)
    if isinstance(value, (dt.date, dt.datetime, pd.Timestamp)):
        return pd.Timestamp(value).normalize().isoformat()
    if isinstance(value, list):
        return tuple(_normalize_cache_part(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_normalize_cache_part(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _normalize_cache_part(raw)) for key, raw in value.items()))
    return value


def _file_snapshot_token(path: Path | str | None) -> tuple[str, int | None, int | None] | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return (str(resolved), None, None)
    stat = resolved.stat()
    return (str(resolved.resolve()), int(stat.st_mtime_ns), int(stat.st_size))


def _directory_snapshot_token(path: Path | str | None, pattern: str = "*.csv") -> tuple[Any, ...] | None:
    if path is None:
        return None
    directory = Path(path)
    if not directory.exists():
        return (str(directory), "missing")
    snapshots = []
    for candidate in sorted(directory.glob(pattern)):
        stat = candidate.stat()
        snapshots.append((candidate.name, int(stat.st_mtime_ns), int(stat.st_size)))
    return (str(directory.resolve()), tuple(snapshots))


def _artifact_cache_key(name: str, **parts: Any) -> tuple[Any, ...]:
    normalized_parts = tuple(sorted((key, _normalize_cache_part(value)) for key, value in parts.items()))
    return (ARTIFACT_CACHE_VERSION, name, normalized_parts)


def _artifact_cache_get(key: tuple[Any, ...]) -> pd.DataFrame | None:
    cached = _ARTIFACT_FRAME_CACHE.get(key)
    if cached is None:
        return None
    result = _clone_with_attrs(cached)
    result.attrs["artifact_cache_status"] = "hit"
    return result


def _artifact_cache_set(key: tuple[Any, ...], df: pd.DataFrame) -> pd.DataFrame:
    stored = _clone_with_attrs(df)
    stored.attrs["artifact_cache_status"] = "miss"
    _ARTIFACT_FRAME_CACHE[key] = stored
    return _clone_with_attrs(stored)


def clear_artifact_frame_cache() -> None:
    _ARTIFACT_FRAME_CACHE.clear()


def _load_existing_artifact_frame(output_path: Path | str | None) -> pd.DataFrame | None:
    if output_path is None:
        return None
    path = Path(output_path)
    if not path.exists():
        return None
    try:
        existing = _normalize_date_frame(pd.read_csv(path))
    except Exception:
        return None
    if existing.empty or "date" not in existing.columns:
        return None
    existing["date"] = pd.to_datetime(existing["date"], errors="coerce").dt.normalize()
    existing = existing.dropna(subset=["date"]).copy()
    if "ticker" in existing.columns:
        existing["ticker"] = existing["ticker"].astype(str).str.upper()
    return existing


def _incremental_history_prefix(
    output_path: Path | str | None,
    *,
    key_columns: tuple[str, ...] | None = None,
    lookback_days: int = INCREMENTAL_CONTEXT_LOOKBACK_DAYS,
) -> tuple[pd.Timestamp | None, pd.DataFrame | None]:
    existing = _load_existing_artifact_frame(output_path)
    if existing is None or existing.empty or "date" not in existing.columns:
        return None, None
    resolved_keys = tuple(column for column in (key_columns or _default_key_columns(existing) or ("date",)) if column in existing.columns)
    if "date" not in resolved_keys:
        resolved_keys = (*resolved_keys, "date")
    existing = (
        existing.sort_values(list(resolved_keys))
        .drop_duplicates(subset=list(resolved_keys), keep="last")
        .reset_index(drop=True)
    )
    if existing.empty:
        return None, None
    rebuild_start = pd.Timestamp(existing["date"].max()).normalize() - pd.Timedelta(days=lookback_days)
    historical_prefix = existing[existing["date"] < rebuild_start].copy()
    return rebuild_start, historical_prefix


def _default_key_columns(df: pd.DataFrame) -> tuple[str, ...]:
    if {"ticker", "date"} <= set(df.columns):
        return ("ticker", "date")
    if "date" in df.columns:
        return ("date",)
    return tuple()


def build_data_quality_contract(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    artifact_path: Path | str | None = None,
    key_columns: tuple[str, ...] | None = None,
    stale_after_days: int | None = None,
) -> dict[str, Any]:
    frame = df.copy()
    resolved_key_columns = key_columns or _default_key_columns(frame)
    if stale_after_days is None:
        stale_after_days = DEFAULT_STALE_AFTER_DAYS.get(Path(str(dataset_name)).name)

    contract: dict[str, Any] = {
        "contract_version": DATA_QUALITY_CONTRACT_VERSION,
        "dataset_name": dataset_name,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "key_columns": list(resolved_key_columns),
        "source_provenance": df.attrs.get("source_provenance"),
        "source_provenance_present": bool(df.attrs.get("source_provenance")),
        "unsupported_source": df.attrs.get("source_provenance") == STUB_PROVENANCE,
    }

    if "date" in frame.columns and not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if "ticker" in resolved_key_columns and "ticker" in frame.columns:
            monotonic = bool(
                frame.sort_values(["ticker", "date"])
                .groupby("ticker", sort=False)["date"]
                .apply(lambda values: values.is_monotonic_increasing)
                .all()
            )
        else:
            monotonic = bool(frame["date"].is_monotonic_increasing)
        contract["date_monotonic"] = monotonic
    else:
        contract["date_monotonic"] = None

    if resolved_key_columns and set(resolved_key_columns) <= set(frame.columns):
        contract["duplicate_key_count"] = int(frame.duplicated(subset=list(resolved_key_columns)).sum())
    else:
        contract["duplicate_key_count"] = None

    total_cells = int(frame.size)
    contract["missing_ratio"] = float(frame.isna().sum().sum() / total_cells) if total_cells else 0.0
    contract["high_missing_columns"] = sorted(
        [
            column
            for column, ratio in frame.isna().mean().items()
            if float(ratio) >= 0.25
        ]
    )

    if artifact_path is not None:
        path = Path(artifact_path)
        contract["artifact_path"] = str(path)
        contract["artifact_exists"] = path.exists()
        if path.exists():
            modified_at = dt.datetime.fromtimestamp(path.stat().st_mtime)
            age_days = max((dt.datetime.now() - modified_at).days, 0)
            contract["artifact_age_days"] = int(age_days)
            contract["stale_after_days"] = stale_after_days
            contract["stale"] = bool(stale_after_days is not None and age_days > stale_after_days)
        else:
            contract["artifact_age_days"] = None
            contract["stale_after_days"] = stale_after_days
            contract["stale"] = None
    else:
        contract["artifact_path"] = None
        contract["artifact_exists"] = None
        contract["artifact_age_days"] = None
        contract["stale_after_days"] = stale_after_days
        contract["stale"] = None

    return contract


def _attach_source_attrs(
    df: pd.DataFrame,
    *,
    provenance: str,
    source_name: str,
    adjustment_status: str | None = None,
    notes: str | None = None,
    artifact_path: Path | str | None = None,
    key_columns: tuple[str, ...] | None = None,
    stale_after_days: int | None = None,
) -> pd.DataFrame:
    df.attrs["source_provenance"] = provenance
    df.attrs["source_name"] = source_name
    if adjustment_status is not None:
        df.attrs["adjustment_status"] = adjustment_status
    if notes:
        df.attrs["source_notes"] = notes
    df.attrs["data_quality_contract"] = build_data_quality_contract(
        df,
        dataset_name=source_name,
        artifact_path=artifact_path,
        key_columns=key_columns,
        stale_after_days=stale_after_days,
    )
    return df


def _stub_frame(source_name: str, reason: str) -> pd.DataFrame:
    logger.debug("context_source_stubbed", source=source_name, reason=reason)
    frame = pd.DataFrame()
    return _attach_source_attrs(
        frame,
        provenance=STUB_PROVENANCE,
        source_name=source_name,
        notes=reason,
    )


def _normalize_date_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for candidate in ("date", "time", "timestamp"):
        if candidate in normalized.columns:
            normalized = normalized.rename(columns={candidate: "date"})
            normalized = _ensure_datetime64ns(normalized, "date")
            return normalized
    if isinstance(normalized.index, pd.DatetimeIndex):
        normalized = normalized.reset_index().rename(columns={normalized.index.name or "index": "date"})
        normalized = _ensure_datetime64ns(normalized, "date")
    return normalized


def _ensure_datetime64ns(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    normalized = frame.copy()
    if column in normalized.columns:
        series = pd.to_datetime(normalized[column], errors="coerce")
        if getattr(series.dt, "tz", None) is not None:
            series = series.dt.tz_localize(None)
        normalized[column] = pd.DatetimeIndex(series).normalize().astype("datetime64[ns]")
    return normalized


def validate_ohlcv(df: pd.DataFrame, ticker: str, min_rows: int = 60) -> pd.DataFrame:
    """Validate OHLCV data structure and quality.
    
    Raises:
        ValueError: If validation fails, with explicit error codes 
                    (e.g., [invalid_ohlcv_input], [insufficient_history]).
    """
    if df is None or df.empty:
        raise ValueError(f"[invalid_ohlcv_input] DataFrame is empty or None for {ticker}")
        
    required_cols = {"date", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"[invalid_ohlcv_input] Missing required columns for {ticker}: {missing}")
        
    for col in ["open", "high", "low", "close", "volume"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                raise ValueError(f"[invalid_ohlcv_input] Column {col} must be numeric for {ticker}")
                
    if "ticker" not in df.columns:
        df["ticker"] = ticker.upper()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    if df.duplicated(subset=["ticker", "date"]).any():
        logger.warning("invalid_ohlcv_input_duplicate_dates", ticker=ticker, action="keeping_last")
        df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
        
    if len(df) < min_rows:
        raise ValueError(f"[insufficient_history] Required {min_rows} rows for {ticker}, got {len(df)}")

    source_name = str(df.attrs.get("source_name", f"{ticker.upper()}.ohlcv"))
    prior_contract = df.attrs.get("data_quality_contract", {}) or {}
    df.attrs["data_quality_contract"] = build_data_quality_contract(
        df,
        dataset_name=source_name,
        artifact_path=prior_contract.get("artifact_path"),
        key_columns=("ticker", "date"),
        stale_after_days=prior_contract.get("stale_after_days"),
    )
    return df


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
        return _attach_source_attrs(
            df,
            provenance=LOCAL_COMPUTATION_PROVENANCE,
            source_name="timescaledb.adjusted_prices",
            adjustment_status="adjusted",
        )
    except Exception as e:
        logger.error("ohlcv_load_error", ticker=ticker, error=str(e))
        raise
    finally:
        engine.dispose()


def load_ohlcv_from_vnstock(ticker: str, num_days: int = 600) -> pd.DataFrame:
    """Fetch recent historical OHLCV data from vnstock_data as a fallback.

    Provides a live API connection when the local TimescaleDB is unavailable.
    Uses the canonical vnstock_data quote-history interfaces via ``VnstockAdapter``.
    """
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=int(num_days * 1.5))
    from src.data.adapters.vnstock_adapter import VnstockAdapter

    adapter = VnstockAdapter(symbol_list=[ticker.upper()])
    df = adapter.get_ohlcv(
        ticker.upper(),
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        interval="1D",
    )
    if df is None or df.empty:
        raise ValueError(f"No data returned from vnstock_data for {ticker}")

    logger.info("ohlcv_loaded_from_vnstock_data", ticker=ticker, rows=len(df))
    return _attach_source_attrs(
        df.sort_values(by="date").reset_index(drop=True),
        provenance=df.attrs.get("source_provenance", DIRECT_VNSTOCK_PROVENANCE),
        source_name=df.attrs.get("source_name", "vnstock_data"),
        adjustment_status=df.attrs.get("adjustment_status", "unknown"),
    )


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
            ts_start = pd.Timestamp(start_date).normalize()
            df = df[df["date"] >= ts_start]
        if end_date:
            ts_end = pd.Timestamp(end_date).normalize()
            df = df[df["date"] <= ts_end]

        # Ensure ticker column
        df["ticker"] = ticker.upper()
        df = df.sort_values("date").reset_index(drop=True)

        logger.debug("ohlcv_loaded_from_csv", ticker=ticker, rows=len(df))
        return _attach_source_attrs(
            df,
            provenance=DERIVED_VNSTOCK_PROVENANCE,
            source_name="csv_cache.daily_market_split_data",
            adjustment_status="unknown",
            artifact_path=csv_path,
        )

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
    if path.exists():
        df = pd.read_csv(path)
        df = _normalize_date_frame(df)
        if start_date:
            ts_start = pd.Timestamp(start_date).normalize()
            df = df[df["date"] >= ts_start]
        if end_date:
            ts_end = pd.Timestamp(end_date).normalize()
            df = df[df["date"] <= ts_end]
        logger.debug("market_proxy_loaded", rows=len(df))
        return _attach_source_attrs(
            df.reset_index(drop=True),
            provenance=DERIVED_VNSTOCK_PROVENANCE,
            source_name="market_proxy.csv",
            notes="Derived from VNINDEX history sourced via vnstock_data.",
            artifact_path=path,
        )

    logger.warning("market_proxy_not_found", path=str(path))
    try:
        from src.data.adapters.vnstock_adapter import VnstockAdapter

        end = end_date or dt.date.today()
        start = start_date or (end - dt.timedelta(days=365 * 6))
        df = VnstockAdapter().get_index_ohlcv(
            "VNINDEX",
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            interval="1D",
        )
        if df is not None and not df.empty:
            proxy = df[["date", "close"]].copy()
            proxy["m_ret"] = pd.to_numeric(proxy["close"], errors="coerce").pct_change()
            proxy = proxy[["date", "m_ret"]].reset_index(drop=True)
            return _attach_source_attrs(
                proxy,
                provenance=DIRECT_VNSTOCK_PROVENANCE,
                source_name="VNINDEX.ohlcv",
            )
    except Exception as exc:
        logger.debug("market_proxy_vnstock_fallback_failed", error=str(exc))

    return _stub_frame("market_proxy", "No market proxy artifact and vnstock_data fallback unavailable.")


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
        return _stub_frame("fundamentals_latest.csv", "No fundamentals artifact found.")

    df = pd.read_csv(path)

    if tickers:
        upper = [t.upper() for t in tickers]
        if "ticker" in df.columns:
            df = df[df["ticker"].str.upper().isin(upper)]

    logger.debug("fundamentals_loaded", rows=len(df))
    return _attach_source_attrs(
        df.reset_index(drop=True),
        provenance=DERIVED_VNSTOCK_PROVENANCE,
        source_name="fundamentals_latest.csv",
        notes="Locally materialized fundamentals artifact.",
        artifact_path=path,
    )


def load_sentiment(
    path: Path | str | None = None,
    tickers: List[str] | None = None,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    lag_days: int = 1,
    *,
    enabled: bool = True,
    require_validated_source: bool = False,
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
    capability_audit = audit_sentiment_capability(path=path, run_live_probe=False)

    def _attach_sentiment_audit(df: pd.DataFrame, integration_status: str) -> pd.DataFrame:
        df.attrs["sentiment_capability_audit"] = capability_audit
        df.attrs["sentiment_integration_status"] = integration_status
        df.attrs["sentiment_main_pipeline_recommendation"] = capability_audit.get("main_pipeline_recommendation", "no_go")
        return df

    if not enabled:
        return _attach_sentiment_audit(
            _stub_frame(
                "sentiment_features.csv",
                "Sentiment integration is explicitly disabled for the main forecasting pipeline.",
            ),
            "disabled",
        )

    if require_validated_source and capability_audit.get("main_pipeline_recommendation") != "go":
        return _attach_sentiment_audit(
            _stub_frame(
                "sentiment_features.csv",
                "Sentiment artifact is present but the source is not validated for the main forecasting pipeline.",
            ),
            "rejected_unvalidated",
        )

    if not path.exists():
        logger.debug("sentiment_not_found", path=str(path))
        return _attach_sentiment_audit(
            _stub_frame("sentiment_features.csv", "No sentiment artifact found."),
            "missing_artifact",
        )

    df = pd.read_csv(path)
    df = _normalize_date_frame(df)

    if tickers and "ticker" in df.columns:
        upper = [t.upper() for t in tickers]
        df = df[df["ticker"].str.upper().isin(upper)]

    if lag_days > 0 and "date" in df.columns:
        sort_keys = ["date"]
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].astype(str).str.upper()
            sort_keys = ["ticker", "date"]
        df = df.sort_values(sort_keys).reset_index(drop=True)
        lag_cols = [c for c in df.columns if c not in {"ticker", "date"}]
        if lag_cols:
            if "ticker" in df.columns:
                df[lag_cols] = df.groupby("ticker", sort=False)[lag_cols].shift(int(lag_days))
            else:
                df[lag_cols] = df[lag_cols].shift(int(lag_days))
            df[lag_cols] = df[lag_cols].fillna(0.0)

    if "date" in df.columns:
        if start_date:
            ts_start = pd.Timestamp(start_date).normalize()
            df = df[df["date"] >= ts_start]
        if end_date:
            ts_end = pd.Timestamp(end_date).normalize()
            df = df[df["date"] <= ts_end]

    logger.debug("sentiment_loaded", rows=len(df))
    result = _attach_source_attrs(
        df.reset_index(drop=True),
        provenance=LOCAL_COMPUTATION_PROVENANCE,
        source_name="sentiment_features.csv",
        notes=f"Daily sentiment aggregates are lagged by {lag_days} trading day(s) for time safety.",
        artifact_path=path,
    )
    return _attach_sentiment_audit(
        result,
        "loaded_unvalidated" if capability_audit.get("main_pipeline_recommendation") != "go" else "validated",
    )


def audit_sentiment_capability(
    path: Path | str | None = None,
    *,
    live_probe_symbol: str = "SSI",
    run_live_probe: bool = True,
) -> dict[str, Any]:
    """Audit sentiment/news capability without assuming the current runtime is production safe."""
    from src.data.adapters.vnstock_adapter import VnstockAdapter

    artifact_path = Path(path) if path else _DEFAULT_SENTIMENT_PATH
    live_news_audit = VnstockAdapter(symbol_list=[live_probe_symbol.upper().strip()]).audit_company_news_capability(
        symbol=live_probe_symbol,
        count=1,
        run_live_probe=run_live_probe,
    )
    audit: dict[str, Any] = {
        "provider_name": "vnstock_data",
        "live_news_endpoint_status": live_news_audit.get("status", "unsupported"),
        "live_news_endpoint_audit": live_news_audit,
        "alternate_vnstock_installed": importlib.util.find_spec("vnstock") is not None,
        "artifact_path": str(artifact_path),
        "artifact_exists": artifact_path.exists(),
        "artifact_status": "unsupported",
        "artifact_rows": 0,
        "artifact_tickers": 0,
        "artifact_start_date": None,
        "artifact_end_date": None,
        "artifact_single_snapshot": None,
        "publication_timestamp_verified": False,
        "entity_mapping_verified": False,
        "aggregation_time_safe": False,
        "integration_status": "no_go",
        "main_pipeline_recommendation": "no_go",
        "notes": [],
    }

    if artifact_path.exists():
        try:
            artifact = _normalize_date_frame(pd.read_csv(artifact_path))
            audit["artifact_rows"] = int(len(artifact))
            if "ticker" in artifact.columns:
                audit["artifact_tickers"] = int(artifact["ticker"].astype(str).str.upper().nunique())
            if "date" in artifact.columns and not artifact.empty:
                dates = pd.to_datetime(artifact["date"], errors="coerce").dropna()
                if not dates.empty:
                    audit["artifact_start_date"] = str(pd.Timestamp(dates.min()).date())
                    audit["artifact_end_date"] = str(pd.Timestamp(dates.max()).date())
                    audit["artifact_single_snapshot"] = bool(dates.dt.normalize().nunique() <= 1)
            audit["artifact_status"] = "unstable_partial" if audit["artifact_rows"] > 0 else "unsupported"
        except Exception as exc:
            audit["artifact_status"] = "unstable_partial"
            audit["notes"].append(f"Sentiment artifact could not be parsed cleanly: {exc}")

    if not live_news_audit.get("provider_runtime_available", False):
        audit["notes"].append("Canonical vnstock_data runtime is unavailable in the active interpreter.")
    elif live_news_audit.get("status") != "live_supported":
        audit["notes"].append("Canonical company/news endpoint is not runtime-proven in the active environment.")

    if audit["alternate_vnstock_installed"]:
        audit["notes"].append("Non-canonical vnstock is installed, but it is not accepted as the primary source of truth.")

    if audit["artifact_exists"]:
        audit["notes"].append(
            "Sentiment artifact is local-computation output only; publication-time alignment and entity mapping are not validated."
        )
        if audit["artifact_single_snapshot"] is True:
            audit["notes"].append("Sentiment artifact currently covers only a single snapshot date.")
        if not audit["aggregation_time_safe"]:
            audit["notes"].append("Daily aggregation cannot be treated as time-safe because publication timestamps are not validated.")

    return audit


def load_ticker_sectors(path: Path | str | None = None) -> pd.DataFrame:
    """Load ticker-to-sector mapping."""
    path = Path(path) if path else _DEFAULT_TICKER_SECTORS_PATH
    cache_key = _artifact_cache_key(
        "load_ticker_sectors",
        artifact=_file_snapshot_token(path),
    )
    cached = _artifact_cache_get(cache_key)
    if cached is not None:
        return cached
    if not path.exists():
        logger.debug("ticker_sectors_not_found", path=str(path))
        try:
            from src.data.adapters.vnstock_adapter import VnstockAdapter

            listing = VnstockAdapter().get_symbols_by_industries()
            if listing is not None and not listing.empty:
                sector_cols = [c for c in ("ticker", "industry_code", "industry_name", "industry") if c in listing.columns]
                if {"ticker", "industry"} <= set(listing.columns):
                    result = _attach_source_attrs(
                        listing[sector_cols].copy(),
                        provenance=DIRECT_VNSTOCK_PROVENANCE,
                        source_name="Listing.symbols_by_industries",
                    )
                    return _artifact_cache_set(cache_key, result)
        except Exception as exc:
            logger.debug("ticker_sector_vnstock_fallback_failed", error=str(exc))
        return _artifact_cache_set(
            cache_key,
            _stub_frame("ticker_sectors", "No ticker-sector mapping artifact available."),
        )
    result = _attach_source_attrs(
        pd.read_csv(path),
        provenance=DERIVED_VNSTOCK_PROVENANCE,
        source_name="ticker_sectors.csv",
        notes="Mapping artifact derived from vnstock_data listing metadata.",
        artifact_path=path,
    )
    return _artifact_cache_set(cache_key, result)


def load_sector_proxies(path: Path | str | None = None) -> pd.DataFrame:
    """Load sector index returns."""
    path = Path(path) if path else _DEFAULT_SECTOR_PROXIES_PATH
    cache_key = _artifact_cache_key(
        "load_sector_proxies",
        artifact=_file_snapshot_token(path),
    )
    cached = _artifact_cache_get(cache_key)
    if cached is not None:
        return cached
    if not path.exists():
        logger.debug("sector_proxies_not_found", path=str(path))
        return build_sector_proxies_from_csv(output_path=path)
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    required_cols = {"ret", "sector_dispersion", "sector_member_count"}
    if not required_cols <= set(df.columns):
        try:
            rebuilt = build_sector_proxies_from_csv(output_path=path)
            if not rebuilt.empty:
                final_key = _artifact_cache_key(
                    "load_sector_proxies",
                    artifact=_file_snapshot_token(path),
                )
                return _artifact_cache_set(final_key, rebuilt)
        except Exception as exc:
            logger.debug("sector_proxy_csv_rebuild_failed", error=str(exc))
    result = _attach_source_attrs(
        df,
        provenance=DERIVED_VNSTOCK_PROVENANCE,
        source_name="sector_proxies.csv",
        notes="Sector proxy returns derived from vnstock_data-based ticker histories.",
        artifact_path=path,
    )
    final_key = _artifact_cache_key(
        "load_sector_proxies",
        artifact=_file_snapshot_token(path),
    )
    return _artifact_cache_set(final_key, result)


def build_sector_proxies_from_csv(
    csv_dir: Path | str | None = None,
    ticker_sectors: pd.DataFrame | None = None,
    output_path: Path | str | None = None,
    incremental_update: bool = True,
    lookback_days: int = INCREMENTAL_CONTEXT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Build sector proxy returns and cross-sectional dispersion from local vnstock CSV caches."""
    csv_dir = Path(csv_dir) if csv_dir else _DEFAULT_DAILY_CSV_DIR
    cache_key = _artifact_cache_key(
        "build_sector_proxies_from_csv",
        csv_dir_snapshot=_directory_snapshot_token(csv_dir),
        ticker_sector_shape=(tuple(sorted(ticker_sectors.columns)), len(ticker_sectors)) if ticker_sectors is not None else None,
        output_path=_file_snapshot_token(output_path),
        incremental_update=incremental_update,
        lookback_days=lookback_days,
    )
    cached = _artifact_cache_get(cache_key)
    if cached is not None:
        return cached

    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        return _stub_frame("sector_proxies.csv", f"No OHLCV csv files found in {csv_dir}")

    rebuild_start, historical_prefix = (
        _incremental_history_prefix(
            output_path,
            key_columns=("industry", "date"),
            lookback_days=lookback_days,
        )
        if incremental_update
        else (None, None)
    )

    sector_map = ticker_sectors.copy() if ticker_sectors is not None else load_ticker_sectors()
    if sector_map is None or sector_map.empty or "ticker" not in sector_map.columns or "industry" not in sector_map.columns:
        return _stub_frame("sector_proxies.csv", "Ticker-sector mapping unavailable for sector proxy construction.")

    sector_map["ticker"] = sector_map["ticker"].astype(str).str.upper()
    ticker_to_industry = {
        str(row["ticker"]).upper(): str(row["industry"])
        for _, row in sector_map.dropna(subset=["ticker", "industry"]).iterrows()
    }

    sector_frames: list[pd.DataFrame] = []
    for csv_path in files:
        ticker = csv_path.stem.upper()
        industry = ticker_to_industry.get(ticker)
        if not industry:
            continue
        try:
            frame = pd.read_csv(csv_path, usecols=lambda c: c in {"time", "date", "close"})
            frame = _normalize_date_frame(frame)
            if "date" not in frame.columns or "close" not in frame.columns:
                continue
            frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            if rebuild_start is not None:
                frame = frame[frame["date"] >= rebuild_start].copy()
            if frame.empty:
                continue
            frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
            frame["ret"] = frame["close"].pct_change()
            frame["ticker"] = ticker
            frame["industry"] = industry
            frame = frame[["date", "ticker", "industry", "ret"]].dropna(subset=["ret"])
            if not frame.empty:
                sector_frames.append(frame)
        except Exception as exc:
            logger.debug("sector_proxy_csv_skip", path=str(csv_path), error=str(exc))

    if not sector_frames:
        return _stub_frame("sector_proxies.csv", "No usable sector return history found in local csv cache.")

    stacked = pd.concat(sector_frames, ignore_index=True)
    grouped = stacked.groupby(["date", "industry"])["ret"]
    proxies = (
        grouped.agg(
            ret="mean",
            sector_dispersion="std",
            sector_member_count="count",
        )
        .reset_index()
        .sort_values(["industry", "date"])
        .reset_index(drop=True)
    )
    for column in ["ret", "sector_dispersion"]:
        proxies[column] = pd.to_numeric(proxies[column], errors="coerce")
    proxies["sector_dispersion"] = proxies["sector_dispersion"].fillna(0.0)
    proxies["sector_member_count"] = pd.to_numeric(proxies["sector_member_count"], errors="coerce").fillna(0).astype(int)

    if historical_prefix is not None and not historical_prefix.empty:
        proxies = _merge_incremental_frames(
            historical_prefix,
            proxies,
            sort_columns=("industry", "date"),
            dedupe_columns=("industry", "date"),
        )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        proxies.to_csv(output_path, index=False)

    result = _attach_source_attrs(
        proxies,
        provenance=DERIVED_VNSTOCK_PROVENANCE,
        source_name="sector_proxies.csv",
        notes="Sector proxy returns and cross-sectional dispersion derived from vnstock_data-based ticker histories.",
        artifact_path=output_path,
    )
    return _artifact_cache_set(cache_key, result)


def build_market_breadth_from_csv(
    csv_dir: Path | str | None = None,
    output_path: Path | str | None = None,
    incremental_update: bool = True,
    lookback_days: int = INCREMENTAL_CONTEXT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Build market breadth metrics from the local vnstock-derived OHLCV universe.

    The resulting file can be cached to avoid recomputing the full universe on
    every training run.
    """
    csv_dir = Path(csv_dir) if csv_dir else _DEFAULT_DAILY_CSV_DIR
    cache_key = _artifact_cache_key(
        "build_market_breadth_from_csv",
        csv_dir_snapshot=_directory_snapshot_token(csv_dir),
        output_path=_file_snapshot_token(output_path),
        incremental_update=incremental_update,
        lookback_days=lookback_days,
    )
    cached = _artifact_cache_get(cache_key)
    if cached is not None:
        return cached

    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        return _stub_frame("market_breadth", f"No OHLCV csv files found in {csv_dir}")

    rebuild_start, historical_prefix = (
        _incremental_history_prefix(
            output_path,
            key_columns=("date",),
            lookback_days=lookback_days,
        )
        if incremental_update
        else (None, None)
    )

    daily_frames: list[pd.DataFrame] = []
    for csv_path in files:
        try:
            frame = pd.read_csv(csv_path, usecols=lambda c: c in {"time", "date", "close", "high", "low", "volume"})
            frame = _normalize_date_frame(frame)
            if "close" not in frame.columns or "date" not in frame.columns:
                continue
            frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            if rebuild_start is not None:
                frame = frame[frame["date"] >= rebuild_start].copy()
            if frame.empty:
                continue
            frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
            frame["high"] = pd.to_numeric(frame["high"], errors="coerce") if "high" in frame.columns else frame["close"]
            frame["low"] = pd.to_numeric(frame["low"], errors="coerce") if "low" in frame.columns else frame["close"]
            frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame.columns else 0.0
            frame["ret"] = frame["close"].pct_change()

            ma20 = frame["close"].rolling(20, min_periods=20).mean()
            ma50 = frame["close"].rolling(50, min_periods=50).mean()
            prev_high_252 = frame["high"].rolling(252, min_periods=60).max().shift(1)
            prev_low_252 = frame["low"].rolling(252, min_periods=60).min().shift(1)

            frame["above_ma20"] = np.where(ma20.notna(), (frame["close"] > ma20).astype(float), np.nan)
            frame["above_ma50"] = np.where(ma50.notna(), (frame["close"] > ma50).astype(float), np.nan)
            frame["new_high_252"] = np.where(prev_high_252.notna(), (frame["high"] >= prev_high_252).astype(float), np.nan)
            frame["new_low_252"] = np.where(prev_low_252.notna(), (frame["low"] <= prev_low_252).astype(float), np.nan)
            frame["up_volume"] = np.where(frame["ret"] > 0, frame["volume"], 0.0)
            frame["down_volume"] = np.where(frame["ret"] < 0, frame["volume"], 0.0)
            frame = frame[
                [
                    "date",
                    "ret",
                    "above_ma20",
                    "above_ma50",
                    "new_high_252",
                    "new_low_252",
                    "up_volume",
                    "down_volume",
                ]
            ].dropna(subset=["ret"])
            if not frame.empty:
                daily_frames.append(frame)
        except Exception as exc:
            logger.debug("market_breadth_csv_skip", path=str(csv_path), error=str(exc))

    if not daily_frames:
        return _stub_frame("market_breadth", "No usable return history found for breadth construction.")

    stacked = pd.concat(daily_frames, ignore_index=True)
    grouped = stacked.groupby("date")["ret"]
    aggregate = stacked.groupby("date").agg(
        advancers=("ret", lambda s: int((s > 0).sum())),
        decliners=("ret", lambda s: int((s < 0).sum())),
        unchanged=("ret", lambda s: int((s == 0).sum())),
        breadth_member_count=("ret", "count"),
        pct_above_ma20=("above_ma20", "mean"),
        pct_above_ma50=("above_ma50", "mean"),
        new_highs_252=("new_high_252", "sum"),
        new_lows_252=("new_low_252", "sum"),
        up_volume=("up_volume", "sum"),
        down_volume=("down_volume", "sum"),
    )
    breadth = aggregate.reset_index()
    breadth["net_advancers"] = breadth["advancers"] - breadth["decliners"]
    breadth["advance_decline_ratio"] = breadth["advancers"] / breadth["decliners"].replace(0, np.nan)
    breadth["market_breadth"] = breadth["net_advancers"] / (
        breadth["advancers"] + breadth["decliners"]
    ).replace(0, np.nan)
    breadth["advancing_share"] = breadth["advancers"] / breadth["breadth_member_count"].replace(0, np.nan)
    breadth["declining_share"] = breadth["decliners"] / breadth["breadth_member_count"].replace(0, np.nan)
    breadth["new_high_low_spread"] = (breadth["new_highs_252"] - breadth["new_lows_252"]) / breadth["breadth_member_count"].replace(0, np.nan)
    breadth["up_down_volume_ratio"] = breadth["up_volume"] / breadth["down_volume"].replace(0, np.nan)
    breadth["market_breadth"] = breadth["market_breadth"].fillna(0.0)
    breadth["advance_decline_ratio"] = breadth["advance_decline_ratio"].replace([np.inf, -np.inf], np.nan)
    breadth["up_down_volume_ratio"] = breadth["up_down_volume_ratio"].replace([np.inf, -np.inf], np.nan)
    breadth["advancing_share"] = breadth["advancing_share"].fillna(0.0)
    breadth["declining_share"] = breadth["declining_share"].fillna(0.0)
    breadth["pct_above_ma20"] = pd.to_numeric(breadth["pct_above_ma20"], errors="coerce")
    breadth["pct_above_ma50"] = pd.to_numeric(breadth["pct_above_ma50"], errors="coerce")
    breadth["new_high_low_spread"] = pd.to_numeric(breadth["new_high_low_spread"], errors="coerce").fillna(0.0)

    if historical_prefix is not None and not historical_prefix.empty:
        breadth = _merge_incremental_frames(
            historical_prefix,
            breadth,
            sort_columns=("date",),
            dedupe_columns=("date",),
        )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        breadth.to_csv(output_path, index=False)

    result = _attach_source_attrs(
        breadth.reset_index(drop=True),
        provenance=DERIVED_VNSTOCK_PROVENANCE,
        source_name="market_breadth.csv",
        notes="Constructed from per-ticker csv caches that were originally sourced from vnstock_data.",
        artifact_path=output_path,
    )
    return _artifact_cache_set(cache_key, result)


def load_market_breadth(
    path: Path | str | None = None,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
) -> pd.DataFrame:
    path = Path(path) if path else _DEFAULT_MARKET_BREADTH_PATH
    cache_key = _artifact_cache_key(
        "load_market_breadth",
        artifact=_file_snapshot_token(path),
        start_date=start_date,
        end_date=end_date,
    )
    cached = _artifact_cache_get(cache_key)
    if cached is not None:
        return cached
    if not path.exists():
        logger.debug("market_breadth_not_found", path=str(path))
        return build_market_breadth_from_csv(output_path=path)

    df = pd.read_csv(path)
    df = _normalize_date_frame(df)
    required_cols = {"pct_above_ma20", "pct_above_ma50", "new_highs_252", "new_lows_252", "up_volume", "down_volume"}
    if not required_cols <= set(df.columns):
        try:
            rebuilt = build_market_breadth_from_csv(output_path=path)
            if not rebuilt.empty:
                final_key = _artifact_cache_key(
                    "load_market_breadth",
                    artifact=_file_snapshot_token(path),
                    start_date=start_date,
                    end_date=end_date,
                )
                return _artifact_cache_set(final_key, rebuilt)
        except Exception as exc:
            logger.debug("market_breadth_csv_rebuild_failed", error=str(exc))
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date).normalize()]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date).normalize()]
    result = _attach_source_attrs(
        df.reset_index(drop=True),
        provenance=DERIVED_VNSTOCK_PROVENANCE,
        source_name="market_breadth.csv",
        artifact_path=path,
    )
    final_key = _artifact_cache_key(
        "load_market_breadth",
        artifact=_file_snapshot_token(path),
        start_date=start_date,
        end_date=end_date,
    )
    return _artifact_cache_set(final_key, result)


def build_macro_context_from_vnstock(
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
) -> pd.DataFrame:
    from src.data.adapters.vnstock_adapter import VnstockAdapter

    end = end_date or dt.date.today()
    start = start_date or (end - dt.timedelta(days=365 * 6))
    adapter = VnstockAdapter()
    series_frames: list[pd.DataFrame] = []

    def _append_series(frame: pd.DataFrame, mapping: dict[str, str]) -> None:
        if frame is None or frame.empty:
            return
        normalized = _normalize_date_frame(frame)
        for source_col, target_col in mapping.items():
            if source_col in normalized.columns:
                tmp = normalized[["date", source_col]].copy()
                tmp[target_col] = pd.to_numeric(tmp[source_col], errors="coerce")
                tmp = tmp[["date", target_col]].dropna()
                if not tmp.empty:
                    series_frames.append(tmp.drop_duplicates(subset=["date"], keep="last"))

    market_eval = adapter.get_market_valuation(metric="evaluation", duration="5Y")
    _append_series(market_eval, {"pe": "market_pe", "pb": "market_pb"})

    fx = adapter.get_macro_exchange_rate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if fx is not None and not fx.empty:
        fx_norm = _normalize_date_frame(fx)
        fx_norm["name"] = fx_norm["name"].astype(str) if "name" in fx_norm.columns else ""
        fx_norm["value"] = pd.to_numeric(fx_norm["value"], errors="coerce") if "value" in fx_norm.columns else np.nan
        fx_main = fx_norm[fx_norm["name"].str.contains("Liên ngân hàng", case=False, na=False)]
        if fx_main.empty:
            fx_main = fx_norm.copy()
        fx_main = fx_main[["date", "value"]].dropna().drop_duplicates(subset=["date"], keep="last")
        if not fx_main.empty:
            fx_main = fx_main.rename(columns={"value": "fx_usdvnd"})
            series_frames.append(fx_main)
        if fx_main.empty and fx_norm["value"].notna().any():
            fx_fallback = fx_norm[["date", "value"]].dropna().drop_duplicates(subset=["date"], keep="last")
            if not fx_fallback.empty:
                fx_fallback = fx_fallback.rename(columns={"value": "fx_usdvnd"})
                series_frames.append(fx_fallback)

    ir = adapter.get_macro_interest_rate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if ir is not None and not ir.empty:
        ir_norm = _normalize_date_frame(ir)
        ir_norm["group_name"] = ir_norm["group_name"].astype(str) if "group_name" in ir_norm.columns else ""
        ir_norm["name"] = ir_norm["name"].astype(str) if "name" in ir_norm.columns else ""
        ir_norm["unit"] = ir_norm["unit"].astype(str) if "unit" in ir_norm.columns else ""
        ir_norm["value"] = pd.to_numeric(ir_norm["value"], errors="coerce") if "value" in ir_norm.columns else np.nan
        overnight = ir_norm[
            ir_norm["group_name"].str.contains("Lãi suất bình quân liên ngân hàng", case=False, na=False)
            & ir_norm["name"].str.contains("Qua đêm", case=False, na=False)
            & ir_norm["unit"].str.contains("%", regex=False, na=False)
        ]
        if overnight.empty:
            overnight = ir_norm[ir_norm["unit"].str.contains("%", regex=False, na=False)]
        overnight = overnight[["date", "value"]].dropna().drop_duplicates(subset=["date"], keep="last")
        if not overnight.empty:
            overnight = overnight.rename(columns={"value": "interest_rate"})
            series_frames.append(overnight)

    gold = adapter.get_commodity_gold(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if gold is not None and not gold.empty:
        gold_norm = _normalize_date_frame(gold)
        if {"buy", "sell"} <= set(gold_norm.columns):
            gold_tmp = gold_norm[["date"]].copy()
            gold_tmp["gold_price"] = (
                pd.to_numeric(gold_norm["buy"], errors="coerce")
                + pd.to_numeric(gold_norm["sell"], errors="coerce")
            ) / 2.0
            gold_tmp = gold_tmp.dropna().drop_duplicates(subset=["date"], keep="last")
            if not gold_tmp.empty:
                series_frames.append(gold_tmp)
        else:
            _append_series(gold, {"close": "gold_price"})

    oil = adapter.get_commodity_oil(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    _append_series(oil, {"close": "oil_price"})

    if not series_frames:
        return _stub_frame(
            "vnstock_data.live_macro_context",
            "Live vnstock_data macro/commodity fetch returned no usable rows.",
        )

    macro = series_frames[0].copy()
    for frame in series_frames[1:]:
        macro = macro.merge(frame, on="date", how="outer")
    macro = _normalize_date_frame(macro)
    macro = macro[(macro["date"] >= pd.Timestamp(start).normalize()) & (macro["date"] <= pd.Timestamp(end).normalize())]
    macro = macro.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return _attach_source_attrs(
        macro,
        provenance=DIRECT_VNSTOCK_PROVENANCE,
        source_name="vnstock_data.live_macro_context",
        notes="Built live from Market.evaluation, Macro.exchange_rate, Macro.interest_rate, and CommodityPrice series.",
    )


def build_foreign_flow_from_vnstock(
    tickers: List[str],
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
) -> pd.DataFrame:
    from src.data.adapters.vnstock_adapter import VnstockAdapter

    normalized_tickers = [str(t).upper().strip() for t in tickers if str(t).strip()]
    if not normalized_tickers:
        return _stub_frame("vnstock_data.live_foreign_flow", "No tickers supplied for live foreign-flow fetch.")

    end = end_date or dt.date.today()
    start = start_date or (end - dt.timedelta(days=365 * 6))
    adapter = VnstockAdapter(symbol_list=normalized_tickers)
    frames: list[pd.DataFrame] = []

    keep_cols = {
        "date",
        "ticker",
        "foreign_buy_volume",
        "foreign_sell_volume",
        "foreign_buy_value",
        "foreign_sell_value",
        "foreign_net_volume",
        "foreign_net_value",
        "foreign_room_pct",
        "foreign_owned_pct",
        "foreign_available_pct",
        "foreign_current_room",
        "foreign_total_room",
    }

    for ticker in normalized_tickers:
        try:
            frame = adapter.get_foreign_flow(
                ticker,
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
            )
            if frame is None or frame.empty:
                continue
            frame = _normalize_date_frame(frame)
            present_cols = [c for c in frame.columns if c in keep_cols]
            if {"date", "ticker"} <= set(present_cols):
                local = frame[present_cols].copy()
                local["ticker"] = local["ticker"].astype(str).str.upper()
                frames.append(local)
        except Exception as exc:
            logger.debug("foreign_flow_live_fetch_failed", ticker=ticker, error=str(exc))

    if not frames:
        return _stub_frame(
            "vnstock_data.live_foreign_flow",
            "Live vnstock_data foreign-flow fetch returned no usable rows.",
        )

    foreign = pd.concat(frames, ignore_index=True)
    foreign = foreign.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"], keep="last")
    return _attach_source_attrs(
        foreign.reset_index(drop=True),
        provenance=DIRECT_VNSTOCK_PROVENANCE,
        source_name="vnstock_data.live_foreign_flow",
        notes="Built live from Trading.foreign_trade with VCI primary and CAFEF fallback inside the adapter.",
    )


def build_macro_context_incremental(
    output_path: Path | str | None = None,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    incremental_update: bool = True,
    lookback_days: int = INCREMENTAL_CONTEXT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Build macro context with optional incremental update.
    
    If incremental_update=True and output_path exists:
      - Preserve historical prefix (before 400 days ago)
      - Recompute only last 400 days from vnstock API
      - Cost: O(400 days) vs O(6 years)
    
    Otherwise:
      - Rebuild full history from API (existing behavior)
    """
    cache_key = _artifact_cache_key(
        "build_macro_context_incremental",
        output_path=_file_snapshot_token(output_path),
        start_date=start_date,
        end_date=end_date,
        incremental_update=incremental_update,
        lookback_days=lookback_days,
    )
    cached = _artifact_cache_get(cache_key)
    if cached is not None:
        return cached
    existing_artifact = _load_existing_artifact_frame(output_path)
    rebuild_start, historical_prefix = (
        _incremental_history_prefix(
            output_path,
            key_columns=("date",),
            lookback_days=lookback_days,
        )
        if incremental_update
        else (None, None)
    )
    
    if rebuild_start is not None:
        # Incremental: only fetch from rebuild_start onward
        fetch_start = rebuild_start.date()
    else:
        # Full rebuild
        fetch_start = start_date
    
    # Fetch macro context for recompute window
    macro_recent = build_macro_context_from_vnstock(
        start_date=fetch_start,
        end_date=end_date,
    )
    
    if macro_recent is None or macro_recent.empty:
        fallback = existing_artifact if existing_artifact is not None and not existing_artifact.empty else macro_recent
        if fallback is None:
            fallback = pd.DataFrame()
        if start_date and "date" in fallback.columns:
            fallback = fallback[fallback["date"] >= pd.Timestamp(start_date).normalize()]
        if end_date and "date" in fallback.columns:
            fallback = fallback[fallback["date"] <= pd.Timestamp(end_date).normalize()]
        result = _attach_source_attrs(
            fallback.reset_index(drop=True),
            provenance=DERIVED_VNSTOCK_PROVENANCE if output_path else DIRECT_VNSTOCK_PROVENANCE,
            source_name="macro_context.csv" if output_path else "vnstock_data.live_macro_context",
            notes="Incremental macro builder returned the existing artifact when live refresh yielded no usable rows.",
            artifact_path=output_path,
        )
        return _artifact_cache_set(cache_key, result)
    
    # Merge: historical prefix + recent rebuild
    macro_final = _merge_incremental_frames(
        historical_prefix,
        macro_recent,
        sort_columns=("date",),
        dedupe_columns=("date",),
    ) if historical_prefix is not None and not historical_prefix.empty else macro_recent.copy().reset_index(drop=True)
    
    # Filter to requested date range
    if start_date:
        ts_start = pd.Timestamp(start_date).normalize()
        macro_final = macro_final[macro_final["date"] >= ts_start]
    
    if end_date:
        ts_end = pd.Timestamp(end_date).normalize()
        macro_final = macro_final[macro_final["date"] <= ts_end]
    
    # Persist if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        macro_final.to_csv(output_path, index=False)
    
    result = _attach_source_attrs(
        macro_final.reset_index(drop=True),
        provenance=DIRECT_VNSTOCK_PROVENANCE,
        source_name="macro_context.csv" if output_path else "vnstock_data.live_macro_context",
        notes="Incremental macro artifact built from vnstock_data macro and commodity series.",
        artifact_path=output_path,
    )
    return _artifact_cache_set(cache_key, result)


def build_foreign_flow_incremental(
    tickers: List[str],
    output_path: Path | str | None = None,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    incremental_update: bool = True,
    lookback_days: int = INCREMENTAL_CONTEXT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Build foreign flow with CSV persistence and optional incremental update.
    
    If incremental_update=True and output_path exists:
      - Load existing CSV
      - Preserve historical prefix (before 400 days ago)
      - Recompute only last 400 days from vnstock API
      - Cost: O(400 days) vs O(6 years)
    
    Otherwise:
      - Rebuild full history from API
      - Persist to CSV for future incremental loads
    """
    normalized_tickers = [str(t).upper().strip() for t in tickers if str(t).strip()]
    if not normalized_tickers:
        return _stub_frame("foreign_flow.csv", "No tickers supplied for foreign-flow fetch.")
    cache_key = _artifact_cache_key(
        "build_foreign_flow_incremental",
        tickers=tuple(sorted(normalized_tickers)),
        output_path=_file_snapshot_token(output_path),
        start_date=start_date,
        end_date=end_date,
        incremental_update=incremental_update,
        lookback_days=lookback_days,
    )
    cached = _artifact_cache_get(cache_key)
    if cached is not None:
        return cached
    existing_artifact = _load_existing_artifact_frame(output_path)
    
    rebuild_start, historical_prefix = (
        _incremental_history_prefix(
            output_path,
            key_columns=("ticker", "date"),
            lookback_days=lookback_days,
        )
        if incremental_update
        else (None, None)
    )
    
    resolved_end_date = end_date or dt.date.today()
    if rebuild_start is not None:
        # Incremental: only fetch from rebuild_start onward
        fetch_start = rebuild_start.date()
    else:
        # Full rebuild
        fetch_start = start_date or (resolved_end_date - dt.timedelta(days=365 * 6))
    
    # Fetch foreign flow for recompute window
    from src.data.adapters.vnstock_adapter import VnstockAdapter
    
    adapter = VnstockAdapter(symbol_list=normalized_tickers)
    frames: list[pd.DataFrame] = []
    
    keep_cols = {
        "date",
        "ticker",
        "foreign_buy_volume",
        "foreign_sell_volume",
        "foreign_buy_value",
        "foreign_sell_value",
        "foreign_net_volume",
        "foreign_net_value",
        "foreign_room_pct",
        "foreign_owned_pct",
        "foreign_available_pct",
        "foreign_current_room",
        "foreign_total_room",
    }
    
    for ticker in normalized_tickers:
        try:
            frame = adapter.get_foreign_flow(
                ticker,
                fetch_start.strftime("%Y-%m-%d"),
                resolved_end_date.strftime("%Y-%m-%d"),
            )
            if frame is None or frame.empty:
                continue
            frame = _normalize_date_frame(frame)
            present_cols = [c for c in frame.columns if c in keep_cols]
            if {"date", "ticker"} <= set(present_cols):
                local = frame[present_cols].copy()
                local["ticker"] = local["ticker"].astype(str).str.upper()
                frames.append(local)
        except Exception as exc:
            logger.debug("foreign_flow_incremental_fetch_failed", ticker=ticker, error=str(exc))
    
    if not frames:
        fallback = existing_artifact if existing_artifact is not None and not existing_artifact.empty else None
        if fallback is not None and not fallback.empty and "ticker" in fallback.columns:
            fallback = fallback[fallback["ticker"].isin(normalized_tickers)]
        if fallback is None or fallback.empty:
            return _artifact_cache_set(
                cache_key,
                _stub_frame("foreign_flow.csv", "Live vnstock_data foreign-flow fetch returned no usable rows."),
            )
        if start_date and "date" in fallback.columns:
            fallback = fallback[fallback["date"] >= pd.Timestamp(start_date).normalize()]
        if end_date and "date" in fallback.columns:
            fallback = fallback[fallback["date"] <= pd.Timestamp(end_date).normalize()]
        result = _attach_source_attrs(
            fallback.reset_index(drop=True),
            provenance=DERIVED_VNSTOCK_PROVENANCE,
            source_name="foreign_flow.csv",
            notes="Incremental foreign-flow builder returned the existing artifact when live refresh yielded no usable rows.",
            artifact_path=output_path,
        )
        return _artifact_cache_set(cache_key, result)
    
    foreign_recent = pd.concat(frames, ignore_index=True)
    foreign_recent = foreign_recent.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"], keep="last")
    
    # Merge: historical prefix + recent rebuild
    foreign_final = _merge_incremental_frames(
        historical_prefix,
        foreign_recent,
        sort_columns=("ticker", "date"),
        dedupe_columns=("ticker", "date"),
    ) if historical_prefix is not None and not historical_prefix.empty else foreign_recent.copy().reset_index(drop=True)
    
    # Filter to requested date range
    if start_date:
        ts_start = pd.Timestamp(start_date).normalize()
        foreign_final = foreign_final[foreign_final["date"] >= ts_start]
    
    if resolved_end_date:
        ts_end = pd.Timestamp(resolved_end_date).normalize()
        foreign_final = foreign_final[foreign_final["date"] <= ts_end]
    
    # Persist if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        foreign_final.to_csv(output_path, index=False)
    
    result = _attach_source_attrs(
        foreign_final.reset_index(drop=True),
        provenance=DIRECT_VNSTOCK_PROVENANCE,
        source_name="foreign_flow.csv" if output_path else "vnstock_data.live_foreign_flow",
        notes="Built with incremental update pattern from vnstock_data Trading.foreign_trade.",
        artifact_path=output_path,
    )
    return _artifact_cache_set(cache_key, result)


def load_macro_context(
    path: Path | str | None = None,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
) -> pd.DataFrame:
    path = Path(path) if path else _DEFAULT_MACRO_CONTEXT_PATH
    cache_key = _artifact_cache_key(
        "load_macro_context",
        artifact=_file_snapshot_token(path),
        start_date=start_date,
        end_date=end_date,
    )
    cached = _artifact_cache_get(cache_key)
    if cached is not None:
        return cached
    if not path.exists():
        logger.debug("macro_context_not_found", path=str(path))
        try:
            # Use incremental builder: preserves history, recomputes only 400-day window
            built = build_macro_context_incremental(
                output_path=path,
                start_date=start_date,
                end_date=end_date,
                incremental_update=True,
                lookback_days=INCREMENTAL_CONTEXT_LOOKBACK_DAYS,
            )
            if built is not None and not built.empty:
                built.attrs["data_quality_contract"] = build_data_quality_contract(
                    built,
                    dataset_name="macro_context.csv",
                    artifact_path=path,
                    key_columns=("date",),
                    stale_after_days=DEFAULT_STALE_AFTER_DAYS.get(path.name),
                )
            final_key = _artifact_cache_key(
                "load_macro_context",
                artifact=_file_snapshot_token(path),
                start_date=start_date,
                end_date=end_date,
            )
            return _artifact_cache_set(final_key, built)
        except Exception as exc:
            logger.debug("macro_context_incremental_build_failed", error=str(exc))
            return _artifact_cache_set(
                cache_key,
                _stub_frame(
                    "macro_context.csv",
                    "Macro/cross-asset artifact is missing and live vnstock_data build failed.",
                ),
            )
    df = pd.read_csv(path)
    df = _normalize_date_frame(df)
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date).normalize()]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date).normalize()]
    result = _attach_source_attrs(
        df.reset_index(drop=True),
        provenance=DERIVED_VNSTOCK_PROVENANCE,
        source_name="macro_context.csv",
        artifact_path=path,
    )
    final_key = _artifact_cache_key(
        "load_macro_context",
        artifact=_file_snapshot_token(path),
        start_date=start_date,
        end_date=end_date,
    )
    return _artifact_cache_set(final_key, result)


def load_foreign_flow(
    path: Path | str | None = None,
    tickers: List[str] | None = None,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
) -> pd.DataFrame:
    path = Path(path) if path else _DEFAULT_FOREIGN_FLOW_PATH
    cache_key = _artifact_cache_key(
        "load_foreign_flow",
        artifact=_file_snapshot_token(path),
        tickers=tuple(sorted({str(t).upper() for t in (tickers or [])})),
        start_date=start_date,
        end_date=end_date,
    )
    cached = _artifact_cache_get(cache_key)
    if cached is not None:
        return cached
    if not path.exists():
        logger.debug("foreign_flow_not_found", path=str(path))
        if tickers:
            try:
                # Use incremental builder with CSV persistence
                built = build_foreign_flow_incremental(
                    tickers=tickers,
                    output_path=path,
                    start_date=start_date,
                    end_date=end_date,
                    incremental_update=True,
                    lookback_days=INCREMENTAL_CONTEXT_LOOKBACK_DAYS,
                )
                if built is not None and not built.empty:
                    built.attrs["data_quality_contract"] = build_data_quality_contract(
                        built,
                        dataset_name="foreign_flow.csv",
                        artifact_path=path,
                        key_columns=("ticker", "date"),
                        stale_after_days=DEFAULT_STALE_AFTER_DAYS.get(path.name),
                    )
                final_key = _artifact_cache_key(
                    "load_foreign_flow",
                    artifact=_file_snapshot_token(path),
                    tickers=tuple(sorted({str(t).upper() for t in (tickers or [])})),
                    start_date=start_date,
                    end_date=end_date,
                )
                return _artifact_cache_set(final_key, built)
            except Exception as exc:
                logger.debug("foreign_flow_incremental_build_failed", error=str(exc))
        return _artifact_cache_set(
            cache_key,
            _stub_frame(
                "foreign_flow.csv",
                "Foreign-flow artifact is missing and live vnstock_data fetch requires an explicit ticker list.",
            ),
        )
    df = pd.read_csv(path)
    df = _normalize_date_frame(df)
    if tickers and "ticker" in df.columns:
        upper = {t.upper() for t in tickers}
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df = df[df["ticker"].isin(upper)]
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date).normalize()]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date).normalize()]
    result = _attach_source_attrs(
        df.reset_index(drop=True),
        provenance=DERIVED_VNSTOCK_PROVENANCE,
        source_name="foreign_flow.csv",
        artifact_path=path,
    )
    final_key = _artifact_cache_key(
        "load_foreign_flow",
        artifact=_file_snapshot_token(path),
        tickers=tuple(sorted({str(t).upper() for t in (tickers or [])})),
        start_date=start_date,
        end_date=end_date,
    )
    return _artifact_cache_set(final_key, result)


def apply_context_features(
    df: pd.DataFrame,
    ticker: str,
    market_df: pd.DataFrame = None,
    sector_df: pd.DataFrame = None,
    ticker_sectors: pd.DataFrame = None,
    breadth_df: pd.DataFrame = None,
    foreign_flow_df: pd.DataFrame = None,
    macro_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Ensure Market and Sector rolling features are present (Training & Inference Parity).

    This logic is centralized here to prevent drift between training and inference scripts.
    It builds:
      - m_ret, m_ret_5d, rel_to_market
      - s_ret, s_ret_5d, rel_to_sector
    """
    frame = df.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date").reset_index(drop=True)

    def _finalize_context_metadata(current: pd.DataFrame, prefix: str) -> pd.DataFrame:
        available_col = f"{prefix}_context_available"
        source_date_col = f"{prefix}_context_source_date"
        missing_col = f"{prefix}_context_missing"
        if available_col not in current.columns:
            current[available_col] = False
        current[available_col] = current[available_col].fillna(False).astype(bool)
        if source_date_col not in current.columns:
            current[source_date_col] = pd.NaT
        current[source_date_col] = pd.to_datetime(current[source_date_col], errors="coerce").dt.normalize()
        current[missing_col] = ~current[available_col]
        return current

    def _relative_return_base(current: pd.DataFrame) -> pd.Series:
        if "pct_return" in current.columns:
            return pd.to_numeric(current["pct_return"], errors="coerce")
        if "close_to_close_return_1d" in current.columns:
            return pd.to_numeric(current["close_to_close_return_1d"], errors="coerce")
        return pd.to_numeric(current["close"], errors="coerce").pct_change().fillna(0.0)

    # 1. Market proxy
    if market_df is not None and not market_df.empty:
        market = _normalize_date_frame(market_df)
        if "m_ret" not in market.columns and "close" in market.columns:
            market = market.copy()
            market["m_ret"] = pd.to_numeric(market["close"], errors="coerce").pct_change()
        if "m_ret" not in frame.columns and "m_ret" in market.columns:
            frame = frame.merge(market[["date", "m_ret"]], on="date", how="left")
        if "m_ret" in frame.columns:
            market_ret = pd.to_numeric(frame["m_ret"], errors="coerce").fillna(0.0)
            rel_base = _relative_return_base(frame)
            frame = _concat_frame_block(
                frame,
                {
                    "m_ret": market_ret,
                    "m_ret_5d": market_ret.rolling(5).mean().fillna(0.0),
                    "m_ret_20d": market_ret.rolling(20).mean().fillna(0.0),
                    "rel_to_market": rel_base - market_ret,
                },
            )

    # 2. Sector proxy
    if sector_df is not None and ticker_sectors is not None and not sector_df.empty and not ticker_sectors.empty:
        ts = ticker_sectors.copy()
        ts["ticker"] = ts["ticker"].astype(str).str.upper()
        industry_matches = ts.loc[ts["ticker"] == ticker.upper(), "industry"]
        industry = str(industry_matches.iloc[0]) if not industry_matches.empty else None

        if industry:
            sector = _normalize_date_frame(sector_df)
            sector_cols = [
                column
                for column in ["ret", "sector_dispersion", "sector_member_count"]
                if column in sector.columns
            ]
            sector = sector[sector["industry"] == industry][["date", *sector_cols]].copy()
            sector = sector.rename(columns={"ret": "s_ret"})

            new_cols = [column for column in sector.columns if column != "date" and column not in frame.columns]
            if new_cols:
                frame = frame.merge(sector[["date", *new_cols]], on="date", how="left")

            sector_block: dict[str, Any] = {}
            if "s_ret" in frame.columns:
                sector_ret = pd.to_numeric(frame["s_ret"], errors="coerce").fillna(0.0)
                sector_block["s_ret"] = sector_ret
                sector_block["s_ret_5d"] = sector_ret.rolling(5).mean().fillna(0.0)
                sector_block["rel_to_sector"] = _relative_return_base(frame) - sector_ret
            if "sector_dispersion" in frame.columns:
                sector_block["sector_dispersion"] = pd.to_numeric(frame["sector_dispersion"], errors="coerce").fillna(0.0)
            if "sector_member_count" in frame.columns:
                sector_block["sector_member_count"] = pd.to_numeric(frame["sector_member_count"], errors="coerce").fillna(0.0)
            if sector_block:
                frame = _concat_frame_block(frame, sector_block)
        else:
            frame = _concat_frame_block(
                frame,
                {
                    column: pd.Series(0.0, index=frame.index)
                    for column in ["s_ret", "s_ret_5d", "rel_to_sector", "sector_dispersion", "sector_member_count"]
                    if column not in frame.columns
                },
            )

    # 3. Market breadth
    if breadth_df is not None:
        frame = frame.drop(columns=[column for column in _BREADTH_CONTEXT_METADATA_COLUMNS if column in frame.columns])
        breadth = _normalize_date_frame(breadth_df) if not breadth_df.empty else pd.DataFrame()
        if not breadth.empty and "date" in breadth.columns:
            breadth = breadth.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
            breadth_meta = breadth[["date"]].copy()
            breadth_meta["breadth_context_available"] = True
            breadth_meta["breadth_context_source_date"] = breadth_meta["date"]
            frame = frame.merge(breadth_meta, on="date", how="left")
        breadth_cols = [
            column
            for column in [
                "advancers",
                "decliners",
                "unchanged",
                "net_advancers",
                "advance_decline_ratio",
                "market_breadth",
                "breadth_member_count",
                "advancing_share",
                "declining_share",
                "pct_above_ma20",
                "pct_above_ma50",
                "new_highs_252",
                "new_lows_252",
                "new_high_low_spread",
                "up_volume",
                "down_volume",
                "up_down_volume_ratio",
            ]
            if column in breadth.columns
        ]
        if breadth_cols:
            missing_cols = [column for column in breadth_cols if column not in frame.columns]
            if missing_cols:
                frame = frame.merge(breadth[["date", *missing_cols]], on="date", how="left")
            breadth_block: dict[str, Any] = {}
            for column in [
                "advancers",
                "decliners",
                "unchanged",
                "net_advancers",
                "market_breadth",
                "breadth_member_count",
                "advancing_share",
                "declining_share",
                "pct_above_ma20",
                "pct_above_ma50",
                "new_highs_252",
                "new_lows_252",
                "new_high_low_spread",
                "up_volume",
                "down_volume",
                "advance_decline_ratio",
                "up_down_volume_ratio",
            ]:
                if column in frame.columns:
                    breadth_block[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
            if breadth_block:
                frame = _concat_frame_block(frame, breadth_block)
        frame = _finalize_context_metadata(frame, "breadth")

    # 4. Foreign flow
    if foreign_flow_df is not None:
        frame = frame.drop(columns=[column for column in _FOREIGN_FLOW_CONTEXT_METADATA_COLUMNS if column in frame.columns])
        foreign_flow = _normalize_date_frame(foreign_flow_df) if not foreign_flow_df.empty else pd.DataFrame()
        if "ticker" in foreign_flow.columns:
            foreign_flow["ticker"] = foreign_flow["ticker"].astype(str).str.upper()
            foreign_flow = foreign_flow[foreign_flow["ticker"] == ticker.upper()]
        if not foreign_flow.empty and "date" in foreign_flow.columns:
            foreign_flow = foreign_flow.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
            foreign_meta = foreign_flow[["date"]].copy()
            foreign_meta["foreign_flow_context_available"] = True
            foreign_meta["foreign_flow_context_source_date"] = foreign_meta["date"]
            frame = frame.merge(foreign_meta, on="date", how="left")
            foreign_cols = [
                column
                for column in foreign_flow.columns
                if column
                not in {
                    "ticker",
                    "date",
                    *_FOREIGN_FLOW_CONTEXT_METADATA_COLUMNS,
                    *_FOREIGN_FLOW_PROVENANCE_COLUMNS,
                }
            ]
            new_cols = [column for column in foreign_cols if column not in frame.columns]
            if new_cols:
                frame = frame.merge(foreign_flow[["date", *new_cols]], on="date", how="left")
                for column in new_cols:
                    if str(column).startswith(("foreign_", "fr_")):
                        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        frame = _finalize_context_metadata(frame, "foreign_flow")

    # 5. Macro / cross-asset context (backward-looking as-of join only)
    if macro_df is not None and not macro_df.empty:
        frame = _ensure_datetime64ns(frame, "date")
        macro = _ensure_datetime64ns(_normalize_date_frame(macro_df), "date")
        macro_cols = [column for column in macro.columns if column != "date" and column not in frame.columns]
        if macro_cols:
            frame = pd.merge_asof(
                frame.sort_values("date"),
                macro[["date", *macro_cols]].sort_values("date"),
                on="date",
                direction="backward",
            )

    return frame


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
                    df = validate_ohlcv(df, ticker, min_rows=60)
                    df["ohlcv_source"] = df.attrs.get("source_name", name)
                    df["price_adjustment_status"] = df.attrs.get("adjustment_status", "unknown")
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
        join_sectors: bool = False,
        join_breadth: bool = False,
        join_macro: bool = False,
        join_foreign_flow: bool = False,
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
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                require_validated_source=True,
            )
            if not sent_df.empty:
                result = result.merge(sent_df, on=["ticker", "date"], how="left")
                logger.debug("sentiment_joined")

        if (join_sectors or join_market or join_breadth or join_macro or join_foreign_flow) and "ticker" in result.columns:
            # We use apply_context_features which handles both Market and Sector
            m_df = load_market_proxy() if join_market else None
            s_proxies = load_sector_proxies() if join_sectors else None
            t_sectors = load_ticker_sectors() if join_sectors else None
            breadth_df = load_market_breadth() if join_breadth else None
            macro_df = load_macro_context(start_date=start_date, end_date=end_date) if join_macro else None
            foreign_flow_df = load_foreign_flow(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
            ) if join_foreign_flow else None
            
            processed_frames = []
            for t in result["ticker"].unique():
                t_df = result[result["ticker"] == t]
                t_df = apply_context_features(
                    t_df,
                    t,
                    market_df=m_df,
                    sector_df=s_proxies,
                    ticker_sectors=t_sectors,
                    breadth_df=breadth_df,
                    foreign_flow_df=foreign_flow_df,
                    macro_df=macro_df,
                )
                processed_frames.append(t_df)
            
            result = pd.concat(processed_frames, ignore_index=True)
            logger.debug(
                "context_features_applied",
                joined_sectors=join_sectors,
                joined_market=join_market,
                joined_breadth=join_breadth,
                joined_macro=join_macro,
                joined_foreign_flow=join_foreign_flow,
            )
        elif join_sectors or join_market or join_breadth or join_macro or join_foreign_flow:
            logger.warning(
                "ticker_missing_context_skipped",
                join_sectors=join_sectors,
                join_market=join_market,
                join_breadth=join_breadth,
                join_macro=join_macro,
                join_foreign_flow=join_foreign_flow,
            )

        result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
        if result.duplicated(subset=["ticker", "date"]).any():
            logger.warning("dataset_duplicate_ticker_date", action="keeping_last")
            result = result.drop_duplicates(subset=["ticker", "date"], keep="last").reset_index(drop=True)

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
        lookback_days: int = 400, # Increased default buffer
        join_market: bool = True,
        join_fundamentals: bool = False,
        join_sentiment: bool = False,
        join_sectors: bool = False,
        join_breadth: bool = False,
        join_macro: bool = False,
        join_foreign_flow: bool = False,
    ) -> pd.DataFrame:
        """Build a dataset suitable for batch inference.
        
        To ensure feature parity with training (especially for rolling features 
        like roc_250), this loads ALL available history for calculation.
        Slicing to lookback_days should be done AFTER feature engineering.
        """
        end = dt.date.today()
        # Pass start_date=None to build_dataset to load full CSV history
        return self.build_dataset(
            tickers=tickers,
            start_date=None, 
            end_date=end,
            join_market=join_market,
            join_fundamentals=join_fundamentals,
            join_sentiment=join_sentiment,
            join_sectors=join_sectors,
            join_breadth=join_breadth,
            join_macro=join_macro,
            join_foreign_flow=join_foreign_flow,
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
    join_sectors: bool = False,
    join_breadth: bool = False,
    join_macro: bool = False,
    join_foreign_flow: bool = False,
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
        join_sectors=join_sectors,
        join_breadth=join_breadth,
        join_macro=join_macro,
        join_foreign_flow=join_foreign_flow,
    )
