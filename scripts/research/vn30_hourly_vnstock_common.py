"""Shared helpers for the VN30 hourly vnstock full-history research track."""

from __future__ import annotations

import csv
import importlib
import importlib.metadata
import importlib.util
import json
import multiprocessing as mp
import time
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.research.vn30_hourly_common import (
    EVAL_END,
    EVAL_END_TEXT,
    EVAL_START,
    EVAL_START_TEXT,
    REPO_ROOT,
    TRAIN_CUTOFF,
    TRAIN_CUTOFF_TEXT,
    TRAIN_START,
    TRAIN_START_TEXT,
    VN30_TICKERS,
    markdown_table,
    read_universe,
    rel,
    timestamp_text,
    write_csv,
    write_json,
)


LOCAL_EXCHANGE_TZ = "Asia/Ho_Chi_Minh"
SOURCE_PRIORITY = ("VCI", "KBS", "VND", "MAS")
PROBE_WINDOWS = (
    ("2024-01-02", "2024-01-05"),
    ("2025-01-02", "2025-01-05"),
    ("2026-05-04", "2026-05-11"),
)
PROBE_SYMBOLS = ("ACB", "HPG", "VNINDEX", "VN30INDEX", "VNXALL")

FETCH_REPORT_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_hourly_vnstock_fetch"
FULL_REPORT_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_hourly_vnstock_full"
RAW_FETCH_DIR = REPO_ROOT / "data" / "raw" / "vnstock_fetch" / "vn30_hourly_2005_2026"
NORMALIZED_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly"
BENCHMARK_OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_vnstock_full_2005_2026_traincutoff"
PAPER_PATH = REPO_ROOT / "reports" / "NCKH_FULL_PAPER_DRAFT_VN30_HOURLY_VNSTOCK_2005_2026_V1_WITH_FIGURES.md"
DOCX_NOTES_PATH = REPO_ROOT / "reports" / "NCKH_VN30_HOURLY_VNSTOCK_2005_2026_DOCX_BUILD_NOTES.md"
MISSING_EVIDENCE_PATH = FETCH_REPORT_ROOT / "vn30_full_benchmark_missing_evidence.md"

VNINDEX_REQUIREMENT_START = pd.Timestamp("2005-01-01 00:00:00")
VN30INDEX_REQUIREMENT_START = pd.Timestamp("2012-02-06 00:00:00")
VNXALL_REQUIREMENT_START = pd.Timestamp("2016-10-24 00:00:00")
INDEX_REQUIREMENTS = {
    "VNINDEX": VNINDEX_REQUIREMENT_START,
    "VN30INDEX": VN30INDEX_REQUIREMENT_START,
    "VNXALL": VNXALL_REQUIREMENT_START,
}
REQUIRED_INDEX_CODES = ("VNINDEX",)
OPTIONAL_INDEX_CODES = ("VN30INDEX", "VNXALL")
ALL_INDEX_CODES = (*REQUIRED_INDEX_CODES, *OPTIONAL_INDEX_CODES)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
NORMALIZED_COLUMNS = ["datetime", "ticker", *OHLCV_COLUMNS, "provider", "source"]


@dataclass(frozen=True)
class ProviderCall:
    package: str
    provider: str
    source: str
    function_used: str
    call: Callable[[], Any]


@dataclass(frozen=True)
class ProviderSpec:
    package: str
    provider: str
    source: str
    function_used: str


@dataclass
class AttemptResult:
    symbol: str
    asset_type: str
    start_date: str
    end_date: str
    package: str
    package_version: str
    provider: str
    source: str
    function_used: str
    returned_rows: int
    standardized_rows: int
    returned_columns: str
    first_timestamp: str
    last_timestamp: str
    success: bool
    exception_type: str
    exception_message: str
    frame: pd.DataFrame

    def to_log_row(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "asset_type": self.asset_type,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "package": self.package,
            "package_version": self.package_version,
            "provider": self.provider,
            "source": self.source,
            "function_used": self.function_used,
            "returned_rows": self.returned_rows,
            "standardized_rows": self.standardized_rows,
            "returned_columns": self.returned_columns,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "success": str(self.success).lower(),
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
        }


def asset_type(symbol: str) -> str:
    return "index" if symbol.upper().strip() in ALL_INDEX_CODES else "stock"


def symbol_requirement_start(symbol: str) -> pd.Timestamp:
    code = symbol.upper().strip()
    if code in INDEX_REQUIREMENTS:
        return INDEX_REQUIREMENTS[code]
    return TRAIN_START


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return ""
    except Exception as exc:
        return f"unknown ({type(exc).__name__}: {exc})"


def package_status_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for package_name in ("vnstock_data", "vnstock"):
        spec = importlib.util.find_spec(package_name)
        rows.append(
            {
                "package": package_name,
                "installed": str(spec is not None).lower(),
                "version": package_version(package_name) if spec is not None else "",
                "origin": getattr(spec, "origin", "") if spec is not None else "",
            }
        )
    return rows


def _import_module_safely(package_name: str) -> Any | None:
    if importlib.util.find_spec(package_name) is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return importlib.import_module(package_name)
    except BaseException:
        return None


def _load_vnstock_data_namespace() -> dict[str, Any]:
    module = _import_module_safely("vnstock_data")
    if module is None:
        return {}
    namespace = {name: getattr(module, name, None) for name in ("Quote", "QuoteHistory")}
    for module_name, attr_name in (
        ("vnstock_data.api.quote", "Quote"),
        ("vnstock_data.api.quote", "QuoteHistory"),
        ("vnstock_data.explorer.vci.quote", "Quote"),
    ):
        if namespace.get(attr_name) is not None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lazy_module = importlib.import_module(module_name)
            namespace[attr_name] = getattr(lazy_module, attr_name, None)
        except Exception:
            continue
    return namespace


def iter_provider_calls(symbol: str, start_date: str, end_date: str) -> Iterator[ProviderCall]:
    """Yield direct hourly provider calls without falling back to daily data."""
    code = symbol.upper().strip()
    namespace = _load_vnstock_data_namespace()
    quote = namespace.get("Quote")
    quote_history = namespace.get("QuoteHistory")
    for source in SOURCE_PRIORITY:
        if quote is not None:
            yield ProviderCall(
                package="vnstock_data",
                provider="vnstock_data",
                source=source,
                function_used="Quote.history(interval=1H)",
                call=lambda quote=quote, source=source: quote(source=source, symbol=code).history(
                    start=start_date,
                    end=end_date,
                    interval="1H",
                    get_all=True,
                ),
            )
        if quote_history is not None:
            yield ProviderCall(
                package="vnstock_data",
                provider="vnstock_data",
                source=source,
                function_used="QuoteHistory.history(timeframe=1H)",
                call=lambda quote_history=quote_history, source=source: quote_history(source=source, symbol=code).history(
                    start_date=start_date,
                    end_date=end_date,
                    timeframe="1H",
                ),
            )

    legacy = _import_module_safely("vnstock")
    if legacy is None:
        return
    legacy_quote = getattr(legacy, "Quote", None)
    if legacy_quote is not None:
        for source in SOURCE_PRIORITY:
            yield ProviderCall(
                package="vnstock",
                provider="vnstock",
                source=source,
                function_used="Quote.history(interval=1H)",
                call=lambda legacy_quote=legacy_quote, source=source: legacy_quote(source=source, symbol=code).history(
                    start=start_date,
                    end=end_date,
                    interval="1H",
                ),
            )
    stock_historical_data = getattr(legacy, "stock_historical_data", None)
    if callable(stock_historical_data):
        for resolution in ("1H", "60"):
            yield ProviderCall(
                package="vnstock",
                provider="vnstock",
                source="legacy_stock_historical_data",
                function_used=f"stock_historical_data(resolution={resolution})",
                call=lambda resolution=resolution: stock_historical_data(
                    symbol=code,
                    start_date=start_date,
                    end_date=end_date,
                    resolution=resolution,
                    type="stock",
                    beautify=False,
                    decor=False,
                ),
            )
    vnstock_class = getattr(legacy, "Vnstock", None)
    if vnstock_class is not None:
        for source in SOURCE_PRIORITY:
            yield ProviderCall(
                package="vnstock",
                provider="vnstock",
                source=source,
                function_used="Vnstock.stock.quote.history(interval=1H)",
                call=lambda vnstock_class=vnstock_class, source=source: vnstock_class()
                .stock(symbol=code, source=source)
                .quote.history(start=start_date, end=end_date, interval="1H"),
            )


def iter_provider_specs(symbol: str) -> Iterator[ProviderSpec]:
    """Yield provider call specs that can be reconstructed in a child process."""
    _ = symbol
    namespace = _load_vnstock_data_namespace()
    if namespace.get("Quote") is not None:
        for source in SOURCE_PRIORITY:
            yield ProviderSpec("vnstock_data", "vnstock_data", source, "Quote.history(interval=1H)")
    if namespace.get("QuoteHistory") is not None:
        for source in SOURCE_PRIORITY:
            yield ProviderSpec("vnstock_data", "vnstock_data", source, "QuoteHistory.history(timeframe=1H)")
    legacy = _import_module_safely("vnstock")
    if legacy is None:
        return
    if getattr(legacy, "Quote", None) is not None:
        for source in SOURCE_PRIORITY:
            yield ProviderSpec("vnstock", "vnstock", source, "Quote.history(interval=1H)")
    if callable(getattr(legacy, "stock_historical_data", None)):
        for resolution in ("1H", "60"):
            yield ProviderSpec("vnstock", "vnstock", "legacy_stock_historical_data", f"stock_historical_data(resolution={resolution})")
    if getattr(legacy, "Vnstock", None) is not None:
        for source in SOURCE_PRIORITY:
            yield ProviderSpec("vnstock", "vnstock", source, "Vnstock.stock.quote.history(interval=1H)")


def _lazy_quote_class(module: Any, class_name: str) -> Any | None:
    cls = getattr(module, class_name, None)
    if cls is not None:
        return cls
    for module_name in ("vnstock_data.api.quote", "vnstock_data.explorer.vci.quote"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lazy_module = importlib.import_module(module_name)
            cls = getattr(lazy_module, class_name, None)
            if cls is not None:
                return cls
        except Exception:
            continue
    return None


def _run_provider_spec(symbol: str, start_date: str, end_date: str, spec: ProviderSpec) -> Any:
    code = symbol.upper().strip()
    if spec.package == "vnstock_data":
        module = importlib.import_module("vnstock_data")
        if spec.function_used.startswith("Quote.history"):
            quote = _lazy_quote_class(module, "Quote")
            if quote is None:
                raise ImportError("vnstock_data Quote class is unavailable")
            return quote(source=spec.source, symbol=code).history(
                start=start_date,
                end=end_date,
                interval="1H",
                get_all=True,
            )
        if spec.function_used.startswith("QuoteHistory.history"):
            quote_history = _lazy_quote_class(module, "QuoteHistory")
            if quote_history is None:
                raise ImportError("vnstock_data QuoteHistory class is unavailable")
            return quote_history(source=spec.source, symbol=code).history(
                start_date=start_date,
                end_date=end_date,
                timeframe="1H",
            )
    if spec.package == "vnstock":
        module = importlib.import_module("vnstock")
        if spec.function_used.startswith("Quote.history"):
            quote = getattr(module, "Quote", None)
            if quote is None:
                raise ImportError("vnstock Quote class is unavailable")
            return quote(source=spec.source, symbol=code).history(start=start_date, end=end_date, interval="1H")
        if spec.function_used.startswith("stock_historical_data"):
            stock_historical_data = getattr(module, "stock_historical_data", None)
            if not callable(stock_historical_data):
                raise ImportError("vnstock stock_historical_data is unavailable")
            resolution = "60" if "resolution=60" in spec.function_used else "1H"
            return stock_historical_data(
                symbol=code,
                start_date=start_date,
                end_date=end_date,
                resolution=resolution,
                type="stock",
                beautify=False,
                decor=False,
            )
        if spec.function_used.startswith("Vnstock.stock"):
            vnstock_class = getattr(module, "Vnstock", None)
            if vnstock_class is None:
                raise ImportError("vnstock Vnstock class is unavailable")
            return vnstock_class().stock(symbol=code, source=spec.source).quote.history(
                start=start_date,
                end=end_date,
                interval="1H",
            )
    raise ValueError(f"Unsupported provider spec: {spec}")


def _provider_timeout_worker(queue: Any, symbol: str, start_date: str, end_date: str, spec: ProviderSpec) -> None:
    try:
        result = _run_provider_spec(symbol, start_date, end_date, spec)
        raw = as_dataframe(result)
        standardized = standardize_provider_frame(raw, symbol, provider=spec.provider, source=spec.source)
        if not standardized.empty:
            standardized = standardized.copy()
            standardized["datetime"] = pd.to_datetime(standardized["datetime"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        queue.put(
            {
                "ok": True,
                "raw_rows": int(len(raw)),
                "columns": ",".join(str(column) for column in raw.columns),
                "records": standardized.to_dict("records"),
            }
        )
    except BaseException as exc:
        queue.put(
            {
                "ok": False,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc).replace("\n", " ")[:800],
            }
        )


def attempt_provider_fetch_with_timeout(symbol: str, start_date: str, end_date: str, timeout_seconds: float) -> list[AttemptResult]:
    rows: list[AttemptResult] = []
    specs = list(iter_provider_specs(symbol))
    for spec in specs:
        queue: Any = mp.Queue()
        process = mp.Process(target=_provider_timeout_worker, args=(queue, symbol, start_date, end_date, spec))
        process.start()
        process.join(timeout=max(1.0, float(timeout_seconds)))
        standardized = pd.DataFrame(columns=NORMALIZED_COLUMNS)
        raw_rows = 0
        columns = ""
        first_ts = ""
        last_ts = ""
        exception_type = ""
        exception_message = ""
        if process.is_alive():
            process.terminate()
            process.join(5)
            exception_type = "TimeoutError"
            exception_message = f"provider call exceeded {timeout_seconds:.1f} seconds"
        else:
            payload = queue.get() if not queue.empty() else {"ok": False, "exception_type": "NoResult", "exception_message": "provider process returned no result"}
            if payload.get("ok"):
                raw_rows = int(payload.get("raw_rows", 0) or 0)
                columns = str(payload.get("columns", ""))
                records = payload.get("records", [])
                standardized = pd.DataFrame(records, columns=NORMALIZED_COLUMNS)
                if not standardized.empty:
                    standardized["datetime"] = pd.to_datetime(standardized["datetime"], errors="coerce")
                    first_ts = timestamp_text(standardized["datetime"].min())
                    last_ts = timestamp_text(standardized["datetime"].max())
            else:
                exception_type = str(payload.get("exception_type", "ProviderError"))
                exception_message = str(payload.get("exception_message", ""))[:800]
        rows.append(
            AttemptResult(
                symbol=symbol.upper().strip(),
                asset_type=asset_type(symbol),
                start_date=start_date,
                end_date=end_date,
                package=spec.package,
                package_version=package_version(spec.package),
                provider=spec.provider,
                source=spec.source,
                function_used=spec.function_used,
                returned_rows=raw_rows,
                standardized_rows=int(len(standardized)),
                returned_columns=columns,
                first_timestamp=first_ts,
                last_timestamp=last_ts,
                success=not standardized.empty,
                exception_type=exception_type,
                exception_message=exception_message,
                frame=standardized,
            )
        )
    if not rows:
        return attempt_provider_fetch(symbol, start_date, end_date)
    return rows


def as_dataframe(result: Any) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result.copy()
    if isinstance(result, pd.Series):
        return result.to_frame(name=result.name or "value").reset_index()
    if isinstance(result, (list, tuple)):
        return pd.DataFrame(result)
    if isinstance(result, dict):
        try:
            return pd.DataFrame(result)
        except Exception:
            return pd.DataFrame([result])
    return pd.DataFrame()


def _column_by_lower(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column).strip().lower(): str(column) for column in frame.columns}


def find_provider_time_column(frame: pd.DataFrame) -> str | None:
    lower = _column_by_lower(frame)
    for candidate in ("datetime", "time", "timestamp", "date", "trading_date", "tradingdate"):
        if candidate in lower:
            return lower[candidate]
    if isinstance(frame.index, pd.DatetimeIndex):
        return "__index__"
    return None


def _parse_exchange_datetime(values: Any) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    try:
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            return parsed.dt.tz_convert(LOCAL_EXCHANGE_TZ).dt.tz_localize(None)
    except Exception:
        pass
    try:
        return parsed.dt.tz_localize(None)
    except (AttributeError, TypeError, ValueError):
        return parsed


def standardize_provider_frame(raw: pd.DataFrame, symbol: str, *, provider: str, source: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    prepared = raw.copy()
    if isinstance(prepared.index, pd.DatetimeIndex) and "datetime" not in prepared.columns:
        prepared = prepared.reset_index().rename(columns={prepared.index.name or "index": "datetime"})
    time_column = find_provider_time_column(prepared)
    if time_column is None:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    if time_column == "__index__":
        prepared = prepared.reset_index().rename(columns={prepared.index.name or "index": "datetime"})
    elif time_column != "datetime":
        prepared = prepared.rename(columns={time_column: "datetime"})

    lower = _column_by_lower(prepared)
    rename_map: dict[str, str] = {}
    for target in OHLCV_COLUMNS:
        source_column = lower.get(target)
        if source_column is not None and source_column != target:
            rename_map[source_column] = target
    if rename_map:
        prepared = prepared.rename(columns=rename_map)

    missing = [column for column in ["datetime", *OHLCV_COLUMNS] if column not in prepared.columns]
    if missing:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)

    standardized = prepared[["datetime", *OHLCV_COLUMNS]].copy()
    standardized["datetime"] = _parse_exchange_datetime(standardized["datetime"])
    standardized["ticker"] = symbol.upper().strip()
    standardized["provider"] = provider
    standardized["source"] = source
    for column in OHLCV_COLUMNS:
        standardized[column] = pd.to_numeric(standardized[column], errors="coerce")
    standardized = standardized.dropna(subset=["datetime", "open", "high", "low", "close", "volume"])
    standardized = standardized[
        (standardized["open"] > 0)
        & (standardized["high"] > 0)
        & (standardized["low"] > 0)
        & (standardized["close"] > 0)
        & (standardized["volume"] >= 0)
    ].copy()
    if standardized.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    standardized = standardized.sort_values(["ticker", "datetime"]).drop_duplicates(["ticker", "datetime"], keep="last")
    return standardized[NORMALIZED_COLUMNS].reset_index(drop=True)


def attempt_provider_fetch(symbol: str, start_date: str, end_date: str) -> list[AttemptResult]:
    rows: list[AttemptResult] = []
    for provider_call in iter_provider_calls(symbol, start_date, end_date):
        raw_rows = 0
        columns = ""
        first_ts = ""
        last_ts = ""
        standardized = pd.DataFrame(columns=NORMALIZED_COLUMNS)
        exception_type = ""
        exception_message = ""
        try:
            result = provider_call.call()
            raw = as_dataframe(result)
            raw_rows = int(len(raw))
            columns = ",".join(str(column) for column in raw.columns)
            standardized = standardize_provider_frame(
                raw,
                symbol,
                provider=provider_call.provider,
                source=provider_call.source,
            )
            if not standardized.empty:
                first_ts = timestamp_text(standardized["datetime"].min())
                last_ts = timestamp_text(standardized["datetime"].max())
        except BaseException as exc:
            exception_type = type(exc).__name__
            exception_message = str(exc).replace("\n", " ")[:800]
        rows.append(
            AttemptResult(
                symbol=symbol.upper().strip(),
                asset_type=asset_type(symbol),
                start_date=start_date,
                end_date=end_date,
                package=provider_call.package,
                package_version=package_version(provider_call.package),
                provider=provider_call.provider,
                source=provider_call.source,
                function_used=provider_call.function_used,
                returned_rows=raw_rows,
                standardized_rows=int(len(standardized)),
                returned_columns=columns,
                first_timestamp=first_ts,
                last_timestamp=last_ts,
                success=not standardized.empty,
                exception_type=exception_type,
                exception_message=exception_message,
                frame=standardized,
            )
        )
    if not rows:
        for package_name in ("vnstock_data", "vnstock"):
            if importlib.util.find_spec(package_name) is None:
                rows.append(
                    AttemptResult(
                        symbol=symbol.upper().strip(),
                        asset_type=asset_type(symbol),
                        start_date=start_date,
                        end_date=end_date,
                        package=package_name,
                        package_version="",
                        provider=package_name,
                        source="not_installed",
                        function_used="package_import",
                        returned_rows=0,
                        standardized_rows=0,
                        returned_columns="",
                        first_timestamp="",
                        last_timestamp="",
                        success=False,
                        exception_type="PackageNotFound",
                        exception_message=f"{package_name} is not installed",
                        frame=pd.DataFrame(columns=NORMALIZED_COLUMNS),
                    )
                )
    return rows


def fetch_first_success(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    retries: int,
    backoff_seconds: float,
    timeout_seconds: float | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    attempt_logs: list[dict[str, Any]] = []
    for attempt_number in range(1, retries + 1):
        if timeout_seconds is not None and timeout_seconds > 0:
            results = attempt_provider_fetch_with_timeout(symbol, start_date, end_date, timeout_seconds)
        else:
            results = attempt_provider_fetch(symbol, start_date, end_date)
        for result in results:
            row = result.to_log_row()
            row["retry_attempt"] = attempt_number
            attempt_logs.append(row)
            if result.success:
                return result.frame, attempt_logs
        if attempt_number < retries:
            time.sleep(max(0.0, backoff_seconds) * attempt_number)
    return pd.DataFrame(columns=NORMALIZED_COLUMNS), attempt_logs


def period_chunks(start: pd.Timestamp, end: pd.Timestamp, level: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if level == "year":
        cursor = start
        while cursor <= end:
            chunk_end = min(pd.Timestamp(cursor.year, 12, 31), end)
            chunks.append((cursor, chunk_end))
            cursor = chunk_end + pd.Timedelta(days=1)
    elif level == "month":
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + pd.offsets.MonthEnd(0), end)
            chunks.append((cursor, pd.Timestamp(chunk_end)))
            cursor = pd.Timestamp(chunk_end) + pd.Timedelta(days=1)
    else:
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + pd.Timedelta(days=6), end)
            chunks.append((cursor, chunk_end))
            cursor = chunk_end + pd.Timedelta(days=1)
    return chunks


def filename_date(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def raw_chunk_path(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    code = symbol.upper().strip()
    return RAW_FETCH_DIR / code / f"{code}_{filename_date(start)}_{filename_date(end)}.csv"


def normalized_cache_path(symbol: str) -> Path:
    return NORMALIZED_CACHE_DIR / f"{symbol.upper().strip()}.csv"


def read_normalized_symbol(symbol: str) -> pd.DataFrame:
    path = normalized_cache_path(symbol)
    if not path.exists():
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    missing = [column for column in NORMALIZED_COLUMNS if column not in frame.columns]
    if missing:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    for column in OHLCV_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["datetime", "ticker", *OHLCV_COLUMNS])
    frame = frame.sort_values(["ticker", "datetime"]).drop_duplicates(["ticker", "datetime"], keep="last")
    return frame[NORMALIZED_COLUMNS].reset_index(drop=True)


def write_normalized_symbol(symbol: str, frame: pd.DataFrame) -> None:
    path = normalized_cache_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output["datetime"] = pd.to_datetime(output["datetime"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    output[NORMALIZED_COLUMNS].to_csv(path, index=False)


def load_fetched_universe_frame(tickers: list[str] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ticker in tickers or VN30_TICKERS:
        frame = read_normalized_symbol(ticker)
        if frame.empty:
            continue
        frames.append(frame[["datetime", "ticker", "open", "high", "low", "close", "volume"]])
    if not frames:
        return pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "datetime"]).reset_index(drop=True)


def coverage_flags(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[bool, bool, str, str]:
    if frame.empty or "datetime" not in frame.columns:
        return False, False, "", ""
    timestamps = pd.to_datetime(frame["datetime"], errors="coerce").dropna()
    if timestamps.empty:
        return False, False, "", ""
    first = pd.Timestamp(timestamps.min())
    last = pd.Timestamp(timestamps.max())
    # The date endpoints can fall on weekends or exchange holidays. The gate
    # still fails material gaps, especially the future 2026-05-31 endpoint.
    start_ok = first <= start + pd.Timedelta(days=10)
    end_ok = last >= end - pd.Timedelta(days=7)
    return start_ok, end_ok, timestamp_text(first), timestamp_text(last)


def build_docx_notes(*, paper_exists: bool, validation_rows: list[dict[str, Any]]) -> None:
    stock_rows = [row for row in validation_rows if row.get("asset_type") == "stock"]
    usable_stocks = [row.get("symbol", "") for row in stock_rows if str(row.get("benchmark_usable", "")).lower() == "true"]
    vnindex_row = next((row for row in validation_rows if row.get("symbol") == "VNINDEX"), {})
    content = [
        "# VN30 Hourly vnstock 2005-2026 DOCX Build Notes",
        "",
        "## Source Markdown",
        "",
        f"- `{rel(PAPER_PATH)}`" if paper_exists else "- Final paper was not written because the fetched-data validation or benchmark gate did not pass.",
        "",
        "## Design",
        "",
        f"- Universe: frozen VN30 tickers from `{rel(REPO_ROOT / 'configs' / 'universes' / 'vn30_constituents_frozen.csv')}`.",
        "- Frequency: hourly only.",
        f"- Training/history period: {TRAIN_START_TEXT} to {TRAIN_CUTOFF_TEXT}.",
        f"- Evaluation/comparison period: {EVAL_START_TEXT} to {EVAL_END_TEXT}.",
        "- Data source: vnstock/vnstock_data fetched normalized cache.",
        "- Daily data and daily-to-hourly resampling are excluded.",
        "- Old VN100 evidence is excluded.",
        "",
        "## Validation Snapshot",
        "",
        f"- Benchmark-usable VN30 stocks: {len(usable_stocks)}/30.",
        f"- VNINDEX benchmark-usable: {str(vnindex_row.get('benchmark_usable', '')).lower() == 'true'}.",
        "- VN30INDEX and VNXALL are optional context indices in this track; unsupported exact codes do not fail the stock+VNINDEX gate.",
        "",
        "## Artifact Directories",
        "",
        f"- Fetch reports: `{rel(FETCH_REPORT_ROOT)}`.",
        f"- Full diagnostics: `{rel(FULL_REPORT_ROOT)}`.",
        f"- Benchmark outputs: `{rel(BENCHMARK_OUTPUT_DIR)}`.",
        "",
        "## Expected DOCX Outputs If Paper Exists",
        "",
        "- `NCKH_VN30_HOURLY_VNSTOCK_2005_2026_FULL_PAPER_VI_APA.docx`",
        "- `NCKH_VN30_HOURLY_VNSTOCK_2005_2026_FULL_PAPER_VI_IEEE.docx`",
        "- `NCKH_VN30_HOURLY_VNSTOCK_2005_2026_FULL_PAPER_EN_APA.docx`",
        "- `NCKH_VN30_HOURLY_VNSTOCK_2005_2026_FULL_PAPER_EN_IEEE.docx`",
        "",
    ]
    DOCX_NOTES_PATH.write_text("\n".join(content), encoding="utf-8")


def write_missing_evidence_report(
    path: Path,
    validation_rows: list[dict[str, Any]],
    *,
    source_script: str,
    benchmark_dir: Path = BENCHMARK_OUTPUT_DIR,
) -> None:
    failed = [row for row in validation_rows if str(row.get("benchmark_usable", "")).lower() != "true"]
    stocks = [row for row in validation_rows if row.get("asset_type") == "stock"]
    usable_stocks = [row for row in stocks if str(row.get("benchmark_usable", "")).lower() == "true"]
    vnindex_row = next((row for row in validation_rows if row.get("symbol") == "VNINDEX"), {})
    content = [
        "# VN30 Hourly vnstock Full Benchmark Missing Evidence",
        "",
        "## Decision",
        "",
        "The full 2005-2026 VN30 hourly benchmark was not run because the fetched-data validation gate did not pass.",
        "",
        "## Required Gate",
        "",
        "- All 30 frozen VN30 stocks must be benchmark-usable.",
        "- VNINDEX hourly coverage must be benchmark-usable.",
        f"- Training/history: {TRAIN_START_TEXT} to {TRAIN_CUTOFF_TEXT}.",
        f"- Evaluation/comparison: {EVAL_START_TEXT} to {EVAL_END_TEXT}.",
        "- Frequency: hourly only.",
        "- No daily data, no daily-to-hourly resampling, no old VN100 evidence, and no fabricated data.",
        "",
        "## Current Validation Snapshot",
        "",
        f"- Benchmark-usable VN30 stocks: {len(usable_stocks)}/30.",
        f"- VNINDEX benchmark-usable: {str(vnindex_row.get('benchmark_usable', '')).lower() == 'true'}.",
        f"- Benchmark output directory reserved: `{rel(benchmark_dir)}`.",
        f"- Source script: `{source_script}`.",
        "",
        "## Failed or Missing Rows",
        "",
        markdown_table(
            [
                "symbol",
                "asset_type",
                "required_start",
                "required_end",
                "first_datetime",
                "last_datetime",
                "row_count",
                "benchmark_usable",
                "failure_reason",
            ],
            failed,
            max_rows=80,
        )
        if failed
        else "No validation rows were available.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def read_validation_rows(path: Path | None = None) -> list[dict[str, Any]]:
    csv_path = path or (FETCH_REPORT_ROOT / "validation" / "vn30_fetched_hourly_validation.csv")
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validation_gate_passed(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    stock_ok = {
        row.get("symbol", ""): str(row.get("benchmark_usable", "")).lower() == "true"
        for row in rows
        if row.get("asset_type") == "stock"
    }
    vnindex = next((row for row in rows if row.get("symbol") == "VNINDEX"), {})
    return len(stock_ok) == 30 and all(stock_ok.get(ticker, False) for ticker in VN30_TICKERS) and str(
        vnindex.get("benchmark_usable", "")
    ).lower() == "true"


def write_small_json(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
