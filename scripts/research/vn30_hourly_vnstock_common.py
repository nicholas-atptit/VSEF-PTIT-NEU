"""Shared helpers for the VN30 hourly vnstock full-history research track."""

from __future__ import annotations

import csv
import importlib
import importlib.metadata
import importlib.util
import json
import time
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
from src.data.providers.vn_price_gateway import ProviderFetchError, fetch_price_history  # noqa: E402
from src.data.providers.vn_provider_contract import AssetType, FetchRequest, Frequency, SourceName  # noqa: E402


LOCAL_EXCHANGE_TZ = "Asia/Ho_Chi_Minh"
SOURCE_PRIORITY = ("KBS", "VCI")
PROBE_WINDOWS = (
    ("2024-01-02", "2024-01-05"),
    ("2025-01-02", "2025-01-05"),
    ("2026-05-04", "2026-05-11"),
)
PROBE_SYMBOLS = ("ACB", "HPG", "VNINDEX", "VN30")

FETCH_REPORT_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_hourly_vnstock_fetch"
FULL_REPORT_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_hourly_vnstock_full"
RAW_FETCH_DIR = REPO_ROOT / "data" / "raw" / "vnstock_fetch" / "vn30_hourly_2005_2026"
NORMALIZED_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly"
BENCHMARK_OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_vnstock_full_2005_2026_traincutoff"
PAPER_PATH = REPO_ROOT / "reports" / "NCKH_FULL_PAPER_DRAFT_VN30_HOURLY_VNSTOCK_2005_2026_V1_WITH_FIGURES.md"
DOCX_NOTES_PATH = REPO_ROOT / "reports" / "NCKH_VN30_HOURLY_VNSTOCK_2005_2026_DOCX_BUILD_NOTES.md"
MISSING_EVIDENCE_PATH = FETCH_REPORT_ROOT / "vn30_full_benchmark_missing_evidence.md"

VNINDEX_REQUIREMENT_START = pd.Timestamp("2005-01-01 00:00:00")
VN30_REQUIREMENT_START = pd.Timestamp("2012-02-06 00:00:00")
INDEX_REQUIREMENTS = {
    "VNINDEX": VNINDEX_REQUIREMENT_START,
    "VN30": VN30_REQUIREMENT_START,
    "HNX30": TRAIN_START,
    "VN100": TRAIN_START,
}
REQUIRED_INDEX_CODES = ("VNINDEX",)
OPTIONAL_INDEX_CODES = ("VN30", "HNX30", "VN100")
ALL_INDEX_CODES = (*REQUIRED_INDEX_CODES, *OPTIONAL_INDEX_CODES)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
NORMALIZED_COLUMNS = ["datetime", "ticker", *OHLCV_COLUMNS, "provider", "source", "frequency"]


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


def attempt_provider_fetch(symbol: str, start_date: str, end_date: str) -> list[AttemptResult]:
    """Fetch through the canonical gateway and adapt metadata to legacy logs."""
    request = FetchRequest(
        symbol=symbol.upper().strip(),
        asset_type=AssetType.INDEX if asset_type(symbol) == "index" else AssetType.STOCK,
        start=start_date,
        end=end_date,
        frequency=Frequency.HOURLY,
        preferred_sources=tuple(SourceName(source) for source in SOURCE_PRIORITY),
        allow_legacy_fallback=True,
        allow_daily=False,
        allow_resample=False,
    )
    try:
        response = fetch_price_history(request)
        frame = response.data.copy()
        if response.asset_type == AssetType.INDEX and "index_code" in frame.columns:
            frame = frame.rename(columns={"index_code": "ticker"})
        frame = frame.reindex(columns=NORMALIZED_COLUMNS)
        return [
            AttemptResult(
                symbol=symbol.upper().strip(),
                asset_type=asset_type(symbol),
                start_date=start_date,
                end_date=end_date,
                package=str(response.provider),
                package_version=package_version("vnstock_data") if "vnstock" in str(response.provider) else "",
                provider=str(response.provider),
                source=str(response.source),
                function_used="src.data.providers.vn_price_gateway.fetch_price_history",
                returned_rows=response.rows,
                standardized_rows=response.rows,
                returned_columns=",".join(str(column) for column in frame.columns),
                first_timestamp=timestamp_text(response.first_datetime) if response.first_datetime is not None else "",
                last_timestamp=timestamp_text(response.last_datetime) if response.last_datetime is not None else "",
                success=response.rows > 0,
                exception_type="",
                exception_message="",
                frame=frame,
            )
        ]
    except ProviderFetchError as exc:
        rows = []
        for attempt in exc.attempts:
            rows.append(
                AttemptResult(
                    symbol=symbol.upper().strip(),
                    asset_type=asset_type(symbol),
                    start_date=start_date,
                    end_date=end_date,
                    package=str(attempt.get("provider", "")),
                    package_version="",
                    provider=str(attempt.get("provider", "")),
                    source=str(attempt.get("source", "")),
                    function_used="src.data.providers.vn_price_gateway.fetch_price_history",
                    returned_rows=int(attempt.get("rows", 0) or 0),
                    standardized_rows=int(attempt.get("rows", 0) or 0),
                    returned_columns="",
                    first_timestamp=str(attempt.get("first_datetime", "")),
                    last_timestamp=str(attempt.get("last_datetime", "")),
                    success=False,
                    exception_type=str(attempt.get("error_type", "ProviderFetchError") or "ProviderFetchError"),
                    exception_message=str(attempt.get("error_message", str(exc)))[:800],
                    frame=pd.DataFrame(columns=NORMALIZED_COLUMNS),
                )
            )
        return rows or [
            AttemptResult(
                symbol=symbol.upper().strip(),
                asset_type=asset_type(symbol),
                start_date=start_date,
                end_date=end_date,
                package="vn_price_gateway",
                package_version="",
                provider="vn_price_gateway",
                source="",
                function_used="src.data.providers.vn_price_gateway.fetch_price_history",
                returned_rows=0,
                standardized_rows=0,
                returned_columns="",
                first_timestamp="",
                last_timestamp="",
                success=False,
                exception_type=type(exc).__name__,
                exception_message=str(exc)[:800],
                frame=pd.DataFrame(columns=NORMALIZED_COLUMNS),
            )
        ]
    except BaseException as exc:
        return [
            AttemptResult(
                symbol=symbol.upper().strip(),
                asset_type=asset_type(symbol),
                start_date=start_date,
                end_date=end_date,
                package="vn_price_gateway",
                package_version="",
                provider="vn_price_gateway",
                source="",
                function_used="src.data.providers.vn_price_gateway.fetch_price_history",
                returned_rows=0,
                standardized_rows=0,
                returned_columns="",
                first_timestamp="",
                last_timestamp="",
                success=False,
                exception_type=type(exc).__name__,
                exception_message=str(exc)[:800],
                frame=pd.DataFrame(columns=NORMALIZED_COLUMNS),
            )
        ]


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
        _ = timeout_seconds
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
    if "frequency" not in output.columns:
        output["frequency"] = "1H"
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
        "- Optional supported context indices are VN30, HNX30, and VN100; unsupported aliases do not fail the stock+VNINDEX gate.",
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
