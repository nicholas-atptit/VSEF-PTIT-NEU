"""Shared helpers for the VN30 hourly listing-aware vnstock research track."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.research.vn30_hourly_common import (
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
from scripts.research.vn30_hourly_vnstock_common import LOCAL_EXCHANGE_TZ


LISTING_DATES_PATH = REPO_ROOT / "configs" / "universes" / "vn30_listing_dates.csv"
REPORT_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_hourly_listing_aware"
RAW_FETCH_DIR = REPO_ROOT / "data" / "raw" / "vnstock_fetch" / "vn30_hourly_listing_aware"
NORMALIZED_CACHE_DIR = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_listing_aware"
BENCHMARK_OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_listing_aware_traincutoff"
PAPER_PATH = REPO_ROOT / "reports" / "NCKH_FULL_PAPER_DRAFT_VN30_HOURLY_LISTING_AWARE_V1_WITH_FIGURES.md"
DOCX_NOTES_PATH = REPO_ROOT / "reports" / "NCKH_VN30_HOURLY_LISTING_AWARE_DOCX_BUILD_NOTES.md"
MISSING_EVIDENCE_PATH = REPORT_ROOT / "vn30_listing_aware_benchmark_missing_evidence.md"

REQUESTED_EVAL_END = pd.Timestamp("2026-05-31 23:59:59")
REQUESTED_EVAL_END_TEXT = "2026-05-31 23:59:59"
MIN_TRAINING_ROWS_PER_TICKER = 1000
MIN_EVALUATION_ROWS_PER_TICKER = 100
TARGET_MODE = "classification"
FREQUENCY = "hourly"
DEFAULT_MODELS = ["xgboost", "lightgbm", "random_forest", "stacking"]
DEFAULT_HORIZONS = [1, 4, 8, 20]

VNINDEX_START = pd.Timestamp("2005-01-01 00:00:00")
VN30_START = pd.Timestamp("2012-02-06 00:00:00")
INDEX_STARTS = {"VNINDEX": VNINDEX_START, "VN30": VN30_START, "HNX30": TRAIN_START, "VN100": TRAIN_START}
OPTIONAL_INDEX_CODES = ("VN30", "HNX30", "VN100")
ALL_INDEX_CODES = ("VNINDEX", *OPTIONAL_INDEX_CODES)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
NORMALIZED_COLUMNS = [
    "datetime",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "provider",
    "source",
    "frequency",
    "requested_start",
    "actual_first_datetime",
    "actual_last_datetime",
]


def current_fetch_end() -> pd.Timestamp:
    now = pd.Timestamp.now(tz=LOCAL_EXCHANGE_TZ).tz_localize(None)
    return min(REQUESTED_EVAL_END, now).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)


def read_listing_dates(path: Path = LISTING_DATES_PATH) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"VN30 listing-date metadata missing: {rel(path)}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"ticker", "company_name", "first_trading_date", "exchange_first_traded", "current_exchange", "source_note", "note"}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Listing-date metadata missing columns: {sorted(missing)}")
    by_ticker = {str(row.get("ticker", "")).strip().upper(): row for row in rows}
    universe = read_universe()
    if sorted(by_ticker) != sorted(universe):
        raise ValueError("Listing-date metadata must contain exactly the frozen VN30 tickers.")
    return by_ticker


def listing_date_for(ticker: str, listing_rows: dict[str, dict[str, str]] | None = None) -> pd.Timestamp | None:
    listing_rows = listing_rows or read_listing_dates()
    value = str(listing_rows.get(ticker.upper(), {}).get("first_trading_date", "")).strip()
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def requested_start_for(symbol: str, listing_rows: dict[str, dict[str, str]] | None = None) -> pd.Timestamp:
    code = symbol.upper().strip()
    if code in INDEX_STARTS:
        return INDEX_STARTS[code]
    listing_date = listing_date_for(code, listing_rows)
    return listing_date if listing_date is not None else TRAIN_START.normalize()


def symbol_type(symbol: str) -> str:
    return "index" if symbol.upper().strip() in ALL_INDEX_CODES else "stock"


def normalized_cache_path(symbol: str) -> Path:
    return NORMALIZED_CACHE_DIR / f"{symbol.upper().strip()}.csv"


def raw_chunk_path(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    code = symbol.upper().strip()
    start_text = pd.Timestamp(start).strftime("%Y%m%d")
    end_text = pd.Timestamp(end).strftime("%Y%m%d")
    return RAW_FETCH_DIR / code / f"{code}_{start_text}_{end_text}.csv"


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
    return frame[NORMALIZED_COLUMNS].sort_values(["ticker", "datetime"]).drop_duplicates(["ticker", "datetime"], keep="last").reset_index(drop=True)


def write_normalized_symbol(symbol: str, frame: pd.DataFrame, requested_start: pd.Timestamp) -> None:
    path = normalized_cache_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output["datetime"] = pd.to_datetime(output["datetime"], errors="coerce")
    output = output.dropna(subset=["datetime"])
    if output.empty:
        return
    actual_first = timestamp_text(output["datetime"].min())
    actual_last = timestamp_text(output["datetime"].max())
    output["requested_start"] = timestamp_text(requested_start)
    output["actual_first_datetime"] = actual_first
    output["actual_last_datetime"] = actual_last
    if "frequency" not in output.columns:
        output["frequency"] = "1H"
    output["datetime"] = output["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    output[NORMALIZED_COLUMNS].to_csv(path, index=False)


def write_listing_raw_chunk(path: Path, frame: pd.DataFrame, requested_start: pd.Timestamp) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output["datetime"] = pd.to_datetime(output["datetime"], errors="coerce")
    output = output.dropna(subset=["datetime"])
    if output.empty:
        return
    output["requested_start"] = timestamp_text(requested_start)
    output["actual_first_datetime"] = timestamp_text(output["datetime"].min())
    output["actual_last_datetime"] = timestamp_text(output["datetime"].max())
    if "frequency" not in output.columns:
        output["frequency"] = "1H"
    output["datetime"] = output["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    output[NORMALIZED_COLUMNS].to_csv(path, index=False)


def load_listing_aware_universe_frame(tickers: list[str] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ticker in tickers or VN30_TICKERS:
        frame = read_normalized_symbol(ticker)
        if frame.empty:
            continue
        frames.append(frame[["datetime", "ticker", "open", "high", "low", "close", "volume"]])
    if not frames:
        return pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])
    combined = pd.concat(frames, ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"], errors="coerce")
    return combined.dropna(subset=["datetime"]).sort_values(["ticker", "datetime"]).drop_duplicates(["ticker", "datetime"], keep="last").reset_index(drop=True)


def compute_actual_eval_end(validation_rows: list[dict[str, Any]]) -> str:
    stock_rows = [row for row in validation_rows if row.get("asset_type") == "stock"]
    timestamps = [
        pd.to_datetime(row.get("last_datetime", ""), errors="coerce")
        for row in stock_rows
        if str(row.get("benchmark_usable", "")).lower() == "true"
    ]
    timestamps = [pd.Timestamp(item) for item in timestamps if not pd.isna(item)]
    if len(timestamps) != 30:
        return ""
    return timestamp_text(min(min(timestamps), REQUESTED_EVAL_END))


def validation_gate_passed(rows: list[dict[str, Any]]) -> bool:
    stock_rows = [row for row in rows if row.get("asset_type") == "stock"]
    if len(stock_rows) != 30:
        return False
    if not all(str(row.get("benchmark_usable", "")).lower() == "true" for row in stock_rows):
        return False
    actual_eval_end = compute_actual_eval_end(rows)
    if not actual_eval_end:
        return False
    vnindex = next((row for row in rows if row.get("symbol") == "VNINDEX"), {})
    if str(vnindex.get("benchmark_usable", "")).lower() != "true":
        return False
    return True


def read_validation_rows(path: Path | None = None) -> list[dict[str, Any]]:
    csv_path = path or (REPORT_ROOT / "validation" / "vn30_listing_aware_validation.csv")
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_missing_evidence_report(path: Path, validation_rows: list[dict[str, Any]], *, source_script: str) -> None:
    failed = [row for row in validation_rows if str(row.get("benchmark_usable", "")).lower() != "true"]
    stock_rows = [row for row in validation_rows if row.get("asset_type") == "stock"]
    usable = [row for row in stock_rows if str(row.get("benchmark_usable", "")).lower() == "true"]
    actual_eval_end = compute_actual_eval_end(validation_rows)
    content = [
        "# VN30 Hourly Listing-Aware Benchmark Missing Evidence",
        "",
        "## Decision",
        "",
        "The VN30 hourly listing-aware benchmark was not run because the validation gate did not pass.",
        "",
        "## Gate",
        "",
        "- All 30 frozen VN30 stocks must be usable under the listing-aware rule.",
        f"- Minimum training rows per ticker: {MIN_TRAINING_ROWS_PER_TICKER}.",
        f"- Minimum evaluation rows per ticker: {MIN_EVALUATION_ROWS_PER_TICKER}.",
        f"- Train cutoff: {TRAIN_CUTOFF_TEXT}.",
        f"- Evaluation start: {EVAL_START_TEXT}.",
        f"- Requested evaluation end: {REQUESTED_EVAL_END_TEXT}.",
        "- Per-ticker start rule: max(first_trading_date, first provider-available hourly timestamp).",
        "- No daily data, daily-to-hourly resampling, VN100 evidence reuse, or fabricated bars.",
        "",
        "## Current Status",
        "",
        f"- Usable VN30 stocks: {len(usable)}/30.",
        f"- actual_eval_end: {actual_eval_end or 'not available'}.",
        f"- Source script: `{source_script}`.",
        "",
        "## Failed or Missing Rows",
        "",
        markdown_table(
            [
                "symbol",
                "asset_type",
                "listing_date_used",
                "requested_start",
                "first_datetime",
                "last_datetime",
                "training_rows",
                "evaluation_rows",
                "benchmark_usable",
                "missing_reason",
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


def write_docx_notes(*, paper_exists: bool, validation_rows: list[dict[str, Any]]) -> None:
    stock_rows = [row for row in validation_rows if row.get("asset_type") == "stock"]
    usable = [row for row in stock_rows if str(row.get("benchmark_usable", "")).lower() == "true"]
    actual_eval_end = compute_actual_eval_end(validation_rows)
    content = [
        "# VN30 Hourly Listing-Aware DOCX Build Notes",
        "",
        "## Source Markdown",
        "",
        f"- `{rel(PAPER_PATH)}`" if paper_exists else "- Final paper was not written because the benchmark gate did not pass.",
        "",
        "## Design",
        "",
        "- Study: VN30 hourly listing-aware historical benchmark.",
        "- Universe: all 30 frozen VN30 tickers.",
        "- Per-ticker start rule: max(first trading/listing date, first provider-available hourly timestamp).",
        f"- Training labels end: {TRAIN_CUTOFF_TEXT}.",
        f"- Evaluation starts: {EVAL_START_TEXT}.",
        f"- actual_eval_end: {actual_eval_end or 'not available'}.",
        "- VNINDEX is market context if fetched and validated; VN30, HNX30, and VN100 are optional supported-index context.",
        "- Old VN100 evidence, daily data, daily-to-hourly resampling, and fabricated bars are excluded.",
        "",
        "## Validation Snapshot",
        "",
        f"- Usable VN30 stocks: {len(usable)}/30.",
        "",
        "## Artifact Directories",
        "",
        f"- Reports: `{rel(REPORT_ROOT)}`.",
        f"- Benchmark outputs: `{rel(BENCHMARK_OUTPUT_DIR)}`.",
        "",
        "## Expected DOCX Outputs If Paper Exists",
        "",
        "- `NCKH_VN30_HOURLY_LISTING_AWARE_FULL_PAPER_VI_APA.docx`",
        "- `NCKH_VN30_HOURLY_LISTING_AWARE_FULL_PAPER_VI_IEEE.docx`",
        "- `NCKH_VN30_HOURLY_LISTING_AWARE_FULL_PAPER_EN_APA.docx`",
        "- `NCKH_VN30_HOURLY_LISTING_AWARE_FULL_PAPER_EN_IEEE.docx`",
        "",
    ]
    DOCX_NOTES_PATH.write_text("\n".join(content), encoding="utf-8")


def write_manifest_json(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
