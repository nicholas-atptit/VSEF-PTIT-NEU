"""Inventory local hourly-source candidates for the joint stock + index panel.

This script is read-only. It scans local data, output, generated-report, and
archive roots for files that mention the joint-panel instruments, then classifies
whether each file can be used as a training input.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_hourly_common import VN30_TICKERS


STOCKS = tuple(VN30_TICKERS)
INDICES = ("VNINDEX", "VN30", "HNXINDEX", "HNX30", "UPCOMINDEX", "VN100")
ALL_CODES = tuple(dict.fromkeys([*STOCKS, *INDICES]))
SCAN_ROOTS = (
    REPO_ROOT / "data",
    REPO_ROOT / "outputs",
    REPO_ROOT / "reports" / "generated",
    REPO_ROOT / "archive" / "generated_data_snapshots",
    REPO_ROOT / "archive" / "reports_superseded",
)
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_stock_index_joint_panel_data_recovery"
TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt", ".log"}
TABLE_SUFFIXES = {".csv", ".parquet", ".feather"}
OHLCV = {"open", "high", "low", "close", "volume"}
PREDICTION_COLUMNS = {
    "prediction",
    "predicted_direction",
    "y_pred",
    "is_correct",
    "actual_direction",
    "target_direction",
    "final_predicted_direction",
}


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[dict[str, Any]], max_rows: int = 60) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows[:max_rows]:
        lines.append("| " + " | ".join(str(row.get(header, "")).replace("|", "\\|") for header in headers) + " |")
    if len(rows) > max_rows:
        lines.append("| " + " | ".join(["..."] + [""] * (len(headers) - 1)) + " |")
    return "\n".join(lines)


def classify_path(path: Path) -> str:
    parts = [part.lower() for part in path.relative_to(REPO_ROOT).parts]
    text = rel(path).lower()
    if parts[:1] == ["data"]:
        if "raw" in parts:
            return "raw_fetch"
        return "active_cache"
    if parts[:1] == ["outputs"]:
        return "output_prediction"
    if parts[:2] == ["reports", "generated"]:
        return "generated_report"
    if parts[:2] == ["archive", "generated_data_snapshots"]:
        return "archive_snapshot"
    if "archive/reports_superseded" in text:
        return "generated_report"
    return "unknown"


def candidate_codes_from_path(path: Path) -> set[str]:
    tokens = set()
    stem_parts = path.stem.upper().replace("-", "_").split("_")
    name = path.name.upper()
    for code in ALL_CODES:
        if path.stem.upper() == code or code in stem_parts or code in name:
            tokens.add(code)
    return tokens


def read_head_text(path: Path, max_bytes: int = 16384) -> str:
    try:
        with path.open("rb") as handle:
            data = handle.read(max_bytes)
        return data.decode("utf-8", errors="ignore").upper()
    except OSError:
        return ""


def file_mentions_codes(path: Path) -> set[str]:
    codes = candidate_codes_from_path(path)
    if codes or path.suffix.lower() not in TEXT_SUFFIXES:
        return codes
    head = read_head_text(path)
    for code in ALL_CODES:
        if code in head:
            codes.add(code)
    return codes


def iter_candidate_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES.union(TABLE_SUFFIXES):
                continue
            text = rel(path).lower()
            if "joint_panel_data_recovery" in text:
                continue
            if file_mentions_codes(path):
                files.append(path)
    return sorted(set(files), key=lambda item: rel(item))


def detect_time_column(columns: list[str]) -> str | None:
    normalized = {column.lower(): column for column in columns}
    for candidate in ("datetime", "time", "timestamp", "date", "trading_date", "target_timestamp"):
        if candidate in normalized:
            return normalized[candidate]
    return None


def detect_code_columns(columns: list[str]) -> list[str]:
    normalized = {column.lower(): column for column in columns}
    found = []
    for candidate in ("instrument_code", "ticker", "symbol", "index_code", "code"):
        if candidate in normalized:
            found.append(normalized[candidate])
    return found


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, low_memory=False)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".feather":
            return pd.read_feather(path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def infer_frequency(datetimes: pd.Series, has_predictions: bool) -> tuple[str, int, str, str, str]:
    if has_predictions:
        return "prediction_output", 0, "", "", ""
    parsed = pd.to_datetime(datetimes, errors="coerce").dropna()
    if parsed.empty:
        return "unknown", 0, "", "", ""
    times = sorted(parsed.dt.strftime("%H:%M:%S").unique().tolist())
    sample = ",".join(times[:10])
    first = str(parsed.min())
    last = str(parsed.max())
    unique_count = len(times)
    if unique_count == 1 and times == ["00:00:00"]:
        return "midnight_only_daily_like", unique_count, sample, first, last
    if unique_count == 1:
        return "daily", unique_count, sample, first, last
    if any(time not in {"00:00:00", "07:00:00"} for time in times):
        return "intraday_hourly", unique_count, sample, first, last
    return "daily", unique_count, sample, first, last


def instrument_codes_for_frame(path: Path, frame: pd.DataFrame) -> set[str]:
    codes = candidate_codes_from_path(path)
    if frame.empty:
        return codes or {"UNKNOWN"}
    for column in detect_code_columns([str(col) for col in frame.columns]):
        values = frame[column].dropna().astype(str).str.upper().str.strip().unique().tolist()
        for value in values:
            if value in ALL_CODES:
                codes.add(value)
    return codes or {"UNKNOWN"}


def analyze_text_file(path: Path, code: str) -> dict[str, Any]:
    text = read_head_text(path)
    has_predictions = any(marker.upper() in text for marker in PREDICTION_COLUMNS)
    return {
        "path": rel(path),
        "path_type": classify_path(path),
        "instrument_code": code,
        "instrument_type": "stock" if code in STOCKS else "index" if code in INDICES else "unknown",
        "frequency_detected": "prediction_output" if has_predictions else "unknown",
        "row_count": "",
        "first_timestamp": "",
        "last_timestamp": "",
        "unique_time_of_day_count": "",
        "sample_time_of_day_values": "",
        "has_ohlcv": "no",
        "has_predictions": "yes" if has_predictions else "no",
        "usable_as_training_input": "no",
        "reason_if_not_usable": "text_report_or_manifest_not_training_input",
    }


def analyze_table_file(path: Path) -> list[dict[str, Any]]:
    frame = read_table(path)
    if frame.empty:
        return [analyze_text_file(path, code) for code in sorted(file_mentions_codes(path) or {"UNKNOWN"})]
    columns = [str(col) for col in frame.columns]
    lower = {column.lower() for column in columns}
    has_ohlcv = OHLCV.issubset(lower)
    has_predictions = bool(PREDICTION_COLUMNS.intersection(lower))
    time_column = detect_time_column(columns)
    if time_column:
        frequency, unique_times, sample_times, first_ts, last_ts = infer_frequency(frame[time_column], has_predictions)
    else:
        frequency, unique_times, sample_times, first_ts, last_ts = ("unknown", 0, "", "", "")
    rows: list[dict[str, Any]] = []
    for code in sorted(instrument_codes_for_frame(path, frame)):
        instrument_type = "stock" if code in STOCKS else "index" if code in INDICES else "unknown"
        reason: list[str] = []
        if has_predictions:
            reason.append("prediction_output_not_training_input")
        if not has_ohlcv:
            reason.append("missing_ohlcv_columns")
        if frequency != "intraday_hourly":
            reason.append(f"frequency_detected={frequency}")
        if code == "UNKNOWN":
            reason.append("instrument_code_not_identified")
        usable = not reason
        rows.append(
            {
                "path": rel(path),
                "path_type": classify_path(path),
                "instrument_code": code,
                "instrument_type": instrument_type,
                "frequency_detected": frequency,
                "row_count": int(len(frame)),
                "first_timestamp": first_ts,
                "last_timestamp": last_ts,
                "unique_time_of_day_count": unique_times,
                "sample_time_of_day_values": sample_times,
                "has_ohlcv": "yes" if has_ohlcv else "no",
                "has_predictions": "yes" if has_predictions else "no",
                "usable_as_training_input": "yes" if usable else "no",
                "reason_if_not_usable": "usable" if usable else "; ".join(reason),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> str:
    usable = [row for row in rows if row.get("usable_as_training_input") == "yes"]
    stock_usable = sorted({row["instrument_code"] for row in usable if row.get("instrument_type") == "stock"})
    index_usable = sorted({row["instrument_code"] for row in usable if row.get("instrument_type") == "index"})
    midnight_indices = sorted(
        {
            row["instrument_code"]
            for row in rows
            if row.get("instrument_type") == "index" and row.get("frequency_detected") == "midnight_only_daily_like"
        }
    )
    top_rows = sorted(
        rows,
        key=lambda row: (
            row.get("usable_as_training_input") != "yes",
            row.get("instrument_type") != "stock",
            row.get("instrument_code", ""),
            row.get("path", ""),
        ),
    )
    content = [
        "# Joint Panel Hourly Data Source Inventory",
        "",
        "- Scan mode: read-only.",
        f"- Candidate file rows: {len(rows)}.",
        f"- Usable stock instruments found: {len(stock_usable)}/30.",
        f"- Usable index instruments found: {len(index_usable)}/6.",
        f"- True intraday hourly index data found for 6/6: {str(len(index_usable) == 6).lower()}.",
        f"- Indices with midnight-only daily-like candidates: {', '.join(midnight_indices) if midnight_indices else 'none'}.",
        "- Benchmark/training run: no.",
        "- Data fetch: no.",
        "",
        "## Usable Candidate Sources",
        "",
        markdown_table(
            ["instrument_code", "instrument_type", "path_type", "frequency_detected", "row_count", "first_timestamp", "last_timestamp", "path"],
            [row for row in top_rows if row.get("usable_as_training_input") == "yes"],
            max_rows=80,
        ),
        "",
        "## Non-Usable Candidate Sources",
        "",
        markdown_table(
            ["instrument_code", "instrument_type", "path_type", "frequency_detected", "row_count", "reason_if_not_usable", "path"],
            [row for row in top_rows if row.get("usable_as_training_input") != "yes"],
            max_rows=80,
        ),
        "",
    ]
    return "\n".join(content)


def main() -> int:
    rows: list[dict[str, Any]] = []
    for path in iter_candidate_files():
        if path.suffix.lower() in TABLE_SUFFIXES:
            rows.extend(analyze_table_file(path))
        else:
            for code in sorted(file_mentions_codes(path) or {"UNKNOWN"}):
                rows.append(analyze_text_file(path, code))
    write_csv(REPORT_DIR / "hourly_data_source_inventory.csv", rows)
    (REPORT_DIR / "hourly_data_source_inventory.md").write_text(summarize(rows), encoding="utf-8")
    print(f"candidate_rows={len(rows)} output={rel(REPORT_DIR / 'hourly_data_source_inventory.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
