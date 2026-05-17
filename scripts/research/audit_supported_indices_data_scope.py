"""Audit local supported-index data scope for daily/hourly benchmark tracks."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.index_benchmark_common import (
    ARCHIVE_ROOT,
    DAILY_CACHE,
    HOURLY_CACHE,
    INDEX_CODES,
    OUTPUT_DIR,
    RAW_HOURLY,
    REPORT_DIR,
    fmt_pct,
    markdown_table,
    read_index_frame,
    rel,
    timestamp_text,
    validate_ohlcv,
    write_csv,
    year_coverage,
)

CSV_PATH = REPORT_DIR / "index_data_scope_audit.csv"
MD_PATH = REPORT_DIR / "index_data_scope_audit.md"


def candidate_paths(code: str, frequency: str) -> list[tuple[Path, str]]:
    if frequency == "1D":
        return [
            (DAILY_CACHE / f"{code}.csv", "market_cache_daily_2015"),
            (REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "daily" / f"{code}.csv", "market_cache_daily"),
        ]
    paths = [
        (HOURLY_CACHE / f"{code}.csv", "market_cache_hourly_2015"),
        (REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly" / f"{code}.csv", "market_cache_hourly"),
    ]
    paths.extend((path, "raw_index_hourly") for path in RAW_HOURLY.rglob(f"*{code}*.csv") if RAW_HOURLY.exists())
    return paths


def scan_archives_and_outputs(code: str, frequency: str) -> list[tuple[Path, str]]:
    needles = [code.lower()]
    freq_terms = ["daily", "1d"] if frequency == "1D" else ["hourly", "1h"]
    roots = [ARCHIVE_ROOT, REPO_ROOT / "outputs"]
    found: list[tuple[Path, str]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            text = path.as_posix().lower()
            if any(needle in text for needle in needles) and any(term in text for term in freq_terms):
                found.append((path, "archive_or_output"))
                if len(found) >= 20:
                    return found
    return found


def summarize_path(code: str, frequency: str, path: Path, provider: str) -> dict[str, Any]:
    expected_frequency = frequency
    row = {
        "index_code": code,
        "frequency": frequency,
        "path": rel(path),
        "file_exists": "yes" if path.exists() else "no",
        "row_count": 0,
        "first_timestamp": "",
        "last_timestamp": "",
        "year_coverage": "",
        "provider_source": provider,
        "usable_for_train_validation_final": "no",
        "reason_unusable": "file_missing",
    }
    if not path.exists():
        return row
    try:
        frame = read_index_frame(path, code, expected_frequency)
        valid, reasons = validate_ohlcv(frame, expected_frequency)
        splits_ok = False
        if valid:
            years = set(frame["datetime"].dt.year.astype(int).tolist())
            if frequency == "1D":
                splits_ok = bool(2015 in years and 2024 in years and any(year >= 2025 for year in years))
            else:
                splits_ok = len(frame) >= 640
        source_values = []
        for col in ("provider", "source"):
            if col in frame.columns and not frame.empty:
                source_values.extend(str(v) for v in frame[col].dropna().astype(str).unique()[:3])
        row.update(
            {
                "row_count": int(len(frame)),
                "first_timestamp": timestamp_text(frame, "min"),
                "last_timestamp": timestamp_text(frame, "max"),
                "year_coverage": year_coverage(frame),
                "provider_source": ";".join(source_values) or provider,
                "usable_for_train_validation_final": "yes" if valid and splits_ok else "no",
                "reason_unusable": "" if valid and splits_ok else "; ".join(reasons or ["insufficient_split_coverage"]),
            }
        )
    except Exception as exc:
        row["reason_unusable"] = f"read_failed:{type(exc).__name__}:{exc}"
    return row


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for code in INDEX_CODES:
        for frequency in ("1D", "1H"):
            paths = candidate_paths(code, frequency)
            existing_primary = [item for item in paths if item[0].exists()]
            archive_paths = scan_archives_and_outputs(code, frequency)
            for path, provider in paths + archive_paths:
                rows.append(summarize_path(code, frequency, path, provider))
            if not existing_primary and not archive_paths:
                rows.append(summarize_path(code, frequency, paths[0][0], paths[0][1]))
    return rows


def best_rows(rows: list[dict[str, Any]], frequency: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for code in INDEX_CODES:
        code_rows = [row for row in rows if row["index_code"] == code and row["frequency"] == frequency and row["file_exists"] == "yes"]
        usable = [row for row in code_rows if row["usable_for_train_validation_final"] == "yes"]
        source = usable or code_rows
        if source:
            out.append(max(source, key=lambda row: int(row["row_count"] or 0)))
    return out


def write_report(rows: list[dict[str, Any]]) -> None:
    fields = [
        "index_code",
        "frequency",
        "path",
        "file_exists",
        "row_count",
        "first_timestamp",
        "last_timestamp",
        "year_coverage",
        "provider_source",
        "usable_for_train_validation_final",
        "reason_unusable",
    ]
    write_csv(CSV_PATH, rows, fields)
    daily_best = best_rows(rows, "1D")
    hourly_best = best_rows(rows, "1H")
    all_daily_dates = [pd.to_datetime(row["first_timestamp"], errors="coerce") for row in daily_best if row["first_timestamp"]]
    all_hourly_dates = [pd.to_datetime(row["first_timestamp"], errors="coerce") for row in hourly_best if row["first_timestamp"]]
    earliest_daily = min(all_daily_dates) if all_daily_dates else pd.NaT
    earliest_hourly = min(all_hourly_dates) if all_hourly_dates else pd.NaT
    daily_2015 = any("2015" in str(row["year_coverage"]).split(",") for row in daily_best)
    hourly_2015 = any("2015" in str(row["year_coverage"]).split(",") for row in hourly_best)
    daily_can_run = any(row["usable_for_train_validation_final"] == "yes" for row in daily_best)
    hourly_can_run = any(row["usable_for_train_validation_final"] == "yes" for row in hourly_best)
    lines = [
        "# Supported Index Data Scope Audit",
        "",
        "- Scope: index-only.",
        "- Stock data used: no.",
        "- Resampling used: no.",
        f"- Earliest daily index date: `{'' if pd.isna(earliest_daily) else earliest_daily}`.",
        f"- Earliest hourly index timestamp: `{'' if pd.isna(earliest_hourly) else earliest_hourly}`.",
        f"- 2015 daily index data exists: {'yes' if daily_2015 else 'no'}.",
        f"- 2015 hourly index data exists: {'yes' if hourly_2015 else 'no'}.",
        f"- Daily benchmark can run: {'yes' if daily_can_run else 'no'}.",
        f"- Hourly benchmark can run: {'yes' if hourly_can_run else 'no'}.",
        "",
        "## Best Local Daily Files",
        "",
        markdown_table(fields, daily_best),
        "",
        "## Best Local Hourly Files",
        "",
        markdown_table(fields, hourly_best),
        "",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows = build_rows()
    write_report(rows)
    print(f"Index data scope audit written: {rel(MD_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
