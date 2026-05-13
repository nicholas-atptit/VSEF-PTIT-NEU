"""Probe hourly support for known vnstock index codes.

No stock tickers, benchmark, model training, daily data, or resampling are used.
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
import traceback
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


INTENDED_EXECUTABLE = Path(r"C:\Users\luong\.venv\Scripts\python.exe")
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "index_hourly_fetch" / "provider_probe"
CSV_REPORT = REPORT_DIR / "vnstock_supported_indices_probe.csv"
MD_REPORT = REPORT_DIR / "vnstock_supported_indices_probe.md"

INDEX_CODES = ("VNINDEX", "HNXINDEX", "UPCOMINDEX", "VN30", "HNX30", "VN100")
SOURCES = ("KBS", "VCI")
INTERVAL = "1H"
WINDOWS = (
    ("2024-01-02", "2024-01-05"),
    ("2025-01-02", "2025-01-06"),
    ("2026-05-04", date.today().isoformat()),
)


def _using_intended_venv() -> bool:
    return Path(sys.executable).resolve() == INTENDED_EXECUTABLE


def _load_provider(package: str) -> Any:
    return importlib.import_module(package)


def _call_history(package: str, source: str, symbol: str, start: str, end: str) -> pd.DataFrame:
    module = _load_provider(package)
    quote_cls = getattr(module, "Quote", None)
    if quote_cls is None:
        raise AttributeError(f"{package}.Quote is unavailable")

    init_attempts = (
        {"symbol": symbol, "source": source},
        {"source": source, "symbol": symbol},
        {"symbol": symbol},
    )
    last_error: Exception | None = None
    quote = None
    for kwargs in init_attempts:
        try:
            quote = quote_cls(**kwargs)
            break
        except Exception as exc:
            last_error = exc
    if quote is None:
        raise RuntimeError(f"could not initialize {package}.Quote") from last_error

    history = getattr(quote, "history", None)
    if history is None:
        raise AttributeError(f"{package}.Quote.history is unavailable")

    call_attempts = (
        {"start": start, "end": end, "interval": INTERVAL},
        {"start": start, "end": end, "interval": INTERVAL, "get_all": False},
        {"start": start, "end": end, "timeframe": INTERVAL},
        {"start_date": start, "end_date": end, "interval": INTERVAL},
        {"start_date": start, "end_date": end, "timeframe": INTERVAL},
    )
    last_error = None
    for kwargs in call_attempts:
        try:
            data = history(**kwargs)
            if data is None:
                return pd.DataFrame()
            if isinstance(data, pd.DataFrame):
                return data
            return pd.DataFrame(data)
        except TypeError as exc:
            last_error = exc
            continue
    raise RuntimeError(f"no compatible {package}.Quote.history signature worked") from last_error


def _timestamp_bounds(df: pd.DataFrame) -> tuple[str, str]:
    if df.empty:
        return "", ""
    for column in ("datetime", "time", "date", "tradingDate"):
        if column in df.columns:
            series = pd.to_datetime(df[column], errors="coerce").dropna()
            if not series.empty:
                return str(series.min()), str(series.max())
    return "", ""


def _attempt(index_code: str, package: str, source: str, start: str, end: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "index_code": index_code,
        "provider_package": package,
        "source": source,
        "interval": INTERVAL,
        "sample_start": start,
        "sample_end": end,
        "success": False,
        "rows": 0,
        "columns": "",
        "first_timestamp": "",
        "last_timestamp": "",
        "exception_type": "",
        "exception_message": "",
    }
    try:
        df = _call_history(package, source, index_code, start, end)
        first_ts, last_ts = _timestamp_bounds(df)
        row.update(
            {
                "success": True,
                "rows": int(len(df)),
                "columns": ",".join(map(str, df.columns)),
                "first_timestamp": first_ts,
                "last_timestamp": last_ts,
            }
        )
    except BaseException as exc:
        row.update(
            {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
        )
    return row


def _classify(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_index: dict[str, dict[str, Any]] = {}
    for code in INDEX_CODES:
        code_rows = [row for row in rows if row["index_code"] == code]
        successful = [row for row in code_rows if row["success"] and int(row["rows"]) > 0]
        years = sorted({row["sample_start"][:4] for row in successful})
        by_index[code] = {
            "any_hourly_rows": bool(successful),
            "years_with_rows": years,
            "source_used": successful[0]["source"] if successful else "",
            "provider_package": successful[0]["provider_package"] if successful else "",
        }
    return by_index


def _write_reports(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "index_code",
        "provider_package",
        "source",
        "interval",
        "sample_start",
        "sample_end",
        "success",
        "rows",
        "columns",
        "first_timestamp",
        "last_timestamp",
        "exception_type",
        "exception_message",
    ]
    with CSV_REPORT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    classified = _classify(rows)
    lines = [
        "# vnstock Supported Indices Hourly Probe",
        "",
        f"- interpreter: `{metadata['sys_executable']}`",
        f"- intended venv used: {'yes' if metadata['using_intended_venv'] else 'no'}",
        f"- vnstock_data importable: {'yes' if metadata['vnstock_data_importable'] else 'no'}",
        f"- vnstock importable: {'yes' if metadata['vnstock_importable'] else 'no'}",
        f"- vnstock version: `{metadata['vnstock_version']}`",
        f"- interval: `{INTERVAL}`",
        "",
        "## Probe Summary",
        "",
        "| index_code | any hourly rows | rows in 2024 | rows in 2025 | rows in 2026 | provider | source |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for code in INDEX_CODES:
        item = classified[code]
        years = set(item["years_with_rows"])
        lines.append(
            f"| `{code}` | {'yes' if item['any_hourly_rows'] else 'no'} | "
            f"{'yes' if '2024' in years else 'no'} | {'yes' if '2025' in years else 'no'} | "
            f"{'yes' if '2026' in years else 'no'} | `{item['provider_package']}` | `{item['source_used']}` |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This probe is index-only.",
            "- It uses hourly requests only.",
            "- It does not treat sample support as full-history support.",
        ]
    )
    MD_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    metadata = {
        "sys_executable": sys.executable,
        "using_intended_venv": _using_intended_venv(),
        "vnstock_data_importable": importlib.util.find_spec("vnstock_data") is not None,
        "vnstock_importable": importlib.util.find_spec("vnstock") is not None,
        "vnstock_version": "",
    }
    try:
        vnstock = importlib.import_module("vnstock")
        metadata["vnstock_version"] = getattr(vnstock, "__version__", "NO __version__")
    except BaseException:
        metadata["vnstock_version"] = "IMPORT_FAILED"

    packages = [pkg for pkg in ("vnstock_data", "vnstock") if importlib.util.find_spec(pkg) is not None]
    rows: list[dict[str, Any]] = []
    for code in INDEX_CODES:
        for start, end in WINDOWS:
            success = False
            for package in packages:
                for source in SOURCES:
                    row = _attempt(code, package, source, start, end)
                    rows.append(row)
                    if row["success"] and int(row["rows"]) > 0:
                        success = True
                        break
                if success:
                    break
    _write_reports(rows, metadata)
    print(json.dumps({"metadata": metadata, "rows": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
