"""Fetch supported vnstock index hourly bars in chunks.

This script is index-only. It does not fetch stock tickers, use daily data,
resample, run benchmarks, or train models.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


INTENDED_EXECUTABLE = Path(r"C:\Users\luong\.venv\Scripts\python.exe")
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPO_ROOT / "data" / "raw" / "vnstock_fetch" / "index_hourly"
CACHE_ROOT = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "index_hourly_fetch" / "fetch"
SUMMARY_CSV = REPORT_DIR / "index_hourly_fetch_summary.csv"
SUMMARY_MD = REPORT_DIR / "index_hourly_fetch_summary.md"
FAILURES_CSV = REPORT_DIR / "index_hourly_fetch_failures.csv"
CHUNK_LOG_CSV = REPORT_DIR / "index_hourly_chunk_log.csv"

INDEX_CODES = ("VNINDEX", "HNXINDEX", "UPCOMINDEX", "VN30", "HNX30", "VN100")
SOURCES = ("KBS", "VCI")
INTERVAL = "1H"


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _date_ranges(start: date, end: date, chunk_days: int) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    current = start
    while current <= end:
        chunk_end = min(end, current + timedelta(days=chunk_days - 1))
        ranges.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return ranges


def _call_history(package: str, source: str, symbol: str, start: str, end: str) -> pd.DataFrame:
    module = importlib.import_module(package)
    quote_cls = getattr(module, "Quote", None)
    if quote_cls is None:
        raise AttributeError(f"{package}.Quote is unavailable")
    last_error: Exception | None = None
    quote = None
    for kwargs in ({"symbol": symbol, "source": source}, {"source": source, "symbol": symbol}, {"symbol": symbol}):
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
    last_error = None
    for kwargs in (
        {"start": start, "end": end, "interval": INTERVAL},
        {"start": start, "end": end, "interval": INTERVAL, "get_all": False},
        {"start": start, "end": end, "timeframe": INTERVAL},
        {"start_date": start, "end_date": end, "interval": INTERVAL},
        {"start_date": start, "end_date": end, "timeframe": INTERVAL},
    ):
        try:
            data = history(**kwargs)
            if data is None:
                return pd.DataFrame()
            if isinstance(data, pd.DataFrame):
                return data.copy()
            return pd.DataFrame(data)
        except TypeError as exc:
            last_error = exc
            continue
    raise RuntimeError(f"no compatible {package}.Quote.history signature worked") from last_error


def _normalize(raw: pd.DataFrame, code: str, package: str, source: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["datetime", "index_code", "open", "high", "low", "close", "volume", "provider", "source"])
    rename = {
        "time": "datetime",
        "date": "datetime",
        "tradingDate": "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = raw.rename(columns={key: value for key, value in rename.items() if key in raw.columns}).copy()
    if "datetime" not in df.columns:
        raise ValueError(f"no timestamp column in provider columns: {list(raw.columns)}")
    required = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}; provider columns: {list(raw.columns)}")
    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(df["datetime"], errors="coerce"),
            "index_code": code,
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df["volume"], errors="coerce").fillna(0),
            "provider": package,
            "source": source,
        }
    )
    out = out.dropna(subset=["datetime", "open", "high", "low", "close"])
    return out.sort_values("datetime")


def _fetch_one_chunk(code: str, start: date, end: date, chunk_days: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    start_s = start.isoformat()
    end_s = end.isoformat()
    packages = [pkg for pkg in ("vnstock_data", "vnstock") if importlib.util.find_spec(pkg) is not None]
    errors: list[str] = []
    for package in packages:
        for source in SOURCES:
            try:
                raw = _call_history(package, source, code, start_s, end_s)
                normalized = _normalize(raw, code, package, source)
                return normalized, {
                    "index_code": code,
                    "chunk_start": start_s,
                    "chunk_end": end_s,
                    "chunk_days": chunk_days,
                    "provider": package,
                    "source": source,
                    "success": True,
                    "rows": int(len(normalized)),
                    "exception_type": "",
                    "exception_message": "",
                    "raw_columns": ",".join(map(str, raw.columns)),
                }
            except BaseException as exc:
                errors.append(f"{package}/{source}: {type(exc).__name__}: {exc}")
    raise RuntimeError(" | ".join(errors))


def _save_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_reports(summary: list[dict[str, Any]], failures: list[dict[str, Any]], chunks: list[dict[str, Any]], stopped_reason: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _save_rows(SUMMARY_CSV, summary, ["index_code", "fetched", "rows", "first_timestamp", "last_timestamp", "provider", "source", "cache_path", "stopped_reason"])
    _save_rows(FAILURES_CSV, failures, ["index_code", "chunk_start", "chunk_end", "chunk_days", "exception_type", "exception_message"])
    _save_rows(CHUNK_LOG_CSV, chunks, ["index_code", "chunk_start", "chunk_end", "chunk_days", "provider", "source", "success", "rows", "exception_type", "exception_message", "raw_columns"])
    lines = [
        "# Index Hourly Fetch Summary",
        "",
        "- Scope: index-only.",
        "- Benchmark run: no.",
        "- Stock data fetched: no.",
        "- Daily data used: no.",
        "- Resampling used: no.",
        f"- Stopped reason: `{stopped_reason}`",
        "",
        "| index_code | fetched | rows | first timestamp | last timestamp | provider | source |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['index_code']}` | {row['fetched']} | {row['rows']} | {row['first_timestamp']} | "
            f"{row['last_timestamp']} | `{row['provider']}` | `{row['source']}` |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="auto")
    parser.add_argument("--chunk-days", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-runtime-seconds", type=int, default=240)
    args = parser.parse_args(argv)

    start_date = _parse_date(args.start)
    end_date = date.today() if args.end == "auto" else _parse_date(args.end)
    started = time.monotonic()
    stopped_reason = "completed"
    chunk_log: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []

    for code in INDEX_CODES:
        frames: list[pd.DataFrame] = []
        cache_path = CACHE_ROOT / f"{code}.csv"
        if args.resume and cache_path.exists() and not args.force:
            try:
                existing = pd.read_csv(cache_path)
                if not existing.empty:
                    frames.append(existing)
            except Exception:
                frames = []

        for chunk_start, chunk_end in _date_ranges(start_date, end_date, args.chunk_days):
            if args.max_runtime_seconds and time.monotonic() - started >= args.max_runtime_seconds:
                stopped_reason = f"max_runtime_seconds={args.max_runtime_seconds}"
                break
            chunk_success = False
            last_error: Exception | None = None
            for days in (args.chunk_days, 3, 1):
                if days > (chunk_end - chunk_start).days + 1:
                    continue
                subranges = _date_ranges(chunk_start, chunk_end, days)
                temp_frames: list[pd.DataFrame] = []
                temp_logs: list[dict[str, Any]] = []
                try:
                    for sub_start, sub_end in subranges:
                        frame, log = _fetch_one_chunk(code, sub_start, sub_end, days)
                        temp_frames.append(frame)
                        temp_logs.append(log)
                        raw_dir = RAW_ROOT / code
                        raw_dir.mkdir(parents=True, exist_ok=True)
                        frame.to_csv(raw_dir / f"{code}_{sub_start.isoformat()}_{sub_end.isoformat()}_{log['provider']}_{log['source']}.csv", index=False)
                    frames.extend(temp_frames)
                    chunk_log.extend(temp_logs)
                    chunk_success = True
                    break
                except Exception as exc:
                    last_error = exc
                    continue
            if not chunk_success:
                failure = {
                    "index_code": code,
                    "chunk_start": chunk_start.isoformat(),
                    "chunk_end": chunk_end.isoformat(),
                    "chunk_days": args.chunk_days,
                    "exception_type": type(last_error).__name__ if last_error else "UnknownError",
                    "exception_message": str(last_error) if last_error else "unknown failure",
                }
                failures.append(failure)
                chunk_log.append({**failure, "provider": "", "source": "", "success": False, "rows": 0, "raw_columns": ""})
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined["datetime"] = pd.to_datetime(combined["datetime"], errors="coerce")
            combined = combined.dropna(subset=["datetime"]).sort_values("datetime")
            duplicate_count = int(combined.duplicated(subset=["datetime", "index_code"], keep="first").sum())
            if duplicate_count:
                chunk_log.append(
                    {
                        "index_code": code,
                        "chunk_start": "",
                        "chunk_end": "",
                        "chunk_days": args.chunk_days,
                        "provider": "dedupe",
                        "source": "",
                        "success": True,
                        "rows": -duplicate_count,
                        "exception_type": "",
                        "exception_message": f"dropped {duplicate_count} exact duplicate datetime/index_code rows",
                        "raw_columns": "",
                    }
                )
            combined = combined.drop_duplicates(subset=["datetime", "index_code"], keep="first")
            CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            combined.to_csv(cache_path, index=False)
            provider = combined["provider"].mode().iloc[0] if "provider" in combined.columns and not combined.empty else ""
            source = combined["source"].mode().iloc[0] if "source" in combined.columns and not combined.empty else ""
            summary.append(
                {
                    "index_code": code,
                    "fetched": "yes",
                    "rows": int(len(combined)),
                    "first_timestamp": str(combined["datetime"].min()) if not combined.empty else "",
                    "last_timestamp": str(combined["datetime"].max()) if not combined.empty else "",
                    "provider": provider,
                    "source": source,
                    "cache_path": str(cache_path.relative_to(REPO_ROOT)),
                    "stopped_reason": stopped_reason,
                }
            )
        else:
            summary.append(
                {
                    "index_code": code,
                    "fetched": "no",
                    "rows": 0,
                    "first_timestamp": "",
                    "last_timestamp": "",
                    "provider": "",
                    "source": "",
                    "cache_path": str(cache_path.relative_to(REPO_ROOT)),
                    "stopped_reason": stopped_reason,
                }
            )

        if stopped_reason != "completed":
            break

    _write_reports(summary, failures, chunk_log, stopped_reason)
    print(json.dumps({"summary": summary, "failures": failures, "stopped_reason": stopped_reason}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
