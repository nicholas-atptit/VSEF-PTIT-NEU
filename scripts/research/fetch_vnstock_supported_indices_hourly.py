"""Fetch supported vnstock index hourly bars in chunks.

This script is index-only. It does not fetch stock tickers, use daily data,
resample, run benchmarks, or train models.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.providers.vn_price_gateway import ProviderFetchError, fetch_price_history  # noqa: E402
from src.data.providers.vn_provider_contract import AssetType, FetchRequest, Frequency, SourceName  # noqa: E402

RAW_ROOT = REPO_ROOT / "data" / "raw" / "vnstock_fetch" / "index_hourly"
CACHE_ROOT = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "index_hourly_fetch" / "fetch"
SUMMARY_CSV = REPORT_DIR / "index_hourly_fetch_summary.csv"
SUMMARY_MD = REPORT_DIR / "index_hourly_fetch_summary.md"
FAILURES_CSV = REPORT_DIR / "index_hourly_fetch_failures.csv"
CHUNK_LOG_CSV = REPORT_DIR / "index_hourly_chunk_log.csv"

INDEX_CODES = ("VNINDEX", "HNXINDEX", "UPCOMINDEX", "VN30", "HNX30", "VN100")
SOURCES = ("KBS", "VCI")
INTERVAL = Frequency.HOURLY.value


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


def _covered_dates(frame: pd.DataFrame) -> set[date]:
    if frame.empty or "datetime" not in frame.columns:
        return set()
    dt = pd.to_datetime(frame["datetime"], errors="coerce").dropna()
    return set(dt.dt.date)


def _fetch_one_chunk(code: str, start: date, end: date, chunk_days: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    start_s = start.isoformat()
    end_s = end.isoformat()
    request = FetchRequest(
        symbol=code,
        asset_type=AssetType.INDEX,
        start=start_s,
        end=end_s,
        frequency=Frequency.HOURLY,
        preferred_sources=tuple(SourceName(source) for source in SOURCES),
        allow_legacy_fallback=True,
        allow_daily=False,
        allow_resample=False,
    )
    try:
        response = fetch_price_history(request)
    except ProviderFetchError as exc:
        errors = [
            f"{attempt.get('provider')}/{attempt.get('source')}: {attempt.get('error_type')}: {attempt.get('error_message')}"
            for attempt in exc.attempts
        ]
        raise RuntimeError(" | ".join(errors) or str(exc)) from exc
    normalized = response.data.copy()
    return normalized, {
        "index_code": code,
        "chunk_start": start_s,
        "chunk_end": end_s,
        "chunk_days": chunk_days,
        "provider": str(response.provider),
        "source": str(response.source),
        "success": True,
        "rows": int(len(normalized)),
        "exception_type": "",
        "exception_message": "",
        "raw_columns": ",".join(map(str, normalized.columns)),
    }


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
    parser.add_argument("--index-code", choices=INDEX_CODES, help="Optional supported index code to fetch.")
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
    index_codes = (args.index_code,) if args.index_code else INDEX_CODES

    for code in index_codes:
        frames: list[pd.DataFrame] = []
        covered_dates: set[date] = set()
        cache_path = CACHE_ROOT / f"{code}.csv"
        if args.resume and cache_path.exists() and not args.force:
            try:
                existing = pd.read_csv(cache_path)
                if not existing.empty:
                    frames.append(existing)
                    covered_dates = _covered_dates(existing)
            except Exception:
                frames = []
                covered_dates = set()

        for chunk_start, chunk_end in _date_ranges(start_date, end_date, args.chunk_days):
            if args.max_runtime_seconds and time.monotonic() - started >= args.max_runtime_seconds:
                stopped_reason = f"max_runtime_seconds={args.max_runtime_seconds}"
                break
            chunk_dates = {chunk_start + timedelta(days=offset) for offset in range((chunk_end - chunk_start).days + 1)}
            if args.resume and not args.force and chunk_dates.issubset(covered_dates):
                continue
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
