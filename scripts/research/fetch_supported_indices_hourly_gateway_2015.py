"""Fetch supported Vietnamese index hourly OHLCV from 2015 in reverse order."""

from __future__ import annotations

import argparse
import csv
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


INDEX_CODES = ("VNINDEX", "HNXINDEX", "UPCOMINDEX", "VN30", "HNX30", "VN100")
SOURCES = (SourceName.KBS, SourceName.VCI)
BASE_START = date(2015, 1, 1)
INDEX_STARTS = {
    "VNINDEX": BASE_START,
    "HNXINDEX": BASE_START,
    "UPCOMINDEX": BASE_START,
    "VN30": BASE_START,
    "HNX30": BASE_START,
    "VN100": BASE_START,
}
RAW_ROOT = REPO_ROOT / "data" / "raw" / "vnstock_fetch" / "index_hourly_2015"
CACHE_ROOT = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly_2015"
REPORT_ROOT = REPO_ROOT / "reports" / "generated" / "index_hourly_2015" / "fetch"
SUMMARY_CSV = REPORT_ROOT / "index_hourly_2015_fetch_summary.csv"
SUMMARY_MD = REPORT_ROOT / "index_hourly_2015_fetch_summary.md"
FAILURES_CSV = REPORT_ROOT / "index_hourly_2015_failures.csv"
CHUNK_LOG_CSV = REPORT_ROOT / "index_hourly_2015_chunk_log.csv"
COLUMNS = ["datetime", "index_code", "open", "high", "low", "close", "volume", "provider", "source", "frequency"]
SUMMARY_FIELDS = [
    "index_code",
    "requested_start",
    "requested_end",
    "direction",
    "fetched",
    "rows",
    "first_timestamp",
    "last_timestamp",
    "provider",
    "source",
    "frequency",
    "cache_path",
    "stopped_by_runtime_cap",
    "stopped_reason",
    "resume_command",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="auto")
    parser.add_argument("--index-code", choices=INDEX_CODES)
    parser.add_argument("--direction", choices=("reverse",), default="reverse")
    parser.add_argument("--year-first", action="store_true")
    parser.add_argument("--monthly-fallback", action="store_true")
    parser.add_argument("--daily-fallback", action="store_true")
    parser.add_argument("--chunk-days", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-usable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-runtime-seconds", type=int, default=7200)
    return parser.parse_args()


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def month_start(value: date) -> date:
    return date(value.year, value.month, 1)


def previous_month(value: date) -> date:
    first = month_start(value)
    return first - timedelta(days=1)


def reverse_year_ranges(start: date, end: date) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    current_end = end
    while current_end >= start:
        current_start = max(start, date(current_end.year, 1, 1))
        ranges.append((current_start, current_end))
        current_end = current_start - timedelta(days=1)
    return ranges


def reverse_month_ranges(start: date, end: date) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    current_end = end
    while current_end >= start:
        current_start = max(start, month_start(current_end))
        ranges.append((current_start, current_end))
        current_end = previous_month(current_end)
    return ranges


def reverse_day_ranges(start: date, end: date, days: int) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    current_end = end
    while current_end >= start:
        current_start = max(start, current_end - timedelta(days=max(1, days) - 1))
        ranges.append((current_start, current_end))
        current_end = current_start - timedelta(days=1)
    return ranges


def raw_chunk_path(code: str, start: date, end: date) -> Path:
    return RAW_ROOT / code / f"{code}_{start:%Y%m%d}_{end:%Y%m%d}.csv"


def read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame(columns=COLUMNS)
    if any(column not in frame.columns for column in COLUMNS):
        return pd.DataFrame(columns=COLUMNS)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    return frame.dropna(subset=["datetime"])[COLUMNS]


def ohlcv_valid(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    frame = frame.copy()
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["open", "high", "low", "close", "volume"]].isna().any().any():
        return False
    prices_ok = bool((frame[["open", "high", "low", "close"]] > 0).all().all())
    volume_ok = bool((frame["volume"] >= 0).all())
    ohlc_ok = bool(
        (frame["high"] >= frame["low"]).all()
        and (frame["high"] >= frame[["open", "close"]].max(axis=1)).all()
        and (frame["low"] <= frame[["open", "close"]].min(axis=1)).all()
    )
    return prices_ok and volume_ok and ohlc_ok


def cache_is_usable(code: str) -> bool:
    cache_path = CACHE_ROOT / f"{code}.csv"
    if not cache_path.exists():
        return False
    frame = read_frame(cache_path)
    if frame.empty:
        return False
    frame = frame[frame["index_code"].astype(str).str.upper().eq(code)].copy()
    if frame.empty:
        return False
    frequency_ok = bool(frame["frequency"].astype(str).eq("1H").all())
    return frequency_ok and ohlcv_valid(frame)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame[COLUMNS].copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)


def fetch_chunk(code: str, start: date, end: date, granularity: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    response = fetch_price_history(
        FetchRequest(
            symbol=code,
            asset_type=AssetType.INDEX,
            start=start.isoformat(),
            end=end.isoformat(),
            frequency=Frequency.HOURLY,
            preferred_sources=SOURCES,
            allow_legacy_fallback=True,
            allow_daily=False,
            allow_resample=False,
        )
    )
    frame = response.data[COLUMNS].copy()
    return frame, {
        "index_code": code,
        "chunk_start": start.isoformat(),
        "chunk_end": end.isoformat(),
        "granularity": granularity,
        "provider": str(response.provider),
        "source": str(response.source),
        "success": "true",
        "rows": int(len(frame)),
        "exception_type": "",
        "exception_message": "",
    }


def attempt_range(code: str, start: date, end: date, granularity: str, args: argparse.Namespace) -> tuple[list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]]]:
    chunk_path = raw_chunk_path(code, start, end)
    if args.resume and not args.force and chunk_path.exists():
        cached = read_frame(chunk_path)
        if not cached.empty:
            return [cached], [], []
    logs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        frame, log = fetch_chunk(code, start, end, granularity)
        logs.append(log)
        if not frame.empty:
            write_frame(chunk_path, frame)
        return [frame], logs, failures
    except Exception as exc:
        failures.append(
            {
                "index_code": code,
                "chunk_start": start.isoformat(),
                "chunk_end": end.isoformat(),
                "granularity": granularity,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc).replace("\n", " ")[:800],
            }
        )
        if isinstance(exc, ProviderFetchError):
            for attempt in exc.attempts:
                logs.append(
                    {
                        "index_code": code,
                        "chunk_start": start.isoformat(),
                        "chunk_end": end.isoformat(),
                        "granularity": granularity,
                        "provider": attempt.get("provider", ""),
                        "source": attempt.get("source", ""),
                        "success": "false",
                        "rows": int(attempt.get("rows", 0) or 0),
                        "exception_type": attempt.get("error_type", ""),
                        "exception_message": str(attempt.get("error_message", ""))[:800],
                    }
                )
        return [], logs, failures


def fetch_range_with_fallback(code: str, start: date, end: date, args: argparse.Namespace, started: float) -> tuple[list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]]]:
    frames, logs, failures = attempt_range(code, start, end, "year", args)
    if frames and any(not frame.empty for frame in frames):
        return frames, logs, failures
    all_frames, all_logs, all_failures = frames, logs, failures
    for month_start_date, month_end_date in reverse_month_ranges(start, end):
        if runtime_exceeded(started, args):
            break
        frames, logs, failures = attempt_range(code, month_start_date, month_end_date, "month", args)
        all_frames.extend(frames)
        all_logs.extend(logs)
        all_failures.extend(failures)
        if frames and any(not frame.empty for frame in frames):
            continue
        for day_start, day_end in reverse_day_ranges(month_start_date, month_end_date, args.chunk_days):
            if runtime_exceeded(started, args):
                break
            frames, logs, failures = attempt_range(code, day_start, day_end, f"{args.chunk_days}day", args)
            all_frames.extend(frames)
            all_logs.extend(logs)
            all_failures.extend(failures)
            if frames and any(not frame.empty for frame in frames):
                continue
            for one_start, one_end in reverse_day_ranges(day_start, day_end, 1):
                if runtime_exceeded(started, args):
                    break
                frames, logs, failures = attempt_range(code, one_start, one_end, "1day", args)
                all_frames.extend(frames)
                all_logs.extend(logs)
                all_failures.extend(failures)
    return all_frames, all_logs, all_failures


def resume_command(code: str, args: argparse.Namespace) -> str:
    return (
        "<repo-approved-venv-python> scripts\\research\\fetch_supported_indices_hourly_gateway_2015.py "
        f"--index-code {code} --direction reverse --start {args.start} --end {args.end} "
        "--year-first --monthly-fallback --daily-fallback --resume "
        f"--max-runtime-seconds {args.max_runtime_seconds}"
    )


def summarize(code: str, frames: list[pd.DataFrame], stopped_reason: str, requested_start: date, requested_end: date, args: argparse.Namespace) -> dict[str, Any]:
    cache_path = CACHE_ROOT / f"{code}.csv"
    combined_frames = [frame for frame in frames if not frame.empty]
    if cache_path.exists():
        existing = read_frame(cache_path)
        if not existing.empty:
            combined_frames.append(existing)
    if combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
        combined["datetime"] = pd.to_datetime(combined["datetime"], errors="coerce")
        combined = combined.dropna(subset=["datetime"])
        combined = combined[combined["index_code"].astype(str).str.upper().eq(code)].copy()
        combined = combined.sort_values("datetime").drop_duplicates(["datetime"], keep="last")
        write_frame(cache_path, combined)
    else:
        combined = pd.DataFrame(columns=COLUMNS)
    timestamps = pd.to_datetime(combined["datetime"], errors="coerce").dropna() if not combined.empty else pd.Series(dtype="datetime64[ns]")
    runtime_cap = stopped_reason == f"max_runtime_seconds={args.max_runtime_seconds}"
    return {
        "index_code": code,
        "requested_start": requested_start.isoformat(),
        "requested_end": requested_end.isoformat(),
        "direction": args.direction,
        "fetched": str(not combined.empty).lower(),
        "rows": int(len(combined)),
        "first_timestamp": "" if timestamps.empty else str(timestamps.min()),
        "last_timestamp": "" if timestamps.empty else str(timestamps.max()),
        "provider": "" if combined.empty else ";".join(sorted(combined["provider"].astype(str).unique())),
        "source": "" if combined.empty else ";".join(sorted(combined["source"].astype(str).unique())),
        "frequency": "" if combined.empty else ";".join(sorted(combined["frequency"].astype(str).unique())),
        "cache_path": str(cache_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "stopped_by_runtime_cap": str(runtime_cap).lower(),
        "stopped_reason": stopped_reason,
        "resume_command": resume_command(code, args) if runtime_cap else "",
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(summary: list[dict[str, Any]], failures: list[dict[str, Any]], logs: list[dict[str, Any]]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(SUMMARY_CSV, summary, SUMMARY_FIELDS)
    write_csv(FAILURES_CSV, failures, ["index_code", "chunk_start", "chunk_end", "granularity", "exception_type", "exception_message"])
    write_csv(CHUNK_LOG_CSV, logs, ["index_code", "chunk_start", "chunk_end", "granularity", "provider", "source", "success", "rows", "exception_type", "exception_message"])
    lines = [
        "# Index Hourly 2015 Reverse Gateway Fetch Summary",
        "",
        "- Provider path: `src.data.providers.vn_price_gateway.fetch_price_history`.",
        "- Direction: reverse, provider-current/latest available timestamp back to configured start.",
        "- Frequency: `1H` only.",
        "- Daily data used: no.",
        "- Resampling used: no.",
        "",
        "| index_code | fetched | rows | first | last | stopped_by_runtime_cap | stopped_reason |",
        "|---|---:|---:|---|---|---:|---|",
    ]
    for row in summary:
        lines.append(f"| `{row['index_code']}` | {row['fetched']} | {row['rows']} | {row['first_timestamp']} | {row['last_timestamp']} | {row['stopped_by_runtime_cap']} | `{row['stopped_reason']}` |")
    stopped = [row for row in summary if row.get("resume_command")]
    if stopped:
        lines.extend(["", "## Resume Commands", ""])
        for row in stopped:
            lines.append(f"- `{row['index_code']}`: `{row['resume_command']}`")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def runtime_exceeded(started: float, args: argparse.Namespace) -> bool:
    return bool(args.max_runtime_seconds and time.monotonic() - started >= args.max_runtime_seconds)


def main() -> int:
    args = parse_args()
    requested_base_start = parse_date(args.start)
    requested_end = date.today() if args.end == "auto" else parse_date(args.end)
    started = time.monotonic()
    codes = (args.index_code,) if args.index_code else INDEX_CODES
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    for code in codes:
        requested_start = max(requested_base_start, INDEX_STARTS[code])
        if args.skip_usable and not args.force and cache_is_usable(code):
            summaries.append(summarize(code, [], "skipped_usable_cache", requested_start, requested_end, args))
            write_reports(summaries, failures, logs)
            continue
        if runtime_exceeded(started, args):
            summaries.append(summarize(code, [], "not_started_runtime_cap", requested_start, requested_end, args))
            write_reports(summaries, failures, logs)
            continue
        stopped = "completed"
        frames: list[pd.DataFrame] = []
        for chunk_start, chunk_end in reverse_year_ranges(requested_start, requested_end):
            if runtime_exceeded(started, args):
                stopped = f"max_runtime_seconds={args.max_runtime_seconds}"
                break
            new_frames, new_logs, new_failures = fetch_range_with_fallback(code, chunk_start, chunk_end, args, started)
            frames.extend(new_frames)
            logs.extend(new_logs)
            failures.extend(new_failures)
            summarize(code, frames, "in_progress", requested_start, requested_end, args)
            if runtime_exceeded(started, args):
                stopped = f"max_runtime_seconds={args.max_runtime_seconds}"
                break
        summaries.append(summarize(code, frames, stopped, requested_start, requested_end, args))
        write_reports(summaries, failures, logs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
