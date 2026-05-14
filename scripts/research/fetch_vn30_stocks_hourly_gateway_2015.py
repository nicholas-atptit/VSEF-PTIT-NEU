"""Fetch frozen VN30 stock hourly OHLCV from 2015 through the gateway."""

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


UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
RAW_ROOT = REPO_ROOT / "data" / "raw" / "vnstock_fetch" / "vn30_hourly_2015"
CACHE_ROOT = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_2015"
REPORT_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015" / "fetch"
SUMMARY_CSV = REPORT_ROOT / "vn30_hourly_2015_fetch_summary.csv"
SUMMARY_MD = REPORT_ROOT / "vn30_hourly_2015_fetch_summary.md"
FAILURES_CSV = REPORT_ROOT / "vn30_hourly_2015_failures.csv"
CHUNK_LOG_CSV = REPORT_ROOT / "vn30_hourly_2015_chunk_log.csv"
SOURCES = (SourceName.KBS, SourceName.VCI)
COLUMNS = ["datetime", "ticker", "open", "high", "low", "close", "volume", "provider", "source", "frequency"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="auto")
    parser.add_argument("--ticker")
    parser.add_argument("--chunk-days", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-runtime-seconds", type=int, default=14400)
    return parser.parse_args()


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def read_universe() -> list[str]:
    with UNIVERSE_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row["ticker"].strip().upper() for row in rows if row.get("ticker")]


def date_ranges(start: date, end: date, chunk_days: int) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    current = start
    while current <= end:
        chunk_end = min(end, current + timedelta(days=max(1, chunk_days) - 1))
        ranges.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return ranges


def raw_chunk_path(ticker: str, start: date, end: date) -> Path:
    return RAW_ROOT / ticker / f"{ticker}_{start:%Y%m%d}_{end:%Y%m%d}.csv"


def read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame(columns=COLUMNS)
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        return pd.DataFrame(columns=COLUMNS)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    return frame.dropna(subset=["datetime"])[COLUMNS]


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame[COLUMNS].copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)


def fetch_chunk(ticker: str, start: date, end: date, chunk_days: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    response = fetch_price_history(
        FetchRequest(
            symbol=ticker,
            asset_type=AssetType.STOCK,
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
        "ticker": ticker,
        "chunk_start": start.isoformat(),
        "chunk_end": end.isoformat(),
        "chunk_days": chunk_days,
        "provider": str(response.provider),
        "source": str(response.source),
        "success": "true",
        "rows": int(len(frame)),
        "exception_type": "",
        "exception_message": "",
    }


def fetch_with_fallback(ticker: str, start: date, end: date, chunk_days: int) -> tuple[list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    logs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for days in (chunk_days, 3, 1):
        if days > (end - start).days + 1:
            continue
        temp_frames: list[pd.DataFrame] = []
        temp_logs: list[dict[str, Any]] = []
        try:
            for sub_start, sub_end in date_ranges(start, end, days):
                frame, log = fetch_chunk(ticker, sub_start, sub_end, days)
                temp_frames.append(frame)
                temp_logs.append(log)
                if not frame.empty:
                    write_frame(raw_chunk_path(ticker, sub_start, sub_end), frame)
            frames.extend(temp_frames)
            logs.extend(temp_logs)
            return frames, logs, failures
        except Exception as exc:
            failures.append(
                {
                    "ticker": ticker,
                    "chunk_start": start.isoformat(),
                    "chunk_end": end.isoformat(),
                    "chunk_days": days,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc).replace("\n", " ")[:800],
                }
            )
            if isinstance(exc, ProviderFetchError):
                for attempt in exc.attempts:
                    logs.append(
                        {
                            "ticker": ticker,
                            "chunk_start": start.isoformat(),
                            "chunk_end": end.isoformat(),
                            "chunk_days": days,
                            "provider": attempt.get("provider", ""),
                            "source": attempt.get("source", ""),
                            "success": "false",
                            "rows": int(attempt.get("rows", 0) or 0),
                            "exception_type": attempt.get("error_type", ""),
                            "exception_message": str(attempt.get("error_message", ""))[:800],
                        }
                    )
            logs.extend(temp_logs)
    return frames, logs, failures


def resume_command(ticker: str, args: argparse.Namespace) -> str:
    return (
        "<repo-approved-venv-python> scripts\\research\\fetch_vn30_stocks_hourly_gateway_2015.py "
        f"--ticker {ticker} --start {args.start} --end {args.end} --chunk-days {args.chunk_days} --resume "
        f"--max-runtime-seconds {args.max_runtime_seconds}"
    )


def summarize(ticker: str, frames: list[pd.DataFrame], stopped_reason: str, requested_start: date, args: argparse.Namespace) -> dict[str, Any]:
    cache_path = CACHE_ROOT / f"{ticker}.csv"
    combined_frames = [frame for frame in frames if not frame.empty]
    if cache_path.exists():
        existing = read_frame(cache_path)
        if not existing.empty:
            combined_frames.append(existing)
    if combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
        combined["datetime"] = pd.to_datetime(combined["datetime"], errors="coerce")
        combined = combined.dropna(subset=["datetime"])
        combined = combined[combined["ticker"].astype(str).str.upper().eq(ticker)].copy()
        combined = combined.sort_values("datetime").drop_duplicates(["datetime"], keep="last")
        write_frame(cache_path, combined)
    else:
        combined = pd.DataFrame(columns=COLUMNS)
    timestamps = pd.to_datetime(combined["datetime"], errors="coerce").dropna() if not combined.empty else pd.Series(dtype="datetime64[ns]")
    return {
        "ticker": ticker,
        "requested_start": requested_start.isoformat(),
        "fetched": str(not combined.empty).lower(),
        "rows": int(len(combined)),
        "first_timestamp": "" if timestamps.empty else str(timestamps.min()),
        "last_timestamp": "" if timestamps.empty else str(timestamps.max()),
        "provider": "" if combined.empty else ";".join(sorted(combined["provider"].astype(str).unique())),
        "source": "" if combined.empty else ";".join(sorted(combined["source"].astype(str).unique())),
        "frequency": "" if combined.empty else ";".join(sorted(combined["frequency"].astype(str).unique())),
        "cache_path": str(cache_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "stopped_by_runtime_cap": str(stopped_reason.startswith("max_runtime_seconds=")).lower(),
        "stopped_reason": stopped_reason,
        "resume_command": resume_command(ticker, args) if stopped_reason.startswith("max_runtime_seconds=") else "",
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(summary: list[dict[str, Any]], failures: list[dict[str, Any]], logs: list[dict[str, Any]]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(SUMMARY_CSV, summary, ["ticker", "requested_start", "fetched", "rows", "first_timestamp", "last_timestamp", "provider", "source", "frequency", "cache_path", "stopped_by_runtime_cap", "stopped_reason", "resume_command"])
    write_csv(FAILURES_CSV, failures, ["ticker", "chunk_start", "chunk_end", "chunk_days", "exception_type", "exception_message"])
    write_csv(CHUNK_LOG_CSV, logs, ["ticker", "chunk_start", "chunk_end", "chunk_days", "provider", "source", "success", "rows", "exception_type", "exception_message"])
    lines = [
        "# VN30 Stock Hourly 2015 Gateway Fetch Summary",
        "",
        "- Provider path: `src.data.providers.vn_price_gateway.fetch_price_history`.",
        "- Data start: `2015-01-01`.",
        "- Frequency: `1H` only.",
        "- Daily data used: no.",
        "- Resampling used: no.",
        "",
        "| ticker | fetched | rows | first | last | provider | source | frequency | stopped_by_runtime_cap | stopped_reason |",
        "|---|---:|---:|---|---|---|---|---|---:|---|",
    ]
    for row in summary:
        lines.append(f"| `{row['ticker']}` | {row['fetched']} | {row['rows']} | {row['first_timestamp']} | {row['last_timestamp']} | `{row['provider']}` | `{row['source']}` | `{row['frequency']}` | {row['stopped_by_runtime_cap']} | `{row['stopped_reason']}` |")
    stopped = [row for row in summary if row.get("resume_command")]
    if stopped:
        lines.extend(["", "## Resume Commands", ""])
        for row in stopped:
            lines.append(f"- `{row['ticker']}`: `{row['resume_command']}`")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    fallback_start = parse_date(args.start)
    end = date.today() if args.end == "auto" else parse_date(args.end)
    tickers = [args.ticker.upper()] if args.ticker else read_universe()
    started = time.monotonic()
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    for ticker in tickers:
        requested_start = fallback_start
        stopped = "completed"
        frames: list[pd.DataFrame] = []
        started_ticker = False
        for chunk_start, chunk_end in date_ranges(requested_start, end, args.chunk_days):
            if args.max_runtime_seconds and time.monotonic() - started >= args.max_runtime_seconds:
                stopped = f"max_runtime_seconds={args.max_runtime_seconds}" if started_ticker else "not_started_runtime_cap"
                break
            started_ticker = True
            chunk_path = raw_chunk_path(ticker, chunk_start, chunk_end)
            if args.resume and not args.force and chunk_path.exists():
                cached = read_frame(chunk_path)
                if not cached.empty:
                    frames.append(cached)
                    continue
            new_frames, new_logs, new_failures = fetch_with_fallback(ticker, chunk_start, chunk_end, args.chunk_days)
            frames.extend(new_frames)
            logs.extend(new_logs)
            failures.extend(new_failures)
        summaries.append(summarize(ticker, frames, stopped, requested_start, args))
    write_reports(summaries, failures, logs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
