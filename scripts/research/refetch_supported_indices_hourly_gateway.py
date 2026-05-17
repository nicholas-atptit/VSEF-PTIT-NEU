"""Refetch supported Vietnamese index hourly OHLCV through the gateway."""

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
RAW_ROOT = REPO_ROOT / "data" / "raw" / "vnstock_fetch" / "index_hourly_gateway"
CACHE_ROOT = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly"
REPORT_ROOT = REPO_ROOT / "reports" / "generated" / "index_hourly_gateway" / "fetch"
SUMMARY_CSV = REPORT_ROOT / "index_hourly_gateway_fetch_summary.csv"
SUMMARY_MD = REPORT_ROOT / "index_hourly_gateway_fetch_summary.md"
FAILURES_CSV = REPORT_ROOT / "index_hourly_gateway_failures.csv"
CHUNK_LOG_CSV = REPORT_ROOT / "index_hourly_gateway_chunk_log.csv"
COLUMNS = ["datetime", "index_code", "open", "high", "low", "close", "volume", "provider", "source", "frequency"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="auto")
    parser.add_argument("--index-code", choices=INDEX_CODES)
    parser.add_argument("--chunk-days", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-runtime-seconds", type=int, default=1800)
    return parser.parse_args()


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def date_ranges(start: date, end: date, chunk_days: int) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    current = start
    while current <= end:
        chunk_end = min(end, current + timedelta(days=max(1, chunk_days) - 1))
        ranges.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
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


def fetch_chunk(code: str, start: date, end: date, chunk_days: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    start_s = start.isoformat()
    end_s = end.isoformat()
    response = fetch_price_history(
        FetchRequest(
            symbol=code,
            asset_type=AssetType.INDEX,
            start=start_s,
            end=end_s,
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
        "chunk_start": start_s,
        "chunk_end": end_s,
        "chunk_days": chunk_days,
        "provider": str(response.provider),
        "source": str(response.source),
        "success": "true",
        "rows": int(len(frame)),
        "exception_type": "",
        "exception_message": "",
    }


def fetch_with_fallback(code: str, start: date, end: date, chunk_days: int) -> tuple[list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]]]:
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
                frame, log = fetch_chunk(code, sub_start, sub_end, days)
                temp_frames.append(frame)
                temp_logs.append(log)
                if not frame.empty:
                    write_frame(raw_chunk_path(code, sub_start, sub_end), frame)
            frames.extend(temp_frames)
            logs.extend(temp_logs)
            return frames, logs, failures
        except Exception as exc:
            failures.append(
                {
                    "index_code": code,
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
                            "index_code": code,
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


def summarize(code: str, frames: list[pd.DataFrame], stopped_reason: str) -> dict[str, Any]:
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
    return {
        "index_code": code,
        "fetched": str(not combined.empty).lower(),
        "rows": int(len(combined)),
        "first_timestamp": "" if timestamps.empty else str(timestamps.min()),
        "last_timestamp": "" if timestamps.empty else str(timestamps.max()),
        "provider": "" if combined.empty else ";".join(sorted(combined["provider"].astype(str).unique())),
        "source": "" if combined.empty else ";".join(sorted(combined["source"].astype(str).unique())),
        "frequency": "" if combined.empty else ";".join(sorted(combined["frequency"].astype(str).unique())),
        "cache_path": str(cache_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "stopped_reason": stopped_reason,
    }


def write_reports(summary: list[dict[str, Any]], failures: list[dict[str, Any]], logs: list[dict[str, Any]]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(SUMMARY_CSV, summary, ["index_code", "fetched", "rows", "first_timestamp", "last_timestamp", "provider", "source", "frequency", "cache_path", "stopped_reason"])
    write_csv(FAILURES_CSV, failures, ["index_code", "chunk_start", "chunk_end", "chunk_days", "exception_type", "exception_message"])
    write_csv(CHUNK_LOG_CSV, logs, ["index_code", "chunk_start", "chunk_end", "chunk_days", "provider", "source", "success", "rows", "exception_type", "exception_message"])
    lines = [
        "# Index Hourly Gateway Fetch Summary",
        "",
        "- Provider path: `src.data.providers.vn_price_gateway.fetch_price_history`.",
        "- Frequency: `1H` only.",
        "- Daily data used: no.",
        "- Resampling used: no.",
        "",
        "| index_code | fetched | rows | first | last | provider | source | frequency | stopped_reason |",
        "|---|---:|---:|---|---|---|---|---|---|",
    ]
    for row in summary:
        lines.append(f"| `{row['index_code']}` | {row['fetched']} | {row['rows']} | {row['first_timestamp']} | {row['last_timestamp']} | `{row['provider']}` | `{row['source']}` | `{row['frequency']}` | `{row['stopped_reason']}` |")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    start = parse_date(args.start)
    end = date.today() if args.end == "auto" else parse_date(args.end)
    started = time.monotonic()
    codes = (args.index_code,) if args.index_code else INDEX_CODES
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    for code in codes:
        stopped = "completed"
        frames: list[pd.DataFrame] = []
        for chunk_start, chunk_end in date_ranges(start, end, args.chunk_days):
            if args.max_runtime_seconds and time.monotonic() - started >= args.max_runtime_seconds:
                stopped = f"max_runtime_seconds={args.max_runtime_seconds}"
                break
            chunk_path = raw_chunk_path(code, chunk_start, chunk_end)
            if args.resume and not args.force and chunk_path.exists():
                cached = read_frame(chunk_path)
                if not cached.empty:
                    frames.append(cached)
                    continue
            new_frames, new_logs, new_failures = fetch_with_fallback(code, chunk_start, chunk_end, args.chunk_days)
            frames.extend(new_frames)
            logs.extend(new_logs)
            failures.extend(new_failures)
        summaries.append(summarize(code, frames, stopped))
    write_reports(summaries, failures, logs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
