"""Fetch frozen VN30 stock hourly OHLCV with adaptive reverse chunks."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_hourly_2015_fetch_plan import (  # noqa: E402
    EVAL_START,
    STOCK_COLUMNS,
    TRAIN_END,
    FetchChunk,
    build_reverse_chunks,
    cache_is_usable,
    chunk_to_dict,
    get_provider_current_end,
    load_effective_starts,
    read_checkpoint,
    read_frame,
    read_universe,
    split_chunk,
    write_checkpoint,
    write_frame,
)
from src.data.providers.vn_price_gateway import ProviderFetchError, fetch_price_history  # noqa: E402
from src.data.providers.vn_provider_contract import AssetType, FetchRequest, Frequency, SourceName  # noqa: E402


RAW_ROOT = REPO_ROOT / "data" / "raw" / "vnstock_fetch" / "vn30_hourly_2015"
CACHE_ROOT = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_2015"
REPORT_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015" / "fetch"
SUMMARY_CSV = REPORT_ROOT / "vn30_hourly_2015_fetch_summary.csv"
SUMMARY_MD = REPORT_ROOT / "vn30_hourly_2015_fetch_summary.md"
FAILURES_CSV = REPORT_ROOT / "vn30_hourly_2015_failures.csv"
CHUNK_LOG_CSV = REPORT_ROOT / "vn30_hourly_2015_chunk_log.csv"
SOURCES = (SourceName.KBS, SourceName.VCI)
SUMMARY_FIELDS = [
    "ticker",
    "effective_start",
    "provider_current_end",
    "rows",
    "first_datetime",
    "last_datetime",
    "rows_before_train_cutoff",
    "rows_after_eval_start",
    "chunks_attempted",
    "chunks_with_rows",
    "chunks_empty",
    "chunks_failed",
    "usable_candidate",
    "skipped_already_usable",
    "stopped_by_runtime_cap",
    "resume_command",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="auto")
    parser.add_argument("--direction", choices=("reverse",), default="reverse")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-runtime-seconds", type=int, default=14400)
    parser.add_argument("--skip-usable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunk-strategy", choices=("adaptive",), default="adaptive")
    parser.add_argument("--year-first", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--monthly-fallback", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--daily-fallback", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--chunk-days", type=int, default=5, help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def raw_chunk_path(ticker: str, chunk: FetchChunk) -> Path:
    return RAW_ROOT / ticker / f"{ticker}_{chunk.start:%Y%m%d}_{chunk.end:%Y%m%d}_{chunk.granularity}.csv"


def cache_path(ticker: str) -> Path:
    return CACHE_ROOT / f"{ticker}.csv"


def runtime_exceeded(started: float, args: argparse.Namespace) -> bool:
    return bool(args.max_runtime_seconds and time.monotonic() - started >= args.max_runtime_seconds)


def resume_command(ticker: str, args: argparse.Namespace) -> str:
    python_exe = f'"{sys.executable}"'
    return (
        f"{python_exe} scripts\\research\\fetch_vn30_stocks_hourly_gateway_2015.py "
        f"--ticker {ticker} --direction reverse --start {args.start} --end {args.end} "
        f"--resume --skip-usable --max-runtime-seconds {args.max_runtime_seconds}"
    )


def fetch_chunk(ticker: str, chunk: FetchChunk) -> tuple[pd.DataFrame, dict[str, Any]]:
    response = fetch_price_history(
        FetchRequest(
            symbol=ticker,
            asset_type=AssetType.STOCK,
            start=chunk.start.isoformat(),
            end=chunk.end.isoformat(),
            frequency=Frequency.HOURLY,
            preferred_sources=SOURCES,
            allow_legacy_fallback=True,
            allow_daily=False,
            allow_resample=False,
        )
    )
    frame = response.data[STOCK_COLUMNS].copy()
    return frame, {
        "ticker": ticker,
        "chunk_start": chunk.start.isoformat(),
        "chunk_end": chunk.end.isoformat(),
        "granularity": chunk.granularity,
        "provider": str(response.provider),
        "source": str(response.source),
        "success": "true",
        "rows": int(len(frame)),
        "exception_type": "",
        "exception_message": "",
    }


def append_to_cache(ticker: str, frame: pd.DataFrame) -> pd.DataFrame:
    existing = read_frame(cache_path(ticker), STOCK_COLUMNS)
    frames = [item for item in (existing, frame) if not item.empty]
    if not frames:
        return pd.DataFrame(columns=STOCK_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"], errors="coerce")
    combined = combined.dropna(subset=["datetime"])
    combined = combined[combined["ticker"].astype(str).str.upper().eq(ticker)].copy()
    combined = combined.sort_values("datetime").drop_duplicates(["datetime"], keep="last")
    write_frame(cache_path(ticker), combined, STOCK_COLUMNS)
    return combined


def cache_stats(ticker: str) -> dict[str, Any]:
    frame = read_frame(cache_path(ticker), STOCK_COLUMNS)
    frame = frame[frame["ticker"].astype(str).str.upper().eq(ticker)].copy() if not frame.empty else frame
    timestamps = frame["datetime"].dropna() if not frame.empty else pd.Series(dtype="datetime64[ns]")
    return {
        "rows": int(len(frame)),
        "first_datetime": "" if timestamps.empty else str(timestamps.min()),
        "last_datetime": "" if timestamps.empty else str(timestamps.max()),
        "rows_before_train_cutoff": int((frame["datetime"] <= TRAIN_END).sum()) if not frame.empty else 0,
        "rows_after_eval_start": int((frame["datetime"] >= EVAL_START).sum()) if not frame.empty else 0,
        "usable_candidate": str(cache_is_usable(ticker, "stock")).lower(),
    }


def provider_attempt_logs(ticker: str, chunk: FetchChunk, exc: Exception) -> list[dict[str, Any]]:
    if not isinstance(exc, ProviderFetchError):
        return []
    rows: list[dict[str, Any]] = []
    for attempt in exc.attempts:
        rows.append(
            {
                "ticker": ticker,
                "chunk_start": chunk.start.isoformat(),
                "chunk_end": chunk.end.isoformat(),
                "granularity": chunk.granularity,
                "provider": attempt.get("provider", ""),
                "source": attempt.get("source", ""),
                "success": "false",
                "rows": int(attempt.get("rows", 0) or 0),
                "exception_type": attempt.get("error_type", ""),
                "exception_message": str(attempt.get("error_message", ""))[:800],
            }
        )
    return rows


def is_empty_data_error(exc: Exception) -> bool:
    if not isinstance(exc, ProviderFetchError):
        return False
    if not exc.attempts:
        return False
    messages = " ".join(
        [str(exc)]
        + [str(attempt.get("error_message", "")) for attempt in exc.attempts]
    ).lower()
    rows = [int(attempt.get("rows", 0) or 0) for attempt in exc.attempts]
    empty_message = (
        "empty ohlcv" in messages
        or "dataset_unavailable" in messages
        or "no rows" in messages
        or "no provider/source returned validated ohlcv rows" in messages
    )
    return all(row == 0 for row in rows) and empty_message


def fetch_symbol(ticker: str, effective_start: date, provider_current_end: date, args: argparse.Namespace, started: float) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    state = read_checkpoint(ticker) if args.resume else {"completed_chunks": [], "failed_chunks": []}
    completed = set(state.get("completed_chunks", []))
    queue = [chunk for chunk in build_reverse_chunks(effective_start, provider_current_end, "year") if chunk.key not in completed]
    logs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    counters = {"attempted": 0, "with_rows": 0, "empty": 0, "failed": 0}
    stopped_by_cap = False
    last_unfinished = ""

    while queue:
        if runtime_exceeded(started, args):
            stopped_by_cap = True
            last_unfinished = queue[0].key
            break
        chunk = queue.pop(0)
        if chunk.key in completed:
            continue
        counters["attempted"] += 1
        raw_path = raw_chunk_path(ticker, chunk)
        if args.resume and not args.force and raw_path.exists():
            frame = read_frame(raw_path, STOCK_COLUMNS)
            if not frame.empty:
                append_to_cache(ticker, frame)
                counters["with_rows"] += 1
                completed.add(chunk.key)
                write_checkpoint(ticker, {"completed_chunks": sorted(completed), "failed_chunks": state.get("failed_chunks", []), "last_unfinished_chunk": queue[0].key if queue else ""})
                continue
        try:
            frame, log = fetch_chunk(ticker, chunk)
            logs.append(log)
            if frame.empty:
                counters["empty"] += 1
            else:
                counters["with_rows"] += 1
                write_frame(raw_path, frame, STOCK_COLUMNS)
                append_to_cache(ticker, frame)
            completed.add(chunk.key)
            write_checkpoint(ticker, {"completed_chunks": sorted(completed), "failed_chunks": state.get("failed_chunks", []), "last_unfinished_chunk": queue[0].key if queue else ""})
        except Exception as exc:
            if is_empty_data_error(exc):
                counters["empty"] += 1
                logs.extend(provider_attempt_logs(ticker, chunk, exc))
                completed.add(chunk.key)
                write_checkpoint(ticker, {"completed_chunks": sorted(completed), "failed_chunks": state.get("failed_chunks", []), "last_unfinished_chunk": queue[0].key if queue else ""})
                continue
            fallback_chunks = split_chunk(chunk)
            if fallback_chunks:
                queue = [item for item in fallback_chunks if item.key not in completed] + queue
            else:
                counters["failed"] += 1
                failures.append(
                    {
                        "ticker": ticker,
                        "chunk_start": chunk.start.isoformat(),
                        "chunk_end": chunk.end.isoformat(),
                        "granularity": chunk.granularity,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc).replace("\n", " ")[:800],
                    }
                )
                logs.extend(provider_attempt_logs(ticker, chunk, exc))
            write_checkpoint(
                ticker,
                {
                    "completed_chunks": sorted(completed),
                    "failed_chunks": [*state.get("failed_chunks", []), chunk.key],
                    "last_unfinished_chunk": queue[0].key if queue else "",
                    "last_failed_chunk": chunk_to_dict(chunk),
                },
            )

    stats = cache_stats(ticker)
    summary = {
        "ticker": ticker,
        "effective_start": effective_start.isoformat(),
        "provider_current_end": provider_current_end.isoformat(),
        **stats,
        "chunks_attempted": counters["attempted"],
        "chunks_with_rows": counters["with_rows"],
        "chunks_empty": counters["empty"],
        "chunks_failed": counters["failed"],
        "skipped_already_usable": "false",
        "stopped_by_runtime_cap": str(stopped_by_cap).lower(),
        "resume_command": resume_command(ticker, args) if stopped_by_cap else "",
    }
    if not stopped_by_cap and not queue:
        write_checkpoint(ticker, {"completed_chunks": sorted(completed), "failed_chunks": state.get("failed_chunks", []), "last_unfinished_chunk": "", "symbol_complete": True})
    return summary, failures, logs


def skipped_summary(ticker: str, effective_start: date, provider_current_end: date) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "effective_start": effective_start.isoformat(),
        "provider_current_end": provider_current_end.isoformat(),
        **cache_stats(ticker),
        "chunks_attempted": 0,
        "chunks_with_rows": 0,
        "chunks_empty": 0,
        "chunks_failed": 0,
        "skipped_already_usable": "true",
        "stopped_by_runtime_cap": "false",
        "resume_command": "",
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
    write_csv(FAILURES_CSV, failures, ["ticker", "chunk_start", "chunk_end", "granularity", "exception_type", "exception_message"])
    write_csv(CHUNK_LOG_CSV, logs, ["ticker", "chunk_start", "chunk_end", "granularity", "provider", "source", "success", "rows", "exception_type", "exception_message"])
    lines = [
        "# VN30 Stock Hourly 2015 Adaptive Reverse Gateway Fetch Summary",
        "",
        "- Provider path: `src.data.providers.vn_price_gateway.fetch_price_history`.",
        "- Direction: reverse from provider-current/latest available timestamp to effective start.",
        "- Effective start rule: `max(2015-01-01, first_trading_date)`.",
        "- Chunk strategy: yearly first, then quarterly/monthly/5-day/1-day only when broader chunks fail.",
        "- Frequency: `1H` only.",
        "- Daily data used: no.",
        "- Resampling used: no.",
        "",
        "| ticker | effective_start | rows | first | last | train_rows | eval_rows | usable_candidate | skipped | chunks_attempted | stopped_by_runtime_cap |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['ticker']}` | {row['effective_start']} | {row['rows']} | {row['first_datetime']} | {row['last_datetime']} | "
            f"{row['rows_before_train_cutoff']} | {row['rows_after_eval_start']} | {row['usable_candidate']} | {row['skipped_already_usable']} | "
            f"{row['chunks_attempted']} | {row['stopped_by_runtime_cap']} |"
        )
    stopped = [row for row in summary if row.get("resume_command")]
    if stopped:
        lines.extend(["", "## Resume Commands", ""])
        for row in stopped:
            lines.append(f"- `{row['ticker']}`: `{row['resume_command']}`")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    requested_start = parse_date(args.start)
    provider_current_end = get_provider_current_end() if args.end == "auto" else min(parse_date(args.end), get_provider_current_end())
    effective_starts = load_effective_starts()
    tickers = [args.ticker.upper()] if args.ticker else read_universe()
    started = time.monotonic()
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []

    for ticker in tickers:
        effective_start = max(requested_start, effective_starts.get(ticker, requested_start))
        if args.skip_usable and not args.force and cache_is_usable(ticker, "stock"):
            summaries.append(skipped_summary(ticker, effective_start, provider_current_end))
            write_reports(summaries, failures, logs)
            continue
        if runtime_exceeded(started, args):
            row = skipped_summary(ticker, effective_start, provider_current_end)
            row["skipped_already_usable"] = "false"
            row["stopped_by_runtime_cap"] = "true"
            row["resume_command"] = resume_command(ticker, args)
            summaries.append(row)
            write_checkpoint(ticker, {"completed_chunks": read_checkpoint(ticker).get("completed_chunks", []), "last_unfinished_chunk": "not_started_runtime_cap"})
            write_reports(summaries, failures, logs)
            break
        row, new_failures, new_logs = fetch_symbol(ticker, effective_start, provider_current_end, args, started)
        summaries.append(row)
        failures.extend(new_failures)
        logs.extend(new_logs)
        write_reports(summaries, failures, logs)
        if row["stopped_by_runtime_cap"] == "true":
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
