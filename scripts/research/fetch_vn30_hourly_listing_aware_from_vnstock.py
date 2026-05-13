"""Fetch VN30 hourly data from vnstock using listing-aware start rules."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_hourly_common import (  # noqa: E402
    EVAL_START,
    TRAIN_CUTOFF,
    TRAIN_START,
    VN30_TICKERS,
    markdown_table,
    read_universe,
    rel,
    timestamp_text,
    write_csv,
)
from scripts.research.vn30_hourly_listing_aware_common import (  # noqa: E402
    ALL_INDEX_CODES,
    MIN_EVALUATION_ROWS_PER_TICKER,
    MIN_TRAINING_ROWS_PER_TICKER,
    NORMALIZED_COLUMNS,
    OPTIONAL_INDEX_CODES,
    RAW_FETCH_DIR,
    REPORT_ROOT,
    current_fetch_end,
    normalized_cache_path,
    raw_chunk_path,
    read_listing_dates,
    read_normalized_symbol,
    requested_start_for,
    symbol_type,
    write_listing_raw_chunk,
    write_normalized_symbol,
)
from scripts.research.vn30_hourly_vnstock_common import fetch_first_success, package_status_rows, period_chunks  # noqa: E402


SUMMARY_COLUMNS = [
    "ticker",
    "asset_type",
    "listing_date_used",
    "requested_start",
    "provider",
    "chunks_attempted",
    "chunks_succeeded",
    "chunks_failed",
    "total_rows",
    "first_datetime",
    "last_datetime",
    "training_rows_before_cutoff",
    "evaluation_rows_after_2025_01_01",
    "usable",
    "normalized_cache_path",
    "missing_reason",
]
FAILURE_COLUMNS = ["ticker", "asset_type", "chunk_start", "chunk_end", "chunk_level", "failure_reason"]
ATTEMPT_COLUMNS = [
    "symbol",
    "asset_type",
    "chunk_start",
    "chunk_end",
    "chunk_level",
    "retry_attempt",
    "package",
    "package_version",
    "provider",
    "source",
    "function_used",
    "returned_rows",
    "standardized_rows",
    "returned_columns",
    "first_timestamp",
    "last_timestamp",
    "success",
    "exception_type",
    "exception_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch VN30 hourly listing-aware data from vnstock/vnstock_data.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--backoff-seconds", type=float, default=1.0)
    parser.add_argument("--provider-timeout-seconds", type=float, default=2.0)
    parser.add_argument("--discovery-years", default="2005,2010,2015,2020,2023,2024")
    parser.add_argument("--max-backfill-years", type=int, default=2)
    parser.add_argument("--skip-optional-indices", action="store_true")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--summarize-existing-only", action="store_true")
    return parser.parse_args()


def selected_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols.strip():
        return [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    tickers = read_universe()
    if tickers != VN30_TICKERS:
        raise ValueError("Frozen VN30 ticker universe does not match the mandatory list.")
    indices = ["VNINDEX"] + ([] if args.skip_optional_indices else list(OPTIONAL_INDEX_CODES))
    return [*tickers, *indices]


def parsed_discovery_years(value: str) -> list[int]:
    years: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        years.append(int(item))
    return sorted(set(years))


def read_cached_chunk(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = raw_chunk_path(symbol, start, end)
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
    frame = frame.dropna(subset=["datetime"])
    return frame[NORMALIZED_COLUMNS].reset_index(drop=True)


def fetch_leaf_chunk(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    level: str,
    args: argparse.Namespace,
    requested_start: pd.Timestamp,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if end < start:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS), [], {
            "ticker": symbol,
            "asset_type": symbol_type(symbol),
            "chunk_start": timestamp_text(start),
            "chunk_end": timestamp_text(end),
            "chunk_level": level,
            "failure_reason": "empty_chunk_after_date_cap",
        }
    path = raw_chunk_path(symbol, start, end)
    if path.exists() and not args.force:
        cached = read_cached_chunk(symbol, start, end)
        if not cached.empty:
            return cached, [], {
                "ticker": symbol,
                "asset_type": symbol_type(symbol),
                "chunk_start": start.strftime("%Y-%m-%d"),
                "chunk_end": end.strftime("%Y-%m-%d"),
                "chunk_level": level,
                "failure_reason": "",
            }

    frame, attempts = fetch_first_success(
        symbol,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        retries=max(1, int(args.retries)),
        backoff_seconds=max(0.0, float(args.backoff_seconds)),
        timeout_seconds=max(1.0, float(args.provider_timeout_seconds)),
    )
    for row in attempts:
        row["chunk_start"] = start.strftime("%Y-%m-%d")
        row["chunk_end"] = end.strftime("%Y-%m-%d")
        row["chunk_level"] = level
    if not frame.empty:
        frame = frame.copy()
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame = frame.dropna(subset=["datetime"])
        frame = frame[frame["datetime"] <= end + pd.Timedelta(days=1)].copy()
    if not frame.empty:
        write_listing_raw_chunk(path, frame, requested_start)
        with_metadata = read_cached_chunk(symbol, start, end)
        return with_metadata if not with_metadata.empty else frame, attempts, {
            "ticker": symbol,
            "asset_type": symbol_type(symbol),
            "chunk_start": start.strftime("%Y-%m-%d"),
            "chunk_end": end.strftime("%Y-%m-%d"),
            "chunk_level": level,
            "failure_reason": "",
        }

    reason = "provider_returned_no_usable_hourly_rows"
    errors = [row.get("exception_message", "") for row in attempts if row.get("exception_message")]
    if errors:
        reason = errors[-1][:300]
    return pd.DataFrame(columns=NORMALIZED_COLUMNS), attempts, {
        "ticker": symbol,
        "asset_type": symbol_type(symbol),
        "chunk_start": start.strftime("%Y-%m-%d"),
        "chunk_end": end.strftime("%Y-%m-%d"),
        "chunk_level": level,
        "failure_reason": reason,
    }


def discovery_windows(requested_start: pd.Timestamp, fetch_end: pd.Timestamp, years: list[int]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    requested_year = pd.Timestamp(requested_start).year
    all_years = sorted(set([requested_year, *years]))
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for year in all_years:
        if year < requested_year or year > fetch_end.year:
            continue
        start = max(pd.Timestamp(year=year, month=1, day=2), pd.Timestamp(requested_start).normalize())
        end = min(start + pd.Timedelta(days=6), fetch_end.normalize())
        if start <= end:
            windows.append((start, end))
    return windows


def fetch_symbol(symbol: str, args: argparse.Namespace, listing_rows: dict[str, dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    requested_start = requested_start_for(symbol, listing_rows)
    listing_date_used = "" if symbol_type(symbol) == "index" else timestamp_text(requested_start) if requested_start > TRAIN_START else ""
    fetch_end = current_fetch_end()
    attempts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    chunks_attempted = chunks_succeeded = chunks_failed = 0

    first_success_year: int | None = None
    for start, end in discovery_windows(requested_start, fetch_end, parsed_discovery_years(args.discovery_years)):
        frame, attempt_rows, status = fetch_leaf_chunk(symbol, start, end, "discovery", args, requested_start)
        chunks_attempted += 1
        attempts.extend(attempt_rows)
        if frame.empty:
            chunks_failed += 1
            failures.append(status)
            continue
        chunks_succeeded += 1
        frames.append(frame)
        first_success_year = int(pd.to_datetime(frame["datetime"], errors="coerce").dropna().min().year)
        break

    if first_success_year is not None:
        backfill_start_year = max(requested_start.year, first_success_year - max(0, int(args.max_backfill_years)))
        for year in range(first_success_year - 1, backfill_start_year - 1, -1):
            start = max(pd.Timestamp(year=year, month=1, day=1), requested_start.normalize())
            end = min(pd.Timestamp(year=year, month=12, day=31), fetch_end.normalize())
            if start > end:
                continue
            frame, attempt_rows, status = fetch_leaf_chunk(symbol, start, end, "backfill_year", args, requested_start)
            chunks_attempted += 1
            attempts.extend(attempt_rows)
            if frame.empty:
                chunks_failed += 1
                failures.append(status)
                continue
            chunks_succeeded += 1
            frames.append(frame)
            first_success_year = min(first_success_year, year)

        for start, end in period_chunks(pd.Timestamp(year=first_success_year, month=1, day=1), fetch_end, "year"):
            start = max(start, requested_start.normalize())
            if start > end:
                continue
            frame, attempt_rows, status = fetch_leaf_chunk(symbol, start, end, "year", args, requested_start)
            chunks_attempted += 1
            attempts.extend(attempt_rows)
            if not frame.empty:
                chunks_succeeded += 1
                frames.append(frame)
                continue
            monthly_frames: list[pd.DataFrame] = []
            monthly_failures: list[dict[str, Any]] = []
            monthly_attempted = monthly_succeeded = monthly_failed = 0
            for month_start, month_end in period_chunks(start, end, "month"):
                month_frame, month_attempts, month_status = fetch_leaf_chunk(symbol, month_start, month_end, "month", args, requested_start)
                monthly_attempted += 1
                attempts.extend(month_attempts)
                if month_frame.empty:
                    monthly_failed += 1
                    monthly_failures.append(month_status)
                    continue
                monthly_succeeded += 1
                monthly_frames.append(month_frame)
            chunks_attempted += monthly_attempted
            chunks_succeeded += monthly_succeeded
            chunks_failed += monthly_failed
            frames.extend(monthly_frames)
            failures.extend(monthly_failures)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined["datetime"] = pd.to_datetime(combined["datetime"], errors="coerce")
        combined = combined.dropna(subset=["datetime"])
        combined["ticker"] = combined["ticker"].astype(str).str.upper().str.strip()
        combined = combined[combined["ticker"] == symbol.upper()].copy()
        combined = combined.sort_values(["ticker", "datetime"]).drop_duplicates(["ticker", "datetime"], keep="last")
        if not combined.empty:
            write_normalized_symbol(symbol, combined, requested_start)
            combined = read_normalized_symbol(symbol)
    else:
        combined = pd.DataFrame(columns=NORMALIZED_COLUMNS)

    timestamps = pd.to_datetime(combined["datetime"], errors="coerce").dropna() if not combined.empty else pd.Series(dtype="datetime64[ns]")
    training_rows = int(((timestamps <= TRAIN_CUTOFF) & (timestamps >= TRAIN_START)).sum()) if not timestamps.empty else 0
    evaluation_rows = int(((timestamps >= EVAL_START) & (timestamps <= fetch_end)).sum()) if not timestamps.empty else 0
    first_ts = timestamp_text(timestamps.min()) if not timestamps.empty else ""
    last_ts = timestamp_text(timestamps.max()) if not timestamps.empty else ""
    provider = ""
    if not combined.empty:
        provider = ";".join(
            sorted(
                {
                    f"{row.provider}/{row.source}"
                    for row in combined[["provider", "source"]].drop_duplicates().itertuples(index=False)
                }
            )
        )
    reasons: list[str] = []
    if combined.empty:
        reasons.append("no_provider_hourly_rows_observed")
    if training_rows < MIN_TRAINING_ROWS_PER_TICKER and symbol_type(symbol) == "stock":
        reasons.append(f"training_rows_below_{MIN_TRAINING_ROWS_PER_TICKER}")
    if evaluation_rows < MIN_EVALUATION_ROWS_PER_TICKER and symbol_type(symbol) == "stock":
        reasons.append(f"evaluation_rows_below_{MIN_EVALUATION_ROWS_PER_TICKER}")
    usable = not reasons and symbol_type(symbol) == "stock"
    if symbol == "VNINDEX":
        usable = not combined.empty and evaluation_rows >= MIN_EVALUATION_ROWS_PER_TICKER
    if symbol in OPTIONAL_INDEX_CODES:
        usable = not combined.empty
    summary = {
        "ticker": symbol,
        "asset_type": symbol_type(symbol),
        "listing_date_used": listing_date_used,
        "requested_start": timestamp_text(requested_start),
        "provider": provider,
        "chunks_attempted": chunks_attempted,
        "chunks_succeeded": chunks_succeeded,
        "chunks_failed": chunks_failed,
        "total_rows": int(len(combined)),
        "first_datetime": first_ts,
        "last_datetime": last_ts,
        "training_rows_before_cutoff": training_rows,
        "evaluation_rows_after_2025_01_01": evaluation_rows,
        "usable": str(bool(usable)).lower(),
        "normalized_cache_path": rel(normalized_cache_path(symbol)),
        "missing_reason": "; ".join(reasons),
    }
    return summary, attempts, failures


def write_report(path: Path, rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    stocks = [row for row in rows if row.get("asset_type") == "stock"]
    usable_stocks = [row for row in stocks if row.get("usable") == "true"]
    actual_eval_candidates = [
        pd.to_datetime(row.get("last_datetime", ""), errors="coerce")
        for row in usable_stocks
        if row.get("last_datetime")
    ]
    actual_eval_candidates = [pd.Timestamp(item) for item in actual_eval_candidates if not pd.isna(item)]
    actual_eval_end = timestamp_text(min(actual_eval_candidates)) if len(actual_eval_candidates) == 30 else ""
    vnindex = next((row for row in rows if row.get("ticker") == "VNINDEX"), {})
    vn30index = next((row for row in rows if row.get("ticker") == "VN30INDEX"), {})
    vnxall = next((row for row in rows if row.get("ticker") == "VNXALL"), {})
    package_rows = []
    for package_row in package_status_rows():
        cleaned = dict(package_row)
        origin = str(cleaned.get("origin", ""))
        cleaned["origin"] = Path(origin).name if origin else ""
        package_rows.append(cleaned)
    content = [
        "# VN30 Hourly Listing-Aware vnstock Fetch Summary",
        "",
        "## Scope",
        "",
        "- Universe: frozen VN30 30 tickers.",
        "- Frequency: hourly only.",
        "- Provider path: vnstock_data if importable, otherwise legacy vnstock.",
        f"- Raw chunk directory: `{rel(RAW_FETCH_DIR)}`.",
        f"- Provider attempt log: `{rel(REPORT_ROOT / 'fetch' / 'vn30_listing_aware_provider_attempt_log.csv')}` for completed provider-call runs; persisted raw chunks and normalized cache are summarized separately after interrupted runs.",
        "- Per-ticker start rule: max(first trading/listing date, first provider-available hourly timestamp).",
        "- Missing pre-listing hours are not required, filled, or synthesized.",
        "",
        "## Package Detection",
        "",
        markdown_table(["package", "installed", "version", "origin"], package_rows),
        "",
        "## Gate Snapshot",
        "",
        f"- Usable VN30 stocks from fetch summary: {len(usable_stocks)}/30.",
        f"- actual_eval_end candidate: {actual_eval_end or 'not available'}.",
        f"- VNINDEX fetched/usable: fetched={bool(vnindex.get('total_rows') and str(vnindex.get('total_rows')) != '0')}, usable={vnindex.get('usable') == 'true'}.",
        f"- VN30INDEX support: {vn30index.get('usable') == 'true'}.",
        f"- VNXALL support: {vnxall.get('usable') == 'true'}.",
        "",
        "## Per-Symbol Summary",
        "",
        markdown_table(
            [
                "ticker",
                "asset_type",
                "listing_date_used",
                "requested_start",
                "provider",
                "total_rows",
                "first_datetime",
                "last_datetime",
                "training_rows_before_cutoff",
                "evaluation_rows_after_2025_01_01",
                "usable",
                "missing_reason",
            ],
            rows,
            max_rows=80,
        ),
        "",
        "## Failure Preview",
        "",
        markdown_table(["ticker", "asset_type", "chunk_start", "chunk_end", "chunk_level", "failure_reason"], failures, max_rows=120)
        if failures
        else "No chunk failures were logged.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def summarize_existing_symbol(symbol: str, listing_rows: dict[str, dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    requested_start = requested_start_for(symbol, listing_rows)
    listing_date_used = "" if symbol_type(symbol) == "index" else timestamp_text(requested_start) if requested_start > TRAIN_START else ""
    frame = read_normalized_symbol(symbol)
    raw_dir = RAW_FETCH_DIR / symbol
    raw_count = len(list(raw_dir.glob("*.csv"))) if raw_dir.exists() else 0
    timestamps = pd.to_datetime(frame["datetime"], errors="coerce").dropna() if not frame.empty else pd.Series(dtype="datetime64[ns]")
    fetch_end = current_fetch_end()
    training_rows = int(((timestamps <= TRAIN_CUTOFF) & (timestamps >= TRAIN_START)).sum()) if not timestamps.empty else 0
    evaluation_rows = int(((timestamps >= EVAL_START) & (timestamps <= fetch_end)).sum()) if not timestamps.empty else 0
    provider = ""
    if not frame.empty:
        provider = ";".join(
            sorted(
                {
                    f"{row.provider}/{row.source}"
                    for row in frame[["provider", "source"]].drop_duplicates().itertuples(index=False)
                }
            )
        )
    reasons: list[str] = []
    if frame.empty:
        reasons.append("no_provider_hourly_rows_observed")
    if symbol_type(symbol) == "stock" and training_rows < MIN_TRAINING_ROWS_PER_TICKER:
        reasons.append(f"training_rows_below_{MIN_TRAINING_ROWS_PER_TICKER}")
    if symbol_type(symbol) == "stock" and evaluation_rows < MIN_EVALUATION_ROWS_PER_TICKER:
        reasons.append(f"evaluation_rows_below_{MIN_EVALUATION_ROWS_PER_TICKER}")
    usable = not reasons and symbol_type(symbol) == "stock"
    if symbol == "VNINDEX":
        usable = not frame.empty and evaluation_rows >= MIN_EVALUATION_ROWS_PER_TICKER
    if symbol in OPTIONAL_INDEX_CODES:
        usable = not frame.empty
    summary = {
        "ticker": symbol,
        "asset_type": symbol_type(symbol),
        "listing_date_used": listing_date_used,
        "requested_start": timestamp_text(requested_start),
        "provider": provider,
        "chunks_attempted": raw_count,
        "chunks_succeeded": raw_count if not frame.empty else 0,
        "chunks_failed": 0,
        "total_rows": int(len(frame)),
        "first_datetime": timestamp_text(timestamps.min()) if not timestamps.empty else "",
        "last_datetime": timestamp_text(timestamps.max()) if not timestamps.empty else "",
        "training_rows_before_cutoff": training_rows,
        "evaluation_rows_after_2025_01_01": evaluation_rows,
        "usable": str(bool(usable)).lower(),
        "normalized_cache_path": rel(normalized_cache_path(symbol)),
        "missing_reason": "; ".join(reasons),
    }
    failures: list[dict[str, Any]] = []
    if frame.empty:
        failures.append(
            {
                "ticker": symbol,
                "asset_type": symbol_type(symbol),
                "chunk_start": timestamp_text(requested_start),
                "chunk_end": timestamp_text(fetch_end),
                "chunk_level": "summarize_existing_only",
                "failure_reason": "no normalized listing-aware cache file exists",
            }
        )
    return summary, failures


def main() -> int:
    args = parse_args()
    listing_rows = read_listing_dates()
    output_dir = REPORT_ROOT / "fetch"
    output_dir.mkdir(parents=True, exist_ok=True)
    symbols = selected_symbols(args)
    summaries: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if args.summarize_existing_only:
        for symbol in symbols:
            summary, symbol_failures = summarize_existing_symbol(symbol, listing_rows)
            summaries.append(summary)
            failures.extend(symbol_failures)
    else:
        for symbol in symbols:
            summary, symbol_attempts, symbol_failures = fetch_symbol(symbol, args, listing_rows)
            summaries.append(summary)
            attempts.extend(symbol_attempts)
            failures.extend(symbol_failures)

    summary_path = output_dir / "vn30_listing_aware_fetch_summary.csv"
    report_path = output_dir / "vn30_listing_aware_fetch_summary.md"
    failure_path = output_dir / "vn30_listing_aware_fetch_failures.csv"
    attempt_path = output_dir / "vn30_listing_aware_provider_attempt_log.csv"
    write_csv(summary_path, summaries, fieldnames=SUMMARY_COLUMNS)
    write_csv(failure_path, failures, fieldnames=FAILURE_COLUMNS)
    if attempts or not attempt_path.exists():
        write_csv(attempt_path, attempts, fieldnames=ATTEMPT_COLUMNS)
    write_report(report_path, summaries, failures)
    print(f"VN30 listing-aware hourly fetch complete: symbols={len(symbols)} report={rel(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
