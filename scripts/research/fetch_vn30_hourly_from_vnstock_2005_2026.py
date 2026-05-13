"""Fetch frozen VN30 hourly data and VNINDEX from vnstock/vnstock_data."""

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
    EVAL_END,
    TRAIN_START,
    VN30_TICKERS,
    markdown_table,
    read_universe,
    rel,
    timestamp_text,
    write_csv,
)
from scripts.research.vn30_hourly_vnstock_common import (  # noqa: E402
    ALL_INDEX_CODES,
    FETCH_REPORT_ROOT,
    NORMALIZED_COLUMNS,
    OPTIONAL_INDEX_CODES,
    RAW_FETCH_DIR,
    asset_type,
    coverage_flags,
    fetch_first_success,
    normalized_cache_path,
    package_status_rows,
    period_chunks,
    raw_chunk_path,
    read_normalized_symbol,
    symbol_requirement_start,
    write_normalized_symbol,
)


SUMMARY_COLUMNS = [
    "symbol",
    "asset_type",
    "provider_used",
    "chunks_attempted",
    "chunks_succeeded",
    "chunks_failed",
    "total_rows",
    "first_datetime",
    "last_datetime",
    "missing_start_gap",
    "missing_end_gap",
    "benchmark_candidate",
    "normalized_cache_path",
    "failure_reason",
]
FAILURE_COLUMNS = [
    "symbol",
    "asset_type",
    "chunk_start",
    "chunk_end",
    "chunk_level",
    "failure_reason",
]
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
    parser = argparse.ArgumentParser(description="Fetch VN30 hourly full-history bars from vnstock/vnstock_data.")
    parser.add_argument("--force", action="store_true", help="Refetch chunks even when raw chunk CSVs already exist.")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--backoff-seconds", type=float, default=1.0)
    parser.add_argument("--provider-timeout-seconds", type=float, default=20.0)
    parser.add_argument(
        "--symbols",
        default="",
        help="Optional comma-separated symbol subset for debugging. Empty means all frozen VN30 stocks plus indices.",
    )
    parser.add_argument("--skip-optional-indices", action="store_true")
    parser.add_argument(
        "--ignore-probe-gate",
        action="store_true",
        help="Ignore provider probe results and attempt full chunk fetching anyway.",
    )
    parser.add_argument(
        "--disable-required-start-probe-gate",
        action="store_true",
        help="Attempt all chunks even if the required-start probe fails.",
    )
    return parser.parse_args()


def selected_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols.strip():
        return [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    tickers = read_universe()
    if tickers != VN30_TICKERS:
        raise ValueError("Frozen VN30 ticker universe does not match the mandatory 30-ticker list.")
    indices = ["VNINDEX"] + ([] if args.skip_optional_indices else list(OPTIONAL_INDEX_CODES))
    return [*tickers, *indices]


def probe_gate_failure_reason() -> str:
    path = FETCH_REPORT_ROOT / "provider_probe" / "vnstock_hourly_provider_probe.csv"
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    stock_ok = any(
        row.get("symbol") in {"ACB", "HPG"} and str(row.get("success", "")).lower() == "true" for row in rows
    )
    vnindex_ok = any(row.get("symbol") == "VNINDEX" and str(row.get("success", "")).lower() == "true" for row in rows)
    if stock_ok and vnindex_ok:
        return ""
    return "provider_probe_did_not_find_hourly_support_for_both_stock_and_vnindex"


def no_provider_failure_reason() -> str:
    installed = [row for row in package_status_rows() if str(row.get("installed", "")).lower() == "true"]
    if installed:
        return ""
    return "neither_vnstock_data_nor_vnstock_is_installed"


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
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
    return frame[NORMALIZED_COLUMNS].reset_index(drop=True)


def write_raw_chunk(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output["datetime"] = pd.to_datetime(output["datetime"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    output[NORMALIZED_COLUMNS].to_csv(path, index=False)


def fetch_leaf_chunk(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    level: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    chunk_start = start.strftime("%Y-%m-%d")
    chunk_end = end.strftime("%Y-%m-%d")
    path = raw_chunk_path(symbol, start, end)
    if path.exists() and not args.force:
        cached = read_cached_chunk(symbol, start, end)
        if not cached.empty:
            return cached, [], {
                "symbol": symbol,
                "asset_type": asset_type(symbol),
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "chunk_level": level,
                "failure_reason": "",
            }

    frame, attempts = fetch_first_success(
        symbol,
        chunk_start,
        chunk_end,
        retries=max(1, int(args.retries)),
        backoff_seconds=max(0.0, float(args.backoff_seconds)),
        timeout_seconds=max(1.0, float(args.provider_timeout_seconds)),
    )
    for row in attempts:
        row["chunk_start"] = chunk_start
        row["chunk_end"] = chunk_end
        row["chunk_level"] = level
    if not frame.empty:
        end_exclusive = pd.Timestamp(end) + pd.Timedelta(days=1)
        frame = frame[
            (pd.to_datetime(frame["datetime"], errors="coerce") >= pd.Timestamp(start))
            & (pd.to_datetime(frame["datetime"], errors="coerce") < end_exclusive)
        ].copy()
    if not frame.empty:
        write_raw_chunk(path, frame)
        return frame, attempts, {
            "symbol": symbol,
            "asset_type": asset_type(symbol),
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "chunk_level": level,
            "failure_reason": "",
        }

    reason = "provider_returned_no_usable_hourly_rows"
    errors = [row.get("exception_message", "") for row in attempts if row.get("exception_message")]
    if errors:
        reason = errors[-1][:300]
    return pd.DataFrame(columns=NORMALIZED_COLUMNS), attempts, {
        "symbol": symbol,
        "asset_type": asset_type(symbol),
        "chunk_start": chunk_start,
        "chunk_end": chunk_end,
        "chunk_level": level,
        "failure_reason": reason,
    }


def fetch_period(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    level: str,
    args: argparse.Namespace,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]], int, int, int]:
    frames: list[pd.DataFrame] = []
    attempts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    attempted = 0
    succeeded = 0
    failed = 0

    frame, attempt_rows, status = fetch_leaf_chunk(symbol, start, end, level, args)
    attempted += 1
    attempts.extend(attempt_rows)
    if not frame.empty:
        frames.append(frame)
        succeeded += 1
        return frames, attempts, failures, attempted, succeeded, failed

    if level == "year":
        for child_start, child_end in period_chunks(start, end, "month"):
            child = fetch_period(symbol, child_start, child_end, "month", args)
            child_frames, child_attempts, child_failures, child_attempted, child_succeeded, child_failed = child
            frames.extend(child_frames)
            attempts.extend(child_attempts)
            failures.extend(child_failures)
            attempted += child_attempted
            succeeded += child_succeeded
            failed += child_failed
        return frames, attempts, failures, attempted, succeeded, failed

    if level == "month":
        for child_start, child_end in period_chunks(start, end, "small"):
            child = fetch_period(symbol, child_start, child_end, "small", args)
            child_frames, child_attempts, child_failures, child_attempted, child_succeeded, child_failed = child
            frames.extend(child_frames)
            attempts.extend(child_attempts)
            failures.extend(child_failures)
            attempted += child_attempted
            succeeded += child_succeeded
            failed += child_failed
        return frames, attempts, failures, attempted, succeeded, failed

    failures.append(status)
    failed += 1
    return frames, attempts, failures, attempted, succeeded, failed


def fetch_symbol(symbol: str, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    required_start = symbol_requirement_start(symbol)
    required_end = EVAL_END
    frames: list[pd.DataFrame] = []
    attempts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    attempted = succeeded = failed = 0

    if not args.disable_required_start_probe_gate:
        probe_end = min(required_start + pd.Timedelta(days=6), required_end)
        probe_frame, probe_attempts, probe_status = fetch_leaf_chunk(symbol, required_start, probe_end, "required_start_probe", args)
        attempted += 1
        attempts.extend(probe_attempts)
        if probe_frame.empty:
            failed += 1
            probe_status["failure_reason"] = f"required_start_probe_failed: {probe_status.get('failure_reason', '')}"
            failures.append(probe_status)
            summary = {
                "symbol": symbol,
                "asset_type": asset_type(symbol),
                "provider_used": "",
                "chunks_attempted": attempted,
                "chunks_succeeded": succeeded,
                "chunks_failed": failed,
                "total_rows": 0,
                "first_datetime": "",
                "last_datetime": "",
                "missing_start_gap": "true",
                "missing_end_gap": "true",
                "benchmark_candidate": "false",
                "normalized_cache_path": rel(normalized_cache_path(symbol)),
                "failure_reason": "required_start_probe_failed; full fetch not attempted for this symbol because full-history coverage cannot pass without required-start evidence",
            }
            return summary, attempts, failures
        frames.append(probe_frame)
        succeeded += 1

    for chunk_start, chunk_end in period_chunks(required_start, required_end, "year"):
        result = fetch_period(symbol, chunk_start, chunk_end, "year", args)
        child_frames, child_attempts, child_failures, child_attempted, child_succeeded, child_failed = result
        frames.extend(child_frames)
        attempts.extend(child_attempts)
        failures.extend(child_failures)
        attempted += child_attempted
        succeeded += child_succeeded
        failed += child_failed

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined["datetime"] = pd.to_datetime(combined["datetime"], errors="coerce")
        combined = combined.dropna(subset=["datetime"])
        combined = combined.sort_values(["ticker", "datetime"]).drop_duplicates(["ticker", "datetime"], keep="last")
        combined = combined[NORMALIZED_COLUMNS].reset_index(drop=True)
        write_normalized_symbol(symbol, combined)
    else:
        combined = pd.DataFrame(columns=NORMALIZED_COLUMNS)

    start_ok, end_ok, first_ts, last_ts = coverage_flags(combined, required_start, required_end)
    provider_used = ""
    if not combined.empty:
        provider_used = ";".join(
            sorted(
                {
                    f"{row.provider}/{row.source}"
                    for row in combined[["provider", "source"]].drop_duplicates().itertuples(index=False)
                }
            )
        )
    reasons: list[str] = []
    if combined.empty:
        reasons.append("no_usable_hourly_rows_fetched")
    if not start_ok:
        reasons.append("missing_required_start_coverage")
    if not end_ok:
        reasons.append("missing_required_end_coverage")
    if failures:
        reasons.append(f"{len(failures)} chunk failures")
    summary = {
        "symbol": symbol,
        "asset_type": asset_type(symbol),
        "provider_used": provider_used,
        "chunks_attempted": attempted,
        "chunks_succeeded": succeeded,
        "chunks_failed": failed,
        "total_rows": int(len(combined)),
        "first_datetime": first_ts,
        "last_datetime": last_ts,
        "missing_start_gap": str(not start_ok).lower(),
        "missing_end_gap": str(not end_ok).lower(),
        "benchmark_candidate": str(bool(start_ok and end_ok and not combined.empty)).lower(),
        "normalized_cache_path": rel(normalized_cache_path(symbol)),
        "failure_reason": "; ".join(reasons),
    }
    return summary, attempts, failures


def write_fast_failure(symbols: list[str], reason: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for symbol in symbols:
        summary = {
            "symbol": symbol,
            "asset_type": asset_type(symbol),
            "provider_used": "",
            "chunks_attempted": 0,
            "chunks_succeeded": 0,
            "chunks_failed": 0,
            "total_rows": 0,
            "first_datetime": "",
            "last_datetime": "",
            "missing_start_gap": "true",
            "missing_end_gap": "true",
            "benchmark_candidate": "false",
            "normalized_cache_path": rel(normalized_cache_path(symbol)),
            "failure_reason": reason,
        }
        summaries.append(summary)
        failures.append(
            {
                "symbol": symbol,
                "asset_type": asset_type(symbol),
                "chunk_start": symbol_requirement_start(symbol).strftime("%Y-%m-%d"),
                "chunk_end": EVAL_END.strftime("%Y-%m-%d"),
                "chunk_level": "full",
                "failure_reason": reason,
            }
        )
    return summaries, failures


def write_summary_report(path: Path, summaries: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    usable = [row for row in summaries if str(row.get("benchmark_candidate", "")).lower() == "true"]
    vn30_usable = [row for row in usable if row.get("asset_type") == "stock"]
    vnindex_row = next((row for row in summaries if row.get("symbol") == "VNINDEX"), {})
    content = [
        "# VN30 Hourly vnstock Fetch Summary",
        "",
        "## Scope",
        "",
        f"- Frozen VN30 stocks: {len(VN30_TICKERS)}.",
        f"- Required market index: VNINDEX.",
        f"- Optional exact-code index probes: {', '.join(code for code in ALL_INDEX_CODES if code != 'VNINDEX')}.",
        f"- Raw chunk directory: `{rel(RAW_FETCH_DIR)}`.",
        "- Frequency: hourly only.",
        "- Daily data and daily-to-hourly resampling are not used.",
        "- Missing bars are not forward-filled or synthesized.",
        "",
        "## Gate Snapshot",
        "",
        f"- Benchmark-candidate VN30 stocks: {len(vn30_usable)}/30.",
        f"- VNINDEX benchmark-candidate: {str(vnindex_row.get('benchmark_candidate', '')).lower() == 'true'}.",
        f"- Total chunk failures: {len(failures)}.",
        "",
        "## Per-Symbol Summary",
        "",
        markdown_table(
            [
                "symbol",
                "asset_type",
                "provider_used",
                "chunks_attempted",
                "chunks_succeeded",
                "chunks_failed",
                "total_rows",
                "first_datetime",
                "last_datetime",
                "benchmark_candidate",
                "failure_reason",
            ],
            summaries,
            max_rows=80,
        ),
        "",
        "## Failure Preview",
        "",
        markdown_table(
            ["symbol", "asset_type", "chunk_start", "chunk_end", "chunk_level", "failure_reason"],
            failures,
            max_rows=80,
        )
        if failures
        else "No chunk failures were logged.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = FETCH_REPORT_ROOT / "fetch"
    output_dir.mkdir(parents=True, exist_ok=True)
    symbols = selected_symbols(args)

    fast_failure_reason = no_provider_failure_reason()
    if not fast_failure_reason and not args.ignore_probe_gate:
        fast_failure_reason = probe_gate_failure_reason()

    all_attempts: list[dict[str, Any]] = []
    if fast_failure_reason:
        summaries, failures = write_fast_failure(symbols, fast_failure_reason)
    else:
        summaries = []
        failures = []
        for symbol in symbols:
            summary, attempts, symbol_failures = fetch_symbol(symbol, args)
            summaries.append(summary)
            all_attempts.extend(attempts)
            failures.extend(symbol_failures)

    summary_path = output_dir / "vn30_hourly_vnstock_fetch_summary.csv"
    report_path = output_dir / "vn30_hourly_vnstock_fetch_summary.md"
    failures_path = output_dir / "vn30_hourly_fetch_failures.csv"
    attempts_path = output_dir / "vn30_hourly_provider_attempt_log.csv"
    write_csv(summary_path, summaries, fieldnames=SUMMARY_COLUMNS)
    write_csv(failures_path, failures, fieldnames=FAILURE_COLUMNS)
    write_csv(attempts_path, all_attempts, fieldnames=ATTEMPT_COLUMNS)
    write_summary_report(report_path, summaries, failures)
    print(
        "VN30 hourly vnstock fetch complete: "
        f"symbols={len(symbols)} failures={len(failures)} report={rel(report_path)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
