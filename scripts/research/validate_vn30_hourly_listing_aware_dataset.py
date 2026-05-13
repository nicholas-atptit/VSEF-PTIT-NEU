"""Validate listing-aware VN30 hourly cache fetched from vnstock/vnstock_data."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_hourly_common import (  # noqa: E402
    EVAL_START,
    EVAL_START_TEXT,
    TRAIN_CUTOFF,
    TRAIN_CUTOFF_TEXT,
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
    MISSING_EVIDENCE_PATH,
    NORMALIZED_COLUMNS,
    OHLCV_COLUMNS,
    OPTIONAL_INDEX_CODES,
    REPORT_ROOT,
    REQUESTED_EVAL_END,
    REQUESTED_EVAL_END_TEXT,
    compute_actual_eval_end,
    current_fetch_end,
    normalized_cache_path,
    read_listing_dates,
    requested_start_for,
    validation_gate_passed,
    write_docx_notes,
    write_missing_evidence_report,
)


FIELDNAMES = [
    "symbol",
    "asset_type",
    "gate_required",
    "fetched",
    "listing_date_used",
    "requested_start",
    "ticker_training_start",
    "actual_eval_end",
    "cache_path",
    "row_count",
    "first_datetime",
    "last_datetime",
    "training_rows",
    "evaluation_rows",
    "duplicate_datetime_count",
    "non_hourly_timestamp_count",
    "ohlcv_numeric_pass",
    "positive_price_pass",
    "volume_nonnegative_pass",
    "provider_source_recorded",
    "benchmark_usable",
    "missing_reason",
]


def bool_text(value: bool) -> str:
    return str(bool(value)).lower()


def load_clean_frame(symbol: str, requested_start: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    path = normalized_cache_path(symbol)
    if not path.exists():
        return pd.DataFrame(columns=NORMALIZED_COLUMNS), ["cache_file_missing"]
    try:
        raw = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS), [f"cache_read_failed: {type(exc).__name__}: {exc}"]
    missing_columns = [column for column in NORMALIZED_COLUMNS if column not in raw.columns]
    if missing_columns:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS), [f"missing_required_columns: {missing_columns}"]
    frame = raw.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame = frame[frame["ticker"] == symbol.upper()].copy()
    for column in OHLCV_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["datetime"])
    frame = frame[frame["datetime"] >= requested_start].copy()
    return frame.sort_values("datetime").reset_index(drop=True), []


def actual_eval_end_from_cache(listing_rows: dict[str, dict[str, str]]) -> str:
    last_values: list[pd.Timestamp] = []
    for ticker in VN30_TICKERS:
        requested_start = requested_start_for(ticker, listing_rows)
        frame, _errors = load_clean_frame(ticker, requested_start)
        if frame.empty:
            return ""
        last_values.append(pd.Timestamp(frame["datetime"].max()))
    if len(last_values) != 30:
        return ""
    return timestamp_text(min(min(last_values), REQUESTED_EVAL_END, current_fetch_end()))


def validate_symbol(symbol: str, listing_rows: dict[str, dict[str, str]], actual_eval_end: str) -> dict[str, Any]:
    requested_start = requested_start_for(symbol, listing_rows)
    gate_required = symbol not in OPTIONAL_INDEX_CODES
    listing_date_used = "" if symbol in ALL_INDEX_CODES else timestamp_text(requested_start) if requested_start > pd.Timestamp("2005-01-01") else ""
    row = {
        "symbol": symbol,
        "asset_type": "index" if symbol in ALL_INDEX_CODES else "stock",
        "gate_required": bool_text(gate_required),
        "fetched": "false",
        "listing_date_used": listing_date_used,
        "requested_start": timestamp_text(requested_start),
        "ticker_training_start": "",
        "actual_eval_end": actual_eval_end,
        "cache_path": rel(normalized_cache_path(symbol)),
        "row_count": 0,
        "first_datetime": "",
        "last_datetime": "",
        "training_rows": 0,
        "evaluation_rows": 0,
        "duplicate_datetime_count": "",
        "non_hourly_timestamp_count": "",
        "ohlcv_numeric_pass": "false",
        "positive_price_pass": "false",
        "volume_nonnegative_pass": "false",
        "provider_source_recorded": "false",
        "benchmark_usable": "false",
        "missing_reason": "",
    }
    frame, errors = load_clean_frame(symbol, requested_start)
    if errors:
        row["missing_reason"] = "; ".join(errors)
        return row
    row["fetched"] = bool_text(not frame.empty)
    if frame.empty:
        row["missing_reason"] = "no_rows_after_requested_start"
        return row

    duplicate_count = int(frame.duplicated(["datetime"]).sum())
    non_hourly = int(
        (
            (frame["datetime"].dt.minute.fillna(-1).astype(int) != 0)
            | (frame["datetime"].dt.second.fillna(-1).astype(int) != 0)
        ).sum()
    )
    numeric_pass = bool(not frame[OHLCV_COLUMNS].isna().any().any())
    positive_pass = bool((frame[["open", "high", "low", "close"]] > 0).all().all())
    volume_pass = bool((frame["volume"] >= 0).all())
    provider_source = bool(
        "provider" in frame.columns
        and "source" in frame.columns
        and frame["provider"].astype(str).str.strip().ne("").all()
        and frame["source"].astype(str).str.strip().ne("").all()
    )
    first_ts = pd.Timestamp(frame["datetime"].min())
    last_ts = pd.Timestamp(frame["datetime"].max())
    training_start = max(requested_start, first_ts)
    eval_end_ts = pd.to_datetime(actual_eval_end, errors="coerce")
    if pd.isna(eval_end_ts):
        eval_end_ts = min(last_ts, REQUESTED_EVAL_END, current_fetch_end())
    eval_end_ts = pd.Timestamp(eval_end_ts)
    training_rows = int(((frame["datetime"] >= training_start) & (frame["datetime"] <= TRAIN_CUTOFF)).sum())
    evaluation_rows = int(((frame["datetime"] >= EVAL_START) & (frame["datetime"] <= eval_end_ts)).sum())

    reasons: list[str] = []
    if duplicate_count:
        reasons.append(f"duplicate_datetime_rows={duplicate_count}")
    if non_hourly:
        reasons.append(f"non_hourly_timestamps={non_hourly}")
    if not numeric_pass:
        reasons.append("ohlcv_numeric_validation_failed")
    if not positive_pass:
        reasons.append("missing_zero_or_negative_prices")
    if not volume_pass:
        reasons.append("negative_volume")
    if not provider_source:
        reasons.append("provider_or_source_missing")
    if symbol not in ALL_INDEX_CODES and training_rows < MIN_TRAINING_ROWS_PER_TICKER:
        reasons.append(f"training_rows_below_{MIN_TRAINING_ROWS_PER_TICKER}")
    if symbol not in ALL_INDEX_CODES and evaluation_rows < MIN_EVALUATION_ROWS_PER_TICKER:
        reasons.append(f"evaluation_rows_below_{MIN_EVALUATION_ROWS_PER_TICKER}")
    if symbol == "VNINDEX" and evaluation_rows < MIN_EVALUATION_ROWS_PER_TICKER:
        reasons.append(f"vnindex_evaluation_rows_below_{MIN_EVALUATION_ROWS_PER_TICKER}")
    if symbol in OPTIONAL_INDEX_CODES and frame.empty:
        reasons.append("optional_index_not_fetched_or_unsupported")

    benchmark_usable = not reasons
    if symbol in OPTIONAL_INDEX_CODES and frame.empty:
        benchmark_usable = False

    row.update(
        {
            "row_count": int(len(frame)),
            "first_datetime": timestamp_text(first_ts),
            "last_datetime": timestamp_text(last_ts),
            "ticker_training_start": timestamp_text(training_start),
            "training_rows": training_rows,
            "evaluation_rows": evaluation_rows,
            "duplicate_datetime_count": duplicate_count,
            "non_hourly_timestamp_count": non_hourly,
            "ohlcv_numeric_pass": bool_text(numeric_pass),
            "positive_price_pass": bool_text(positive_pass),
            "volume_nonnegative_pass": bool_text(volume_pass),
            "provider_source_recorded": bool_text(provider_source),
            "benchmark_usable": bool_text(benchmark_usable),
            "missing_reason": "; ".join(reasons),
        }
    )
    return row


def build_rows() -> list[dict[str, Any]]:
    tickers = read_universe()
    if tickers != VN30_TICKERS:
        raise ValueError("Frozen VN30 ticker universe does not match the mandatory list.")
    listing_rows = read_listing_dates()
    actual_eval_end = actual_eval_end_from_cache(listing_rows)
    rows: list[dict[str, Any]] = []
    for symbol in [*VN30_TICKERS, "VNINDEX", *OPTIONAL_INDEX_CODES]:
        rows.append(validate_symbol(symbol, listing_rows, actual_eval_end))
    return rows


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    gate = validation_gate_passed(rows)
    stocks = [row for row in rows if row.get("asset_type") == "stock"]
    usable_stocks = [row for row in stocks if row.get("benchmark_usable") == "true"]
    actual_eval_end = compute_actual_eval_end(rows) or (rows[0].get("actual_eval_end", "") if rows else "")
    vnindex = next((row for row in rows if row.get("symbol") == "VNINDEX"), {})
    vn30index = next((row for row in rows if row.get("symbol") == "VN30INDEX"), {})
    vnxall = next((row for row in rows if row.get("symbol") == "VNXALL"), {})
    failed = [row for row in rows if row.get("gate_required") == "true" and row.get("benchmark_usable") != "true"]
    content = [
        "# VN30 Hourly Listing-Aware Validation",
        "",
        "## Gate Decision",
        "",
        f"- Listing-aware validation gate passed: {gate}.",
        f"- Usable VN30 stocks: {len(usable_stocks)}/30.",
        f"- actual_eval_end: {actual_eval_end or 'not available'}.",
        f"- VNINDEX fetched/usable: fetched={vnindex.get('fetched') == 'true'}, usable={vnindex.get('benchmark_usable') == 'true'}.",
        f"- VN30INDEX support: {vn30index.get('benchmark_usable') == 'true'}.",
        f"- VNXALL support: {vnxall.get('benchmark_usable') == 'true'}.",
        "",
        "## Thresholds and Rules",
        "",
        f"- Minimum training rows per stock: {MIN_TRAINING_ROWS_PER_TICKER}.",
        f"- Minimum evaluation rows per stock: {MIN_EVALUATION_ROWS_PER_TICKER}.",
        "- Per-ticker training start: max(first trading/listing date, first provider-available hourly timestamp).",
        f"- Training labels end at: {TRAIN_CUTOFF_TEXT}.",
        f"- Evaluation starts at: {EVAL_START_TEXT}.",
        f"- Requested evaluation end: {REQUESTED_EVAL_END_TEXT}.",
        "- actual_eval_end is computed from available provider timestamps, not assumed future data.",
        "- No daily data, daily-to-hourly resampling, VN100 evidence reuse, or fabricated bars are used.",
        "",
        "## Required Failures",
        "",
        markdown_table(
            [
                "symbol",
                "asset_type",
                "listing_date_used",
                "requested_start",
                "ticker_training_start",
                "first_datetime",
                "last_datetime",
                "training_rows",
                "evaluation_rows",
                "benchmark_usable",
                "missing_reason",
            ],
            failed,
            max_rows=80,
        )
        if failed
        else "No required validation failures.",
        "",
        "## Per-Symbol Validation",
        "",
        markdown_table(
            [
                "symbol",
                "asset_type",
                "gate_required",
                "fetched",
                "row_count",
                "first_datetime",
                "last_datetime",
                "training_rows",
                "evaluation_rows",
                "benchmark_usable",
                "missing_reason",
            ],
            rows,
            max_rows=80,
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> int:
    output_dir = REPORT_ROOT / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    csv_path = output_dir / "vn30_listing_aware_validation.csv"
    report_path = output_dir / "vn30_listing_aware_validation.md"
    write_csv(csv_path, rows, fieldnames=FIELDNAMES)
    write_report(report_path, rows)
    if not validation_gate_passed(rows):
        write_missing_evidence_report(MISSING_EVIDENCE_PATH, rows, source_script=Path(__file__).name)
        write_docx_notes(paper_exists=False, validation_rows=rows)
    print(f"VN30 listing-aware validation complete: gate_passed={validation_gate_passed(rows)} report={rel(report_path)}")
    return 0 if validation_gate_passed(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
