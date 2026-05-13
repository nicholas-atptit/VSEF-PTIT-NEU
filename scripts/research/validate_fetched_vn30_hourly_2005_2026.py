"""Validate vnstock-fetched normalized VN30 hourly cache for the full design."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_hourly_common import (  # noqa: E402
    EVAL_END,
    EVAL_END_TEXT,
    EVAL_START,
    EVAL_START_TEXT,
    TRAIN_CUTOFF,
    TRAIN_CUTOFF_TEXT,
    TRAIN_START,
    TRAIN_START_TEXT,
    VN30_TICKERS,
    markdown_table,
    read_universe,
    rel,
    timestamp_text,
    write_csv,
)
from scripts.research.vn30_hourly_vnstock_common import (  # noqa: E402
    FETCH_REPORT_ROOT,
    INDEX_REQUIREMENTS,
    NORMALIZED_COLUMNS,
    OHLCV_COLUMNS,
    OPTIONAL_INDEX_CODES,
    REQUIRED_INDEX_CODES,
    asset_type,
    build_docx_notes,
    normalized_cache_path,
    read_normalized_symbol,
    validation_gate_passed,
)


FIELDNAMES = [
    "symbol",
    "asset_type",
    "gate_required",
    "optional_supported",
    "cache_path",
    "file_exists",
    "row_count",
    "first_datetime",
    "last_datetime",
    "required_start",
    "required_end",
    "training_rows",
    "evaluation_rows",
    "duplicate_datetime_count",
    "non_hourly_timestamp_count",
    "ohlcv_numeric_pass",
    "positive_price_pass",
    "volume_nonnegative_pass",
    "sorted_datetime_pass",
    "training_coverage_pass",
    "evaluation_coverage_pass",
    "benchmark_usable",
    "failure_reason",
]


def bool_text(value: bool) -> str:
    return str(bool(value)).lower()


def validate_symbol(symbol: str, *, required: bool) -> dict[str, Any]:
    path = normalized_cache_path(symbol)
    file_exists = path.exists()
    required_start = INDEX_REQUIREMENTS.get(symbol, TRAIN_START)
    required_end = EVAL_END
    row = {
        "symbol": symbol,
        "asset_type": asset_type(symbol),
        "gate_required": bool_text(required),
        "optional_supported": bool_text(file_exists) if symbol in OPTIONAL_INDEX_CODES else "",
        "cache_path": rel(path),
        "file_exists": bool_text(file_exists),
        "row_count": 0,
        "first_datetime": "",
        "last_datetime": "",
        "required_start": timestamp_text(required_start),
        "required_end": timestamp_text(required_end),
        "training_rows": 0,
        "evaluation_rows": 0,
        "duplicate_datetime_count": "",
        "non_hourly_timestamp_count": "",
        "ohlcv_numeric_pass": "false",
        "positive_price_pass": "false",
        "volume_nonnegative_pass": "false",
        "sorted_datetime_pass": "false",
        "training_coverage_pass": "false",
        "evaluation_coverage_pass": "false",
        "benchmark_usable": "false",
        "failure_reason": "",
    }
    if not file_exists:
        row["failure_reason"] = "required_cache_file_missing" if required else "optional_index_not_fetched_or_unsupported"
        return row

    try:
        raw = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        row["failure_reason"] = f"cache_read_failed: {type(exc).__name__}: {exc}"
        return row

    missing_columns = [column for column in NORMALIZED_COLUMNS if column not in raw.columns]
    if missing_columns:
        row["failure_reason"] = f"missing_required_columns: {missing_columns}"
        return row

    frame = raw.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    for column in OHLCV_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    symbol_frame = frame[frame["ticker"].eq(symbol)].copy()
    symbol_frame = symbol_frame.dropna(subset=["datetime"])
    duplicate_count = int(symbol_frame.duplicated(["datetime"]).sum())
    sorted_pass = symbol_frame["datetime"].is_monotonic_increasing if not symbol_frame.empty else False
    non_hourly = int(
        (
            (symbol_frame["datetime"].dt.minute.fillna(-1).astype(int) != 0)
            | (symbol_frame["datetime"].dt.second.fillna(-1).astype(int) != 0)
        ).sum()
    )
    numeric_pass = bool(not symbol_frame[OHLCV_COLUMNS].isna().any().any()) if not symbol_frame.empty else False
    positive_price_pass = bool((symbol_frame[["open", "high", "low", "close"]] > 0).all().all()) if not symbol_frame.empty else False
    volume_pass = bool((symbol_frame["volume"] >= 0).all()) if not symbol_frame.empty else False
    if not symbol_frame.empty:
        first_ts = pd.Timestamp(symbol_frame["datetime"].min())
        last_ts = pd.Timestamp(symbol_frame["datetime"].max())
        training_rows = int(((symbol_frame["datetime"] >= TRAIN_START) & (symbol_frame["datetime"] <= TRAIN_CUTOFF)).sum())
        evaluation_rows = int(((symbol_frame["datetime"] >= EVAL_START) & (symbol_frame["datetime"] <= EVAL_END)).sum())
        training_coverage = (
            first_ts <= required_start + pd.Timedelta(days=10)
            and last_ts >= TRAIN_CUTOFF - pd.Timedelta(days=7)
            and training_rows > 0
        )
        evaluation_coverage = (
            last_ts >= EVAL_END - pd.Timedelta(days=7)
            and evaluation_rows > 0
            and first_ts <= EVAL_START + pd.Timedelta(days=10)
        )
        row.update(
            {
                "row_count": int(len(symbol_frame)),
                "first_datetime": timestamp_text(first_ts),
                "last_datetime": timestamp_text(last_ts),
                "training_rows": training_rows,
                "evaluation_rows": evaluation_rows,
            }
        )
    else:
        training_coverage = False
        evaluation_coverage = False

    quality_pass = numeric_pass and positive_price_pass and volume_pass and duplicate_count == 0 and non_hourly == 0 and sorted_pass
    benchmark_usable = bool(quality_pass and training_coverage and evaluation_coverage)
    reasons: list[str] = []
    if symbol_frame.empty:
        reasons.append("no_rows_for_symbol")
    if duplicate_count:
        reasons.append(f"duplicate_datetime_rows={duplicate_count}")
    if non_hourly:
        reasons.append(f"non_hourly_timestamps={non_hourly}")
    if not numeric_pass:
        reasons.append("ohlcv_numeric_validation_failed")
    if not positive_price_pass:
        reasons.append("missing_zero_or_negative_prices")
    if not volume_pass:
        reasons.append("negative_volume")
    if not sorted_pass:
        reasons.append("timestamps_not_sorted")
    if not training_coverage:
        reasons.append("training_coverage_missing")
    if not evaluation_coverage:
        reasons.append("evaluation_coverage_missing")

    row.update(
        {
            "duplicate_datetime_count": duplicate_count,
            "non_hourly_timestamp_count": non_hourly,
            "ohlcv_numeric_pass": bool_text(numeric_pass),
            "positive_price_pass": bool_text(positive_price_pass),
            "volume_nonnegative_pass": bool_text(volume_pass),
            "sorted_datetime_pass": bool_text(sorted_pass),
            "training_coverage_pass": bool_text(training_coverage),
            "evaluation_coverage_pass": bool_text(evaluation_coverage),
            "benchmark_usable": bool_text(benchmark_usable),
            "failure_reason": "; ".join(reasons),
        }
    )
    return row


def validation_rows() -> list[dict[str, Any]]:
    tickers = read_universe()
    if tickers != VN30_TICKERS:
        raise ValueError("Frozen VN30 ticker universe does not match the mandatory list.")
    rows: list[dict[str, Any]] = []
    rows.extend(validate_symbol(ticker, required=True) for ticker in tickers)
    rows.extend(validate_symbol(index_code, required=True) for index_code in REQUIRED_INDEX_CODES)
    rows.extend(validate_symbol(index_code, required=False) for index_code in OPTIONAL_INDEX_CODES)
    return rows


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    gate_pass = validation_gate_passed(rows)
    stocks = [row for row in rows if row.get("asset_type") == "stock"]
    stock_usable = [row for row in stocks if row.get("benchmark_usable") == "true"]
    vnindex = next((row for row in rows if row.get("symbol") == "VNINDEX"), {})
    vn30index = next((row for row in rows if row.get("symbol") == "VN30INDEX"), {})
    vnxall = next((row for row in rows if row.get("symbol") == "VNXALL"), {})
    failed = [row for row in rows if row.get("gate_required") == "true" and row.get("benchmark_usable") != "true"]
    content = [
        "# VN30 Hourly vnstock Fetched Data Validation",
        "",
        "## Gate Decision",
        "",
        f"- Full fetched stock+VNINDEX gate passed: {gate_pass}.",
        f"- Benchmark-usable VN30 stocks: {len(stock_usable)}/30.",
        f"- VNINDEX benchmark-usable: {vnindex.get('benchmark_usable') == 'true'}.",
        f"- VN30INDEX support/usable if fetched: supported={vn30index.get('optional_supported')}, usable={vn30index.get('benchmark_usable')}.",
        f"- VNXALL support/usable if fetched: supported={vnxall.get('optional_supported')}, usable={vnxall.get('benchmark_usable')}.",
        "",
        "## Required Coverage",
        "",
        f"- VN30 stocks: {TRAIN_START_TEXT} to {EVAL_END_TEXT}; train cutoff {TRAIN_CUTOFF_TEXT}.",
        f"- VNINDEX: {TRAIN_START_TEXT} to {EVAL_END_TEXT}.",
        "- VN30INDEX optional context if fetched: 2012-02-06 00:00:00 to 2026-05-31 23:59:59.",
        "- VNXALL optional context if fetched: 2016-10-24 00:00:00 to 2026-05-31 23:59:59.",
        f"- Common evaluation/comparison window: {EVAL_START_TEXT} to {EVAL_END_TEXT}.",
        "- Optional VN30INDEX/VNXALL absence does not fail the stock+VNINDEX gate.",
        "",
        "## Required Failures",
        "",
        markdown_table(
            [
                "symbol",
                "asset_type",
                "row_count",
                "first_datetime",
                "last_datetime",
                "training_rows",
                "evaluation_rows",
                "benchmark_usable",
                "failure_reason",
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
                "row_count",
                "first_datetime",
                "last_datetime",
                "benchmark_usable",
                "failure_reason",
            ],
            rows,
            max_rows=80,
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> int:
    output_dir = FETCH_REPORT_ROOT / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = validation_rows()
    csv_path = output_dir / "vn30_fetched_hourly_validation.csv"
    report_path = output_dir / "vn30_fetched_hourly_validation.md"
    write_csv(csv_path, rows, fieldnames=FIELDNAMES)
    write_report(report_path, rows)
    build_docx_notes(paper_exists=False, validation_rows=rows)
    gate = validation_gate_passed(rows)
    print(f"VN30 fetched hourly validation complete: gate_passed={gate} report={rel(report_path)}")
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
