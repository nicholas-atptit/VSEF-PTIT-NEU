"""Validate combined external VN30 stock and market-index hourly data."""

from __future__ import annotations

import argparse
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


OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_external_data_validation"
STOCK_OUTPUT = OUTPUT_DIR / "vn30_external_stock_hourly_validation.csv"
INDEX_OUTPUT = OUTPUT_DIR / "vn30_external_index_hourly_validation.csv"
COMBINED_REPORT = OUTPUT_DIR / "vn30_external_combined_validation.md"

REQUIRED_COLUMNS = ["timestamp", "ticker", "index_code", "open", "high", "low", "close", "volume"]
OPTIONAL_COLUMNS = ["source_status"]
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
ALLOWED_SOURCE_STATUS = {"official_live", "vendor_backfilled", "vendor_reconstructed", "unknown"}
PRE_START_ALLOWED_SOURCE_STATUS = {"vendor_backfilled", "vendor_reconstructed"}
INDEX_REQUIREMENTS = {
    "VNINDEX": pd.Timestamp("2005-01-01 00:00:00"),
    "VN30INDEX": pd.Timestamp("2012-02-06 00:00:00"),
    "VNXALL": pd.Timestamp("2016-10-24 00:00:00"),
}

STOCK_ROW_COLUMNS = [
    "ticker",
    "present",
    "required_start",
    "required_end",
    "first_timestamp",
    "last_timestamp",
    "row_count",
    "training_rows",
    "evaluation_rows",
    "timestamp_parse_errors",
    "hourly_alignment_errors",
    "sorted_by_timestamp",
    "duplicate_timestamps",
    "numeric_errors",
    "missing_price_rows",
    "zero_or_negative_price_rows",
    "ohlc_consistency_errors",
    "missing_volume_rows",
    "negative_volume_rows",
    "zero_volume_rows",
    "coverage_ratio_vs_union",
    "missing_period_count_vs_union",
    "missing_period_examples",
    "benchmark_usable",
    "blocking_reasons",
]
INDEX_ROW_COLUMNS = [
    "index_code",
    "present",
    "required_start",
    "required_end",
    "first_timestamp",
    "last_timestamp",
    "row_count",
    "required_range_rows",
    "evaluation_rows",
    "pre_start_rows",
    "invalid_pre_start_source_status_rows",
    "timestamp_parse_errors",
    "hourly_alignment_errors",
    "sorted_by_timestamp",
    "duplicate_timestamps",
    "numeric_errors",
    "missing_price_rows",
    "zero_or_negative_price_rows",
    "ohlc_consistency_errors",
    "missing_volume_rows",
    "negative_volume_rows",
    "zero_volume_rows",
    "invalid_source_status_rows",
    "coverage_ratio_vs_union",
    "missing_period_count_vs_union",
    "missing_period_examples",
    "benchmark_usable",
    "blocking_reasons",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one combined external VN30 hourly stock/index CSV or Parquet dataset.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="CSV or Parquet with timestamp,ticker,index_code,open,high,low,close,volume.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--min-coverage-ratio-vs-union",
        type=float,
        default=0.95,
        help="Minimum per-symbol coverage ratio against the dataset's inferred hourly timestamp union.",
    )
    parser.add_argument(
        "--coverage-start-tolerance-days",
        type=int,
        default=10,
        help="Tolerance for first expected trading bar after each required start when no calendar file is supplied.",
    )
    parser.add_argument(
        "--coverage-end-tolerance-days",
        type=int,
        default=7,
        help="Tolerance for last expected trading bar before 2026-05-31 when no calendar file is supplied.",
    )
    return parser.parse_args()


def read_external(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"External dataset does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input type '{suffix}'. Use CSV or Parquet.")


def normalize_code_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].fillna("").astype(str).str.upper().str.strip()


def normalize_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing_columns:
        return raw.copy(), missing_columns
    frame = raw.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["ticker"] = normalize_code_column(frame, "ticker")
    frame["index_code"] = normalize_code_column(frame, "index_code")
    if "source_status" not in frame.columns:
        frame["source_status"] = "unknown"
    frame["source_status"] = frame["source_status"].fillna("unknown").astype(str).str.lower().str.strip()
    frame.loc[frame["source_status"].eq(""), "source_status"] = "unknown"
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, []


def timestamp_set(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> set[pd.Timestamp]:
    if frame.empty or "timestamp" not in frame.columns:
        return set()
    values = frame.loc[(frame["timestamp"] >= start) & (frame["timestamp"] <= end), "timestamp"].dropna()
    return set(pd.Timestamp(value) for value in values)


def missing_period_examples(expected: set[pd.Timestamp], actual: set[pd.Timestamp], limit: int = 5) -> str:
    missing = sorted(expected.difference(actual))
    if not missing:
        return ""
    return "; ".join(timestamp_text(item) for item in missing[:limit])


def quality_metrics(group: pd.DataFrame, *, duplicate_key: str) -> dict[str, Any]:
    if group.empty:
        return {
            "timestamp_parse_errors": 0,
            "hourly_alignment_errors": 0,
            "sorted_by_timestamp": False,
            "duplicate_timestamps": 0,
            "numeric_errors": 0,
            "missing_price_rows": 0,
            "zero_or_negative_price_rows": 0,
            "ohlc_consistency_errors": 0,
            "missing_volume_rows": 0,
            "negative_volume_rows": 0,
            "zero_volume_rows": 0,
        }
    valid_timestamps = group["timestamp"].dropna()
    return {
        "timestamp_parse_errors": int(group["timestamp"].isna().sum()),
        "hourly_alignment_errors": int((valid_timestamps.dt.floor("h") != valid_timestamps).sum()),
        "sorted_by_timestamp": bool(valid_timestamps.is_monotonic_increasing),
        "duplicate_timestamps": int(group.duplicated([duplicate_key, "timestamp"], keep=False).sum()),
        "numeric_errors": int(group[NUMERIC_COLUMNS].isna().any(axis=1).sum()),
        "missing_price_rows": int(group[PRICE_COLUMNS].isna().any(axis=1).sum()),
        "zero_or_negative_price_rows": int((group[PRICE_COLUMNS] <= 0.0).any(axis=1).sum()),
        "ohlc_consistency_errors": int(
            (
                (group["high"] < group["low"])
                | (group["high"] < group["open"])
                | (group["high"] < group["close"])
                | (group["low"] > group["open"])
                | (group["low"] > group["close"])
            ).sum()
        ),
        "missing_volume_rows": int(group["volume"].isna().sum()),
        "negative_volume_rows": int((group["volume"] < 0.0).sum()),
        "zero_volume_rows": int((group["volume"] == 0.0).sum()),
    }


def quality_reasons(metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in (
        "timestamp_parse_errors",
        "hourly_alignment_errors",
        "duplicate_timestamps",
        "numeric_errors",
        "missing_price_rows",
        "zero_or_negative_price_rows",
        "ohlc_consistency_errors",
        "missing_volume_rows",
        "negative_volume_rows",
    ):
        value = int(metrics.get(key, 0))
        if value:
            reasons.append(f"{key}:{value}")
    if not bool(metrics.get("sorted_by_timestamp")):
        reasons.append("timestamps_not_sorted")
    return reasons


def empty_stock_row(ticker: str, expected_count: int, reasons: list[str]) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "present": False,
        "required_start": TRAIN_START_TEXT,
        "required_end": EVAL_END_TEXT,
        "first_timestamp": "",
        "last_timestamp": "",
        "row_count": 0,
        "training_rows": 0,
        "evaluation_rows": 0,
        "timestamp_parse_errors": 0,
        "hourly_alignment_errors": 0,
        "sorted_by_timestamp": False,
        "duplicate_timestamps": 0,
        "numeric_errors": 0,
        "missing_price_rows": 0,
        "zero_or_negative_price_rows": 0,
        "ohlc_consistency_errors": 0,
        "missing_volume_rows": 0,
        "negative_volume_rows": 0,
        "zero_volume_rows": 0,
        "coverage_ratio_vs_union": 0.0,
        "missing_period_count_vs_union": expected_count,
        "missing_period_examples": "",
        "benchmark_usable": False,
        "blocking_reasons": "; ".join(dict.fromkeys(reasons)),
    }


def validate_stock_ticker(
    ticker: str,
    frame: pd.DataFrame,
    expected_union: set[pd.Timestamp],
    *,
    min_coverage_ratio: float,
    start_tolerance_days: int,
    end_tolerance_days: int,
) -> dict[str, Any]:
    group = frame[frame["ticker"].eq(ticker)].copy() if "ticker" in frame.columns else pd.DataFrame()
    if group.empty:
        return empty_stock_row(ticker, len(expected_union), ["ticker_missing"])

    metrics = quality_metrics(group, duplicate_key="ticker")
    valid_timestamps = group["timestamp"].dropna()
    first_ts = pd.Timestamp(valid_timestamps.min()) if not valid_timestamps.empty else None
    last_ts = pd.Timestamp(valid_timestamps.max()) if not valid_timestamps.empty else None
    training_rows = int(((group["timestamp"] >= TRAIN_START) & (group["timestamp"] <= TRAIN_CUTOFF)).sum())
    evaluation_rows = int(((group["timestamp"] >= EVAL_START) & (group["timestamp"] <= EVAL_END)).sum())
    actual_set = set(pd.Timestamp(value) for value in valid_timestamps)
    missing_vs_union = expected_union.difference(actual_set)
    coverage_ratio = (len(actual_set.intersection(expected_union)) / len(expected_union)) if expected_union else 0.0

    reasons = quality_reasons(metrics)
    if first_ts is None or first_ts > TRAIN_START + pd.Timedelta(days=int(start_tolerance_days)):
        reasons.append(f"training_start_coverage_gap:first={timestamp_text(first_ts)}")
    if last_ts is None or last_ts < EVAL_END - pd.Timedelta(days=int(end_tolerance_days)):
        reasons.append(f"evaluation_end_coverage_gap:last={timestamp_text(last_ts)}")
    if training_rows == 0:
        reasons.append("no_training_rows")
    if evaluation_rows == 0:
        reasons.append("no_evaluation_rows")
    if coverage_ratio < float(min_coverage_ratio):
        reasons.append(f"coverage_ratio_below_min:{coverage_ratio:.4f}<{float(min_coverage_ratio):.4f}")

    return {
        "ticker": ticker,
        "present": True,
        "required_start": TRAIN_START_TEXT,
        "required_end": EVAL_END_TEXT,
        "first_timestamp": timestamp_text(first_ts),
        "last_timestamp": timestamp_text(last_ts),
        "row_count": int(len(group)),
        "training_rows": training_rows,
        "evaluation_rows": evaluation_rows,
        **metrics,
        "coverage_ratio_vs_union": coverage_ratio,
        "missing_period_count_vs_union": int(len(missing_vs_union)),
        "missing_period_examples": missing_period_examples(expected_union, actual_set),
        "benchmark_usable": not reasons,
        "blocking_reasons": "usable" if not reasons else "; ".join(dict.fromkeys(reasons)),
    }


def empty_index_row(index_code: str, required_start: pd.Timestamp, expected_count: int, reasons: list[str]) -> dict[str, Any]:
    return {
        "index_code": index_code,
        "present": False,
        "required_start": timestamp_text(required_start),
        "required_end": EVAL_END_TEXT,
        "first_timestamp": "",
        "last_timestamp": "",
        "row_count": 0,
        "required_range_rows": 0,
        "evaluation_rows": 0,
        "pre_start_rows": 0,
        "invalid_pre_start_source_status_rows": 0,
        "timestamp_parse_errors": 0,
        "hourly_alignment_errors": 0,
        "sorted_by_timestamp": False,
        "duplicate_timestamps": 0,
        "numeric_errors": 0,
        "missing_price_rows": 0,
        "zero_or_negative_price_rows": 0,
        "ohlc_consistency_errors": 0,
        "missing_volume_rows": 0,
        "negative_volume_rows": 0,
        "zero_volume_rows": 0,
        "invalid_source_status_rows": 0,
        "coverage_ratio_vs_union": 0.0,
        "missing_period_count_vs_union": expected_count,
        "missing_period_examples": "",
        "benchmark_usable": False,
        "blocking_reasons": "; ".join(dict.fromkeys(reasons)),
    }


def validate_index_code(
    index_code: str,
    frame: pd.DataFrame,
    expected_union: set[pd.Timestamp],
    *,
    min_coverage_ratio: float,
    start_tolerance_days: int,
    end_tolerance_days: int,
) -> dict[str, Any]:
    required_start = INDEX_REQUIREMENTS[index_code]
    group = frame[frame["index_code"].eq(index_code)].copy() if "index_code" in frame.columns else pd.DataFrame()
    if group.empty:
        return empty_index_row(index_code, required_start, len(expected_union), ["index_missing"])

    metrics = quality_metrics(group, duplicate_key="index_code")
    valid_timestamps = group["timestamp"].dropna()
    first_ts = pd.Timestamp(valid_timestamps.min()) if not valid_timestamps.empty else None
    last_ts = pd.Timestamp(valid_timestamps.max()) if not valid_timestamps.empty else None
    required_range_rows = int(((group["timestamp"] >= required_start) & (group["timestamp"] <= EVAL_END)).sum())
    evaluation_rows = int(((group["timestamp"] >= EVAL_START) & (group["timestamp"] <= EVAL_END)).sum())
    pre_start = group[group["timestamp"] < required_start].copy()
    pre_start_rows = int(len(pre_start))
    invalid_pre_start = int((~pre_start["source_status"].isin(PRE_START_ALLOWED_SOURCE_STATUS)).sum()) if pre_start_rows else 0
    invalid_source_status_rows = int((~group["source_status"].isin(ALLOWED_SOURCE_STATUS)).sum())
    required_actual_set = timestamp_set(group, required_start, EVAL_END)
    missing_vs_union = expected_union.difference(required_actual_set)
    coverage_ratio = (len(required_actual_set.intersection(expected_union)) / len(expected_union)) if expected_union else 0.0

    reasons = quality_reasons(metrics)
    if invalid_source_status_rows:
        reasons.append(f"invalid_source_status_rows:{invalid_source_status_rows}")
    if invalid_pre_start:
        reasons.append(f"invalid_pre_start_source_status_rows:{invalid_pre_start}")
    if first_ts is None or first_ts > required_start + pd.Timedelta(days=int(start_tolerance_days)):
        reasons.append(f"required_start_coverage_gap:first={timestamp_text(first_ts)}")
    if last_ts is None or last_ts < EVAL_END - pd.Timedelta(days=int(end_tolerance_days)):
        reasons.append(f"evaluation_end_coverage_gap:last={timestamp_text(last_ts)}")
    if required_range_rows == 0:
        reasons.append("no_required_range_rows")
    if evaluation_rows == 0:
        reasons.append("no_evaluation_rows")
    if coverage_ratio < float(min_coverage_ratio):
        reasons.append(f"coverage_ratio_below_min:{coverage_ratio:.4f}<{float(min_coverage_ratio):.4f}")

    return {
        "index_code": index_code,
        "present": True,
        "required_start": timestamp_text(required_start),
        "required_end": EVAL_END_TEXT,
        "first_timestamp": timestamp_text(first_ts),
        "last_timestamp": timestamp_text(last_ts),
        "row_count": int(len(group)),
        "required_range_rows": required_range_rows,
        "evaluation_rows": evaluation_rows,
        "pre_start_rows": pre_start_rows,
        "invalid_pre_start_source_status_rows": invalid_pre_start,
        **metrics,
        "invalid_source_status_rows": invalid_source_status_rows,
        "coverage_ratio_vs_union": coverage_ratio,
        "missing_period_count_vs_union": int(len(missing_vs_union)),
        "missing_period_examples": missing_period_examples(expected_union, required_actual_set),
        "benchmark_usable": not reasons,
        "blocking_reasons": "usable" if not reasons else "; ".join(dict.fromkeys(reasons)),
    }


def rows_usable(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(bool(row.get("benchmark_usable")) for row in rows)


def write_combined_report(
    path: Path,
    input_path: Path,
    missing_columns: list[str],
    stock_rows: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
    extra_tickers: list[str],
    extra_indices: list[str],
) -> None:
    stock_ready = rows_usable(stock_rows) and not missing_columns
    index_ready = rows_usable(index_rows) and not missing_columns
    combined_ready = stock_ready and index_ready
    stock_usable = [row["ticker"] for row in stock_rows if bool(row.get("benchmark_usable"))]
    index_usable = [row["index_code"] for row in index_rows if bool(row.get("benchmark_usable"))]
    content = [
        "# VN30 External Hourly Combined Validation",
        "",
        "## Scope",
        "",
        f"- Input path: `{rel(input_path)}`.",
        "- Input shape: one combined CSV or Parquet file with stock rows and index rows.",
        "- Required stock columns: `timestamp`, `ticker`, `index_code`, `open`, `high`, `low`, `close`, `volume`.",
        "- Required index columns: `timestamp`, `index_code`, `open`, `high`, `low`, `close`, `volume`.",
        "- Optional index column: `source_status`.",
        "- Daily data, daily-to-hourly resampling, VN100 evidence reuse, and fabricated data are not used.",
        "",
        "## Corrected Required Index Ranges",
        "",
        "- VNINDEX required start: 2005-01-01 00:00:00.",
        "- VN30INDEX required start: 2012-02-06 00:00:00.",
        "- VNXALL required start: 2016-10-24 00:00:00.",
        f"- Required end for all indices: {EVAL_END_TEXT}.",
        f"- Common evaluation/comparison window: {EVAL_START_TEXT} to {EVAL_END_TEXT}.",
        "- Pre-start VN30INDEX/VNXALL rows are optional vendor-backfilled/reconstructed rows and are not required for readiness.",
        "",
        "## Stock Requirement",
        "",
        f"- Frozen VN30 stock training/history range: {TRAIN_START_TEXT} to {TRAIN_CUTOFF_TEXT}.",
        f"- Frozen VN30 stock evaluation/comparison range: {EVAL_START_TEXT} to {EVAL_END_TEXT}.",
        f"- Benchmark-usable stock tickers: {len(stock_usable)} of 30.",
        f"- Stock readiness passed: {str(stock_ready).lower()}.",
        "",
        "## Index Requirement",
        "",
        f"- Benchmark-usable indices: {len(index_usable)} of 3.",
        f"- Index readiness passed: {str(index_ready).lower()}.",
        f"- Combined readiness passed: {str(combined_ready).lower()}.",
        "",
        "## Schema Result",
        "",
        f"- Required columns present: {str(not missing_columns).lower()}.",
        f"- Missing required columns: {', '.join(missing_columns) if missing_columns else 'None'}.",
        f"- Extra non-frozen stock tickers present: {', '.join(extra_tickers) if extra_tickers else 'None'}.",
        f"- Extra market index codes present: {', '.join(extra_indices) if extra_indices else 'None'}.",
        "",
        "## Stock Validation",
        "",
        markdown_table(
            [
                "ticker",
                "first_timestamp",
                "last_timestamp",
                "row_count",
                "training_rows",
                "evaluation_rows",
                "coverage_ratio_vs_union",
                "benchmark_usable",
                "blocking_reasons",
            ],
            stock_rows,
        ),
        "",
        "## Index Validation",
        "",
        markdown_table(
            [
                "index_code",
                "required_start",
                "first_timestamp",
                "last_timestamp",
                "required_range_rows",
                "evaluation_rows",
                "pre_start_rows",
                "invalid_pre_start_source_status_rows",
                "coverage_ratio_vs_union",
                "benchmark_usable",
                "blocking_reasons",
            ],
            index_rows,
        ),
        "",
        "## Blocking Boundary",
        "",
    ]
    if combined_ready:
        content.append("The external dataset passed stock, index, and common comparison/evaluation readiness gates.")
    else:
        content.append("The full VN30 hourly 2005-2026 benchmark must not proceed until stock readiness and corrected index readiness both pass.")
    content.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def empty_rows_for_missing_schema(missing_columns: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reasons = [f"required_columns_missing:{','.join(missing_columns)}"]
    stock_rows = [empty_stock_row(ticker, 0, reasons) for ticker in VN30_TICKERS]
    index_rows = [empty_index_row(code, start, 0, reasons) for code, start in INDEX_REQUIREMENTS.items()]
    return stock_rows, index_rows


def main() -> int:
    args = parse_args()
    tickers = read_universe()
    if tickers != VN30_TICKERS:
        raise ValueError("Frozen VN30 universe file does not match the mandatory 30-ticker list.")
    raw = read_external(args.input_path)
    frame, missing_columns = normalize_frame(raw)
    if missing_columns:
        stock_rows, index_rows = empty_rows_for_missing_schema(missing_columns)
        extra_tickers: list[str] = []
        extra_indices: list[str] = []
    else:
        stock_frame = frame[frame["ticker"].isin(tickers)].copy()
        index_frame = frame[frame["index_code"].isin(INDEX_REQUIREMENTS)].copy()
        extra_tickers = sorted(set(frame.loc[frame["ticker"].ne(""), "ticker"]).difference(tickers))
        extra_indices = sorted(set(frame.loc[frame["index_code"].ne(""), "index_code"]).difference(INDEX_REQUIREMENTS))
        stock_expected_union = timestamp_set(stock_frame, TRAIN_START, EVAL_END)
        stock_rows = [
            validate_stock_ticker(
                ticker,
                stock_frame,
                stock_expected_union,
                min_coverage_ratio=args.min_coverage_ratio_vs_union,
                start_tolerance_days=args.coverage_start_tolerance_days,
                end_tolerance_days=args.coverage_end_tolerance_days,
            )
            for ticker in tickers
        ]
        index_rows = []
        for index_code, required_start in INDEX_REQUIREMENTS.items():
            index_expected_union = timestamp_set(index_frame, required_start, EVAL_END)
            index_rows.append(
                validate_index_code(
                    index_code,
                    index_frame,
                    index_expected_union,
                    min_coverage_ratio=args.min_coverage_ratio_vs_union,
                    start_tolerance_days=args.coverage_start_tolerance_days,
                    end_tolerance_days=args.coverage_end_tolerance_days,
                )
            )

    output_dir = args.output_dir
    stock_path = output_dir / STOCK_OUTPUT.name
    index_path = output_dir / INDEX_OUTPUT.name
    report_path = output_dir / COMBINED_REPORT.name
    write_csv(stock_path, stock_rows, fieldnames=STOCK_ROW_COLUMNS)
    write_csv(index_path, index_rows, fieldnames=INDEX_ROW_COLUMNS)
    write_combined_report(report_path, args.input_path, missing_columns, stock_rows, index_rows, extra_tickers, extra_indices)
    stock_ready = rows_usable(stock_rows) and not missing_columns
    index_ready = rows_usable(index_rows) and not missing_columns
    combined_ready = stock_ready and index_ready
    print(
        "VN30 external hourly combined validation complete: "
        f"stock_ready={str(stock_ready).lower()} index_ready={str(index_ready).lower()} "
        f"combined_ready={str(combined_ready).lower()} report={rel(report_path)}"
    )
    if not combined_ready:
        print(
            "VN30 external hourly validation failed: stock readiness and corrected index readiness must both pass.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
