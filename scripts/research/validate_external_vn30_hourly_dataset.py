"""Validate an external frozen VN30 hourly dataset for the 2005-2026 design."""

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
CSV_OUTPUT = OUTPUT_DIR / "vn30_external_hourly_validation.csv"
MD_OUTPUT = OUTPUT_DIR / "vn30_external_hourly_validation.md"
REQUIRED_COLUMNS = ["timestamp", "ticker", "open", "high", "low", "close", "volume"]
OPTIONAL_COLUMNS = ["adjusted_close", "source", "exchange", "session", "corporate_action_flag"]
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
ROW_COLUMNS = [
    "ticker",
    "present",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an external VN30 hourly CSV or Parquet dataset.")
    parser.add_argument("input_path", type=Path, help="CSV or Parquet file with timestamp,ticker,open,high,low,close,volume.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--min-coverage-ratio-vs-union",
        type=float,
        default=0.95,
        help="Minimum per-ticker coverage ratio against the dataset's inferred hourly timestamp union.",
    )
    parser.add_argument(
        "--coverage-start-tolerance-days",
        type=int,
        default=10,
        help="Tolerance for the first expected trading bar after 2005-01-01 when no calendar file is supplied.",
    )
    parser.add_argument(
        "--coverage-end-tolerance-days",
        type=int,
        default=7,
        help="Tolerance for the last expected trading bar before 2026-05-31 when no calendar file is supplied.",
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


def normalize_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing_columns:
        return raw.copy(), missing_columns
    frame = raw.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, []


def missing_period_examples(expected: set[pd.Timestamp], actual: set[pd.Timestamp], limit: int = 5) -> str:
    missing = sorted(expected.difference(actual))
    if not missing:
        return ""
    return "; ".join(timestamp_text(item) for item in missing[:limit])


def validate_ticker(
    ticker: str,
    frame: pd.DataFrame,
    expected_union: set[pd.Timestamp],
    *,
    min_coverage_ratio: float,
    start_tolerance_days: int,
    end_tolerance_days: int,
) -> dict[str, Any]:
    group = frame[frame["ticker"].eq(ticker)].copy() if "ticker" in frame.columns else pd.DataFrame()
    present = not group.empty
    reasons: list[str] = []
    if not present:
        reasons.append("ticker_missing")
        return {
            "ticker": ticker,
            "present": False,
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
            "missing_period_count_vs_union": len(expected_union),
            "missing_period_examples": "",
            "benchmark_usable": False,
            "blocking_reasons": "; ".join(reasons),
        }

    timestamp_parse_errors = int(group["timestamp"].isna().sum())
    valid_timestamps = group["timestamp"].dropna()
    first_ts = pd.Timestamp(valid_timestamps.min()) if not valid_timestamps.empty else None
    last_ts = pd.Timestamp(valid_timestamps.max()) if not valid_timestamps.empty else None
    hourly_alignment_errors = int((valid_timestamps.dt.floor("h") != valid_timestamps).sum())
    sorted_by_timestamp = bool(valid_timestamps.is_monotonic_increasing)
    duplicate_timestamps = int(group.duplicated(["ticker", "timestamp"], keep=False).sum())
    numeric_errors = int(group[NUMERIC_COLUMNS].isna().any(axis=1).sum())
    missing_price_rows = int(group[PRICE_COLUMNS].isna().any(axis=1).sum())
    zero_or_negative_price_rows = int((group[PRICE_COLUMNS] <= 0.0).any(axis=1).sum())
    ohlc_consistency_errors = int(
        (
            (group["high"] < group["low"])
            | (group["high"] < group["open"])
            | (group["high"] < group["close"])
            | (group["low"] > group["open"])
            | (group["low"] > group["close"])
        ).sum()
    )
    missing_volume_rows = int(group["volume"].isna().sum())
    negative_volume_rows = int((group["volume"] < 0.0).sum())
    zero_volume_rows = int((group["volume"] == 0.0).sum())
    training_rows = int(((group["timestamp"] >= TRAIN_START) & (group["timestamp"] <= TRAIN_CUTOFF)).sum())
    evaluation_rows = int(((group["timestamp"] >= EVAL_START) & (group["timestamp"] <= EVAL_END)).sum())
    actual_set = set(pd.Timestamp(value) for value in valid_timestamps)
    missing_vs_union = expected_union.difference(actual_set)
    coverage_ratio = (len(actual_set.intersection(expected_union)) / len(expected_union)) if expected_union else 0.0

    if timestamp_parse_errors:
        reasons.append(f"timestamp_parse_errors:{timestamp_parse_errors}")
    if hourly_alignment_errors:
        reasons.append(f"hourly_alignment_errors:{hourly_alignment_errors}")
    if not sorted_by_timestamp:
        reasons.append("timestamps_not_sorted")
    if duplicate_timestamps:
        reasons.append(f"duplicate_timestamps:{duplicate_timestamps}")
    if numeric_errors:
        reasons.append(f"numeric_errors:{numeric_errors}")
    if missing_price_rows:
        reasons.append(f"missing_price_rows:{missing_price_rows}")
    if zero_or_negative_price_rows:
        reasons.append(f"zero_or_negative_price_rows:{zero_or_negative_price_rows}")
    if ohlc_consistency_errors:
        reasons.append(f"ohlc_consistency_errors:{ohlc_consistency_errors}")
    if missing_volume_rows:
        reasons.append(f"missing_volume_rows:{missing_volume_rows}")
    if negative_volume_rows:
        reasons.append(f"negative_volume_rows:{negative_volume_rows}")
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
        "first_timestamp": timestamp_text(first_ts),
        "last_timestamp": timestamp_text(last_ts),
        "row_count": int(len(group)),
        "training_rows": training_rows,
        "evaluation_rows": evaluation_rows,
        "timestamp_parse_errors": timestamp_parse_errors,
        "hourly_alignment_errors": hourly_alignment_errors,
        "sorted_by_timestamp": sorted_by_timestamp,
        "duplicate_timestamps": duplicate_timestamps,
        "numeric_errors": numeric_errors,
        "missing_price_rows": missing_price_rows,
        "zero_or_negative_price_rows": zero_or_negative_price_rows,
        "ohlc_consistency_errors": ohlc_consistency_errors,
        "missing_volume_rows": missing_volume_rows,
        "negative_volume_rows": negative_volume_rows,
        "zero_volume_rows": zero_volume_rows,
        "coverage_ratio_vs_union": coverage_ratio,
        "missing_period_count_vs_union": int(len(missing_vs_union)),
        "missing_period_examples": missing_period_examples(expected_union, actual_set),
        "benchmark_usable": not reasons,
        "blocking_reasons": "usable" if not reasons else "; ".join(dict.fromkeys(reasons)),
    }


def write_report(path: Path, input_path: Path, missing_columns: list[str], rows: list[dict[str, Any]], extra_tickers: list[str]) -> None:
    usable = [row["ticker"] for row in rows if bool(row.get("benchmark_usable"))]
    failed = [row for row in rows if not bool(row.get("benchmark_usable"))]
    content = [
        "# VN30 External Hourly Data Validation",
        "",
        "## Scope",
        "",
        f"- Input path: `{rel(input_path)}`.",
        "- Universe: frozen VN30, exactly 30 tickers.",
        "- Frequency: hourly only.",
        f"- Training period: {TRAIN_START_TEXT} to {TRAIN_CUTOFF_TEXT}.",
        f"- Evaluation period: {EVAL_START_TEXT} to {EVAL_END_TEXT}.",
        "- No data fetching or fabrication is performed by this validator.",
        "",
        "## Result",
        "",
        f"- Required columns present: {str(not missing_columns).lower()}.",
        f"- Missing required columns: {', '.join(missing_columns) if missing_columns else 'None'}.",
        f"- Extra non-frozen tickers present: {', '.join(extra_tickers) if extra_tickers else 'None'}.",
        f"- Benchmark-usable tickers: {len(usable)} of 30.",
        f"- All-30 usability passed: {str(len(usable) == 30 and not missing_columns).lower()}.",
        "",
        "## Per-Ticker Validation",
        "",
        markdown_table(
            [
                "ticker",
                "first_timestamp",
                "last_timestamp",
                "row_count",
                "training_rows",
                "evaluation_rows",
                "duplicate_timestamps",
                "coverage_ratio_vs_union",
                "missing_period_count_vs_union",
                "benchmark_usable",
                "blocking_reasons",
            ],
            rows,
        ),
        "",
        "## Blocking Boundary",
        "",
    ]
    if len(usable) != 30 or missing_columns:
        content.append("The full VN30 hourly 2005-2026 benchmark must not proceed until all 30 frozen tickers are benchmark-usable.")
    else:
        content.append("The external dataset passed the all-30 validator gate. The next step is cache import and rerunning the 2005-2026 audit.")
    content.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> int:
    args = parse_args()
    tickers = read_universe()
    if tickers != VN30_TICKERS:
        raise ValueError("Frozen VN30 universe file does not match the mandatory 30-ticker list.")
    raw = read_external(args.input_path)
    frame, missing_columns = normalize_frame(raw)
    if missing_columns:
        rows = [
            validate_ticker(
                ticker,
                pd.DataFrame(columns=REQUIRED_COLUMNS),
                set(),
                min_coverage_ratio=args.min_coverage_ratio_vs_union,
                start_tolerance_days=args.coverage_start_tolerance_days,
                end_tolerance_days=args.coverage_end_tolerance_days,
            )
            for ticker in tickers
        ]
        extra_tickers: list[str] = []
    else:
        frame = frame[frame["ticker"].notna()].copy()
        extra_tickers = sorted(set(frame["ticker"].dropna().astype(str)).difference(tickers))
        expected_union = set(
            pd.Timestamp(value)
            for value in frame.loc[
                frame["ticker"].isin(tickers)
                & (frame["timestamp"] >= TRAIN_START)
                & (frame["timestamp"] <= EVAL_END),
                "timestamp",
            ].dropna()
        )
        rows = [
            validate_ticker(
                ticker,
                frame,
                expected_union,
                min_coverage_ratio=args.min_coverage_ratio_vs_union,
                start_tolerance_days=args.coverage_start_tolerance_days,
                end_tolerance_days=args.coverage_end_tolerance_days,
            )
            for ticker in tickers
        ]
    output_dir = args.output_dir
    csv_path = output_dir / "vn30_external_hourly_validation.csv"
    md_path = output_dir / "vn30_external_hourly_validation.md"
    write_csv(csv_path, rows, fieldnames=ROW_COLUMNS)
    write_report(md_path, args.input_path, missing_columns, rows, extra_tickers)
    usable_count = sum(1 for row in rows if bool(row.get("benchmark_usable")))
    print(
        "VN30 external hourly validation complete: "
        f"usable={usable_count}/30 csv={rel(csv_path)} report={rel(md_path)}"
    )
    if missing_columns or usable_count < 30:
        print(
            "VN30 external hourly validation failed: all 30 frozen tickers are not benchmark-usable.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
