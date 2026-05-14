"""Validate frozen VN30 stock hourly gateway cache files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
CACHE_ROOT = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_gateway"
REPORT_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_hourly_gateway" / "validation"
CSV_PATH = REPORT_ROOT / "vn30_hourly_gateway_validation.csv"
MD_PATH = REPORT_ROOT / "vn30_hourly_gateway_validation.md"
EVAL_START = pd.Timestamp("2025-01-01")
TRAIN_END = pd.Timestamp("2024-12-31 23:59:59")
OHLCV = ["open", "high", "low", "close", "volume"]
COLUMNS = ["datetime", "ticker", *OHLCV, "provider", "source", "frequency"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-training-rows", type=int, default=1000)
    parser.add_argument("--min-evaluation-rows", type=int, default=100)
    return parser.parse_args()


def read_universe() -> list[str]:
    with UNIVERSE_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row["ticker"].strip().upper() for row in rows if row.get("ticker")]


def validate_ticker(ticker: str, min_training_rows: int, min_evaluation_rows: int) -> dict[str, Any]:
    path = CACHE_ROOT / f"{ticker}.csv"
    row: dict[str, Any] = {
        "ticker": ticker,
        "cache_path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "file_exists": str(path.exists()).lower(),
        "row_count": 0,
        "first_datetime": "",
        "last_datetime": "",
        "training_rows_before_2025": 0,
        "evaluation_rows_from_2025": 0,
        "sorted_datetime": "false",
        "duplicate_datetime_count": "",
        "frequency_1h": "false",
        "ohlcv_numeric": "false",
        "positive_prices": "false",
        "volume_nonnegative": "false",
        "ohlc_consistent": "false",
        "usable": "false",
        "missing_reason": "",
    }
    if not path.exists():
        row["missing_reason"] = "cache_file_missing"
        return row
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        row["missing_reason"] = f"cache_read_failed: {type(exc).__name__}: {exc}"
        return row
    missing_columns = [column for column in COLUMNS if column not in frame.columns]
    if missing_columns:
        row["missing_reason"] = f"missing_columns: {missing_columns}"
        return row
    frame = frame[COLUMNS].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame = frame[frame["ticker"].astype(str).str.upper().eq(ticker)].copy()
    for column in OHLCV:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["datetime"])
    duplicate_count = int(frame.duplicated(["datetime"]).sum())
    sorted_ok = bool(frame["datetime"].is_monotonic_increasing) if not frame.empty else False
    frequency_ok = bool(frame["frequency"].astype(str).eq("1H").all()) if not frame.empty else False
    numeric_ok = bool(not frame[OHLCV].isna().any().any()) if not frame.empty else False
    positive_ok = bool((frame[["open", "high", "low", "close"]] > 0).all().all()) if not frame.empty else False
    volume_ok = bool((frame["volume"] >= 0).all()) if not frame.empty else False
    ohlc_ok = bool(
        (frame["high"] >= frame["low"]).all()
        and (frame["high"] >= frame[["open", "close"]].max(axis=1)).all()
        and (frame["low"] <= frame[["open", "close"]].min(axis=1)).all()
    ) if not frame.empty else False
    train_rows = int((frame["datetime"] <= TRAIN_END).sum()) if not frame.empty else 0
    eval_rows = int((frame["datetime"] >= EVAL_START).sum()) if not frame.empty else 0
    reasons: list[str] = []
    if frame.empty:
        reasons.append("no_rows_for_ticker")
    if duplicate_count:
        reasons.append(f"duplicate_datetime_rows={duplicate_count}")
    if not sorted_ok:
        reasons.append("datetime_not_sorted")
    if not frequency_ok:
        reasons.append("frequency_not_1H")
    if not numeric_ok:
        reasons.append("ohlcv_not_numeric")
    if not positive_ok:
        reasons.append("non_positive_price")
    if not volume_ok:
        reasons.append("negative_volume")
    if not ohlc_ok:
        reasons.append("ohlc_inconsistent")
    if train_rows < min_training_rows:
        reasons.append(f"training_rows_below_{min_training_rows}")
    if eval_rows < min_evaluation_rows:
        reasons.append(f"evaluation_rows_below_{min_evaluation_rows}")
    timestamps = frame["datetime"].dropna()
    usable = not reasons
    row.update(
        {
            "row_count": int(len(frame)),
            "first_datetime": "" if timestamps.empty else str(timestamps.min()),
            "last_datetime": "" if timestamps.empty else str(timestamps.max()),
            "training_rows_before_2025": train_rows,
            "evaluation_rows_from_2025": eval_rows,
            "sorted_datetime": str(sorted_ok).lower(),
            "duplicate_datetime_count": duplicate_count,
            "frequency_1h": str(frequency_ok).lower(),
            "ohlcv_numeric": str(numeric_ok).lower(),
            "positive_prices": str(positive_ok).lower(),
            "volume_nonnegative": str(volume_ok).lower(),
            "ohlc_consistent": str(ohlc_ok).lower(),
            "usable": str(usable).lower(),
            "missing_reason": "; ".join(reasons),
        }
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = [validate_ticker(ticker, args.min_training_rows, args.min_evaluation_rows) for ticker in read_universe()]
    fields = list(rows[0])
    write_csv(CSV_PATH, rows, fields)
    usable = [row for row in rows if row["usable"] == "true"]
    lines = [
        "# VN30 Stock Hourly Gateway Validation",
        "",
        f"- Usable tickers: {len(usable)}/{len(rows)}.",
        f"- Minimum training rows: {args.min_training_rows}.",
        f"- Minimum evaluation rows: {args.min_evaluation_rows}.",
        "- Daily data used: no.",
        "- Resampling used: no.",
        "",
        "| ticker | usable | rows | training_rows | evaluation_rows | first | last | missing_reason |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| `{row['ticker']}` | {row['usable']} | {row['row_count']} | {row['training_rows_before_2025']} | {row['evaluation_rows_from_2025']} | {row['first_datetime']} | {row['last_datetime']} | {row['missing_reason']} |")
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
