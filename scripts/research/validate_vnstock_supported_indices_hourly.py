"""Validate normalized supported-index hourly cache files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "index_hourly_fetch" / "validation"
CSV_REPORT = REPORT_DIR / "index_hourly_validation.csv"
MD_REPORT = REPORT_DIR / "index_hourly_validation.md"
INDEX_CODES = ("VNINDEX", "HNXINDEX", "UPCOMINDEX", "VN30", "HNX30", "VN100")


def _validate_one(code: str) -> dict[str, Any]:
    path = CACHE_ROOT / f"{code}.csv"
    row: dict[str, Any] = {
        "index_code": code,
        "path": str(path.relative_to(REPO_ROOT)),
        "exists": path.exists(),
        "timestamp_parse": False,
        "hourly_alignment": False,
        "sorted_timestamps": False,
        "duplicate_datetime": False,
        "numeric_ohlcv": False,
        "positive_ohlc": False,
        "high_low_valid": False,
        "volume_nonnegative": False,
        "actual_first_datetime": "",
        "actual_last_datetime": "",
        "actual_rows": 0,
        "usable": False,
        "missing_reason": "",
    }
    if not path.exists():
        row["missing_reason"] = "normalized cache missing"
        return row
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        row["missing_reason"] = f"read failed: {type(exc).__name__}: {exc}"
        return row
    row["actual_rows"] = int(len(df))
    required = ["datetime", "index_code", "open", "high", "low", "close", "volume", "provider", "source"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        row["missing_reason"] = f"missing columns: {missing}"
        return row
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    row["timestamp_parse"] = bool(dt.notna().all() and len(dt) > 0)
    if row["timestamp_parse"]:
        row["actual_first_datetime"] = str(dt.min())
        row["actual_last_datetime"] = str(dt.max())
        row["hourly_alignment"] = bool((dt.dt.minute.eq(0) & dt.dt.second.eq(0)).all())
        row["sorted_timestamps"] = bool(dt.is_monotonic_increasing)
        row["duplicate_datetime"] = bool(dt.duplicated().any())
    numeric = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    row["numeric_ohlcv"] = bool(numeric.notna().all().all())
    if row["numeric_ohlcv"]:
        row["positive_ohlc"] = bool((numeric[["open", "high", "low", "close"]] > 0).all().all())
        row["high_low_valid"] = bool(
            (numeric["high"] >= numeric["low"]).all()
            and (numeric["high"] >= numeric["open"]).all()
            and (numeric["high"] >= numeric["close"]).all()
            and (numeric["low"] <= numeric["open"]).all()
            and (numeric["low"] <= numeric["close"]).all()
        )
        row["volume_nonnegative"] = bool((numeric["volume"] >= 0).all())
    checks = [
        row["timestamp_parse"],
        row["hourly_alignment"],
        row["sorted_timestamps"],
        not row["duplicate_datetime"],
        row["numeric_ohlcv"],
        row["positive_ohlc"],
        row["high_low_valid"],
        row["volume_nonnegative"],
        row["actual_rows"] > 0,
    ]
    row["usable"] = all(checks)
    if not row["usable"]:
        failed = [
            name
            for name, ok in (
                ("timestamp_parse", row["timestamp_parse"]),
                ("hourly_alignment", row["hourly_alignment"]),
                ("sorted_timestamps", row["sorted_timestamps"]),
                ("no_duplicate_datetime", not row["duplicate_datetime"]),
                ("numeric_ohlcv", row["numeric_ohlcv"]),
                ("positive_ohlc", row["positive_ohlc"]),
                ("high_low_valid", row["high_low_valid"]),
                ("volume_nonnegative", row["volume_nonnegative"]),
                ("rows_gt_zero", row["actual_rows"] > 0),
            )
            if not ok
        ]
        row["missing_reason"] = ", ".join(failed)
    return row


def _write_reports(rows: list[dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with CSV_REPORT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Index Hourly Validation",
        "",
        "| index_code | usable | rows | first datetime | last datetime | missing reason |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['index_code']}` | {'yes' if row['usable'] else 'no'} | {row['actual_rows']} | "
            f"{row['actual_first_datetime']} | {row['actual_last_datetime']} | {row['missing_reason']} |"
        )
    MD_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = [_validate_one(code) for code in INDEX_CODES]
    _write_reports(rows)
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
