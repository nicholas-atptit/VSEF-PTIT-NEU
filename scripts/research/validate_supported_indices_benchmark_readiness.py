"""Validate supported index daily/hourly benchmark readiness."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.index_benchmark_common import (
    DAILY_CACHE,
    HOURLY_CACHE,
    INDEX_CODES,
    REPORT_DIR,
    fmt_pct,
    markdown_table,
    read_index_frame,
    readiness_thresholds,
    rel,
    split_frame,
    timestamp_text,
    validate_ohlcv,
    write_csv,
)

CSV_PATH = REPORT_DIR / "index_readiness.csv"
MD_PATH = REPORT_DIR / "index_readiness.md"


def validate_one(code: str, frequency: str) -> dict[str, Any]:
    cache_root = DAILY_CACHE if frequency == "1D" else HOURLY_CACHE
    path = cache_root / f"{code}.csv"
    row = {
        "index_code": code,
        "frequency": frequency,
        "cache_path": rel(path),
        "file_exists": "yes" if path.exists() else "no",
        "row_count": 0,
        "first_timestamp": "",
        "last_timestamp": "",
        "train_rows": 0,
        "validation_rows": 0,
        "final_rows": 0,
        "train_start": "",
        "train_end": "",
        "validation_start": "",
        "validation_end": "",
        "final_start": "",
        "final_end": "",
        "duplicate_timestamps": 0,
        "valid_ohlcv": "no",
        "frequency_correct": "no",
        "usable": "no",
        "reason_unusable": "file_missing",
    }
    if not path.exists():
        return row
    frame = read_index_frame(path, code, frequency)
    valid, reasons = validate_ohlcv(frame, frequency)
    splits = split_frame(frame, frequency)
    min_train, min_val, min_final = readiness_thresholds(frequency)
    split_reasons = []
    if len(splits["train"]) < min_train:
        split_reasons.append(f"train_rows<{min_train}")
    if len(splits["validation"]) < min_val:
        split_reasons.append(f"validation_rows<{min_val}")
    if len(splits["final"]) < min_final:
        split_reasons.append(f"final_rows<{min_final}")
    usable = valid and not split_reasons
    row.update(
        {
            "row_count": int(len(frame)),
            "first_timestamp": timestamp_text(frame, "min"),
            "last_timestamp": timestamp_text(frame, "max"),
            "train_rows": int(len(splits["train"])),
            "validation_rows": int(len(splits["validation"])),
            "final_rows": int(len(splits["final"])),
            "train_start": timestamp_text(splits["train"], "min"),
            "train_end": timestamp_text(splits["train"], "max"),
            "validation_start": timestamp_text(splits["validation"], "min"),
            "validation_end": timestamp_text(splits["validation"], "max"),
            "final_start": timestamp_text(splits["final"], "min"),
            "final_end": timestamp_text(splits["final"], "max"),
            "duplicate_timestamps": int(frame.duplicated(["datetime"]).sum()) if not frame.empty else 0,
            "valid_ohlcv": "yes" if valid else "no",
            "frequency_correct": "yes" if not frame.empty and frame["frequency"].astype(str).eq(frequency).all() else "no",
            "usable": "yes" if usable else "no",
            "reason_unusable": "" if usable else "; ".join(reasons + split_reasons),
        }
    )
    return row


def main() -> int:
    rows = [validate_one(code, frequency) for frequency in ("1D", "1H") for code in INDEX_CODES]
    fields = list(rows[0])
    write_csv(CSV_PATH, rows, fields)
    daily_usable = [row for row in rows if row["frequency"] == "1D" and row["usable"] == "yes"]
    hourly_usable = [row for row in rows if row["frequency"] == "1H" and row["usable"] == "yes"]
    lines = [
        "# Supported Index Benchmark Readiness",
        "",
        "- Scope: index-only.",
        "- Daily split: train 2015-01-01 to 2023-12-31; validation 2024; final 2025-01-01 to provider-current.",
        "- Hourly split: actual available timestamp window only, split 60% train / 20% validation / 20% final by time order.",
        "- Daily-to-hourly resampling used: no.",
        "- Hourly-to-daily resampling used: no.",
        f"- Daily benchmark ready: {'yes' if daily_usable else 'no'} ({len(daily_usable)}/{len(INDEX_CODES)} indices).",
        f"- Hourly benchmark ready: {'yes' if hourly_usable else 'no'} ({len(hourly_usable)}/{len(INDEX_CODES)} indices).",
        "",
        "## Readiness Rows",
        "",
        markdown_table(fields, rows),
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Index readiness written: {rel(MD_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
