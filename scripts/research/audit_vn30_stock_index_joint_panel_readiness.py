"""Audit readiness for a joint VN30 stock + supported-index hourly panel."""

from __future__ import annotations

from collections import Counter
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.index_benchmark_common import read_index_frame
from scripts.research.vn30_hourly_common import REPO_ROOT, standardize_hourly_frame
from scripts.research.vn30_stock_index_joint_panel_features import (
    HORIZONS,
    OHLCV,
    REPORT_DIR,
    SUPPORTED_INDICES,
    VN30_TICKERS,
    markdown_table,
    rel,
    write_csv,
)


MIN_ROWS = max(HORIZONS) + 30
REPAIRED_STOCK_CACHE = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_2015"
FALLBACK_STOCK_CACHE = REPO_ROOT / "data" / "hourly_market_split_data"
ARCHIVE_INDEX_CACHE = (
    REPO_ROOT
    / "archive"
    / "generated_data_snapshots"
    / "vn30_hourly_pre_benchmark_20260514_062528"
    / "data"
    / "market_cache"
    / "vnstock_data"
    / "indices"
    / "hourly"
)
ACTIVE_INDEX_CACHE = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly_2015"


def infer_frequency_status(datetimes: pd.Series, expected_intraday: bool) -> tuple[str, str]:
    if datetimes.empty:
        return "missing", "no timestamps"
    hours = sorted(pd.to_datetime(datetimes, errors="coerce").dropna().dt.hour.unique().tolist())
    if expected_intraday and hours == [0]:
        return "not_intraday_hourly", "all timestamps are midnight"
    if expected_intraday and len(hours) < 2:
        return "weak_intraday_hourly", f"observed_hours={hours}"
    return "ok", f"observed_hours={hours}"


def is_intraday_frame(frame: pd.DataFrame) -> bool:
    if frame.empty or "datetime" not in frame.columns:
        return False
    datetimes = pd.to_datetime(frame["datetime"], errors="coerce").dropna()
    if datetimes.empty:
        return False
    hours = sorted(datetimes.dt.hour.unique().tolist())
    return len(hours) >= 2 and hours != [0]


def load_repaired_stock_frame(code: str) -> tuple[pd.DataFrame, str]:
    candidates = [REPAIRED_STOCK_CACHE / f"{code}.csv", FALLBACK_STOCK_CACHE / f"{code}.csv"]
    loaded: list[tuple[pd.DataFrame, str, bool]] = []
    for path in candidates:
        if not path.exists():
            continue
        raw = pd.read_csv(path, low_memory=False)
        frame = standardize_hourly_frame(raw, code)
        if frame.empty:
            continue
        frame = frame.rename(columns={"ticker": "instrument_code"})
        frame["instrument_type"] = "stock"
        frame = frame[["datetime", "instrument_code", "instrument_type", *OHLCV]].copy()
        loaded.append((frame, rel(path), is_intraday_frame(frame)))
    if not loaded:
        return pd.DataFrame(columns=["datetime", "instrument_code", "instrument_type", *OHLCV]), ""
    intraday = [item for item in loaded if item[2]]
    selected = max(intraday or loaded, key=lambda item: len(item[0]))
    return selected[0], selected[1]


def load_repaired_index_frame(code: str) -> tuple[pd.DataFrame, str]:
    candidates = [ARCHIVE_INDEX_CACHE / f"{code}.csv", ACTIVE_INDEX_CACHE / f"{code}.csv"]
    loaded: list[tuple[pd.DataFrame, str, bool]] = []
    for path in candidates:
        if not path.exists():
            continue
        frame = read_index_frame(path, code=code, frequency="1H")
        if frame.empty:
            continue
        frame = frame.rename(columns={"index_code": "instrument_code"})
        frame["instrument_type"] = "index"
        frame = frame[["datetime", "instrument_code", "instrument_type", *OHLCV]].copy()
        loaded.append((frame, rel(path), is_intraday_frame(frame)))
    if not loaded:
        return pd.DataFrame(columns=["datetime", "instrument_code", "instrument_type", *OHLCV]), ""
    intraday = [item for item in loaded if item[2]]
    selected = max(intraday or loaded, key=lambda item: len(item[0]))
    return selected[0], selected[1]


def audit_one(code: str, instrument_type: str) -> dict[str, Any]:
    if instrument_type == "stock":
        frame, source_path = load_repaired_stock_frame(code)
    else:
        frame, source_path = load_repaired_index_frame(code)
    reasons: list[str] = []
    if frame.empty:
        reasons.append("missing_or_empty")
        datetimes = pd.Series(dtype="datetime64[ns]")
    else:
        datetimes = pd.to_datetime(frame["datetime"], errors="coerce").dropna()
    duplicates = int(frame.duplicated(["datetime"]).sum()) if not frame.empty else 0
    missing_ohlcv = int(frame[list(OHLCV)].isna().sum().sum()) if not frame.empty else 0
    if duplicates:
        reasons.append(f"duplicate_timestamps={duplicates}")
    if missing_ohlcv:
        reasons.append(f"missing_ohlcv={missing_ohlcv}")
    if len(frame) < MIN_ROWS:
        reasons.append(f"insufficient_rows_for_h120:{len(frame)}<{MIN_ROWS}")
    frequency_status, frequency_note = infer_frequency_status(datetimes, expected_intraday=True)
    if frequency_status != "ok":
        reasons.append(frequency_status)
    n = len(frame)
    train_rows = int(n * 0.60)
    validation_rows = int(n * 0.20)
    final_rows = n - train_rows - validation_rows
    if min(train_rows, validation_rows, final_rows) <= max(HORIZONS):
        reasons.append("insufficient_split_rows_after_horizon")
    usable = not reasons
    return {
        "instrument_code": code,
        "instrument_type": instrument_type,
        "row_count": n,
        "first_timestamp": "" if datetimes.empty else str(datetimes.min()),
        "last_timestamp": "" if datetimes.empty else str(datetimes.max()),
        "train_rows_estimated": train_rows,
        "validation_rows_estimated": validation_rows,
        "final_rows_estimated": final_rows,
        "duplicate_timestamps": duplicates,
        "missing_ohlcv_cells": missing_ohlcv,
        "frequency_status": frequency_status,
        "frequency_note": frequency_note,
        "source_path": source_path,
        "usable": usable,
        "reason": "usable" if usable else "; ".join(dict.fromkeys(reasons)),
    }


def main() -> int:
    rows = [audit_one(code, "stock") for code in VN30_TICKERS]
    rows.extend(audit_one(code, "index") for code in SUPPORTED_INDICES)
    write_csv(REPORT_DIR / "joint_panel_readiness_repaired.csv", rows)
    counts = Counter(row["instrument_type"] for row in rows)
    usable = [row for row in rows if row["usable"]]
    stock_usable = [row for row in usable if row["instrument_type"] == "stock"]
    index_usable = [row for row in usable if row["instrument_type"] == "index"]
    can_run = len(stock_usable) == 30 and len(index_usable) == 6
    failed = [row for row in rows if not row["usable"]]
    content = [
        "# VN30 Stock + Index Joint Panel Readiness",
        "",
        f"- Stock instrument count: {counts.get('stock', 0)}.",
        f"- Index instrument count: {counts.get('index', 0)}.",
        f"- Total instrument count: {len(rows)}.",
        f"- Usable stock instruments: {len(stock_usable)}/30.",
        f"- Usable index instruments: {len(index_usable)}/6.",
        f"- Joint panel can run with 36/36 instruments: {str(can_run).lower()}.",
        f"- Repaired stock cache checked: `{rel(REPAIRED_STOCK_CACHE)}`.",
        f"- Repaired index archive checked: `{rel(ARCHIVE_INDEX_CACHE)}`.",
        f"- Active index fallback checked: `{rel(ACTIVE_INDEX_CACHE)}`.",
        "- Benchmark/training run in this phase: no.",
        "",
        "## Failed Or Risky Instruments",
        "",
        markdown_table(
            [
                "instrument_code",
                "instrument_type",
                "row_count",
                "first_timestamp",
                "last_timestamp",
                "frequency_status",
                "source_path",
                "reason",
            ],
            failed if failed else rows,
            max_rows=80,
        ),
        "",
        "## Decision",
        "",
        "The joint 36-instrument hourly panel is ready." if can_run else "The joint 36-instrument hourly panel is not validation-ready from current cache.",
        "",
    ]
    (REPORT_DIR / "joint_panel_readiness_repaired.md").write_text("\n".join(content), encoding="utf-8")
    print(f"joint_panel_can_run_36={str(can_run).lower()} usable={len(usable)}/36")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
