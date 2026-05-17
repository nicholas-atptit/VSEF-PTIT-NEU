"""Fetch supported Vietnamese index daily OHLCV through the canonical gateway."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.index_benchmark_common import DAILY_CACHE, INDEX_CODES, INDEX_COLUMNS, REPORT_DIR, rel, read_index_frame, validate_ohlcv  # noqa: E402
from src.data.providers.vn_price_gateway import ProviderFetchError, _quote_history, fetch_price_history  # noqa: E402
from src.data.providers.vn_provider_contract import AssetType, FetchRequest, Frequency, ProviderName, SourceName  # noqa: E402

SUMMARY_CSV = REPORT_DIR / "index_daily_gateway_fetch_summary.csv"
SUMMARY_MD = REPORT_DIR / "index_daily_gateway_fetch_summary.md"
SOURCES = (SourceName.KBS, SourceName.VCI)


def normalize_frame(raw: pd.DataFrame, code: str, provider: Any, source: Any) -> pd.DataFrame:
    frame = raw.copy()
    if "time" in frame.columns and "datetime" not in frame.columns:
        frame = frame.rename(columns={"time": "datetime"})
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["index_code"] = code
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["provider"] = str(provider)
    frame["source"] = str(source)
    frame["frequency"] = "1D"
    frame = frame[list(INDEX_COLUMNS)].dropna(subset=["datetime"]).sort_values("datetime")
    frame = frame.drop_duplicates(["datetime"], keep="last").reset_index(drop=True)
    return frame


def filter_valid_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    valid = (
        frame["datetime"].notna()
        & (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (frame["volume"] >= 0)
        & (frame["high"] >= frame["low"])
        & (frame["high"] >= frame[["open", "close"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close"]].min(axis=1))
    )
    return frame[valid].copy().sort_values("datetime").drop_duplicates(["datetime"], keep="last").reset_index(drop=True)


def cache_is_usable(code: str) -> bool:
    path = DAILY_CACHE / f"{code}.csv"
    if not path.exists():
        return False
    frame = read_index_frame(path, code, "1D")
    valid, _ = validate_ohlcv(frame, "1D")
    years = set(frame["datetime"].dt.year.astype(int).tolist()) if not frame.empty else set()
    return valid and 2015 in years and 2024 in years and any(year >= 2025 for year in years)


def fetch_raw_fallback(code: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
    last_error = ""
    for source in SOURCES:
        try:
            raw = _quote_history("vnstock_data", code, source, start, end, Frequency.DAILY)
            if raw is None or raw.empty:
                last_error = f"{source}: empty response"
                continue
            return normalize_frame(raw, code, ProviderName.VNSTOCK_DATA, source), f"raw_fallback:{source}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(last_error or "raw fallback failed")


def fetch_index(code: str, start: str, end: str, force: bool) -> dict[str, Any]:
    path = DAILY_CACHE / f"{code}.csv"
    row = {
        "index_code": code,
        "cache_path": rel(path),
        "status": "failed",
        "rows": 0,
        "first_timestamp": "",
        "last_timestamp": "",
        "provider": "",
        "source": "",
        "error": "",
    }
    if path.exists() and cache_is_usable(code) and not force:
        frame = read_index_frame(path, code, "1D")
        row.update(
            {
                "status": "skipped_existing_usable",
                "rows": int(len(frame)),
                "first_timestamp": str(frame["datetime"].min()),
                "last_timestamp": str(frame["datetime"].max()),
                "provider": ";".join(frame["provider"].dropna().astype(str).unique()[:3]),
                "source": ";".join(frame["source"].dropna().astype(str).unique()[:3]),
            }
        )
        return row
    try:
        response = fetch_price_history(
            FetchRequest(
                symbol=code,
                asset_type=AssetType.INDEX,
                start=start,
                end=end,
                frequency=Frequency.DAILY,
                preferred_sources=SOURCES,
                allow_legacy_fallback=True,
                allow_daily=True,
                allow_resample=False,
            )
        )
        frame = normalize_frame(response.data, code, response.provider, response.source)
        fetch_source = "gateway"
    except Exception as exc:
        try:
            frame, fetch_source = fetch_raw_fallback(code, start, end)
        except Exception as fallback_exc:
            row["error"] = f"gateway:{type(exc).__name__}:{exc}; fallback:{type(fallback_exc).__name__}:{fallback_exc}"
            return row
    valid, reasons = validate_ohlcv(frame, "1D")
    if not valid:
        filtered = filter_valid_ohlcv(frame)
        filtered_valid, filtered_reasons = validate_ohlcv(filtered, "1D")
        if filtered_valid and len(filtered) >= 100:
            frame = filtered
        else:
            row["status"] = "unusable"
            row["rows"] = int(len(filtered))
            row["first_timestamp"] = "" if filtered.empty else str(filtered["datetime"].min())
            row["last_timestamp"] = "" if filtered.empty else str(filtered["datetime"].max())
            row["error"] = "; ".join(reasons + filtered_reasons)
            return row
    DAILY_CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    row.update(
        {
            "status": "success",
            "rows": int(len(frame)),
            "first_timestamp": str(frame["datetime"].min()),
            "last_timestamp": str(frame["datetime"].max()),
            "provider": ";".join(frame["provider"].dropna().astype(str).unique()[:3]),
            "source": fetch_source,
        }
    )
    return row


def write_reports(rows: list[dict[str, Any]]) -> None:
    fields = ["index_code", "cache_path", "status", "rows", "first_timestamp", "last_timestamp", "provider", "source", "error"]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Supported Index Daily Gateway Fetch Summary",
        "",
        "- Scope: index-only.",
        "- Frequency: daily only.",
        "- Stock data fetched: no.",
        "- Hourly data fetched: no.",
        "- Resampling used: no.",
        "",
        "| index_code | status | rows | first timestamp | last timestamp | source | error |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| `{row['index_code']}` | {row['status']} | {row['rows']} | {row['first_timestamp']} | {row['last_timestamp']} | `{row['source']}` | {row['error']} |")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-code", choices=INDEX_CODES)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-runtime-seconds", type=int, default=7200)
    args = parser.parse_args()
    started = time.monotonic()
    codes = (args.index_code,) if args.index_code else INDEX_CODES
    rows: list[dict[str, Any]] = []
    for code in codes:
        if args.max_runtime_seconds and time.monotonic() - started > args.max_runtime_seconds:
            break
        print(f"Fetching daily index {code}...")
        row = fetch_index(code, args.start, args.end, args.force)
        rows.append(row)
        print(f"  {row['status']}: {row['rows']} rows ({row['first_timestamp']} to {row['last_timestamp']})")
        time.sleep(0.25)
    write_reports(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
