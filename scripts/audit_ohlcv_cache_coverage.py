"""Read-only audit for local per-ticker OHLCV cache coverage.

This script inspects CSV files under ``data/daily_market_split_data`` by
default. It does not fetch provider data, write cache files, or modify input
artifacts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OHLCV_DIR = PROJECT_ROOT / "data" / "daily_market_split_data"
SUPPORT_COVERAGE_THRESHOLD = 0.95


def _parse_tickers(raw: str | list[str]) -> list[str]:
    if isinstance(raw, str):
        values = raw.replace(",", " ").split()
    else:
        values = []
        for item in raw:
            values.extend(str(item).replace(",", " ").split())
    return [ticker.upper().strip() for ticker in values if ticker.strip()]


def _detect_provider_availability() -> dict[str, Any]:
    """Detect provider module availability without importing provider clients."""
    return {
        "provider_fetch_attempted": False,
        "canonical_provider": "vnstock_data",
        "vnstock_data_available": importlib.util.find_spec("vnstock_data") is not None,
        "alternate_vnstock_available": importlib.util.find_spec("vnstock") is not None,
    }


def _normalize_date_column(frame: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    normalized = frame.copy()
    for candidate in ("date", "time", "timestamp"):
        if candidate in normalized.columns:
            if candidate != "date":
                normalized = normalized.rename(columns={candidate: "date"})
            normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
            return normalized, candidate
    if isinstance(normalized.index, pd.DatetimeIndex):
        normalized = normalized.reset_index().rename(columns={normalized.index.name or "index": "date"})
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
        return normalized, "index"
    return normalized, None


def _safe_iso(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def inspect_ticker_cache_coverage(
    ticker: str,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    data_dir: Path | str = DEFAULT_OHLCV_DIR,
    support_coverage_threshold: float = SUPPORT_COVERAGE_THRESHOLD,
) -> dict[str, Any]:
    """Return read-only coverage diagnostics for one cached OHLCV CSV file."""
    symbol = ticker.upper().strip()
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    requested_dates = pd.bdate_range(start_ts, end_ts)
    requested_date_count = int(len(requested_dates))
    cache_dir = Path(data_dir)
    source_path = cache_dir / f"{symbol}.csv"

    base: dict[str, Any] = {
        "ticker": symbol,
        "source_path": str(source_path),
        "fallback_file_present": source_path.exists(),
        "source_status": "missing_file",
        "date_column": None,
        "row_count": 0,
        "date_min": None,
        "date_max": None,
        "requested_business_day_count": requested_date_count,
        "matched_date_count": 0,
        "missing_date_count": requested_date_count,
        "requested_date_coverage_rate": 0.0,
        "supports_requested_window": False,
        "support_coverage_threshold": float(support_coverage_threshold),
        "read_error": None,
    }
    if not source_path.exists():
        return base

    try:
        frame = pd.read_csv(source_path)
    except Exception as exc:
        base["source_status"] = "read_error"
        base["read_error"] = str(exc)
        return base

    base["row_count"] = int(len(frame))
    normalized, date_column = _normalize_date_column(frame)
    base["date_column"] = date_column
    if date_column is None or "date" not in normalized.columns:
        base["source_status"] = "missing_date_column"
        return base

    valid_dates = pd.DatetimeIndex(normalized["date"].dropna().dt.normalize().unique()).sort_values()
    if len(valid_dates) == 0:
        base["source_status"] = "no_valid_dates"
        return base

    date_min = pd.Timestamp(valid_dates.min()).normalize()
    date_max = pd.Timestamp(valid_dates.max()).normalize()
    matched_dates = valid_dates[(valid_dates >= start_ts) & (valid_dates <= end_ts)]
    matched_set = set(pd.Timestamp(value).normalize() for value in matched_dates)
    requested_set = set(pd.Timestamp(value).normalize() for value in requested_dates)
    matched_count = int(len(requested_set & matched_set))
    missing_count = int(max(requested_date_count - matched_count, 0))
    coverage_rate = float(matched_count / requested_date_count) if requested_date_count else 0.0
    spans_requested_range = bool(date_min <= start_ts and date_max >= end_ts)

    base.update(
        {
            "source_status": "loaded",
            "date_min": _safe_iso(date_min),
            "date_max": _safe_iso(date_max),
            "matched_date_count": matched_count,
            "missing_date_count": missing_count,
            "requested_date_coverage_rate": coverage_rate,
            "supports_requested_window": bool(
                spans_requested_range and coverage_rate >= support_coverage_threshold
            ),
        }
    )
    return base


def inspect_ohlcv_cache_coverage(
    tickers: list[str] | str,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    data_dir: Path | str = DEFAULT_OHLCV_DIR,
    detect_provider: bool = True,
    support_coverage_threshold: float = SUPPORT_COVERAGE_THRESHOLD,
) -> dict[str, Any]:
    """Return read-only OHLCV cache coverage diagnostics for multiple tickers."""
    resolved_tickers = _parse_tickers(tickers)
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    requested_business_day_count = int(len(pd.bdate_range(start_ts, end_ts)))
    results = [
        inspect_ticker_cache_coverage(
            ticker,
            start_date=start_ts,
            end_date=end_ts,
            data_dir=data_dir,
            support_coverage_threshold=support_coverage_threshold,
        )
        for ticker in resolved_tickers
    ]
    supporting = sum(1 for item in results if item["supports_requested_window"])
    return {
        "requested_start_date": start_ts.strftime("%Y-%m-%d"),
        "requested_end_date": end_ts.strftime("%Y-%m-%d"),
        "requested_business_day_count": requested_business_day_count,
        "data_dir": str(Path(data_dir)),
        "provider_availability": (
            _detect_provider_availability()
            if detect_provider
            else {"provider_fetch_attempted": False, "provider_detection_attempted": False}
        ),
        "ticker_results": results,
        "summary": {
            "ticker_count": int(len(results)),
            "supporting_ticker_count": int(supporting),
            "missing_file_count": int(sum(1 for item in results if not item["fallback_file_present"])),
            "all_tickers_support_requested_window": bool(len(results) > 0 and supporting == len(results)),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit local OHLCV cache coverage without fetching data.")
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker list, comma-separated or space-separated.")
    parser.add_argument("--start-date", required=True, help="Requested start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Requested end date, YYYY-MM-DD.")
    parser.add_argument(
        "--data-dir",
        "--cache-dir",
        dest="data_dir",
        default=str(DEFAULT_OHLCV_DIR),
        help="Directory containing per-ticker OHLCV CSV files.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = inspect_ohlcv_cache_coverage(
        args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        data_dir=args.data_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
