"""Read-only probe for foreign-flow source coverage.

The script checks whether a local foreign-flow CSV can support exact
ticker/date joins for requested OHLCV rows. It does not fetch provider data and
does not create or modify artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.settings import PROJECT_ROOT
from src.ml.backtest.foreign_flow_validation import validate_foreign_flow_artifact


DEFAULT_FOREIGN_FLOW_PATH = PROJECT_ROOT / "data" / "foreign_flow.csv"
DEFAULT_OHLCV_DIR = PROJECT_ROOT / "data" / "daily_market_split_data"


def _normalize_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _normalize_tickers(values: list[str] | tuple[str, ...] | str) -> list[str]:
    if isinstance(values, str):
        raw_values = values.split(",")
    else:
        raw_values = list(values)
    return [str(value).strip().upper() for value in raw_values if str(value).strip()]


def load_foreign_flow_source(path: Path | str) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_path = Path(path)
    status: dict[str, Any] = {
        "source_path": str(source_path),
        "source_exists": source_path.exists(),
        "provider_fetch_attempted": False,
    }
    if not source_path.exists():
        status["source_status"] = "missing_local_artifact"
        return pd.DataFrame(), status

    frame = pd.read_csv(source_path)
    if "time" in frame.columns and "date" not in frame.columns:
        frame = frame.rename(columns={"time": "date"})
    if "date" in frame.columns:
        frame["date"] = _normalize_date_series(frame["date"])
    if "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    status["source_status"] = "loaded_local_artifact"
    status["row_count"] = int(len(frame))
    status["sample_columns"] = list(frame.columns)
    if "ticker" in frame.columns:
        status["ticker_coverage"] = sorted(frame["ticker"].dropna().astype(str).unique().tolist())
    if "date" in frame.columns and frame["date"].notna().any():
        status["date_min"] = str(frame["date"].min().date())
        status["date_max"] = str(frame["date"].max().date())
    return frame, status


def inspect_foreign_flow_coverage(
    *,
    tickers: list[str] | tuple[str, ...] | str,
    start_date: str,
    end_date: str,
    foreign_flow_path: Path | str = DEFAULT_FOREIGN_FLOW_PATH,
    ohlcv_dir: Path | str = DEFAULT_OHLCV_DIR,
) -> dict[str, Any]:
    """Return exact ticker/date coverage diagnostics for a local artifact."""

    normalized_tickers = _normalize_tickers(tickers)
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    foreign_flow, source_status = load_foreign_flow_source(foreign_flow_path)
    artifact_validation = validate_foreign_flow_artifact(
        foreign_flow,
        normalized_tickers,
        start_ts,
        end_ts,
    )

    rows: list[dict[str, Any]] = []
    source_has_join_keys = {"ticker", "date"} <= set(foreign_flow.columns)
    if source_has_join_keys:
        foreign_keys = set(
            zip(
                foreign_flow["ticker"].astype(str).str.upper(),
                _normalize_date_series(foreign_flow["date"]),
            )
        )
    else:
        foreign_keys = set()

    for ticker in normalized_tickers:
        ticker_rows = foreign_flow[foreign_flow["ticker"] == ticker] if source_has_join_keys else pd.DataFrame()
        if source_has_join_keys and not ticker_rows.empty:
            ticker_dates = _normalize_date_series(ticker_rows["date"])
            source_ticker_min = str(ticker_dates.min().date()) if ticker_dates.notna().any() else None
            source_ticker_max = str(ticker_dates.max().date()) if ticker_dates.notna().any() else None
        else:
            source_ticker_min = None
            source_ticker_max = None

        ohlcv_path = Path(ohlcv_dir) / f"{ticker}.csv"
        try:
            ohlcv = pd.read_csv(ohlcv_path) if ohlcv_path.exists() else pd.DataFrame()
            if "time" in ohlcv.columns and "date" not in ohlcv.columns:
                ohlcv = ohlcv.rename(columns={"time": "date"})
            if "date" in ohlcv.columns:
                ohlcv["date"] = _normalize_date_series(ohlcv["date"])
                ohlcv = ohlcv[(ohlcv["date"] >= start_ts) & (ohlcv["date"] <= end_ts)]
        except Exception:
            ohlcv = pd.DataFrame()
        if ohlcv is None or ohlcv.empty or "date" not in ohlcv.columns:
            requested_dates = pd.DatetimeIndex([])
            ohlcv_status = "missing_ohlcv"
        else:
            requested_dates = pd.DatetimeIndex(_normalize_date_series(ohlcv["date"]).dropna().unique())
            ohlcv_status = "loaded"

        if len(requested_dates):
            matched_count = sum((ticker, date) in foreign_keys for date in requested_dates)
            requested_count = int(len(requested_dates))
            missing_count = int(requested_count - matched_count)
            missing_ratio = float(missing_count / requested_count)
        else:
            matched_count = 0
            requested_count = 0
            missing_count = 0
            missing_ratio = None

        rows.append(
            {
                "ticker": ticker,
                "ohlcv_status": ohlcv_status,
                "requested_ohlcv_dates": requested_count,
                "foreign_flow_rows_for_ticker": int(len(ticker_rows)),
                "foreign_flow_ticker_date_min": source_ticker_min,
                "foreign_flow_ticker_date_max": source_ticker_max,
                "exact_join_match_count": int(matched_count),
                "exact_join_missing_count": missing_count,
                "exact_join_missing_ratio": missing_ratio,
                "exact_join_would_succeed": bool(matched_count > 0),
            }
        )

    return {
        "requested_tickers": normalized_tickers,
        "requested_start_date": str(start_ts.date()),
        "requested_end_date": str(end_ts.date()),
        "foreign_flow_source": source_status,
        "artifact_validation": artifact_validation,
        "ticker_results": rows,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit local foreign-flow coverage for exact ticker/date joins.")
    parser.add_argument("--tickers", required=True, help="Comma-separated ticker list, for example SSI,FPT,ACB,HPG.")
    parser.add_argument("--start-date", required=True, help="Requested start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Requested end date, YYYY-MM-DD.")
    parser.add_argument("--foreign-flow-path", default=str(DEFAULT_FOREIGN_FLOW_PATH))
    parser.add_argument("--ohlcv-dir", default=str(DEFAULT_OHLCV_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = inspect_foreign_flow_coverage(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        foreign_flow_path=args.foreign_flow_path,
        ohlcv_dir=args.ohlcv_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
