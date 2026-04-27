from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.audit_ohlcv_cache_coverage import inspect_ohlcv_cache_coverage


def _write_ohlcv(path: Path, ticker: str, dates: pd.DatetimeIndex) -> None:
    frame = pd.DataFrame(
        {
            "time": dates.strftime("%Y-%m-%d"),
            "open": [10.0 + idx for idx in range(len(dates))],
            "high": [10.5 + idx for idx in range(len(dates))],
            "low": [9.5 + idx for idx in range(len(dates))],
            "close": [10.2 + idx for idx in range(len(dates))],
            "volume": [1000 + idx for idx in range(len(dates))],
            "ticker": ticker,
        }
    )
    frame.to_csv(path / f"{ticker}.csv", index=False)


def _ticker_result(result: dict, ticker: str) -> dict:
    return next(item for item in result["ticker_results"] if item["ticker"] == ticker)


def test_coverage_audit_detects_full_coverage_for_synthetic_data(tmp_path: Path) -> None:
    dates = pd.bdate_range("2024-01-01", "2024-01-10")
    _write_ohlcv(tmp_path, "AAA", dates)

    result = inspect_ohlcv_cache_coverage(
        "AAA",
        start_date="2024-01-01",
        end_date="2024-01-10",
        data_dir=tmp_path,
        detect_provider=False,
    )

    ticker = _ticker_result(result, "AAA")
    assert ticker["fallback_file_present"] is True
    assert ticker["source_status"] == "loaded"
    assert ticker["matched_date_count"] == ticker["requested_business_day_count"]
    assert ticker["missing_date_count"] == 0
    assert ticker["requested_date_coverage_rate"] == 1.0
    assert ticker["supports_requested_window"] is True


def test_coverage_audit_detects_truncated_later_start(tmp_path: Path) -> None:
    _write_ohlcv(tmp_path, "BBB", pd.bdate_range("2024-01-05", "2024-01-10"))

    result = inspect_ohlcv_cache_coverage(
        ["BBB"],
        start_date="2024-01-01",
        end_date="2024-01-10",
        data_dir=tmp_path,
        detect_provider=False,
    )

    ticker = _ticker_result(result, "BBB")
    assert ticker["date_min"] == "2024-01-05"
    assert ticker["missing_date_count"] > 0
    assert ticker["requested_date_coverage_rate"] < 1.0
    assert ticker["supports_requested_window"] is False


def test_coverage_audit_detects_missing_file(tmp_path: Path) -> None:
    result = inspect_ohlcv_cache_coverage(
        "MISSING",
        start_date="2024-01-01",
        end_date="2024-01-10",
        data_dir=tmp_path,
        detect_provider=False,
    )

    ticker = _ticker_result(result, "MISSING")
    assert ticker["fallback_file_present"] is False
    assert ticker["source_status"] == "missing_file"
    assert ticker["row_count"] == 0
    assert ticker["date_min"] is None
    assert ticker["date_max"] is None
    assert ticker["supports_requested_window"] is False


def test_coverage_audit_reports_row_count_and_date_range(tmp_path: Path) -> None:
    _write_ohlcv(tmp_path, "CCC", pd.bdate_range("2024-02-01", periods=3))

    result = inspect_ohlcv_cache_coverage(
        "CCC",
        start_date="2024-02-01",
        end_date="2024-02-05",
        data_dir=tmp_path,
        detect_provider=False,
    )

    ticker = _ticker_result(result, "CCC")
    assert ticker["row_count"] == 3
    assert ticker["date_min"] == "2024-02-01"
    assert ticker["date_max"] == "2024-02-05"
    assert ticker["date_column"] == "time"


def test_coverage_audit_does_not_fetch_or_modify_data(tmp_path: Path) -> None:
    _write_ohlcv(tmp_path, "DDD", pd.bdate_range("2024-01-01", "2024-01-05"))
    source_path = tmp_path / "DDD.csv"
    before = source_path.read_bytes()
    before_mtime = source_path.stat().st_mtime_ns

    result = inspect_ohlcv_cache_coverage(
        "DDD",
        start_date="2024-01-01",
        end_date="2024-01-05",
        data_dir=tmp_path,
        detect_provider=True,
    )

    assert source_path.read_bytes() == before
    assert source_path.stat().st_mtime_ns == before_mtime
    assert result["provider_availability"]["provider_fetch_attempted"] is False


def test_coverage_audit_handles_multiple_tickers(tmp_path: Path) -> None:
    _write_ohlcv(tmp_path, "EEE", pd.bdate_range("2024-01-01", "2024-01-05"))
    _write_ohlcv(tmp_path, "FFF", pd.bdate_range("2024-01-03", "2024-01-05"))

    result = inspect_ohlcv_cache_coverage(
        "EEE,FFF,GGG",
        start_date="2024-01-01",
        end_date="2024-01-05",
        data_dir=tmp_path,
        detect_provider=False,
    )

    assert result["summary"]["ticker_count"] == 3
    assert result["summary"]["supporting_ticker_count"] == 1
    assert result["summary"]["missing_file_count"] == 1
    assert _ticker_result(result, "EEE")["supports_requested_window"] is True
    assert _ticker_result(result, "FFF")["supports_requested_window"] is False
    assert _ticker_result(result, "GGG")["source_status"] == "missing_file"


def test_audit_script_runs_against_synthetic_temporary_csvs(tmp_path: Path) -> None:
    _write_ohlcv(tmp_path, "HHH", pd.bdate_range("2024-03-01", "2024-03-08"))
    script_path = Path("scripts/audit_ohlcv_cache_coverage.py")

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--tickers",
            "HHH",
            "--start-date",
            "2024-03-01",
            "--end-date",
            "2024-03-08",
            "--data-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    ticker = _ticker_result(result, "HHH")
    assert result["summary"]["all_tickers_support_requested_window"] is True
    assert ticker["matched_date_count"] == ticker["requested_business_day_count"]
