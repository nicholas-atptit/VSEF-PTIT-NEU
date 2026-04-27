from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.ml.backtest.foreign_flow_validation import (
    classify_foreign_flow_artifact,
    summarize_foreign_flow_coverage,
    validate_foreign_flow_artifact,
)


def _valid_rows(ticker: str = "AAA") -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", "2024-01-04")
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates,
            "foreign_net_value": [100.0, 0.0, -50.0],
            "source": "synthetic_unit_test",
            "source_date": dates,
            "retrieved_at": "2024-01-05T00:00:00Z",
            "provider": "unit_test",
            "coverage_note": "complete synthetic test fixture",
        }
    )


def test_valid_artifact_classifies_as_usable_for_requested_window() -> None:
    result = validate_foreign_flow_artifact(_valid_rows(), ["AAA"], "2024-01-02", "2024-01-04")

    assert result["artifact_classification"] == "usable_for_requested_window"
    assert result["requested_ticker_date_coverage_rate"] == 1.0
    assert result["suitable_for_foreign_feature_interpretation"] is True
    assert result["provenance_complete"] is True


def test_test_only_artifact_classifies_as_fixture_only() -> None:
    frame = _valid_rows("TEST")

    result = validate_foreign_flow_artifact(frame, ["SSI"], "2024-01-02", "2024-01-04")

    assert result["artifact_classification"] == "fixture_only"
    assert result["only_fixture_tickers"] is True
    assert result["suitable_for_foreign_feature_interpretation"] is False


def test_missing_required_columns_classifies_as_schema_invalid() -> None:
    frame = pd.DataFrame({"date": pd.bdate_range("2024-01-02", "2024-01-04"), "foreign_net_value": [1, 2, 3]})

    result = validate_foreign_flow_artifact(frame, ["AAA"], "2024-01-02", "2024-01-04")

    assert result["artifact_classification"] == "schema_invalid"
    assert result["missing_required_columns"] == ["ticker"]


def test_empty_artifact_classifies_as_empty_or_missing() -> None:
    assert classify_foreign_flow_artifact(pd.DataFrame(), ["AAA"], "2024-01-02", "2024-01-04") == "empty_or_missing"


def test_partial_ticker_date_coverage_classifies_as_partial_coverage() -> None:
    frame = _valid_rows().iloc[[0, 2]].copy()

    result = validate_foreign_flow_artifact(frame, ["AAA"], "2024-01-02", "2024-01-04")

    assert result["artifact_classification"] == "partial_coverage"
    assert result["matched_ticker_date_count"] == 2
    assert result["requested_ticker_date_coverage_rate"] == 2 / 3


def test_missing_provenance_columns_are_reported_without_failing() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "date": pd.bdate_range("2024-01-02", "2024-01-04"),
            "foreign_net_value": [1.0, 2.0, 3.0],
        }
    )

    result = validate_foreign_flow_artifact(frame, ["AAA"], "2024-01-02", "2024-01-04")

    assert result["artifact_classification"] == "usable_for_requested_window"
    assert result["provenance_complete"] is False
    assert "provider" in result["missing_provenance_columns"]


def test_coverage_summary_reports_requested_window_counts() -> None:
    summary = summarize_foreign_flow_coverage(_valid_rows(), "AAA", "2024-01-02", "2024-01-04")

    assert summary["requested_business_date_count"] == 3
    assert summary["requested_ticker_date_count"] == 3
    assert summary["matched_ticker_date_count"] == 3


def test_diagnostic_script_reports_fixture_only_status(tmp_path: Path) -> None:
    ohlcv_dir = tmp_path / "ohlcv"
    ohlcv_dir.mkdir()
    pd.DataFrame(
        {
            "ticker": ["SSI", "SSI", "SSI"],
            "date": pd.bdate_range("2024-01-02", "2024-01-04"),
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [100, 100, 100],
        }
    ).to_csv(ohlcv_dir / "SSI.csv", index=False)
    foreign_path = tmp_path / "foreign_flow.csv"
    _valid_rows("TEST").to_csv(foreign_path, index=False)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_foreign_flow_coverage.py",
            "--tickers",
            "SSI",
            "--start-date",
            "2024-01-02",
            "--end-date",
            "2024-01-04",
            "--foreign-flow-path",
            str(foreign_path),
            "--ohlcv-dir",
            str(ohlcv_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["artifact_validation"]["artifact_classification"] == "fixture_only"
    assert report["artifact_validation"]["suitable_for_foreign_feature_interpretation"] is False


def test_curated_sample_fixture_validates_for_controlled_window() -> None:
    sample_path = Path("tests/fixtures/foreign_flow_sample.csv")
    sample = pd.read_csv(sample_path)

    result = validate_foreign_flow_artifact(sample, ["SSI", "FPT"], "2025-01-02", "2025-01-10")

    assert result["artifact_classification"] == "usable_for_requested_window"
    assert result["requested_ticker_date_coverage_rate"] == 1.0
    assert result["fixture_or_sample_source"] is True
    assert result["real_provider_evidence"] is False
    assert result["suitable_for_performance_interpretation"] is False
    assert "fixture" in result["artifact_usage_warning"].lower()


def test_diagnostic_script_reads_explicit_curated_sample_path(tmp_path: Path) -> None:
    ohlcv_dir = tmp_path / "ohlcv"
    ohlcv_dir.mkdir()
    dates = pd.bdate_range("2025-01-02", "2025-01-10")
    for ticker in ("SSI", "FPT"):
        pd.DataFrame(
            {
                "ticker": ticker,
                "date": dates,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 100,
            }
        ).to_csv(ohlcv_dir / f"{ticker}.csv", index=False)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_foreign_flow_coverage.py",
            "--tickers",
            "SSI,FPT",
            "--start-date",
            "2025-01-02",
            "--end-date",
            "2025-01-10",
            "--foreign-flow-path",
            "tests/fixtures/foreign_flow_sample.csv",
            "--ohlcv-dir",
            str(ohlcv_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["artifact_validation"]["artifact_classification"] == "usable_for_requested_window"
    assert report["artifact_validation"]["fixture_or_sample_source"] is True
    assert report["artifact_validation"]["real_provider_evidence"] is False
    assert report["artifact_validation"]["suitable_for_performance_interpretation"] is False
    assert all(row["exact_join_missing_ratio"] == 0.0 for row in report["ticker_results"])
