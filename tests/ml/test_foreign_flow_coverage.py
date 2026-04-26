from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_foreign_flow_coverage import inspect_foreign_flow_coverage
from src.ml.backtest.context_coverage_diagnostics import build_context_coverage_rows
from src.ml.data_loader import apply_context_features


def _base_ohlcv(ticker: str = "AAA", dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    dates = dates if dates is not None else pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    close = pd.Series(np.linspace(100.0, 102.0, len(dates)))
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1_000.0, 1_200.0, len(dates)),
        }
    )


def _fold_context() -> dict[str, object]:
    return {
        "ticker": "AAA",
        "fold_id": "fold_001",
        "step_size": 1,
        "forecast_sequence_index": 0,
        "prediction_date": "2024-01-04",
        "horizon": "short_5d",
        "train_start": "2024-01-02",
        "train_end": "2024-01-04",
        "eval_start": "2024-01-04",
        "eval_end": "2024-01-11",
    }


def _write_ohlcv(path: Path, ticker: str = "AAA") -> None:
    frame = _base_ohlcv(ticker=ticker)
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / f"{ticker}.csv", index=False)


def test_foreign_flow_coverage_reports_full_missing_when_no_source_rows_exist() -> None:
    contextual = apply_context_features(_base_ohlcv(), "AAA", foreign_flow_df=pd.DataFrame())

    coverage = build_context_coverage_rows(feature_frame=contextual, fold_context=_fold_context()).iloc[0]

    assert coverage["row_count"] == 3
    assert coverage["foreign_flow_available_count"] == 0
    assert coverage["foreign_flow_missing_count"] == 3
    assert coverage["foreign_flow_missing_rate"] == 1.0
    assert coverage["coverage_warning_level"] == "weak_coverage"


def test_exact_ticker_date_foreign_flow_join_sets_availability_metadata() -> None:
    foreign_flow = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            "foreign_net_value": [10_000.0, -5_000.0],
        }
    )

    result = apply_context_features(_base_ohlcv(), "AAA", foreign_flow_df=foreign_flow).set_index("date")

    assert bool(result.loc[pd.Timestamp("2024-01-02"), "foreign_flow_context_available"]) is True
    assert bool(result.loc[pd.Timestamp("2024-01-03"), "foreign_flow_context_available"]) is False
    assert bool(result.loc[pd.Timestamp("2024-01-04"), "foreign_flow_context_available"]) is True
    assert result.loc[pd.Timestamp("2024-01-04"), "foreign_net_value"] == -5_000.0


def test_ticker_mismatch_produces_missing_foreign_flow_coverage() -> None:
    foreign_flow = pd.DataFrame(
        {
            "ticker": ["BBB"],
            "date": pd.to_datetime(["2024-01-02"]),
            "foreign_net_value": [10_000.0],
        }
    )

    result = apply_context_features(_base_ohlcv(), "AAA", foreign_flow_df=foreign_flow)

    assert result["foreign_flow_context_available"].sum() == 0
    assert result["foreign_flow_context_missing"].sum() == 3


def test_date_mismatch_produces_missing_foreign_flow_coverage() -> None:
    foreign_flow = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "date": pd.to_datetime(["2024-02-01"]),
            "foreign_net_value": [10_000.0],
        }
    )

    result = apply_context_features(_base_ohlcv(), "AAA", foreign_flow_df=foreign_flow)

    assert result["foreign_flow_context_available"].sum() == 0
    assert result["foreign_flow_context_missing"].sum() == 3


def test_matching_source_rows_produce_non_missing_coverage() -> None:
    foreign_flow = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "foreign_net_value": [10_000.0, 0.0, -5_000.0],
        }
    )

    contextual = apply_context_features(_base_ohlcv(), "AAA", foreign_flow_df=foreign_flow)
    coverage = build_context_coverage_rows(feature_frame=contextual, fold_context=_fold_context()).iloc[0]

    assert coverage["foreign_flow_available_count"] == 3
    assert coverage["foreign_flow_missing_count"] == 0
    assert coverage["foreign_flow_missing_rate"] == 0.0


def test_probe_reports_missing_source_without_crashing(tmp_path: Path) -> None:
    ohlcv_dir = tmp_path / "ohlcv"
    _write_ohlcv(ohlcv_dir)

    report = inspect_foreign_flow_coverage(
        tickers=["AAA"],
        start_date="2024-01-02",
        end_date="2024-01-04",
        foreign_flow_path=tmp_path / "missing_foreign_flow.csv",
        ohlcv_dir=ohlcv_dir,
    )

    assert report["foreign_flow_source"]["source_status"] == "missing_local_artifact"
    assert report["foreign_flow_source"]["provider_fetch_attempted"] is False
    assert report["ticker_results"][0]["exact_join_missing_ratio"] == 1.0


def test_probe_runs_against_synthetic_local_paths_without_provider(tmp_path: Path) -> None:
    ohlcv_dir = tmp_path / "ohlcv"
    _write_ohlcv(ohlcv_dir)
    foreign_path = tmp_path / "foreign_flow.csv"
    pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "date": ["2024-01-02", "2024-01-04", "2024-01-03"],
            "foreign_net_value": [1.0, 2.0, 3.0],
        }
    ).to_csv(foreign_path, index=False)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_foreign_flow_coverage.py",
            "--tickers",
            "AAA",
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

    assert report["foreign_flow_source"]["source_status"] == "loaded_local_artifact"
    assert report["ticker_results"][0]["exact_join_match_count"] == 2
    assert report["ticker_results"][0]["exact_join_missing_count"] == 1
    assert report["ticker_results"][0]["exact_join_would_succeed"] is True
