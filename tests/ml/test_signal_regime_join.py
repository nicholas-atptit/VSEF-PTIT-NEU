from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.ml.backtest.signal_regime_join import (
    JOIN_GOVERNANCE_REQUIRES_SOURCE_REVIEW,
    JOIN_GOVERNANCE_SAFE_IF_TRAILING,
    JOIN_MODE_DATE,
    JOIN_MODE_TICKER_DATE,
    JOIN_SUMMARY_FIELDS,
    join_regime_labels,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "prediction_date": "2025-01-02",
                "ticker": "SSI",
                "model_name": "stacking_final",
                "horizon": "short_10d",
                "predicted_return": 0.02,
                "actual_return": 0.01,
            },
            {
                "prediction_date": "2025-01-02",
                "ticker": "FPT",
                "model_name": "stacking_final",
                "horizon": "short_10d",
                "predicted_return": 0.03,
                "actual_return": -0.01,
            },
            {
                "prediction_date": "2025-01-03",
                "ticker": "SSI",
                "model_name": "stacking_final",
                "horizon": "short_10d",
                "predicted_return": 0.01,
                "actual_return": 0.02,
            },
        ]
    )


def test_date_join_works() -> None:
    regimes = pd.DataFrame(
        [
            {"date": "2025-01-02", "market_regime": "bull"},
            {"date": "2025-01-03", "market_regime": "bear"},
        ]
    )

    enriched, summary = join_regime_labels(_predictions(), regimes)

    assert enriched["regime"].tolist() == ["bull", "bull", "bear"]
    assert summary["matched_prediction_rows"] == 3
    assert summary["unmatched_prediction_rows"] == 0
    assert summary["matched_rate"] == pytest.approx(1.0)
    assert summary["join_governance"] == JOIN_GOVERNANCE_SAFE_IF_TRAILING


def test_ticker_date_join_works() -> None:
    regimes = pd.DataFrame(
        [
            {"date": "2025-01-02", "ticker": "SSI", "regime": "bull"},
            {"date": "2025-01-02", "ticker": "FPT", "regime": "sideway"},
            {"date": "2025-01-03", "ticker": "SSI", "regime": "bear"},
        ]
    )

    enriched, summary = join_regime_labels(_predictions(), regimes, join_mode=JOIN_MODE_TICKER_DATE)

    assert enriched["regime"].tolist() == ["bull", "sideway", "bear"]
    assert summary["join_mode"] == JOIN_MODE_TICKER_DATE
    assert summary["ticker_column"] == "ticker"


def test_unmatched_rows_are_preserved_with_missing_regime() -> None:
    regimes = pd.DataFrame([{"date": "2025-01-02", "regime": "bull"}])

    enriched, summary = join_regime_labels(_predictions(), regimes)

    assert len(enriched) == 3
    assert enriched["regime"].tolist()[:2] == ["bull", "bull"]
    assert pd.isna(enriched["regime"].iloc[2])
    assert summary["matched_prediction_rows"] == 2
    assert summary["unmatched_prediction_rows"] == 1


def test_existing_regime_column_is_not_overwritten_by_default() -> None:
    predictions = _predictions()
    predictions["regime"] = ["existing"] * len(predictions)
    regimes = pd.DataFrame([{"date": "2025-01-02", "regime": "bull"}, {"date": "2025-01-03", "regime": "bear"}])

    enriched, summary = join_regime_labels(predictions, regimes)

    assert enriched["regime"].tolist() == ["existing", "existing", "existing"]
    assert summary["existing_regime_column_present"] is True
    assert summary["existing_regime_values_preserved"] is True
    assert summary["join_applied"] is False


def test_overwrite_regime_replaces_existing_regime_column() -> None:
    predictions = _predictions()
    predictions["regime"] = ["existing"] * len(predictions)
    regimes = pd.DataFrame([{"date": "2025-01-02", "regime": "bull"}, {"date": "2025-01-03", "regime": "bear"}])

    enriched, summary = join_regime_labels(predictions, regimes, overwrite_regime=True)

    assert enriched["regime"].tolist() == ["bull", "bull", "bear"]
    assert summary["existing_regime_values_preserved"] is False
    assert summary["join_applied"] is True


def test_duplicate_regime_keys_are_detected() -> None:
    regimes = pd.DataFrame(
        [
            {"date": "2025-01-02", "regime": "bull"},
            {"date": "2025-01-02", "regime": "sideway"},
            {"date": "2025-01-03", "regime": "bear"},
        ]
    )

    enriched, summary = join_regime_labels(_predictions(), regimes)

    assert len(enriched) == 3
    assert summary["duplicate_regime_keys_exist"] is True
    assert summary["duplicate_regime_key_count"] == 2
    assert summary["join_governance"] == JOIN_GOVERNANCE_REQUIRES_SOURCE_REVIEW


def test_suspicious_future_looking_columns_are_flagged() -> None:
    regimes = pd.DataFrame(
        [
            {"date": "2025-01-02", "regime": "bull", "future_return": 0.05},
            {"date": "2025-01-03", "regime": "bear", "future_return": -0.01},
        ]
    )

    _enriched, summary = join_regime_labels(_predictions(), regimes)

    assert summary["suspicious_columns_present"] is True
    assert "future_return" in summary["suspicious_columns"]
    assert summary["join_governance"] == JOIN_GOVERNANCE_REQUIRES_SOURCE_REVIEW


def test_missing_prediction_date_column_fails_clearly() -> None:
    predictions = _predictions().drop(columns=["prediction_date"])
    regimes = pd.DataFrame([{"date": "2025-01-02", "regime": "bull"}])

    with pytest.raises(ValueError, match="Predictions missing required date column"):
        join_regime_labels(predictions, regimes)


def test_missing_regime_column_fails_clearly() -> None:
    regimes = pd.DataFrame([{"date": "2025-01-02", "not_regime": "bull"}])

    with pytest.raises(ValueError, match="Regime input missing regime label column"):
        join_regime_labels(_predictions(), regimes)


def test_cli_script_runs_on_synthetic_csvs(tmp_path: Path) -> None:
    prediction_path = tmp_path / "predictions.csv"
    regime_path = tmp_path / "regimes.csv"
    output_path = tmp_path / "enriched.csv"
    summary_path = tmp_path / "summary.json"
    _predictions().to_csv(prediction_path, index=False)
    pd.DataFrame(
        [
            {"date": "2025-01-02", "regime": "bull"},
            {"date": "2025-01-03", "regime": "bear"},
        ]
    ).to_csv(regime_path, index=False)
    repo_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/join_regime_to_predictions.py",
            "--predictions-path",
            str(prediction_path),
            "--regime-path",
            str(regime_path),
            "--output-path",
            str(output_path),
            "--summary-path",
            str(summary_path),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    enriched = pd.read_csv(output_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert enriched["regime"].tolist() == ["bull", "bull", "bear"]
    assert summary["matched_prediction_rows"] == 3
    assert "Signal regime join completed." in completed.stdout


def test_output_summary_has_stable_schema() -> None:
    regimes = pd.DataFrame(
        [
            {"date": "2025-01-02", "regime": "bull"},
            {"date": "2025-01-03", "regime": "bear"},
        ]
    )

    _enriched, summary = join_regime_labels(_predictions(), regimes, join_mode=JOIN_MODE_DATE)

    assert list(summary.keys()) == JOIN_SUMMARY_FIELDS
