from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.ml.backtest.signal_effectiveness import (
    EVALUATION_MODE_FRONTIER,
    EVALUATION_MODE_HELDOUT_THRESHOLD_SELECTION,
    POLICY_RETURN_THRESHOLD,
    SUCCESS_RAW_POSITIVE,
    SignalEffectivenessConfig,
    SignalEffectivenessRunner,
)


def _row(date: str, ticker: str, predicted: float, realized: float, model: str = "cart") -> dict[str, object]:
    return {
        "prediction_date": date,
        "target_date": str((pd.Timestamp(date) + pd.Timedelta(days=5)).date()),
        "ticker": ticker,
        "horizon": "short_10d",
        "model_name": model,
        "predicted_return": predicted,
        "predicted_direction": int(predicted > 0.0),
        "actual_return": realized,
    }


def _selection_and_test_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, realized in enumerate([0.01, 0.01, 0.01, -0.01, -0.01], start=1):
        rows.append(_row(f"2024-01-0{idx}", f"L{idx}", 0.006, realized))
    for idx, realized in enumerate([0.03, 0.03, 0.02, 0.01], start=1):
        rows.append(_row(f"2024-02-0{idx}", f"H{idx}", 0.035, realized))
    for idx, realized in enumerate([0.02, 0.02, 0.01, -0.01], start=1):
        rows.append(_row(f"2025-01-0{idx}", f"T{idx}", 0.035, realized))
    for idx, realized in enumerate([0.01, -0.01], start=1):
        rows.append(_row(f"2025-02-0{idx}", f"U{idx}", 0.006, realized))
    return pd.DataFrame(rows)


def _heldout_config(tmp_path: Path, *, minimum_signal_counts: list[int] | None = None) -> SignalEffectivenessConfig:
    return SignalEffectivenessConfig(
        output_dir=str(tmp_path / "heldout"),
        evaluation_mode=EVALUATION_MODE_HELDOUT_THRESHOLD_SELECTION,
        policy=POLICY_RETURN_THRESHOLD,
        predicted_return_thresholds=[0.005, 0.03],
        cost_per_trade_values=[0.0],
        slippage_values=[0.0],
        success_definition=SUCCESS_RAW_POSITIVE,
        minimum_signal_counts=minimum_signal_counts or [3],
        selection_start="2024-01-01",
        selection_end="2024-12-31",
        test_start="2025-01-01",
        test_end="2025-12-31",
        precision_targets=[0.60, 0.70, 0.80],
    )


def test_selection_period_chooses_intended_threshold(tmp_path) -> None:
    result = SignalEffectivenessRunner(_heldout_config(tmp_path)).run(_selection_and_test_frame())
    selected = result["selected_thresholds"].iloc[0]

    assert selected["selected_predicted_return_threshold"] == pytest.approx(0.03)
    assert selected["selection_buy_precision"] == pytest.approx(1.0)
    assert selected["selection_buy_count"] == 4


def test_heldout_metrics_use_only_test_period_rows(tmp_path) -> None:
    result = SignalEffectivenessRunner(_heldout_config(tmp_path)).run(_selection_and_test_frame())
    heldout = result["heldout_buy_precision"].iloc[0]
    heldout_rows = result["heldout_signal_rows"]

    assert heldout["heldout_buy_count"] == 4
    assert heldout["heldout_buy_precision"] == pytest.approx(0.75)
    assert pd.to_datetime(heldout_rows["prediction_date"]).min() >= pd.Timestamp("2025-01-01")
    assert pd.to_datetime(heldout_rows["prediction_date"]).max() <= pd.Timestamp("2025-12-31")


def test_heldout_realized_returns_do_not_affect_selection(tmp_path) -> None:
    base = _selection_and_test_frame()
    modified = base.copy()
    modified.loc[pd.to_datetime(modified["prediction_date"]) >= pd.Timestamp("2025-01-01"), "actual_return"] = -0.99

    first = SignalEffectivenessRunner(_heldout_config(tmp_path / "first")).run(base)
    second = SignalEffectivenessRunner(_heldout_config(tmp_path / "second")).run(modified)

    assert first["selected_thresholds"]["selected_predicted_return_threshold"].tolist() == second["selected_thresholds"][
        "selected_predicted_return_threshold"
    ].tolist()


def test_minimum_signal_count_constraint_is_enforced(tmp_path) -> None:
    rows: list[dict[str, object]] = []
    for idx, realized in enumerate([0.01, 0.01, 0.01], start=1):
        rows.append(_row(f"2024-01-0{idx}", f"L{idx}", 0.006, realized))
    for idx, realized in enumerate([0.03, 0.03], start=1):
        rows.append(_row(f"2024-02-0{idx}", f"H{idx}", 0.035, realized))
    for idx, realized in enumerate([0.02, 0.02], start=1):
        rows.append(_row(f"2025-01-0{idx}", f"T{idx}", 0.035, realized))

    result = SignalEffectivenessRunner(_heldout_config(tmp_path, minimum_signal_counts=[3])).run(pd.DataFrame(rows))
    selected = result["selected_thresholds"].iloc[0]

    assert selected["selected_predicted_return_threshold"] == pytest.approx(0.005)
    assert selected["selection_buy_count"] == 5


def test_precision_target_pass_fail_is_computed(tmp_path) -> None:
    result = SignalEffectivenessRunner(_heldout_config(tmp_path)).run(_selection_and_test_frame())
    pass_fail = result["precision_target_pass_fail"].sort_values("precision_target")

    assert pass_fail["precision_target"].tolist() == [0.60, 0.70, 0.80]
    assert pass_fail["pass_fail"].tolist() == [True, True, False]


def test_empty_selection_candidate_case_is_safe(tmp_path) -> None:
    config = _heldout_config(tmp_path, minimum_signal_counts=[100])
    result = SignalEffectivenessRunner(config).run(_selection_and_test_frame())

    assert result["selected_thresholds"].empty
    assert result["heldout_buy_precision"].empty
    assert result["precision_target_pass_fail"].empty
    assert not result["threshold_selection_trace"].empty
    assert result["threshold_selection_trace"]["selected"].sum() == 0


def test_empty_heldout_buy_case_is_safe(tmp_path) -> None:
    frame = _selection_and_test_frame()
    test_mask = pd.to_datetime(frame["prediction_date"]) >= pd.Timestamp("2025-01-01")
    frame.loc[test_mask, "predicted_return"] = 0.001
    frame.loc[test_mask, "predicted_direction"] = 1

    result = SignalEffectivenessRunner(_heldout_config(tmp_path)).run(frame)
    heldout = result["heldout_buy_precision"].iloc[0]
    pass_fail = result["precision_target_pass_fail"]

    assert heldout["heldout_buy_count"] == 0
    assert pd.isna(heldout["heldout_buy_precision"])
    assert pass_fail["pass_fail"].tolist() == [False, False, False]


def test_cli_heldout_mode_runs_on_synthetic_csv(tmp_path) -> None:
    input_path = tmp_path / "predictions.csv"
    output_dir = tmp_path / "heldout_cli"
    _selection_and_test_frame().to_csv(input_path, index=False)
    repo_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_signal_effectiveness_backtest.py",
            "--predictions-path",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--evaluation-mode",
            EVALUATION_MODE_HELDOUT_THRESHOLD_SELECTION,
            "--selection-start",
            "2024-01-01",
            "--selection-end",
            "2024-12-31",
            "--test-start",
            "2025-01-01",
            "--test-end",
            "2025-12-31",
            "--policy",
            POLICY_RETURN_THRESHOLD,
            "--threshold-grid",
            "0.005,0.03",
            "--cost-per-trade",
            "0.0",
            "--slippage",
            "0.0",
            "--success-definition",
            SUCCESS_RAW_POSITIVE,
            "--minimum-signal-count",
            "3",
            "--precision-targets",
            "0.60,0.70,0.80",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "selected_thresholds.csv").exists()
    assert (output_dir / "heldout_buy_precision.csv").exists()
    assert (output_dir / "precision_target_pass_fail.csv").exists()
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["evaluation_mode"] == EVALUATION_MODE_HELDOUT_THRESHOLD_SELECTION


def test_default_frontier_mode_remains_unchanged(tmp_path) -> None:
    config = SignalEffectivenessConfig(
        output_dir=str(tmp_path / "frontier"),
        evaluation_mode=EVALUATION_MODE_FRONTIER,
        policy=POLICY_RETURN_THRESHOLD,
        predicted_return_thresholds=[0.005],
        cost_per_trade_values=[0.0],
        slippage_values=[0.0],
        success_definition=SUCCESS_RAW_POSITIVE,
        minimum_signal_counts=[1],
    )
    result = SignalEffectivenessRunner(config).run(_selection_and_test_frame())

    assert "precision_coverage_frontier" in result
    assert "heldout_buy_precision" not in result
    assert (tmp_path / "frontier" / "precision_coverage_frontier.csv").exists()
