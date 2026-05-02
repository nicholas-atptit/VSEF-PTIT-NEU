from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import scripts.run_quant_core as runner
from tests.scenario.conftest import sample_forecasts, sample_regime, sample_risk, sample_signals, sample_strategy


def _annotate(frame: pd.DataFrame, core_row: dict[str, object], run_mode: str) -> pd.DataFrame:
    annotated = frame.copy()
    for column in (
        "core_run_id",
        "preset",
        "group_name",
        "horizon",
        "target_name",
        "target_type",
        "target_column",
        "target_family",
        "target_tradable",
        "ticker_count",
        "ticker_group_members",
    ):
        if column in core_row:
            annotated[column] = core_row[column]
    annotated["run_mode"] = run_mode
    return annotated


def _fake_quant_core_result(core_row: dict[str, object], *, run_mode: str) -> dict[str, object]:
    forecasts = _annotate(sample_forecasts(), core_row, run_mode)
    risk = _annotate(sample_risk(), core_row, run_mode)
    regime = _annotate(sample_regime(), core_row, run_mode)
    signals = _annotate(sample_signals(), core_row, run_mode)
    strategy = _annotate(sample_strategy(), core_row, run_mode)
    forecast_summary = pd.DataFrame(
        [
            {
                "model_name": "lightgbm",
                "rmse": 0.01,
                "mae": 0.01,
                "directional_accuracy": 1.0,
                "observations": 1,
            }
        ]
    )
    forecast_summary_by_horizon = pd.DataFrame(
        [
            {
                "model_name": "lightgbm",
                "horizon": core_row["horizon"],
                "target_type": core_row["target_type"],
                "rmse": 0.01,
                "mae": 0.01,
                "directional_accuracy": 1.0,
                "observations": 1,
            }
        ]
    )
    window_summary = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "window_id": "window_001",
                "train_start": "2023-01-01",
                "train_end": "2023-12-31",
                "test_start": "2024-01-02",
                "test_end": "2024-01-02",
            }
        ]
    )
    execution_log = pd.DataFrame(
        [
            {
                "model_name": "lightgbm",
                "model_family": "boosting",
                "model_role": "primary_research",
                "model_status": "active",
                "run_success": True,
                "warning_count": 0,
                "missing_output_count": 0,
                "failure_reason": "",
            },
            {
                "model_name": "xgboost",
                "model_family": "boosting",
                "model_role": "primary_research",
                "model_status": "active",
                "run_success": True,
                "warning_count": 0,
                "missing_output_count": 0,
                "failure_reason": "",
            },
            {
                "model_name": "weighted_ensemble",
                "model_family": "ensemble",
                "model_role": "ensemble",
                "model_status": "derived",
                "run_success": True,
                "warning_count": 0,
                "missing_output_count": 0,
                "failure_reason": "",
            },
        ]
    )
    return {
        "forecasts": forecasts,
        "forecast_summary": _annotate(forecast_summary, core_row, run_mode),
        "forecast_summary_by_horizon": _annotate(forecast_summary_by_horizon, core_row, run_mode),
        "window_summary": _annotate(window_summary, core_row, run_mode),
        "risk_summary": risk,
        "regime_summary": regime,
        "signals": signals,
        "positions": signals,
        "trades": pd.DataFrame(),
        "strategy_metrics": strategy,
        "equity_curve": pd.DataFrame(),
        "policy_summary": strategy,
        "model_execution_log": _annotate(execution_log, core_row, run_mode),
        "evaluated_models": ["lightgbm", "xgboost", "weighted_ensemble"],
        "skipped_models": [],
    }


def test_quant_core_runner_writes_scenario_outputs_when_enabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_quant_core.py",
            "--preset",
            "smoke",
            "--run-mode",
            "research_core",
            "--output-dir",
            str(tmp_path),
            "--enable-scenario-engine",
            "--scenario-calibration-lookback",
            "0",
        ],
    )
    monkeypatch.setattr(
        runner,
        "run_quant_core_scenario",
        lambda core_row, **kwargs: _fake_quant_core_result(core_row, run_mode=str(kwargs["run_mode"])),
    )
    monkeypatch.setattr(runner, "collect_git_metadata", lambda path: {"branch": "test", "commit_hash": "abc123", "is_dirty": False})
    monkeypatch.setattr(runner, "collect_runtime_metadata", lambda: {"python_executable": sys.executable})
    monkeypatch.setattr(runner, "collect_dependency_versions", lambda packages: {"pandas": pd.__version__})

    assert runner.main() == 0

    assert (tmp_path / "scenario_probability.csv").exists()
    assert (tmp_path / "scenario_rankings.csv").exists()
    assert (tmp_path / "scenario_dominance_summary.csv").exists()
    packet = json.loads((tmp_path / "analysis_packets.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "scenario_summary" in packet
    assert "dominant_scenario" in packet
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert "scenario_probability" in manifest["artifact_paths"]
    assert manifest["run_counts"]["scenario_probability_rows"] == 7
