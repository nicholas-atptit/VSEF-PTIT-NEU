from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import scripts.run_quant_core as runner
from tests.scenario.test_quant_core_scenario_integration import _fake_quant_core_result


def _patch_lightweight_quant_core(monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "run_quant_core_scenario",
        lambda core_row, **kwargs: _fake_quant_core_result(core_row, run_mode=str(kwargs["run_mode"])),
    )
    monkeypatch.setattr(
        runner,
        "collect_git_metadata",
        lambda path: {"branch": "test", "commit_hash": "abc123", "is_dirty": False},
    )
    monkeypatch.setattr(runner, "collect_runtime_metadata", lambda: {"python_executable": sys.executable})
    monkeypatch.setattr(runner, "collect_dependency_versions", lambda packages: {"pandas": pd.__version__})


def test_quant_core_runner_writes_phase3_router_outputs_when_enabled(tmp_path: Path, monkeypatch) -> None:
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
            "--enable-risk-governance",
            "--enable-portfolio-allocator",
            "--enable-phase3-router",
        ],
    )
    _patch_lightweight_quant_core(monkeypatch)

    assert runner.main() == 0

    assert (tmp_path / "router_decisions.csv").exists()
    assert (tmp_path / "router_summary.csv").exists()
    assert (tmp_path / "router_manifest.json").exists()

    decisions = pd.read_csv(tmp_path / "router_decisions.csv")
    assert set(decisions["route_decision"]) <= {"route_allocation_candidate", "hold", "reject", "no_candidate"}
    assert decisions["diagnostic_only_authority"].all()
    assert decisions["no_buy_sell_recommendation_authority"].all()

    run_manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert "router_decisions" in run_manifest["artifact_paths"]
    assert "router_manifest" in run_manifest["artifact_paths"]
    assert run_manifest["run_counts"]["router_decision_rows"] == len(decisions)
