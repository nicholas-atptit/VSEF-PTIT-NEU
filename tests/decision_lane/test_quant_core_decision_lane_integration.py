from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import scripts.run_quant_core as runner
from tests.scenario.test_quant_core_scenario_integration import _fake_quant_core_result


def test_quant_core_runner_writes_decision_lane_enriched_outputs_when_risk_governance_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
        ],
    )
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

    assert runner.main() == 0

    assert (tmp_path / "decision_lane_candidates.csv").exists()
    assert (tmp_path / "decision_lane_enriched_candidates.csv").exists()
    assert (tmp_path / "decision_lane_manifest.json").exists()
    legacy = pd.read_csv(tmp_path / "decision_lane_candidates.csv")
    enriched = pd.read_csv(tmp_path / "decision_lane_enriched_candidates.csv")
    assert len(enriched) == len(legacy)
    assert "candidate_id" in enriched.columns
    assert "source_packet_id" in enriched.columns
    assert "reason_summary" in enriched.columns

    decision_manifest = json.loads((tmp_path / "decision_lane_manifest.json").read_text(encoding="utf-8"))
    assert decision_manifest["diagnostic_only_authority"]
    assert decision_manifest["no_buy_sell_recommendation_authority"]

    run_manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert "decision_lane_enriched_candidates" in run_manifest["artifact_paths"]
    assert "decision_lane_manifest" in run_manifest["artifact_paths"]
    assert run_manifest["run_counts"]["decision_lane_enriched_candidate_rows"] == len(enriched)
