from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import scripts.run_quant_core as runner
from tests.scenario.test_quant_core_scenario_integration import _fake_quant_core_result


def test_quant_core_runner_writes_risk_governance_outputs_when_enabled(tmp_path: Path, monkeypatch) -> None:
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

    assert (tmp_path / "risk_governance_summary.csv").exists()
    assert (tmp_path / "risk_adjusted_candidates.csv").exists()
    assert (tmp_path / "risk_override_log.csv").exists()
    assert (tmp_path / "risk_manifest.json").exists()
    adjusted = pd.read_csv(tmp_path / "risk_adjusted_candidates.csv")
    assert "risk_adjusted_candidate_score" in adjusted.columns
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert "risk_adjusted_candidates" in manifest["artifact_paths"]
    assert "risk_manifest" in manifest["artifact_paths"]
    assert manifest["run_counts"]["risk_governance_rows"] == len(adjusted)
