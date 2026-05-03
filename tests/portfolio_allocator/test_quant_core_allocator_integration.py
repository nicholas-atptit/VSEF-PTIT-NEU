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


def test_quant_core_runner_writes_portfolio_allocator_outputs_when_enabled(
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
            "--enable-portfolio-allocator",
        ],
    )
    _patch_lightweight_quant_core(monkeypatch)

    assert runner.main() == 0

    assert (tmp_path / "portfolio_allocation.csv").exists()
    assert (tmp_path / "portfolio_summary.csv").exists()
    assert (tmp_path / "portfolio_risk_summary.csv").exists()
    assert (tmp_path / "portfolio_decision_cards.jsonl").exists()
    assert (tmp_path / "allocator_manifest.json").exists()

    allocation = pd.read_csv(tmp_path / "portfolio_allocation.csv")
    assert set(allocation["allocation_status"]) <= {"allocation_candidate", "no_allocation"}

    allocator_manifest = json.loads((tmp_path / "allocator_manifest.json").read_text(encoding="utf-8"))
    assert allocator_manifest["diagnostic_only_authority"]
    assert allocator_manifest["no_buy_sell_recommendation_authority"]

    run_manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert "portfolio_allocation" in run_manifest["artifact_paths"]
    assert "allocator_manifest" in run_manifest["artifact_paths"]
    assert run_manifest["run_counts"]["portfolio_allocation_rows"] == len(allocation)


def test_quant_core_allocator_flag_without_enriched_candidates_writes_no_allocation(
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
            "--enable-portfolio-allocator",
        ],
    )
    _patch_lightweight_quant_core(monkeypatch)

    assert runner.main() == 0

    allocation = pd.read_csv(tmp_path / "portfolio_allocation.csv")
    manifest = json.loads((tmp_path / "allocator_manifest.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(tmp_path / "portfolio_summary.csv")

    assert allocation.loc[0, "allocation_status"] == "no_allocation"
    assert allocation.loc[0, "no_allocation_reason"] == "missing_enriched_candidates"
    assert summary.loc[0, "portfolio_status"] == "all_cash"
    assert manifest["missing_enriched_candidates"]
