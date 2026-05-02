from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.risk_governance import run_risk_governance, write_risk_governance_outputs
from tests.risk_governance.conftest import build_governance_inputs, build_scenario_governance_frames


def test_missing_scenario_data_does_not_crash() -> None:
    inputs = build_governance_inputs()

    result = run_risk_governance(
        candidates_df=inputs["candidates"],
        packets_df=inputs["packets"],
        risk_df=inputs["risk"],
        consensus_df=inputs["consensus"],
        model_health_df=inputs["health"],
    )

    assert len(result.risk_governance_summary) == len(inputs["candidates"])
    assert "scenario_dispersion_component" in result.risk_governance_summary.columns
    assert result.risk_governance_summary["scenario_dispersion_component"].notna().all()


def test_output_artifacts_are_written(tmp_path: Path) -> None:
    inputs = build_governance_inputs()
    scenario = build_scenario_governance_frames(inputs["packets"].iloc[0])
    result = run_risk_governance(
        candidates_df=inputs["candidates"],
        packets_df=inputs["packets"],
        risk_df=inputs["risk"],
        consensus_df=inputs["consensus"],
        model_health_df=inputs["health"],
        scenario_dominance_df=scenario["dominance"],
        scenario_uncertainty_df=scenario["uncertainty"],
        scenario_probability_df=scenario["probability"],
    )

    paths = write_risk_governance_outputs(tmp_path, result)

    assert (tmp_path / "risk_governance_summary.csv").exists()
    assert (tmp_path / "risk_adjusted_candidates.csv").exists()
    assert (tmp_path / "risk_override_log.csv").exists()
    assert (tmp_path / "risk_manifest.json").exists()
    assert set(paths) == {
        "risk_governance_summary",
        "risk_adjusted_candidates",
        "risk_override_log",
        "risk_manifest",
    }
    adjusted = pd.read_csv(tmp_path / "risk_adjusted_candidates.csv")
    assert "risk_adjusted_candidate_score" in adjusted.columns
    manifest = json.loads((tmp_path / "risk_manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_type"] == "risk_governance_layer_v1_manifest"
    assert manifest["diagnostic_only_authority"]
    assert manifest["no_buy_sell_recommendation_authority"]


def test_deterministic_reproducibility_for_identical_inputs() -> None:
    inputs = build_governance_inputs()
    scenario = build_scenario_governance_frames(inputs["packets"].iloc[0])
    kwargs = {
        "candidates_df": inputs["candidates"],
        "packets_df": inputs["packets"],
        "risk_df": inputs["risk"],
        "consensus_df": inputs["consensus"],
        "model_health_df": inputs["health"],
        "scenario_dominance_df": scenario["dominance"],
        "scenario_uncertainty_df": scenario["uncertainty"],
        "scenario_probability_df": scenario["probability"],
    }

    first = run_risk_governance(**kwargs)
    second = run_risk_governance(**kwargs)

    pd.testing.assert_frame_equal(first.risk_governance_summary, second.risk_governance_summary)
    pd.testing.assert_frame_equal(first.risk_adjusted_candidates, second.risk_adjusted_candidates)
    pd.testing.assert_frame_equal(first.risk_override_log, second.risk_override_log)
    assert first.manifest == second.manifest
