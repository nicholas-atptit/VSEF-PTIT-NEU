from __future__ import annotations

from src.risk_governance import run_risk_governance
from tests.risk_governance.conftest import build_governance_inputs, build_scenario_governance_frames


def test_model_health_affects_risk_score() -> None:
    healthy_inputs = build_governance_inputs()
    failing_inputs = build_governance_inputs(health_status="failing")

    healthy = run_risk_governance(
        candidates_df=healthy_inputs["candidates"],
        packets_df=healthy_inputs["packets"],
        risk_df=healthy_inputs["risk"],
        consensus_df=healthy_inputs["consensus"],
        model_health_df=healthy_inputs["health"],
    ).risk_governance_summary
    failing = run_risk_governance(
        candidates_df=failing_inputs["candidates"],
        packets_df=failing_inputs["packets"],
        risk_df=failing_inputs["risk"],
        consensus_df=failing_inputs["consensus"],
        model_health_df=failing_inputs["health"],
    ).risk_governance_summary

    assert failing.loc[0, "model_health_component"] == 1.0
    assert failing.loc[0, "risk_score"] > healthy.loc[0, "risk_score"]


def test_scenario_dispersion_affects_risk_score() -> None:
    inputs = build_governance_inputs()
    scenario = build_scenario_governance_frames(inputs["packets"].iloc[0], uncertainty_score=0.85, dominance_score=0.10)

    without_scenario = run_risk_governance(
        candidates_df=inputs["candidates"],
        packets_df=inputs["packets"],
        risk_df=inputs["risk"],
        consensus_df=inputs["consensus"],
        model_health_df=inputs["health"],
    ).risk_governance_summary
    with_scenario = run_risk_governance(
        candidates_df=inputs["candidates"],
        packets_df=inputs["packets"],
        risk_df=inputs["risk"],
        consensus_df=inputs["consensus"],
        model_health_df=inputs["health"],
        scenario_dominance_df=scenario["dominance"],
        scenario_uncertainty_df=scenario["uncertainty"],
        scenario_probability_df=scenario["probability"],
    ).risk_governance_summary

    assert with_scenario.loc[0, "scenario_dispersion_component"] >= 0.85
    assert with_scenario.loc[0, "risk_score"] > without_scenario.loc[0, "risk_score"]


def test_disagreement_and_sign_conflict_affect_risk_score() -> None:
    calm_inputs = build_governance_inputs()
    conflict_inputs = build_governance_inputs(
        consensus_updates={
            "agreement_score": 0.20,
            "disagreement_score": 0.80,
            "agreement_bucket": "low",
            "sign_conflict": True,
        }
    )

    calm = run_risk_governance(
        candidates_df=calm_inputs["candidates"],
        packets_df=calm_inputs["packets"],
        risk_df=calm_inputs["risk"],
        consensus_df=calm_inputs["consensus"],
        model_health_df=calm_inputs["health"],
    ).risk_governance_summary
    conflict = run_risk_governance(
        candidates_df=conflict_inputs["candidates"],
        packets_df=conflict_inputs["packets"],
        risk_df=conflict_inputs["risk"],
        consensus_df=conflict_inputs["consensus"],
        model_health_df=conflict_inputs["health"],
    ).risk_governance_summary

    assert conflict.loc[0, "disagreement_component"] == 1.0
    assert "sign_conflict" in conflict.loc[0, "risk_reason_codes"]
    assert conflict.loc[0, "risk_score"] > calm.loc[0, "risk_score"]
