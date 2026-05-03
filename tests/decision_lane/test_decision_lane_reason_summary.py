from __future__ import annotations

from src.reporting.decision_lane import build_enriched_decision_lane_candidates
from tests.risk_governance.conftest import build_governance_inputs, build_scenario_governance_frames


def test_reason_summary_is_generated() -> None:
    inputs = build_governance_inputs(candidate_updates={"agreement_bucket": "medium"})
    risk_adjusted = inputs["candidates"].copy()
    risk_adjusted["confidence_adjustment_factor"] = 0.80
    risk_adjusted["risk_action"] = "adjust_confidence"
    risk_adjusted["risk_reason_codes"] = "elevated_drawdown"
    risk_adjusted["risk_adjusted_candidate_score"] = risk_adjusted["candidate_score"] * 0.80
    scenario = build_scenario_governance_frames(inputs["packets"].iloc[0], dominant_scenario="bull")

    enriched = build_enriched_decision_lane_candidates(
        inputs["candidates"],
        inputs["packets"],
        risk_adjusted_candidates_df=risk_adjusted,
        scenario_dominance_df=scenario["dominance"],
        scenario_probability_df=scenario["probability"],
    )

    summary = enriched.loc[0, "reason_summary"]
    assert summary.startswith("Candidate is aligned with bull scenario")
    assert "medium model agreement" in summary
    assert "confidence-adjusted" in summary
    assert "scenario_aligned" in enriched.loc[0, "reason_codes"]
    assert "risk_adjusted" in enriched.loc[0, "reason_codes"]
    assert "elevated_drawdown" in enriched.loc[0, "reason_codes"]


def test_missing_scenario_and_risk_data_does_not_crash() -> None:
    inputs = build_governance_inputs()

    enriched = build_enriched_decision_lane_candidates(inputs["candidates"], inputs["packets"])

    assert len(enriched) == len(inputs["candidates"])
    assert enriched.loc[0, "scenario_alignment"] == "unknown"
    assert enriched.loc[0, "candidate_status"] == "diagnostic_candidate"
    assert enriched.loc[0, "reason_summary"]
