from __future__ import annotations

import pandas as pd
import pytest

from src.reporting.analysis_packets import build_decision_lane_candidates as legacy_candidate_builder
from src.reporting.decision_lane import (
    build_decision_lane_candidates,
    build_enriched_decision_lane_candidates,
)
from tests.risk_governance.conftest import build_governance_inputs, build_scenario_governance_frames


def test_candidate_id_is_generated_and_source_packet_id_is_preserved() -> None:
    inputs = build_governance_inputs()
    scenario = build_scenario_governance_frames(inputs["packets"].iloc[0], dominant_scenario="bull")

    enriched = build_enriched_decision_lane_candidates(
        inputs["candidates"],
        inputs["packets"],
        scenario_dominance_df=scenario["dominance"],
        scenario_probability_df=scenario["probability"],
    )

    assert enriched.loc[0, "candidate_id"] == f"decision_lane_v2|{inputs['candidates'].loc[0, 'packet_id']}"
    assert enriched.loc[0, "source_packet_id"] == inputs["candidates"].loc[0, "packet_id"]


@pytest.mark.parametrize("scenario_label", ["bull", "recovery"])
def test_scenario_alignment_is_aligned_for_bull_or_recovery(scenario_label: str) -> None:
    inputs = build_governance_inputs()
    scenario = build_scenario_governance_frames(inputs["packets"].iloc[0], dominant_scenario=scenario_label)

    enriched = build_enriched_decision_lane_candidates(
        inputs["candidates"],
        inputs["packets"],
        scenario_dominance_df=scenario["dominance"],
        scenario_probability_df=scenario["probability"],
    )

    assert enriched.loc[0, "scenario_alignment"] == "aligned"


@pytest.mark.parametrize("scenario_label", ["bear", "drawdown", "high_volatility"])
def test_scenario_alignment_is_misaligned_or_risky_for_downside_scenarios(scenario_label: str) -> None:
    inputs = build_governance_inputs()
    scenario = build_scenario_governance_frames(inputs["packets"].iloc[0], dominant_scenario=scenario_label)

    enriched = build_enriched_decision_lane_candidates(
        inputs["candidates"],
        inputs["packets"],
        scenario_dominance_df=scenario["dominance"],
        scenario_probability_df=scenario["probability"],
    )

    assert enriched.loc[0, "scenario_alignment"] == "misaligned_or_risky"


def test_risk_adjusted_confidence_uses_confidence_adjustment_factor() -> None:
    inputs = build_governance_inputs()
    risk_adjusted = inputs["candidates"].copy()
    risk_adjusted["confidence_adjustment_factor"] = 0.25
    risk_adjusted["risk_action"] = "adjust_confidence"
    risk_adjusted["risk_adjusted_candidate_score"] = risk_adjusted["candidate_score"] * 0.25

    enriched = build_enriched_decision_lane_candidates(
        inputs["candidates"],
        inputs["packets"],
        risk_adjusted_candidates_df=risk_adjusted,
    )

    assert enriched.loc[0, "risk_adjusted_confidence"] == pytest.approx(
        inputs["candidates"].loc[0, "model_agreement_score"] * 0.25
    )


@pytest.mark.parametrize(
    ("risk_fields", "expected_status"),
    [
        ({"force_hold": True, "block_candidate": True, "risk_action": "force_hold"}, "force_hold"),
        ({"force_hold": False, "block_candidate": True, "risk_action": "block_candidate"}, "blocked"),
        ({"force_hold": False, "block_candidate": False, "risk_action": "reduce_candidate"}, "reduced"),
        ({"force_hold": False, "block_candidate": False, "risk_action": "adjust_confidence"}, "adjusted"),
    ],
)
def test_risk_statuses_are_assigned_correctly(risk_fields: dict[str, object], expected_status: str) -> None:
    inputs = build_governance_inputs()
    risk_adjusted = inputs["candidates"].copy()
    for column, value in risk_fields.items():
        risk_adjusted[column] = value
    risk_adjusted["confidence_adjustment_factor"] = 0.50
    risk_adjusted["risk_adjusted_candidate_score"] = risk_adjusted["candidate_score"] * 0.50

    enriched = build_enriched_decision_lane_candidates(
        inputs["candidates"],
        inputs["packets"],
        risk_adjusted_candidates_df=risk_adjusted,
    )

    assert enriched.loc[0, "candidate_status"] == expected_status


def test_disagreement_score_is_present() -> None:
    inputs = build_governance_inputs()

    enriched = build_enriched_decision_lane_candidates(inputs["candidates"], inputs["packets"])

    assert "disagreement_score" in enriched.columns
    assert pd.notna(enriched.loc[0, "disagreement_score"])


def test_existing_decision_lane_candidates_behavior_is_not_broken() -> None:
    inputs = build_governance_inputs()

    legacy = legacy_candidate_builder(inputs["packets"])
    mirrored = build_decision_lane_candidates(inputs["packets"])

    pd.testing.assert_frame_equal(legacy, mirrored)
    assert list(legacy.columns) == [
        "packet_id",
        "timestamp",
        "ticker",
        "group_name",
        "horizon",
        "target_type",
        "run_mode",
        "primary_model_name",
        "primary_prediction",
        "model_agreement_score",
        "agreement_bucket",
        "regime_label",
        "volatility_bucket",
        "active_signal_count",
        "top_policy_model",
        "top_policy_sharpe",
        "candidate_score",
    ]
