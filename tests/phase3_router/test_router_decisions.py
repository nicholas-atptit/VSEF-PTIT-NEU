from __future__ import annotations

import pandas as pd

from src.phase3_router import run_phase3_router


def allocation_frame(*records: dict[str, object]) -> pd.DataFrame:
    base = {
        "allocation_id": "portfolio_allocator_v1|decision_lane_v2|packet_001",
        "candidate_id": "decision_lane_v2|packet_001",
        "source_packet_id": "packet_001",
        "timestamp": "2026-01-02",
        "ticker": "AAA",
        "horizon": 5,
        "allocation_status": "allocation_candidate",
        "final_weight": 0.05,
        "risk_level": "level_1_soft_adjustment",
        "risk_score": 0.20,
        "risk_adjusted_confidence": 0.80,
        "disagreement_score": 0.10,
        "dominance_score": 0.35,
        "scenario_alignment": "aligned",
        "dominant_scenario": "bull",
    }
    rows = []
    for index, updates in enumerate(records or ({},), start=1):
        packet_id = f"packet_{index:03d}"
        row = {
            **base,
            "allocation_id": f"portfolio_allocator_v1|decision_lane_v2|{packet_id}",
            "candidate_id": f"decision_lane_v2|{packet_id}",
            "source_packet_id": packet_id,
            **updates,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def portfolio_summary(**updates: object) -> pd.DataFrame:
    row = {
        "portfolio_status": "allocation_candidate",
        "total_exposure": 0.05,
        "cash_weight": 0.95,
        "min_cash_buffer": 0.30,
        "max_total_exposure": 0.70,
        "effective_max_exposure": 0.70,
    }
    row.update(updates)
    return pd.DataFrame([row])


def test_valid_low_risk_high_confidence_candidate_routes() -> None:
    result = run_phase3_router(allocation_frame(), portfolio_summary_df=portfolio_summary())

    decisions = result.router_decisions
    assert decisions.loc[0, "route_decision"] == "route_allocation_candidate"
    assert decisions.loc[0, "route_reason"] == "valid_allocation_candidate"
    assert decisions.loc[0, "diagnostic_only_authority"]
    assert decisions.loc[0, "no_buy_sell_recommendation_authority"]


def test_allocator_no_allocation_row_becomes_no_candidate() -> None:
    result = run_phase3_router(
        allocation_frame({"allocation_status": "no_allocation", "final_weight": 0.0}),
        portfolio_summary_df=portfolio_summary(portfolio_status="all_cash", total_exposure=0.0, cash_weight=1.0),
    )

    assert result.router_decisions.loc[0, "route_decision"] == "no_candidate"
    assert result.router_decisions.loc[0, "route_reason"] == "allocator_status_not_allocation_candidate"


def test_hard_override_or_high_risk_rejects_candidate() -> None:
    hard = run_phase3_router(
        allocation_frame({"risk_level": "level_3_hard_override", "risk_score": 0.20}),
        portfolio_summary_df=portfolio_summary(),
    )
    high_score = run_phase3_router(allocation_frame({"risk_score": 0.70}), portfolio_summary_df=portfolio_summary())

    assert hard.router_decisions.loc[0, "route_decision"] == "reject"
    assert hard.router_decisions.loc[0, "route_reason"] == "level_3_hard_override"
    assert high_score.router_decisions.loc[0, "route_decision"] == "reject"
    assert high_score.router_decisions.loc[0, "route_reason"] == "risk_score_reject_threshold"


def test_adverse_dominant_scenario_biases_candidate_to_hold() -> None:
    result = run_phase3_router(
        allocation_frame({"dominant_scenario": "bear", "scenario_alignment": "weakly_aligned"}),
        portfolio_summary_df=portfolio_summary(),
    )

    assert result.router_decisions.loc[0, "route_decision"] == "hold"
    assert "adverse_dominant_scenario" in result.router_decisions.loc[0, "route_reason_codes"]
