from __future__ import annotations

import pandas as pd

from src.portfolio_allocator import PortfolioAllocatorConfig, run_portfolio_allocator
from tests.portfolio_allocator.conftest import allocation_candidates


def test_min_cash_buffer_is_enforced() -> None:
    frame = allocation_candidates(*[{"ticker": f"T{i:02d}"} for i in range(10)])

    result = run_portfolio_allocator(frame)

    assert result.summary.loc[0, "total_exposure"] <= 0.70 + 1e-12
    assert result.summary.loc[0, "cash_weight"] >= 0.30


def test_total_exposure_cap_is_enforced() -> None:
    frame = allocation_candidates(
        *[
            {
                "ticker": f"T{i:02d}",
                "risk_adjusted_confidence": 1.0,
                "disagreement_score": 0.0,
                "dominance_score": 0.50,
                "risk_score": 0.0,
            }
            for i in range(6)
        ]
    )

    result = run_portfolio_allocator(
        frame,
        config=PortfolioAllocatorConfig(max_total_exposure=0.50, min_cash_buffer=0.30),
    )

    assert result.summary.loc[0, "total_exposure"] <= 0.50 + 1e-12
    assert result.summary.loc[0, "cash_weight"] >= 0.50


def test_no_candidates_results_in_all_cash_portfolio_summary() -> None:
    columns = allocation_candidates().columns

    result = run_portfolio_allocator(pd.DataFrame(columns=columns))

    assert result.summary.loc[0, "portfolio_status"] == "all_cash"
    assert result.summary.loc[0, "total_exposure"] == 0.0
    assert result.summary.loc[0, "cash_weight"] == 1.0
    assert result.allocation.empty


def test_multiple_candidates_are_ranked_by_lower_disagreement_and_higher_dominance() -> None:
    frame = allocation_candidates(
        {
            "candidate_id": "decision_lane_v2|packet_low_priority",
            "source_packet_id": "packet_low_priority",
            "ticker": "LOW",
            "disagreement_score": 0.20,
            "dominance_score": 0.30,
        },
        {
            "candidate_id": "decision_lane_v2|packet_high_priority",
            "source_packet_id": "packet_high_priority",
            "ticker": "HIGH",
            "disagreement_score": 0.05,
            "dominance_score": 0.45,
        },
    )

    result = run_portfolio_allocator(frame)

    allocated = result.allocation[result.allocation["allocation_status"] == "allocation_candidate"]
    assert allocated.iloc[0]["ticker"] == "HIGH"
    assert allocated.iloc[0]["allocation_priority_score"] > allocated.iloc[1]["allocation_priority_score"]
