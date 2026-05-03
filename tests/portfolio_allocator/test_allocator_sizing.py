from __future__ import annotations

import pytest

from src.portfolio_allocator import PortfolioAllocatorConfig, calculate_raw_weight, run_portfolio_allocator
from tests.portfolio_allocator.conftest import allocation_candidates


def test_aligned_strong_candidate_becomes_allocation_candidate() -> None:
    result = run_portfolio_allocator(allocation_candidates())

    row = result.allocation.iloc[0]
    assert row["allocation_status"] == "allocation_candidate"
    assert row["final_weight"] == pytest.approx(0.144)
    assert result.summary.loc[0, "portfolio_status"] == "allocation_candidate"


def test_raw_weight_uses_required_multipliers() -> None:
    frame = allocation_candidates(
        {
            "risk_adjusted_confidence": 0.90,
            "dominance_score": 0.20,
            "disagreement_score": 0.20,
            "risk_level": "level_2_candidate_filtering",
            "scenario_alignment": "weakly_aligned",
        }
    )

    weight = calculate_raw_weight(frame.iloc[0], PortfolioAllocatorConfig())

    assert weight == pytest.approx(0.20 * 0.90 * 0.70 * 0.80 * 0.50 * 0.50)


def test_max_position_weight_is_enforced() -> None:
    result = run_portfolio_allocator(
        allocation_candidates(
            {
                "risk_adjusted_confidence": 1.0,
                "disagreement_score": 0.0,
                "dominance_score": 0.50,
                "risk_score": 0.0,
            }
        ),
        config=PortfolioAllocatorConfig(max_position_weight=0.05),
    )

    row = result.allocation.iloc[0]
    assert row["allocation_status"] == "allocation_candidate"
    assert row["final_weight"] <= 0.05
