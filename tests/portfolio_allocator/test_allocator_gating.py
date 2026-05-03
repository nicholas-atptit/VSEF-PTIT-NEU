from __future__ import annotations

import pytest

from src.portfolio_allocator import run_portfolio_allocator
from tests.portfolio_allocator.conftest import allocation_candidates


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"risk_level": "level_3_hard_override"}, "risk_level_3_hard_override"),
        ({"risk_adjusted_confidence": 0.10}, "risk_adjusted_confidence_below_threshold"),
        ({"disagreement_score": 0.80}, "disagreement_score_above_threshold"),
        ({"dominance_score": 0.10}, "dominance_score_below_threshold"),
        ({"candidate_status": "blocked"}, "candidate_status_blocked"),
        ({"candidate_status": "force_hold"}, "candidate_status_force_hold"),
        ({"risk_action": "force_hold"}, "risk_action_force_hold"),
        ({"scenario_alignment": "misaligned_or_risky"}, "scenario_alignment_misaligned_or_risky"),
        ({"scenario_confidence_bucket": "low"}, "scenario_confidence_bucket_low"),
    ],
)
def test_allocator_gates_emit_no_allocation(updates: dict[str, object], reason: str) -> None:
    result = run_portfolio_allocator(allocation_candidates(updates))

    row = result.allocation.iloc[0]
    assert row["allocation_status"] == "no_allocation"
    assert row["no_allocation_reason"] == reason
    assert reason in row["allocation_reason_codes"]
