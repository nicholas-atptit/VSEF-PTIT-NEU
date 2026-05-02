from __future__ import annotations

import pytest

from src.risk_governance import run_risk_governance
from tests.risk_governance.conftest import build_governance_inputs


def test_low_risk_produces_soft_adjustment_or_pass() -> None:
    inputs = build_governance_inputs()

    result = run_risk_governance(
        candidates_df=inputs["candidates"],
        packets_df=inputs["packets"],
        risk_df=inputs["risk"],
        consensus_df=inputs["consensus"],
        model_health_df=inputs["health"],
    )
    row = result.risk_governance_summary.iloc[0]

    assert row["risk_level"] == "level_1_soft_adjustment"
    assert row["risk_action"] in {"pass", "adjust_confidence"}
    assert not row["block_candidate"]
    assert not row["force_hold"]
    assert row["confidence_adjustment_factor"] == pytest.approx(1.0 - row["risk_score"])


def test_elevated_risk_blocks_weak_candidates() -> None:
    inputs = build_governance_inputs(
        risk_updates={
            "drawdown_state": "elevated",
            "vol_forecast": 0.08,
            "cvar_loss_95": 0.06,
        },
        health_status="weak",
        candidate_updates={
            "candidate_score": 0.004,
            "primary_prediction": 0.010,
            "agreement_bucket": "medium",
            "active_signal_count": 1,
        },
    )

    result = run_risk_governance(
        candidates_df=inputs["candidates"],
        packets_df=inputs["packets"],
        risk_df=inputs["risk"],
        consensus_df=inputs["consensus"],
        model_health_df=inputs["health"],
    )
    row = result.risk_governance_summary.iloc[0]

    assert 0.35 <= row["risk_score"] < 0.70
    assert row["risk_level"] == "level_2_candidate_filtering"
    assert row["risk_action"] == "block_candidate"
    assert row["block_candidate"]
    assert not row["force_hold"]


def test_severe_risk_forces_hold() -> None:
    inputs = build_governance_inputs(
        risk_updates={
            "drawdown_state": "severe",
            "vol_forecast": 0.12,
            "cvar_loss_95": 0.14,
        },
        consensus_updates={
            "agreement_score": 0.20,
            "disagreement_score": 0.80,
            "agreement_bucket": "low",
            "sign_conflict": True,
        },
        health_status="failing",
    )

    result = run_risk_governance(
        candidates_df=inputs["candidates"],
        packets_df=inputs["packets"],
        risk_df=inputs["risk"],
        consensus_df=inputs["consensus"],
        model_health_df=inputs["health"],
    )
    row = result.risk_governance_summary.iloc[0]

    assert row["risk_score"] >= 0.70
    assert row["risk_level"] == "level_3_hard_override"
    assert row["risk_action"] == "force_hold"
    assert row["block_candidate"]
    assert row["force_hold"]
