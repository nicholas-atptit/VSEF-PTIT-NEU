"""Risk action selection for Risk Governance Layer v1."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src.risk_governance.schema import RiskGovernanceConfig
from src.risk_governance.scoring import bounded, safe_float


def confidence_adjustment_factor(risk_score: float) -> float:
    """Return max(0, 1 - risk_score) as a stable rounded factor."""

    return round(max(0.0, 1.0 - bounded(risk_score)), 6)


def risk_adjusted_candidate_score(candidate_score: Any, risk_score: float) -> float:
    """Apply the governance confidence adjustment to a candidate score."""

    score = safe_float(candidate_score, default=0.0)
    return round(float(score) * confidence_adjustment_factor(risk_score), 6)


def is_weak_candidate(
    candidate: Mapping[str, Any],
    config: RiskGovernanceConfig | None = None,
) -> bool:
    """Return True when level-2 risk should block rather than reduce a candidate."""

    resolved = config or RiskGovernanceConfig()
    candidate_score = safe_float(candidate.get("candidate_score"), default=0.0)
    primary_prediction = abs(safe_float(candidate.get("primary_prediction"), default=0.0))
    agreement_bucket = str(candidate.get("agreement_bucket", "") or "").strip().lower()
    active_signal_count = safe_float(candidate.get("active_signal_count"), default=0.0)
    if agreement_bucket in {"low", "unknown", ""}:
        return True
    if active_signal_count <= 0.0:
        return True
    if candidate_score < resolved.weak_candidate_score_threshold and primary_prediction < 0.02:
        return True
    return False


def determine_risk_action(
    *,
    risk_score: float,
    candidate: Mapping[str, Any] | None = None,
    config: RiskGovernanceConfig | None = None,
) -> dict[str, Any]:
    """Select v1 risk level, action, block flag, and force-hold flag."""

    resolved = config or RiskGovernanceConfig()
    score = bounded(risk_score)
    if score >= float(resolved.score_thresholds["level_3_min"]):
        return {
            "risk_level": "level_3_hard_override",
            "risk_action": "force_hold",
            "block_candidate": True,
            "force_hold": True,
        }

    if score >= float(resolved.score_thresholds["level_2_min"]):
        weak = is_weak_candidate(candidate or {}, resolved)
        return {
            "risk_level": "level_2_candidate_filtering",
            "risk_action": "block_candidate" if weak else "reduce_candidate",
            "block_candidate": bool(weak),
            "force_hold": False,
        }

    return {
        "risk_level": "level_1_soft_adjustment",
        "risk_action": "pass" if score <= float(resolved.pass_risk_score_threshold) else "adjust_confidence",
        "block_candidate": False,
        "force_hold": False,
    }


def apply_risk_action_fields(
    candidate: Mapping[str, Any],
    *,
    risk_score: float,
    config: RiskGovernanceConfig | None = None,
) -> dict[str, Any]:
    """Build action fields plus confidence and adjusted score for one candidate."""

    action = determine_risk_action(risk_score=risk_score, candidate=candidate, config=config)
    factor = confidence_adjustment_factor(risk_score)
    return {
        **action,
        "confidence_adjustment_factor": factor,
        "risk_adjusted_candidate_score": risk_adjusted_candidate_score(candidate.get("candidate_score"), risk_score),
    }
