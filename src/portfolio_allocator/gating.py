"""Candidate gating rules for Portfolio Allocator v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.portfolio_allocator.schema import PortfolioAllocatorConfig, bounded, normalize_text, safe_float


@dataclass(frozen=True)
class AllocationGateDecision:
    allocation_status: str
    no_allocation_reason: str
    reason_codes: list[str]

    @property
    def passed(self) -> bool:
        return self.allocation_status == "allocation_candidate"


def split_reason_codes(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text.lower() == "none":
        return []
    return [code.strip() for code in text.split("|") if code.strip() and code.strip().lower() != "none"]


def combine_reason_codes(*groups: Any) -> str:
    codes: list[str] = []
    for group in groups:
        if isinstance(group, (list, tuple, set)):
            candidates = [str(item).strip() for item in group if str(item).strip()]
        else:
            candidates = split_reason_codes(group)
        for code in candidates:
            if code and code not in codes:
                codes.append(code)
    return "|".join(codes) if codes else "none"


def evaluate_allocation_gate(row: pd.Series | dict[str, Any], config: PortfolioAllocatorConfig) -> AllocationGateDecision:
    """Apply deterministic Portfolio Allocator v1 gates to one candidate."""

    record = row if isinstance(row, dict) else row.to_dict()
    reasons: list[str] = []

    candidate_status = normalize_text(record.get("candidate_status"), default="diagnostic_candidate").lower()
    if candidate_status == "blocked":
        reasons.append("candidate_status_blocked")
    if candidate_status == "force_hold":
        reasons.append("candidate_status_force_hold")

    risk_level = normalize_text(record.get("risk_level")).lower()
    if risk_level == "level_3_hard_override":
        reasons.append("risk_level_3_hard_override")

    risk_action = normalize_text(record.get("risk_action"), default="pass").lower()
    if risk_action == "force_hold":
        reasons.append("risk_action_force_hold")

    risk_adjusted_confidence = safe_float(record.get("risk_adjusted_confidence"), default=0.0)
    if risk_adjusted_confidence < float(config.confidence_threshold):
        reasons.append("risk_adjusted_confidence_below_threshold")

    disagreement_score = bounded(record.get("disagreement_score"), default=1.0)
    if disagreement_score > float(config.disagreement_threshold):
        reasons.append("disagreement_score_above_threshold")

    dominance_score = bounded(record.get("dominance_score"), default=0.0)
    if dominance_score < float(config.dominance_threshold):
        reasons.append("dominance_score_below_threshold")

    scenario_alignment = normalize_text(record.get("scenario_alignment"), default="unknown").lower()
    if scenario_alignment == "misaligned_or_risky":
        reasons.append("scenario_alignment_misaligned_or_risky")

    scenario_confidence_bucket = normalize_text(record.get("scenario_confidence_bucket"), default="unknown").lower()
    if scenario_confidence_bucket == "low":
        reasons.append("scenario_confidence_bucket_low")

    if reasons:
        return AllocationGateDecision(
            allocation_status="no_allocation",
            no_allocation_reason=reasons[0],
            reason_codes=reasons,
        )
    return AllocationGateDecision(
        allocation_status="allocation_candidate",
        no_allocation_reason="",
        reason_codes=["passed_allocator_gates"],
    )
