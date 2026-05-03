"""Deterministic route guards for Phase 3 Router v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.phase3_router.schema import Phase3RouterConfig, PortfolioContext, bounded, normalize_text, safe_float


@dataclass(frozen=True)
class RouteGuardDecision:
    route_decision: str
    reason_codes: tuple[str, ...]

    @property
    def route_reason(self) -> str:
        return self.reason_codes[0] if self.reason_codes else "routed_by_default_hold"


def _row_text(row: Mapping[str, Any], column: str, default: str = "") -> str:
    return normalize_text(row.get(column), default=default).strip().lower()


def _row_float(row: Mapping[str, Any], column: str, default: float = 0.0) -> float:
    return bounded(row.get(column), default=default)


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = normalize_text(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    return bool(value)


def is_adverse_scenario(row: Mapping[str, Any], config: Phase3RouterConfig) -> bool:
    scenario = _row_text(row, "dominant_scenario")
    return scenario in {value.lower() for value in config.adverse_scenarios}


def has_high_volatility_regime(row: Mapping[str, Any]) -> bool:
    for column in (
        "volatility_regime",
        "volatility_bucket",
        "market_volatility_regime",
        "regime_volatility",
    ):
        if _row_text(row, column) == "high":
            return True
    return _row_text(row, "regime_label") == "high_volatility"


def conflict_level(row: Mapping[str, Any]) -> str:
    for column in ("conflict_level", "conflict_severity", "scenario_conflict", "model_conflict"):
        value = _row_text(row, column)
        if value:
            return value
    if _is_truthy(row.get("sign_conflict")):
        return "mild"
    reason_codes = _row_text(row, "allocation_reason_codes")
    if "severe_conflict" in reason_codes or "hard_conflict" in reason_codes:
        return "severe"
    if "mild_conflict" in reason_codes or "sign_conflict" in reason_codes:
        return "mild"
    return "none"


def evaluate_route_guard(
    row: Mapping[str, Any],
    context: PortfolioContext,
    config: Phase3RouterConfig | None = None,
) -> RouteGuardDecision:
    """Evaluate one standardized allocator row into a v1 route decision."""

    resolved = config or Phase3RouterConfig()
    allocation_status = _row_text(row, "allocation_status")
    risk_level = _row_text(row, "risk_level")
    risk_score = _row_float(row, "risk_score", default=1.0)
    confidence = _row_float(row, "risk_adjusted_confidence", default=0.0)
    disagreement = _row_float(row, "disagreement_score", default=1.0)
    dominance = _row_float(row, "dominance_score", default=1.0)
    final_weight = safe_float(row.get("final_weight"), default=0.0)
    scenario_alignment = _row_text(row, "scenario_alignment", default="unknown")
    conflict = conflict_level(row)
    adverse_scenario = is_adverse_scenario(row, resolved)
    high_volatility = has_high_volatility_regime(row)

    if allocation_status != "allocation_candidate":
        return RouteGuardDecision("no_candidate", ("allocator_status_not_allocation_candidate",))

    if risk_level == "level_3_hard_override":
        return RouteGuardDecision("reject", ("level_3_hard_override",))
    if risk_score >= float(resolved.risk_reject_threshold):
        return RouteGuardDecision("reject", ("risk_score_reject_threshold",))
    if conflict in {"severe", "hard"}:
        return RouteGuardDecision("reject", ("severe_conflict",))

    if final_weight <= 0.0:
        return RouteGuardDecision("hold", ("non_positive_final_weight",))

    if adverse_scenario or high_volatility:
        reason_codes: list[str] = []
        if adverse_scenario:
            reason_codes.append("adverse_dominant_scenario")
        if high_volatility:
            reason_codes.append("high_volatility_regime")
        elevated = (
            risk_level == "level_2_candidate_filtering"
            or risk_score >= float(resolved.risk_elevated_threshold)
            or scenario_alignment == "misaligned_or_risky"
            or disagreement > float(resolved.disagreement_medium_threshold)
            or confidence < float(resolved.confidence_medium_threshold)
        )
        if elevated:
            return RouteGuardDecision("reject", tuple(reason_codes + ["elevated_risk_in_adverse_context"]))
        return RouteGuardDecision("hold", tuple(reason_codes + ["market_regime_hold_bias"]))

    if confidence < float(resolved.confidence_medium_threshold):
        return RouteGuardDecision("reject", ("low_risk_adjusted_confidence",))
    if disagreement > float(resolved.disagreement_medium_threshold):
        return RouteGuardDecision("reject", ("high_disagreement",))

    if confidence < float(resolved.confidence_high_threshold):
        return RouteGuardDecision("hold", ("medium_risk_adjusted_confidence",))
    if disagreement > float(resolved.disagreement_low_threshold):
        return RouteGuardDecision("hold", ("medium_disagreement",))
    if conflict == "mild":
        return RouteGuardDecision("hold", ("mild_conflict",))
    if dominance < float(resolved.dominance_min_threshold):
        return RouteGuardDecision("hold", ("weak_scenario_dominance",))
    if not context.exposure_room_available(resolved):
        return RouteGuardDecision("hold", ("portfolio_context_near_exposure_constraint",))
    if risk_score > float(resolved.risk_low_threshold) or risk_level != "level_1_soft_adjustment":
        return RouteGuardDecision("hold", ("risk_not_low",))

    return RouteGuardDecision("route_allocation_candidate", ("valid_allocation_candidate",))
