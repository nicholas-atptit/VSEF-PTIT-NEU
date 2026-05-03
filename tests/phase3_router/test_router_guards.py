from __future__ import annotations

from src.phase3_router.guards import evaluate_route_guard
from src.phase3_router.schema import Phase3RouterConfig, PortfolioContext


def _candidate(**updates: object) -> dict[str, object]:
    row = {
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
    row.update(updates)
    return row


def _context(**updates: object) -> PortfolioContext:
    values = {
        "portfolio_status": "allocation_candidate",
        "total_exposure": 0.05,
        "cash_weight": 0.95,
        "min_cash_buffer": 0.30,
        "max_total_exposure": 0.70,
        "effective_max_exposure": 0.70,
    }
    values.update(updates)
    return PortfolioContext(**values)


def test_medium_confidence_disagreement_or_mild_conflict_holds() -> None:
    config = Phase3RouterConfig()

    medium_confidence = evaluate_route_guard(_candidate(risk_adjusted_confidence=0.55), _context(), config)
    medium_disagreement = evaluate_route_guard(_candidate(disagreement_score=0.40), _context(), config)
    mild_conflict = evaluate_route_guard(_candidate(conflict_level="mild"), _context(), config)

    assert medium_confidence.route_decision == "hold"
    assert medium_confidence.route_reason == "medium_risk_adjusted_confidence"
    assert medium_disagreement.route_decision == "hold"
    assert medium_disagreement.route_reason == "medium_disagreement"
    assert mild_conflict.route_decision == "hold"
    assert mild_conflict.route_reason == "mild_conflict"


def test_near_exposure_or_cash_constraint_downgrades_route_to_hold() -> None:
    decision = evaluate_route_guard(
        _candidate(),
        _context(total_exposure=0.69, cash_weight=0.31),
        Phase3RouterConfig(exposure_near_limit_buffer=0.02),
    )

    assert decision.route_decision == "hold"
    assert decision.route_reason == "portfolio_context_near_exposure_constraint"


def test_high_volatility_regime_biases_hold_or_reject() -> None:
    hold = evaluate_route_guard(_candidate(volatility_regime="high"), _context(), Phase3RouterConfig())
    reject = evaluate_route_guard(
        _candidate(volatility_regime="high", risk_score=0.60),
        _context(),
        Phase3RouterConfig(),
    )

    assert hold.route_decision == "hold"
    assert hold.route_reason == "high_volatility_regime"
    assert reject.route_decision == "reject"
    assert reject.route_reason == "high_volatility_regime"
