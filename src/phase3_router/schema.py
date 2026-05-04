"""Schema and deterministic configuration for Phase 3 Router v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


PHASE3_ROUTER_VERSION = "v1"

ROUTER_ARTIFACT_FILENAMES: dict[str, str] = {
    "router_decisions": "router_decisions.csv",
    "router_summary": "router_summary.csv",
    "router_manifest": "router_manifest.json",
}

LEGACY_ROUTER_ARTIFACT_FILENAMES: dict[str, str] = {
    "route_decision": "route_decision.csv",
    "phase3_decision_cards": "phase3_decision_cards.jsonl",
    "routing_summary": "routing_summary.csv",
    "routing_manifest": "routing_manifest.json",
}

ROUTE_DECISIONS: tuple[str, ...] = (
    "route_allocation_candidate",
    "hold",
    "reject",
    "no_candidate",
)

ADVERSE_DOMINANT_SCENARIOS: tuple[str, ...] = (
    "bear",
    "drawdown",
    "high_volatility",
)

ROUTER_DECISION_COLUMNS: tuple[str, ...] = (
    "router_decision_id",
    "allocation_id",
    "candidate_id",
    "source_packet_id",
    "timestamp",
    "ticker",
    "horizon",
    "route_decision",
    "route_reason",
    "allocation_status",
    "final_weight",
    "risk_level",
    "risk_score",
    "risk_adjusted_confidence",
    "disagreement_score",
    "dominance_score",
    "scenario_alignment",
    "dominant_scenario",
    "portfolio_status",
    "total_exposure",
    "cash_weight",
    "route_reason_codes",
    "diagnostic_only_authority",
    "no_buy_sell_recommendation_authority",
)

ROUTER_SUMMARY_COLUMNS: tuple[str, ...] = (
    "router_status",
    "source_allocation_count",
    "route_allocation_candidate_count",
    "hold_count",
    "reject_count",
    "no_candidate_count",
    "routed_final_weight",
    "total_exposure",
    "cash_weight",
    "diagnostic_only_authority",
    "no_buy_sell_recommendation_authority",
)


@dataclass(frozen=True)
class Phase3RouterConfig:
    """Conservative deterministic Phase 3 Router v1 thresholds."""

    version: str = PHASE3_ROUTER_VERSION
    risk_reject_threshold: float = 0.70
    risk_low_threshold: float = 0.35
    risk_elevated_threshold: float = 0.55
    confidence_high_threshold: float = 0.70
    confidence_medium_threshold: float = 0.45
    disagreement_low_threshold: float = 0.25
    disagreement_medium_threshold: float = 0.50
    dominance_min_threshold: float = 0.15
    exposure_near_limit_buffer: float = 0.02
    adverse_scenarios: tuple[str, ...] = field(default_factory=lambda: ADVERSE_DOMINANT_SCENARIOS)

    def __post_init__(self) -> None:
        for name in (
            "risk_reject_threshold",
            "risk_low_threshold",
            "risk_elevated_threshold",
            "confidence_high_threshold",
            "confidence_medium_threshold",
            "disagreement_low_threshold",
            "disagreement_medium_threshold",
            "dominance_min_threshold",
            "exposure_near_limit_buffer",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if float(self.confidence_medium_threshold) > float(self.confidence_high_threshold):
            raise ValueError("confidence_medium_threshold cannot exceed confidence_high_threshold")
        if float(self.disagreement_low_threshold) > float(self.disagreement_medium_threshold):
            raise ValueError("disagreement_low_threshold cannot exceed disagreement_medium_threshold")
        if float(self.risk_low_threshold) > float(self.risk_reject_threshold):
            raise ValueError("risk_low_threshold cannot exceed risk_reject_threshold")

    @property
    def thresholds(self) -> dict[str, float]:
        return {
            "risk_reject_threshold": float(self.risk_reject_threshold),
            "risk_low_threshold": float(self.risk_low_threshold),
            "risk_elevated_threshold": float(self.risk_elevated_threshold),
            "confidence_high_threshold": float(self.confidence_high_threshold),
            "confidence_medium_threshold": float(self.confidence_medium_threshold),
            "disagreement_low_threshold": float(self.disagreement_low_threshold),
            "disagreement_medium_threshold": float(self.disagreement_medium_threshold),
            "dominance_min_threshold": float(self.dominance_min_threshold),
            "exposure_near_limit_buffer": float(self.exposure_near_limit_buffer),
        }

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["adverse_scenarios"] = list(self.adverse_scenarios)
        return value


@dataclass(frozen=True)
class PortfolioContext:
    """Portfolio-level context used to downgrade route decisions."""

    portfolio_status: str = "unknown"
    total_exposure: float = 0.0
    cash_weight: float = 1.0
    min_cash_buffer: float = 0.0
    max_total_exposure: float = 1.0
    effective_max_exposure: float = 1.0

    def exposure_room_available(self, config: Phase3RouterConfig) -> bool:
        buffer = float(config.exposure_near_limit_buffer)
        max_exposure = min(float(self.max_total_exposure), float(self.effective_max_exposure))
        near_max_exposure = float(self.total_exposure) >= max(0.0, max_exposure - buffer)
        near_min_cash = float(self.cash_weight) <= float(self.min_cash_buffer) + buffer
        return not near_max_exposure and not near_min_cash


@dataclass
class Phase3RouterResult:
    """In-memory Phase 3 Router v1 outputs before or after writing artifacts."""

    router_decisions: pd.DataFrame
    router_summary: pd.DataFrame
    manifest: dict[str, Any]
    output_paths: dict[str, Path] = field(default_factory=dict)

    @property
    def decisions(self) -> pd.DataFrame:
        return self.router_decisions

    @property
    def summary(self) -> pd.DataFrame:
        return self.router_summary


def empty_router_decisions_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(ROUTER_DECISION_COLUMNS))


def normalize_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    return text


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if pd.isna(number):
        return float(default)
    if number == float("inf") or number == float("-inf"):
        return float(default)
    return float(number)


def bounded(value: Any, default: float = 0.0) -> float:
    number = safe_float(value, default=default)
    return max(0.0, min(1.0, number))
