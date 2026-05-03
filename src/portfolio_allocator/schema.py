"""Schema and deterministic configuration for Portfolio Allocator v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


PORTFOLIO_ALLOCATOR_VERSION = "v1"

PORTFOLIO_ALLOCATOR_ARTIFACT_FILENAMES: dict[str, str] = {
    "portfolio_allocation": "portfolio_allocation.csv",
    "portfolio_summary": "portfolio_summary.csv",
    "portfolio_risk_summary": "portfolio_risk_summary.csv",
    "portfolio_decision_cards": "portfolio_decision_cards.jsonl",
    "allocator_manifest": "allocator_manifest.json",
}

ALLOCATION_STATUSES: tuple[str, ...] = (
    "allocation_candidate",
    "no_allocation",
)

RISK_MULTIPLIERS: dict[str, float] = {
    "level_1_soft_adjustment": 1.00,
    "level_2_candidate_filtering": 0.50,
    "level_3_hard_override": 0.00,
}

SCENARIO_ALIGNMENT_MULTIPLIERS: dict[str, float] = {
    "aligned": 1.00,
    "weakly_aligned": 0.50,
    "misaligned_or_risky": 0.00,
    "unknown": 0.40,
}

REQUIRED_ALLOCATION_COLUMNS: tuple[str, ...] = (
    "allocation_id",
    "candidate_id",
    "source_packet_id",
    "timestamp",
    "ticker",
    "horizon",
    "target_type",
    "run_mode",
    "allocation_status",
    "no_allocation_reason",
    "risk_adjusted_confidence",
    "risk_adjusted_candidate_score",
    "risk_score",
    "risk_level",
    "risk_action",
    "disagreement_score",
    "dominance_score",
    "dominant_scenario",
    "dominant_scenario_probability",
    "scenario_alignment",
    "raw_weight",
    "final_weight",
    "exposure_before_allocation",
    "exposure_after_allocation",
    "cash_buffer_after_allocation",
    "allocation_reason_codes",
)

ALLOCATION_OUTPUT_COLUMNS: tuple[str, ...] = (
    *REQUIRED_ALLOCATION_COLUMNS,
    "allocation_priority_score",
    "normalized_risk_adjusted_candidate_score",
)

PORTFOLIO_SUMMARY_COLUMNS: tuple[str, ...] = (
    "portfolio_status",
    "candidate_count",
    "allocation_candidate_count",
    "no_allocation_count",
    "total_exposure",
    "cash_weight",
    "min_cash_buffer",
    "max_total_exposure",
    "effective_max_exposure",
    "diagnostic_only_authority",
    "no_buy_sell_recommendation_authority",
    "no_forced_trade_rule",
)

PORTFOLIO_RISK_SUMMARY_COLUMNS: tuple[str, ...] = (
    "portfolio_status",
    "allocation_candidate_count",
    "total_exposure",
    "cash_weight",
    "max_position_weight",
    "max_single_position_weight",
    "weighted_average_risk_score",
    "max_allocated_risk_score",
    "level_1_soft_adjustment_count",
    "level_2_candidate_filtering_count",
    "level_3_hard_override_count",
    "no_allocation_count",
)


@dataclass(frozen=True)
class PortfolioAllocatorConfig:
    """Default deterministic Portfolio Allocator v1 thresholds."""

    version: str = PORTFOLIO_ALLOCATOR_VERSION
    max_position_weight: float = 0.20
    min_position_weight: float = 0.02
    min_cash_buffer: float = 0.30
    max_total_exposure: float = 0.70
    confidence_threshold: float = 0.45
    disagreement_threshold: float = 0.50
    dominance_threshold: float = 0.15

    def __post_init__(self) -> None:
        for name in (
            "max_position_weight",
            "min_position_weight",
            "min_cash_buffer",
            "max_total_exposure",
            "confidence_threshold",
            "disagreement_threshold",
            "dominance_threshold",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if float(self.min_position_weight) > float(self.max_position_weight):
            raise ValueError("min_position_weight cannot exceed max_position_weight")

    @property
    def thresholds(self) -> dict[str, float]:
        return {
            "confidence_threshold": float(self.confidence_threshold),
            "disagreement_threshold": float(self.disagreement_threshold),
            "dominance_threshold": float(self.dominance_threshold),
        }

    @property
    def effective_max_exposure(self) -> float:
        return max(0.0, min(float(self.max_total_exposure), 1.0 - float(self.min_cash_buffer)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"effective_max_exposure": self.effective_max_exposure}


@dataclass
class PortfolioAllocatorResult:
    """In-memory Portfolio Allocator v1 outputs before or after writing artifacts."""

    allocation: pd.DataFrame
    portfolio_summary: pd.DataFrame
    portfolio_risk_summary: pd.DataFrame
    decision_cards: list[dict[str, Any]]
    manifest: dict[str, Any]
    output_paths: dict[str, Path] = field(default_factory=dict)

    @property
    def summary(self) -> pd.DataFrame:
        return self.portfolio_summary

    @property
    def risk_summary(self) -> pd.DataFrame:
        return self.portfolio_risk_summary


def empty_allocation_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(ALLOCATION_OUTPUT_COLUMNS))


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


def normalize_text(value: Any, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    return text
