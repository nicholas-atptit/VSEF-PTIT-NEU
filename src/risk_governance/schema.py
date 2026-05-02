"""Shared schema and configuration for Risk Governance Layer v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


RISK_GOVERNANCE_VERSION = "v1"

RISK_LEVELS: tuple[str, ...] = (
    "level_1_soft_adjustment",
    "level_2_candidate_filtering",
    "level_3_hard_override",
)

RISK_ACTIONS: tuple[str, ...] = (
    "pass",
    "adjust_confidence",
    "reduce_candidate",
    "block_candidate",
    "force_hold",
)

RISK_CONTEXT_COLUMNS: tuple[str, ...] = (
    "packet_id",
    "timestamp",
    "ticker",
    "horizon",
    "target_type",
    "run_mode",
    "core_run_id",
)

JOIN_CONTEXT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "ticker",
    "horizon",
    "target_type",
    "run_mode",
    "core_run_id",
)

RISK_COMPONENT_COLUMNS: tuple[str, ...] = (
    "drawdown_component",
    "volatility_component",
    "downside_risk_component",
    "model_health_component",
    "scenario_dispersion_component",
    "disagreement_component",
    "calibration_component",
)

RISK_GOVERNANCE_SUMMARY_COLUMNS: tuple[str, ...] = (
    "packet_id",
    "timestamp",
    "ticker",
    "horizon",
    "target_type",
    "run_mode",
    "core_run_id",
    "risk_score",
    "risk_level",
    "risk_action",
    "confidence_adjustment_factor",
    "block_candidate",
    "force_hold",
    "risk_reason_codes",
    *RISK_COMPONENT_COLUMNS,
)

RISK_GOVERNANCE_ARTIFACT_FILENAMES: dict[str, str] = {
    "risk_governance_summary": "risk_governance_summary.csv",
    "risk_adjusted_candidates": "risk_adjusted_candidates.csv",
    "risk_override_log": "risk_override_log.csv",
    "risk_manifest": "risk_manifest.json",
}

DEFAULT_SCORING_WEIGHTS: dict[str, float] = {
    "drawdown_component": 0.20,
    "volatility_component": 0.20,
    "downside_risk_component": 0.15,
    "model_health_component": 0.15,
    "scenario_dispersion_component": 0.15,
    "disagreement_component": 0.10,
    "calibration_component": 0.05,
}

DEFAULT_SCORE_THRESHOLDS: dict[str, float] = {
    "level_2_min": 0.35,
    "level_3_min": 0.70,
}

DEFAULT_DRAW_DOWN_COMPONENT_MAP: dict[str, float] = {
    "normal": 0.00,
    "elevated": 0.50,
    "severe": 1.00,
}

DEFAULT_MODEL_HEALTH_COMPONENT_MAP: dict[str, float] = {
    "healthy": 0.00,
    "brittle": 0.35,
    "weak": 0.70,
    "failing": 1.00,
}

DEFAULT_AGREEMENT_COMPONENT_MAP: dict[str, float] = {
    "high": 0.00,
    "medium": 0.35,
    "low": 0.80,
    "unknown": 0.60,
}

DEFAULT_SCENARIO_CONFIDENCE_COMPONENT_MAP: dict[str, float] = {
    "high": 0.00,
    "medium": 0.30,
    "low": 0.75,
    "uncalibrated": 0.60,
    "risk_overridden": 1.00,
}


@dataclass(frozen=True)
class RiskGovernanceConfig:
    """Deterministic Risk Governance Layer v1 configuration."""

    version: str = RISK_GOVERNANCE_VERSION
    scoring_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SCORING_WEIGHTS))
    score_thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SCORE_THRESHOLDS))
    drawdown_component_map: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_DRAW_DOWN_COMPONENT_MAP))
    model_health_component_map: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_MODEL_HEALTH_COMPONENT_MAP)
    )
    agreement_component_map: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_AGREEMENT_COMPONENT_MAP))
    scenario_confidence_component_map: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SCENARIO_CONFIDENCE_COMPONENT_MAP)
    )
    volatility_reference: float = 0.08
    downside_risk_reference: float = 0.12
    calibration_error_reference: float = 0.25
    weak_candidate_score_threshold: float = 0.015
    pass_risk_score_threshold: float = 0.05


@dataclass
class RiskGovernanceResult:
    """Risk governance outputs kept in memory before writing artifacts."""

    risk_governance_summary: pd.DataFrame
    risk_adjusted_candidates: pd.DataFrame
    risk_override_log: pd.DataFrame
    manifest: dict[str, Any]


def present_columns(frame: pd.DataFrame, preferred: tuple[str, ...] | list[str]) -> list[str]:
    """Return preferred columns that are present in a frame."""

    return [column for column in preferred if column in frame.columns]


def empty_summary_frame() -> pd.DataFrame:
    """Return an empty governance summary with stable v1 columns."""

    return pd.DataFrame(columns=list(RISK_GOVERNANCE_SUMMARY_COLUMNS))
