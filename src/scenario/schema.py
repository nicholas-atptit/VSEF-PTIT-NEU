"""Shared schema and configuration for Scenario Evaluation Engine v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


SCENARIO_LABELS: tuple[str, ...] = (
    "bull",
    "bear",
    "sideway",
    "high_volatility",
    "drawdown",
    "recovery",
    "uncertain",
)

DOMINANCE_LABELS: tuple[str, ...] = (
    "dominant",
    "weak_dominance",
    "no_clear_dominance",
    "uncalibrated_dominance",
    "risk_overrides_dominance",
)

SCENARIO_CONTEXT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "ticker",
    "horizon",
    "target_type",
    "run_mode",
    "core_run_id",
)

PACKET_CONTEXT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "ticker",
    "horizon",
    "target_type",
    "window_id",
    "target_timestamp",
    "core_run_id",
    "preset",
    "group_name",
    "target_name",
    "target_column",
    "target_family",
    "target_tradable",
    "ticker_count",
    "ticker_group_members",
    "run_mode",
)

SCENARIO_REQUIRED_FIELDS: tuple[str, ...] = (
    "scenario_id",
    "timestamp",
    "ticker",
    "horizon",
    "target_type",
    "run_mode",
    "core_run_id",
    "scenario_label",
    "scenario_probability",
    "confidence_adjusted_probability",
    "expected_outcome",
    "downside_risk",
    "confidence_interval_low",
    "confidence_interval_high",
    "uncertainty_score",
    "dispersion_score",
    "dominance_score",
    "dominant_scenario_flag",
    "calibration_error",
    "historical_hit_rate",
    "source_model",
)

SCENARIO_ARTIFACT_FILENAMES: dict[str, str] = {
    "scenario_probability": "scenario_probability.csv",
    "scenario_rankings": "scenario_rankings.csv",
    "scenario_dominance_summary": "scenario_dominance_summary.csv",
    "scenario_uncertainty_summary": "scenario_uncertainty_summary.csv",
    "scenario_calibration_summary": "scenario_calibration_summary.csv",
    "scenario_manifest": "scenario_manifest.json",
}


@dataclass(frozen=True)
class ScenarioEngineConfig:
    """Configuration for deterministic Scenario Evaluation Engine v1."""

    probability_method: str = "deterministic_v1"
    calibration_lookback: int | None = 252
    calibration_bins: int = 5


@dataclass
class ScenarioEvaluationResult:
    """Scenario engine outputs kept in memory before writing artifacts."""

    scenario_probability: pd.DataFrame
    scenario_rankings: pd.DataFrame
    scenario_dominance_summary: pd.DataFrame
    scenario_uncertainty_summary: pd.DataFrame
    scenario_calibration_summary: pd.DataFrame
    analysis_packets: pd.DataFrame
    manifest: dict[str, Any]


def empty_probability_frame() -> pd.DataFrame:
    """Return an empty probability frame with stable Scenario v1 columns."""

    return pd.DataFrame(columns=list(SCENARIO_REQUIRED_FIELDS))


def present_columns(frame: pd.DataFrame, preferred: tuple[str, ...] | list[str]) -> list[str]:
    """Return preferred columns that are present in the provided frame."""

    return [column for column in preferred if column in frame.columns]
