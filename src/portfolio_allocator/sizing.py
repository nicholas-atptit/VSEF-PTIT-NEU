"""Ranking and sizing utilities for Portfolio Allocator v1."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.portfolio_allocator.schema import (
    RISK_MULTIPLIERS,
    SCENARIO_ALIGNMENT_MULTIPLIERS,
    PortfolioAllocatorConfig,
    bounded,
    normalize_text,
)


def dominance_multiplier(dominance_score: Any) -> float:
    score = bounded(dominance_score, default=0.0)
    if score >= 0.30:
        return 1.00
    if score >= 0.15:
        return 0.70
    return 0.00


def agreement_multiplier(disagreement_score: Any) -> float:
    return 1.0 - bounded(disagreement_score, default=1.0)


def risk_multiplier(risk_level: Any) -> float:
    level = normalize_text(risk_level, default="unknown").lower()
    return float(RISK_MULTIPLIERS.get(level, 0.50))


def scenario_alignment_multiplier(scenario_alignment: Any) -> float:
    alignment = normalize_text(scenario_alignment, default="unknown").lower()
    return float(SCENARIO_ALIGNMENT_MULTIPLIERS.get(alignment, SCENARIO_ALIGNMENT_MULTIPLIERS["unknown"]))


def calculate_raw_weight(row: pd.Series | dict[str, Any], config: PortfolioAllocatorConfig) -> float:
    record = row if isinstance(row, dict) else row.to_dict()
    weight = (
        float(config.max_position_weight)
        * bounded(record.get("risk_adjusted_confidence"), default=0.0)
        * dominance_multiplier(record.get("dominance_score"))
        * agreement_multiplier(record.get("disagreement_score"))
        * risk_multiplier(record.get("risk_level"))
        * scenario_alignment_multiplier(record.get("scenario_alignment"))
    )
    return round(max(0.0, min(float(config.max_position_weight), weight)), 10)


def attach_allocation_priority_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach normalized candidate scores and v1 allocation priority score."""

    scored = frame.copy()
    if scored.empty:
        scored["normalized_risk_adjusted_candidate_score"] = pd.Series(dtype="float64")
        scored["allocation_priority_score"] = pd.Series(dtype="float64")
        return scored

    default_series = pd.Series(0.0, index=scored.index)
    raw_scores = pd.to_numeric(scored.get("risk_adjusted_candidate_score", default_series), errors="coerce").fillna(0.0)
    score_min = float(raw_scores.min())
    score_max = float(raw_scores.max())
    if score_max > score_min:
        normalized = (raw_scores - score_min) / (score_max - score_min)
    else:
        normalized = raw_scores.map(lambda value: 1.0 if float(value) > 0.0 else 0.0)

    confidence = pd.to_numeric(scored.get("risk_adjusted_confidence", default_series), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    dominance = pd.to_numeric(scored.get("dominance_score", default_series), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    disagreement = pd.to_numeric(scored.get("disagreement_score", pd.Series(1.0, index=scored.index)), errors="coerce").fillna(1.0).clip(0.0, 1.0)
    risk_score = pd.to_numeric(scored.get("risk_score", pd.Series(1.0, index=scored.index)), errors="coerce").fillna(1.0).clip(0.0, 1.0)

    scored["normalized_risk_adjusted_candidate_score"] = normalized.round(10)
    scored["allocation_priority_score"] = (
        0.35 * confidence
        + 0.25 * scored["normalized_risk_adjusted_candidate_score"]
        + 0.20 * dominance
        + 0.10 * (1.0 - disagreement)
        + 0.10 * (1.0 - risk_score)
    ).round(10)
    return scored
