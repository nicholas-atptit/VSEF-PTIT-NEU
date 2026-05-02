from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluation.consensus import build_model_consensus_summary
from src.reporting.analysis_packets import build_analysis_packets, build_decision_lane_candidates
from tests.scenario.conftest import sample_forecasts, sample_regime, sample_risk, sample_signals, sample_strategy


def _apply_updates(frame: pd.DataFrame, updates: dict[str, Any] | None) -> pd.DataFrame:
    result = frame.copy()
    for column, value in dict(updates or {}).items():
        result[column] = value
    return result


def build_governance_inputs(
    *,
    risk_updates: dict[str, Any] | None = None,
    consensus_updates: dict[str, Any] | None = None,
    health_status: str = "healthy",
    candidate_updates: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    forecasts = sample_forecasts()
    signals = sample_signals()
    risk = _apply_updates(sample_risk(), risk_updates)
    consensus = _apply_updates(build_model_consensus_summary(forecasts, signals_df=signals), consensus_updates)
    packets = build_analysis_packets(
        forecasts,
        build_model_consensus_summary(forecasts, signals_df=signals),
        risk_df=risk,
        regime_df=sample_regime(),
        signals_df=signals,
        positions_df=signals,
        strategy_metrics_df=sample_strategy(),
    )
    candidates = _apply_updates(build_decision_lane_candidates(packets), candidate_updates)
    health = pd.DataFrame(
        [
            {"model_name": "lightgbm", "health_status": health_status, "run_success_rate": 1.0},
            {"model_name": "xgboost", "health_status": health_status, "run_success_rate": 1.0},
            {"model_name": "weighted_ensemble", "health_status": health_status, "run_success_rate": 1.0},
        ]
    )
    return {
        "candidates": candidates,
        "packets": packets,
        "risk": risk,
        "consensus": consensus,
        "health": health,
    }


def build_scenario_governance_frames(
    packet: pd.Series,
    *,
    uncertainty_score: float = 0.80,
    dominance_score: float = 0.20,
    calibration_error: float = 0.18,
    confidence_bucket: str = "low",
    dominant_scenario: str = "bull",
    downside_risk: float = 0.08,
) -> dict[str, pd.DataFrame]:
    context = {
        "timestamp": packet["timestamp"],
        "ticker": packet["ticker"],
        "horizon": packet["horizon"],
        "target_type": packet["target_type"],
        "run_mode": packet["run_mode"],
        "core_run_id": packet["core_run_id"],
    }
    dominance = pd.DataFrame(
        [
            {
                **context,
                "dominant_scenario": dominant_scenario,
                "dominant_scenario_probability": 0.60,
                "dominant_scenario_adjusted_probability": 0.58,
                "dominance_score": dominance_score,
                "uncertainty_score": uncertainty_score,
                "calibration_error": calibration_error,
                "scenario_confidence_bucket": confidence_bucket,
            }
        ]
    )
    uncertainty = pd.DataFrame(
        [
            {
                **context,
                "scenario_count": 7,
                "probability_entropy": uncertainty_score,
                "top_probability": 0.30,
                "second_probability": 0.25,
                "probability_gap": 0.05,
                "uncertainty_score": uncertainty_score,
                "dispersion_score": uncertainty_score,
                "mean_calibration_error": calibration_error,
                "missing_calibration_share": 0.0,
                "confidence_bucket": confidence_bucket,
            }
        ]
    )
    probability = pd.DataFrame(
        [
            {
                **context,
                "scenario_label": dominant_scenario,
                "scenario_probability": 0.60,
                "confidence_adjusted_probability": 0.58,
                "downside_risk": downside_risk,
                "calibration_error": calibration_error,
            }
        ]
    )
    return {
        "dominance": dominance,
        "uncertainty": uncertainty,
        "probability": probability,
    }
