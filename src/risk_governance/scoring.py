"""Deterministic risk component scoring for Risk Governance Layer v1."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.risk_governance.schema import RISK_COMPONENT_COLUMNS, RiskGovernanceConfig


def safe_float(value: Any, default: float = float("nan")) -> float:
    """Return a finite float or a caller-provided default."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric) or pd.isna(numeric):
        return default
    return float(numeric)


def bounded(value: Any, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp numeric values to a stable 0-1 component interval by default."""

    numeric = safe_float(value, default=lower)
    return float(min(max(numeric, lower), upper))


def normalized_abs(value: Any, reference: float) -> float:
    """Normalize an absolute value against a positive reference threshold."""

    numeric = safe_float(value)
    if pd.isna(numeric) or reference <= 0.0:
        return 0.0
    return bounded(abs(numeric) / float(reference))


def normalized_drawdown_component(source: Mapping[str, Any], config: RiskGovernanceConfig | None = None) -> float:
    """Map drawdown state or current drawdown into a 0-1 risk component."""

    resolved = config or RiskGovernanceConfig()
    state = str(source.get("drawdown_state", "") or "").strip().lower()
    if state in resolved.drawdown_component_map:
        return bounded(resolved.drawdown_component_map[state])

    current_drawdown = safe_float(source.get("current_drawdown"))
    if pd.isna(current_drawdown):
        max_drawdown = safe_float(source.get("max_drawdown"))
        current_drawdown = max_drawdown if pd.notna(max_drawdown) else float("nan")
    if pd.isna(current_drawdown):
        return 0.0
    if float(current_drawdown) <= -0.10:
        return 1.0
    if float(current_drawdown) <= -0.05:
        return 0.50
    return 0.0


def normalized_volatility_component(source: Mapping[str, Any], config: RiskGovernanceConfig | None = None) -> float:
    """Normalize volatility forecast diagnostics to 0-1."""

    resolved = config or RiskGovernanceConfig()
    explicit = source.get("volatility_component")
    if explicit is not None and pd.notna(explicit):
        return bounded(explicit)

    numeric_component = 0.0
    for column in ("vol_forecast", "volatility_forecast", "realized_volatility", "volatility"):
        value = source.get(column)
        normalized = normalized_abs(value, resolved.volatility_reference)
        numeric_component = max(numeric_component, normalized)

    bucket = str(source.get("volatility_bucket", "") or "").strip().lower()
    bucket_component = {"low": 0.10, "medium": 0.40, "high": 0.80}.get(bucket, 0.0)
    return bounded(max(numeric_component, bucket_component))


def normalized_downside_risk_component(source: Mapping[str, Any], config: RiskGovernanceConfig | None = None) -> float:
    """Normalize VaR, CVaR, and scenario downside diagnostics to 0-1."""

    resolved = config or RiskGovernanceConfig()
    explicit = source.get("downside_risk_component")
    if explicit is not None and pd.notna(explicit):
        return bounded(explicit)

    component = 0.0
    for column in (
        "cvar_loss_95",
        "var_loss_95",
        "expected_shortfall",
        "downside_risk",
        "scenario_downside_risk",
    ):
        component = max(component, normalized_abs(source.get(column), resolved.downside_risk_reference))
    return bounded(component)


def normalized_model_health_component(status: Any, config: RiskGovernanceConfig | None = None) -> float:
    """Map model health status into a 0-1 risk component."""

    resolved = config or RiskGovernanceConfig()
    health_status = str(status or "").strip().lower()
    return bounded(resolved.model_health_component_map.get(health_status, 0.0))


def normalized_disagreement_component(source: Mapping[str, Any], config: RiskGovernanceConfig | None = None) -> float:
    """Normalize model disagreement and sign conflict diagnostics to 0-1."""

    resolved = config or RiskGovernanceConfig()
    explicit = source.get("disagreement_component")
    if explicit is not None and pd.notna(explicit):
        return bounded(explicit)

    bucket = str(source.get("agreement_bucket", "") or "").strip().lower()
    bucket_component = resolved.agreement_component_map.get(bucket, 0.0)
    disagreement_score = safe_float(source.get("disagreement_score"))
    if pd.isna(disagreement_score):
        disagreement_score = safe_float(source.get("model_disagreement_score"))
    if pd.isna(disagreement_score):
        agreement_score = safe_float(source.get("agreement_score"))
        if pd.isna(agreement_score):
            agreement_score = safe_float(source.get("model_agreement_score"))
        disagreement_score = 1.0 - agreement_score if pd.notna(agreement_score) else 0.0
    sign_conflict_component = 1.0 if bool(source.get("sign_conflict", False)) else 0.0
    return bounded(max(bucket_component, disagreement_score, sign_conflict_component))


def normalized_scenario_dispersion_component(
    source: Mapping[str, Any],
    config: RiskGovernanceConfig | None = None,
) -> float:
    """Normalize scenario uncertainty, dispersion, and weak dominance to 0-1."""

    _ = config or RiskGovernanceConfig()
    explicit = source.get("scenario_dispersion_component")
    if explicit is not None and pd.notna(explicit):
        return bounded(explicit)

    component = 0.0
    for column in (
        "scenario_uncertainty_score",
        "uncertainty_score",
        "probability_entropy",
        "scenario_dispersion_score",
        "dispersion_score",
    ):
        value = source.get(column)
        numeric = safe_float(value)
        if pd.notna(numeric):
            component = max(component, bounded(numeric))

    dominance_score = safe_float(source.get("dominance_score"))
    if pd.notna(dominance_score):
        component = max(component, bounded(1.0 - dominance_score))

    probability_gap = safe_float(source.get("probability_gap"))
    if pd.notna(probability_gap):
        component = max(component, bounded(1.0 - probability_gap))

    return bounded(component)


def normalized_calibration_component(source: Mapping[str, Any], config: RiskGovernanceConfig | None = None) -> float:
    """Normalize calibration error and scenario confidence bucket to 0-1."""

    resolved = config or RiskGovernanceConfig()
    explicit = source.get("calibration_component")
    if explicit is not None and pd.notna(explicit):
        return bounded(explicit)

    component = 0.0
    for column in (
        "scenario_calibration_error",
        "calibration_error",
        "mean_calibration_error",
    ):
        component = max(component, normalized_abs(source.get(column), resolved.calibration_error_reference))

    bucket = str(
        source.get("scenario_confidence_bucket")
        or source.get("confidence_bucket")
        or source.get("calibration_bucket")
        or ""
    ).strip().lower()
    component = max(component, resolved.scenario_confidence_component_map.get(bucket, 0.0))
    return bounded(component)


def calculate_weighted_risk_score(
    components: Mapping[str, Any],
    config: RiskGovernanceConfig | None = None,
) -> float:
    """Calculate deterministic weighted risk score from normalized components."""

    resolved = config or RiskGovernanceConfig()
    score = 0.0
    for column in RISK_COMPONENT_COLUMNS:
        score += float(resolved.scoring_weights[column]) * bounded(components.get(column, 0.0))
    return round(bounded(score), 6)


def build_risk_components(
    source: Mapping[str, Any],
    *,
    model_health_status: Any = None,
    config: RiskGovernanceConfig | None = None,
) -> dict[str, float]:
    """Build all normalized risk components from merged governance sources."""

    resolved = config or RiskGovernanceConfig()
    components = {
        "drawdown_component": normalized_drawdown_component(source, resolved),
        "volatility_component": normalized_volatility_component(source, resolved),
        "downside_risk_component": normalized_downside_risk_component(source, resolved),
        "model_health_component": normalized_model_health_component(model_health_status, resolved),
        "scenario_dispersion_component": normalized_scenario_dispersion_component(source, resolved),
        "disagreement_component": normalized_disagreement_component(source, resolved),
        "calibration_component": normalized_calibration_component(source, resolved),
    }
    return {column: round(float(np.clip(value, 0.0, 1.0)), 6) for column, value in components.items()}


def build_reason_codes(
    source: Mapping[str, Any],
    components: Mapping[str, float],
    *,
    model_health_status: Any = None,
) -> str:
    """Build deterministic pipe-separated reason codes for elevated risk components."""

    codes: list[str] = []
    drawdown_state = str(source.get("drawdown_state", "") or "").strip().lower()
    if drawdown_state == "severe" or components.get("drawdown_component", 0.0) >= 1.0:
        codes.append("severe_drawdown")
    elif drawdown_state == "elevated" or components.get("drawdown_component", 0.0) >= 0.50:
        codes.append("elevated_drawdown")

    if components.get("volatility_component", 0.0) >= 0.70:
        codes.append("volatility_spike")
    if components.get("downside_risk_component", 0.0) >= 0.65:
        codes.append("high_downside_risk")

    health_status = str(model_health_status or "").strip().lower()
    if health_status == "failing" or components.get("model_health_component", 0.0) >= 1.0:
        codes.append("failing_model_health")
    elif health_status in {"brittle", "weak"} or components.get("model_health_component", 0.0) >= 0.35:
        codes.append("weak_model_health")

    if components.get("scenario_dispersion_component", 0.0) >= 0.60:
        codes.append("high_scenario_dispersion")
    if str(source.get("agreement_bucket", "") or "").strip().lower() == "low" or components.get(
        "disagreement_component", 0.0
    ) >= 0.75:
        codes.append("low_model_agreement")
    if bool(source.get("sign_conflict", False)):
        codes.append("sign_conflict")
    if components.get("calibration_component", 0.0) >= 0.60:
        codes.append("poor_calibration")

    confidence_bucket = str(source.get("scenario_confidence_bucket", "") or "").strip().lower()
    dominance_label = str(source.get("dominance_label", "") or "").strip().lower()
    if confidence_bucket == "risk_overridden" or dominance_label == "risk_overrides_dominance":
        codes.append("risk_overrides_scenario")
    elif (
        components.get("drawdown_component", 0.0) >= 1.0
        and str(source.get("dominant_scenario", "") or "").strip().lower() in {"bull", "recovery"}
    ):
        codes.append("risk_overrides_scenario")

    if not codes:
        return "none"
    return "|".join(dict.fromkeys(codes))
