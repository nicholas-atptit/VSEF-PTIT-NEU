"""Position sizing helpers for Phase 1 threshold-based strategies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.core.contracts import validate_position_frame, validate_signal_frame


RISK_JOIN_PRIORITY = ["timestamp", "ticker", "model_name", "window_id"]


def _prepare_risk_frame(risk_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if risk_df is None or risk_df.empty:
        return None
    prepared = risk_df.copy()
    if "timestamp" in prepared.columns:
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
    if "ticker" in prepared.columns:
        prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    if "model_name" in prepared.columns:
        prepared["model_name"] = prepared["model_name"].astype(str)
    return prepared


def _merge_signal_and_risk(signal_df: pd.DataFrame, risk_df: pd.DataFrame | None) -> pd.DataFrame:
    if risk_df is None or risk_df.empty:
        return signal_df.copy()
    prepared_risk = _prepare_risk_frame(risk_df)
    join_keys = [column for column in RISK_JOIN_PRIORITY if column in signal_df.columns and column in prepared_risk.columns]
    if not join_keys:
        raise ValueError("risk_df must share at least one join key with the signal frame")
    additional_columns = [column for column in prepared_risk.columns if column not in join_keys and column not in signal_df.columns]
    if not additional_columns:
        return signal_df.copy()
    return signal_df.merge(prepared_risk[[*join_keys, *additional_columns]], on=join_keys, how="left", suffixes=("", "_risk"))


def _row_position_size(
    row: pd.Series,
    *,
    sizing_mode: str,
    fixed_position_size: float,
    risk_budget: float,
    max_position_size: float,
    min_position_size: float,
    volatility_floor: float,
    regime_size_multipliers: dict[str, float],
    drawdown_state_multipliers: dict[str, float],
) -> float:
    signal = float(row.get("signal", 0.0) or 0.0)
    if signal == 0.0:
        return 0.0
    if sizing_mode == "fixed_fraction":
        return float(np.clip(fixed_position_size, 0.0, max_position_size))
    if sizing_mode != "adaptive":
        raise ValueError(f"Unsupported sizing_mode '{sizing_mode}'")

    barrier = max(float(row.get("threshold", 0.0) or 0.0), 0.0)
    signal_strength = float(abs(row.get("y_pred", 0.0)))
    conviction = signal_strength / max(barrier, volatility_floor) if barrier > 0 else min(signal_strength, 1.0)
    conviction = float(np.clip(conviction, 0.0, 1.0))

    risk_scale = max_position_size
    for column in ("vol_forecast", "volatility", "cvar_loss_95", "var_loss_95"):
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if pd.notna(value) and float(value) > 0:
            risk_scale = min(max_position_size, float(risk_budget) / max(float(value), volatility_floor))
            break

    raw_regime_label = row.get("regime_label")
    regime_label = str(raw_regime_label).lower() if pd.notna(raw_regime_label) else ""
    drawdown_state = str(row.get("drawdown_state", "normal") or "normal").lower()
    regime_multiplier = float(regime_size_multipliers.get(regime_label, 1.0)) if regime_label else 1.0
    drawdown_multiplier = float(drawdown_state_multipliers.get(drawdown_state, 1.0))

    size = max(min_position_size, conviction * risk_scale * regime_multiplier * drawdown_multiplier)
    return float(np.clip(size, 0.0, max_position_size))


def size_positions(
    signal_df: pd.DataFrame,
    *,
    risk_df: pd.DataFrame | None = None,
    capital_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Size threshold-based signals from volatility or risk-budget constraints."""

    validated = validate_signal_frame(signal_df)
    config = dict(capital_config or {})
    sizing_mode = str(config.get("sizing_mode", "adaptive") or "adaptive").lower()
    risk_budget = float(config.get("risk_budget", 0.02))
    max_position_size = float(config.get("max_position_size", 1.0))
    min_position_size = float(config.get("min_position_size", 0.0))
    volatility_floor = float(config.get("volatility_floor", 0.005))
    fixed_position_size = float(config.get("fixed_position_size", max_position_size))
    regime_size_multipliers = {
        "bull": 1.0,
        "sideway": 0.75,
        "bear": 0.35,
    }
    regime_size_multipliers.update(
        {str(key).lower(): float(value) for key, value in dict(config.get("regime_size_multipliers", {})).items()}
    )
    drawdown_state_multipliers = {
        "normal": 1.0,
        "elevated": 0.6,
        "severe": 0.25,
    }
    drawdown_state_multipliers.update(
        {str(key).lower(): float(value) for key, value in dict(config.get("drawdown_state_multipliers", {})).items()}
    )

    merged = _merge_signal_and_risk(validated, risk_df)
    merged["position_size"] = merged.apply(
        _row_position_size,
        axis=1,
        sizing_mode=sizing_mode,
        fixed_position_size=fixed_position_size,
        risk_budget=risk_budget,
        max_position_size=max_position_size,
        min_position_size=min_position_size,
        volatility_floor=volatility_floor,
        regime_size_multipliers=regime_size_multipliers,
        drawdown_state_multipliers=drawdown_state_multipliers,
    )
    scale = max(max_position_size, 1e-12)
    merged["size_multiplier"] = pd.to_numeric(merged["position_size"], errors="coerce").fillna(0.0) / scale
    merged["sizing_mode"] = sizing_mode
    position_columns = [
        "timestamp",
        "ticker",
        "model_name",
        "signal",
        "position_size",
    ]
    optional_columns = [
        "threshold",
        "y_pred",
        "y_true",
        "target_type",
        "horizon",
        "window_id",
        "target_timestamp",
        "regime_label",
        "regime_prob_bull",
        "regime_prob_bear",
        "regime_prob_sideway",
        "regime_source_model",
        "vol_forecast",
        "volatility",
        "var_loss_95",
        "cvar_loss_95",
        "drawdown_state",
        "current_drawdown",
        "max_drawdown",
        "risk_source_model",
        "risk_model",
        "size_multiplier",
        "sizing_mode",
    ]
    keep_columns = position_columns + [column for column in optional_columns if column in merged.columns]
    return validate_position_frame(merged[keep_columns])
