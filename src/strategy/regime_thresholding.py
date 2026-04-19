"""Regime-aware thresholding for Phase 2 strategy conditioning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.contracts import validate_forecast_frame, validate_signal_frame
from src.strategy.thresholding import generate_signal_value


def prepare_context_frame(frame: pd.DataFrame | None, *, source: str) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return None
    prepared = frame.copy()
    if "timestamp" in prepared.columns:
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
    if "ticker" in prepared.columns:
        prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    if "window_id" in prepared.columns:
        prepared["window_id"] = prepared["window_id"].astype(str)
    if "model_name" in prepared.columns:
        prepared["model_name"] = prepared["model_name"].astype(str)
    rename_map: dict[str, str] = {}
    if source == "risk" and "source_model" in prepared.columns:
        rename_map["source_model"] = "risk_source_model"
    if source == "regime" and "source_model" in prepared.columns:
        rename_map["source_model"] = "regime_source_model"
    if rename_map:
        prepared = prepared.rename(columns=rename_map)
    return prepared


def merge_strategy_context(
    forecast_df: pd.DataFrame,
    *,
    risk_df: pd.DataFrame | None = None,
    regime_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    merged = forecast_df.copy()
    prepared_risk = prepare_context_frame(risk_df, source="risk")
    if prepared_risk is not None:
        join_keys = [
            column
            for column in ("timestamp", "ticker", "window_id", "model_name")
            if column in merged.columns and column in prepared_risk.columns
        ]
        if join_keys:
            merged = merged.merge(prepared_risk, on=join_keys, how="left", suffixes=("", "_risk"))
    prepared_regime = prepare_context_frame(regime_df, source="regime")
    if prepared_regime is not None:
        join_keys = [
            column
            for column in ("timestamp", "ticker", "window_id")
            if column in merged.columns and column in prepared_regime.columns
        ]
        if join_keys:
            merged = merged.merge(prepared_regime, on=join_keys, how="left", suffixes=("", "_regime"))
    return merged


def resolve_regime_threshold(
    row: pd.Series,
    *,
    base_threshold: float,
    regime_thresholds: dict[str, float] | None = None,
    regime_threshold_multipliers: dict[str, float] | None = None,
) -> float:
    label = str(row.get("regime_label", "sideway") or "sideway").lower()
    direct_thresholds = {str(key).lower(): float(value) for key, value in (regime_thresholds or {}).items()}
    if label in direct_thresholds:
        return max(direct_thresholds[label], 0.0)

    multipliers = {"bull": 0.85, "sideway": 1.0, "bear": 1.25}
    multipliers.update({str(key).lower(): float(value) for key, value in (regime_threshold_multipliers or {}).items()})
    return max(float(base_threshold) * float(multipliers.get(label, 1.0)), 0.0)


def generate_regime_aware_signals(
    forecast_df: pd.DataFrame,
    *,
    threshold: float = 0.0,
    allow_short: bool = False,
    risk_df: pd.DataFrame | None = None,
    regime_df: pd.DataFrame | None = None,
    regime_thresholds: dict[str, float] | None = None,
    regime_threshold_multipliers: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Generate thresholded signals after joining regime and risk context."""

    validated = validate_forecast_frame(forecast_df, require_y_true=False)
    merged = merge_strategy_context(validated, risk_df=risk_df, regime_df=regime_df)
    applied_threshold = merged.apply(
        resolve_regime_threshold,
        axis=1,
        base_threshold=max(float(threshold), 0.0),
        regime_thresholds=regime_thresholds,
        regime_threshold_multipliers=regime_threshold_multipliers,
    )

    signals = pd.DataFrame(
        {
            "timestamp": merged["timestamp"].to_numpy(),
            "ticker": merged["ticker"].to_numpy(),
            "model_name": merged["model_name"].to_numpy(),
            "signal": [
                generate_signal_value(value, threshold=barrier, allow_short=allow_short)
                for value, barrier in zip(
                    pd.to_numeric(merged["y_pred"], errors="coerce").fillna(0.0),
                    applied_threshold.to_numpy(),
                    strict=False,
                )
            ],
            "threshold": applied_threshold.to_numpy(dtype=float),
            "y_pred": pd.to_numeric(merged["y_pred"], errors="coerce").astype(float).to_numpy(),
            "y_true": pd.to_numeric(merged["y_true"], errors="coerce").astype(float).to_numpy(),
            "target_type": merged["target_type"].to_numpy(),
            "horizon": merged["horizon"].astype(int).to_numpy(),
            "window_id": merged["window_id"].astype(str).to_numpy(),
        }
    )
    if "target_timestamp" in merged.columns:
        signals["target_timestamp"] = pd.to_datetime(merged["target_timestamp"], errors="coerce").to_numpy()
    signals["signal_strength"] = np.where(
        signals["threshold"] > 0,
        np.abs(signals["y_pred"]) / signals["threshold"],
        np.abs(signals["y_pred"]),
    )

    passthrough_columns = [
        "regime_label",
        "regime_prob_bull",
        "regime_prob_bear",
        "regime_prob_sideway",
        "regime_source_model",
        "vol_forecast",
        "volatility",
        "var_loss_95",
        "cvar_loss_95",
        "var_loss_99",
        "cvar_loss_99",
        "drawdown_state",
        "current_drawdown",
        "max_drawdown",
        "risk_source_model",
        "risk_model",
    ]
    for column in passthrough_columns:
        if column in merged.columns:
            signals[column] = merged[column].to_numpy()
    return validate_signal_frame(signals)
