"""Threshold-based signal generation for standardized forecast outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.contracts import validate_forecast_frame, validate_signal_frame


def generate_signal_value(
    predicted_return: float,
    *,
    threshold: float,
    allow_short: bool = False,
) -> float:
    """Map predicted return to a directional signal."""

    value = float(predicted_return)
    barrier = max(float(threshold), 0.0)
    if value > barrier:
        return 1.0
    if allow_short and value < -barrier:
        return -1.0
    return 0.0


def generate_threshold_signals(
    forecast_df: pd.DataFrame,
    *,
    threshold: float = 0.0,
    allow_short: bool = False,
) -> pd.DataFrame:
    """Convert a forecast frame into a signal frame with explicit thresholds."""

    validated = validate_forecast_frame(forecast_df, require_y_true=False)
    barrier = max(float(threshold), 0.0)
    signals = pd.DataFrame(
        {
            "timestamp": validated["timestamp"].to_numpy(),
            "ticker": validated["ticker"].to_numpy(),
            "model_name": validated["model_name"].to_numpy(),
            "signal": [
                generate_signal_value(value, threshold=barrier, allow_short=allow_short)
                for value in pd.to_numeric(validated["y_pred"], errors="coerce").fillna(0.0)
            ],
            "threshold": barrier,
            "y_pred": pd.to_numeric(validated["y_pred"], errors="coerce").astype(float).to_numpy(),
            "y_true": pd.to_numeric(validated["y_true"], errors="coerce").astype(float).to_numpy(),
            "target_type": validated["target_type"].to_numpy(),
            "horizon": validated["horizon"].astype(int).to_numpy(),
            "window_id": validated["window_id"].astype(str).to_numpy(),
        }
    )
    if "target_timestamp" in validated.columns:
        signals["target_timestamp"] = pd.to_datetime(validated["target_timestamp"], errors="coerce").to_numpy()
    signals["signal_strength"] = np.where(
        barrier > 0,
        np.abs(signals["y_pred"]) / barrier,
        np.abs(signals["y_pred"]),
    )
    return validate_signal_frame(signals)
