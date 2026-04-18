"""Explicit target specifications for forecast-layer rehabilitation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastTargetSpec:
    """Serializable description of one evaluation target."""

    name: str
    target_column: str
    target_type: str
    target_family: str
    description: str
    tradable_output: bool
    neutral_threshold: float = 0.0
    annualize_volatility: bool = True


TARGET_SPECS: dict[str, dict[str, Any]] = {
    "forward_return": {
        "target_column": "target_forward_return",
        "target_type": "forward_return",
        "target_family": "return_regression",
        "description": "Forward close-to-close return over the requested horizon.",
        "tradable_output": True,
    },
    "forward_log_return": {
        "target_column": "target_forward_log_return",
        "target_type": "forward_log_return",
        "target_family": "return_regression",
        "description": "Forward log return over the requested horizon.",
        "tradable_output": True,
    },
    "direction_binary": {
        "target_column": "target_direction_binary",
        "target_type": "direction_binary",
        "target_family": "direction_classification",
        "description": "Signed up/down direction derived from the forward return.",
        "tradable_output": False,
        "neutral_threshold": 0.0,
    },
    "future_realized_volatility": {
        "target_column": "target_future_realized_volatility",
        "target_type": "future_realized_volatility",
        "target_family": "volatility_regression",
        "description": "Future realized volatility over the requested horizon.",
        "tradable_output": False,
        "annualize_volatility": True,
    },
}


def supported_target_specs() -> list[str]:
    return sorted(TARGET_SPECS)


def build_target_spec(
    name: str,
    *,
    target_column: str | None = None,
    neutral_threshold: float | None = None,
    annualize_volatility: bool | None = None,
) -> ForecastTargetSpec:
    key = str(name).strip().lower()
    if key not in TARGET_SPECS:
        raise ValueError(f"Unsupported target spec '{name}'. Available: {supported_target_specs()}")
    payload = dict(TARGET_SPECS[key])
    if target_column is not None:
        payload["target_column"] = str(target_column)
    if neutral_threshold is not None:
        payload["neutral_threshold"] = float(neutral_threshold)
    if annualize_volatility is not None:
        payload["annualize_volatility"] = bool(annualize_volatility)
    return ForecastTargetSpec(name=key, **payload)


def target_spec_manifest() -> dict[str, dict[str, Any]]:
    return {
        name: {
            **payload,
            "documented_behavior": "explicit_leakage_safe_target_builder",
        }
        for name, payload in TARGET_SPECS.items()
    }


def apply_target_spec(
    frame: pd.DataFrame,
    *,
    horizon: int,
    target_spec: ForecastTargetSpec,
) -> pd.DataFrame:
    """Apply one leakage-safe target definition to a prepared frame."""

    if frame.empty:
        raise ValueError("Cannot apply a target spec to an empty frame")
    if "timestamp" not in frame.columns:
        raise ValueError("Target generation requires a timestamp column")
    if "close" not in frame.columns:
        raise ValueError("Target generation requires a close column")

    prepared = frame.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
    prepared = prepared.sort_values("timestamp").reset_index(drop=True)

    close = pd.to_numeric(prepared["close"], errors="coerce")
    if close.isna().all():
        raise ValueError("Target generation requires usable close values")

    horizon = int(horizon)
    forward_return = close.shift(-horizon) / close - 1.0
    prepared["target_timestamp"] = prepared["timestamp"].shift(-horizon)
    prepared["daily_return"] = close.pct_change()
    prepared["current_log_return_1d"] = np.log(close / close.shift(1))
    prepared["current_direction_1d"] = np.sign(prepared["daily_return"]).astype(float)

    if target_spec.name == "forward_return":
        prepared[target_spec.target_column] = forward_return.astype(float)
    elif target_spec.name == "forward_log_return":
        prepared[target_spec.target_column] = np.log(close.shift(-horizon) / close).astype(float)
    elif target_spec.name == "direction_binary":
        neutral_threshold = float(target_spec.neutral_threshold)
        labels = np.where(
            forward_return > neutral_threshold,
            1.0,
            np.where(forward_return < -neutral_threshold, -1.0, 0.0),
        )
        prepared[target_spec.target_column] = pd.Series(labels, index=prepared.index, dtype=float)
        prepared.loc[forward_return.isna(), target_spec.target_column] = np.nan
    elif target_spec.name == "future_realized_volatility":
        log_return = np.log(close / close.shift(1))
        realized_vol = log_return.rolling(horizon).std().shift(-horizon)
        if target_spec.annualize_volatility:
            realized_vol = realized_vol * np.sqrt(252.0)
        prepared[target_spec.target_column] = realized_vol.astype(float)
    else:  # pragma: no cover - build_target_spec guards this path
        raise ValueError(f"Unsupported target spec '{target_spec.name}'")

    return prepared
