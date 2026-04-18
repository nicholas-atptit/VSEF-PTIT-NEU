"""Realized drawdown analytics for return or pnl distributions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.ml.risk.core_risk import compute_drawdown, compute_max_drawdown_series, compute_rolling_drawdown
from src.risk.base import RiskModel


def _underwater_durations(drawdown: pd.Series) -> list[int]:
    durations: list[int] = []
    current = 0
    for value in pd.to_numeric(drawdown, errors="coerce").fillna(0.0):
        if float(value) < 0.0:
            current += 1
        elif current > 0:
            durations.append(current)
            current = 0
    if current > 0:
        durations.append(current)
    return durations


class DrawdownRiskModel(RiskModel):
    """Drawdown analytics over realized returns or pnl."""

    model_name = "drawdown"

    def __init__(self, *, rolling_window: int = 63) -> None:
        super().__init__()
        self.rolling_window = int(rolling_window)

    def _fit_model(self, returns_series: pd.Series) -> None:
        self.total_return_ = float((1.0 + returns_series).prod() - 1.0)

    def forecast_risk(self, horizon: int = 1) -> dict[str, Any]:
        if self.returns_ is None:
            raise RuntimeError(f"{self.model_name} is not fitted")

        drawdown = compute_drawdown(self.returns_)
        max_drawdown = compute_max_drawdown_series(self.returns_)
        rolling_drawdown = compute_rolling_drawdown(
            self.returns_,
            window=self.rolling_window,
            min_periods=min(self.rolling_window, len(self.returns_)),
        )
        negative_drawdowns = drawdown[drawdown < 0.0]
        durations = _underwater_durations(drawdown)
        max_dd_value = float(max_drawdown.iloc[-1]) if not max_drawdown.empty else 0.0
        recovery_factor = (
            float(self.total_return_ / abs(max_dd_value))
            if abs(max_dd_value) > 1e-12
            else float("inf")
        )
        return {
            "risk_model": self.model_name,
            "horizon": int(horizon),
            "current_drawdown": float(drawdown.iloc[-1]) if not drawdown.empty else 0.0,
            "max_drawdown": max_dd_value,
            "average_drawdown": float(negative_drawdowns.mean()) if not negative_drawdowns.empty else 0.0,
            "latest_rolling_drawdown": float(rolling_drawdown.iloc[-1]) if not rolling_drawdown.empty else 0.0,
            "underwater_periods": int(len(durations)),
            "max_underwater_duration": int(max(durations)) if durations else 0,
            "total_return": float(self.total_return_),
            "recovery_factor": float(recovery_factor),
        }
