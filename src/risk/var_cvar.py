"""VaR and CVaR models for realized return distributions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.risk.base import RiskModel


def _confidence_key(confidence_level: float) -> str:
    pct = confidence_level * 100.0
    return str(int(round(pct))) if abs(pct - round(pct)) < 1e-9 else str(pct).replace(".", "_")


def aggregate_horizon_returns(returns: pd.Series, horizon: int) -> pd.Series:
    """Build realized multi-step return windows from a return series."""

    clean = pd.to_numeric(pd.Series(returns), errors="coerce").dropna().astype(float)
    if horizon <= 1:
        return clean.reset_index(drop=True)
    compounded = (1.0 + clean).rolling(window=int(horizon), min_periods=int(horizon)).apply(np.prod, raw=True) - 1.0
    return compounded.dropna().reset_index(drop=True)


def compute_var_cvar(distribution: pd.Series | np.ndarray | list[float], confidence_level: float) -> dict[str, float]:
    """Compute return-based and loss-based VaR/CVaR from a realized distribution."""

    clean = pd.to_numeric(pd.Series(distribution), errors="coerce").dropna().astype(float)
    if clean.empty:
        raise ValueError("compute_var_cvar requires a non-empty distribution")
    alpha = 1.0 - float(confidence_level)
    var_return = float(np.quantile(clean, alpha))
    tail = clean[clean <= var_return]
    cvar_return = float(tail.mean()) if not tail.empty else var_return
    return {
        "var_return": var_return,
        "cvar_return": cvar_return,
        "var_loss": float(max(0.0, -var_return)),
        "cvar_loss": float(max(0.0, -cvar_return)),
    }


class VaRCVaRRiskModel(RiskModel):
    """Historical VaR/CVaR estimator over return or pnl distributions."""

    model_name = "var_cvar"

    def __init__(
        self,
        *,
        confidence_levels: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.confidence_levels = list(confidence_levels or [0.95, 0.99])

    def _fit_model(self, returns_series: pd.Series) -> None:
        self.mean_return_ = float(returns_series.mean())
        self.volatility_ = float(returns_series.std(ddof=0))

    def forecast_risk(self, horizon: int = 1) -> dict[str, Any]:
        if self.returns_ is None:
            raise RuntimeError(f"{self.model_name} is not fitted")

        realized = aggregate_horizon_returns(self.returns_, int(horizon))
        result: dict[str, Any] = {
            "risk_model": self.model_name,
            "horizon": int(horizon),
            "observations": int(len(realized)),
            "expected_return": float(realized.mean()),
            "volatility": float(realized.std(ddof=0)),
        }
        for conf in self.confidence_levels:
            metrics = compute_var_cvar(realized, float(conf))
            key = _confidence_key(float(conf))
            result[f"var_return_{key}"] = metrics["var_return"]
            result[f"cvar_return_{key}"] = metrics["cvar_return"]
            result[f"var_loss_{key}"] = metrics["var_loss"]
            result[f"cvar_loss_{key}"] = metrics["cvar_loss"]
        return result
