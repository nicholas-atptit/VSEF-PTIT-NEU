"""Risk analytics package for time-series-safe ML trading workflows."""

from __future__ import annotations

from typing import Any

import numpy as np

from .core_risk import (
    build_system_return_series,
    compute_cvar,
    compute_drawdown,
    compute_historical_var,
    compute_max_drawdown_series,
    compute_monte_carlo_cvar,
    compute_monte_carlo_var,
    compute_rolling_drawdown,
)
from .covar import compute_covar, compute_delta_covar, compute_var
from .risk_engine import RiskEngine, RiskSummary


class MonteCarloRiskSimulator:
    """Deterministic point-estimate Monte Carlo wrapper used at inference time."""

    def __init__(self, simulations: int = 10000, random_seed: int = 42) -> None:
        self.simulations = simulations
        self.random_seed = random_seed

    def simulate_risk(
        self,
        forecast_mean: float,
        volatility_proxy: float,
        horizon: str | int = 1,
        confidence_levels: list[float] | None = None,
    ) -> dict[str, Any]:
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]
        if volatility_proxy <= 0:
            raise ValueError("Volatility proxy must be strictly positive.")

        rng = np.random.default_rng(self.random_seed)
        scenarios = rng.normal(
            loc=float(forecast_mean),
            scale=float(volatility_proxy),
            size=int(self.simulations),
        )
        sorted_scenarios = np.sort(scenarios)

        var_metrics: dict[str, float] = {}
        cvar_metrics: dict[str, float] = {}
        for conf in confidence_levels:
            if not 0.0 < conf < 1.0:
                raise ValueError("Confidence levels must be in range (0, 1).")
            alpha = 1.0 - conf
            idx = int(np.floor(alpha * self.simulations))
            idx = min(max(idx, 0), len(sorted_scenarios) - 1)
            var_val = float(sorted_scenarios[idx])
            cvar_val = float(np.mean(sorted_scenarios[: max(idx, 1)]))
            key = f"{conf * 100:.1f}"
            var_metrics[key] = var_val
            cvar_metrics[key] = cvar_val

        return {
            "var": var_metrics,
            "cvar": cvar_metrics,
            "summary": {
                "mean": float(np.mean(scenarios)),
                "std": float(np.std(scenarios)),
                "min": float(sorted_scenarios[0]),
                "max": float(sorted_scenarios[-1]),
                "median": float(np.median(scenarios)),
            },
            "metadata": {
                "assumptions": "Normal distribution of residuals around forecast mean",
                "simulations": int(self.simulations),
                "horizon": horizon,
                "random_seed": int(self.random_seed),
            },
        }


__all__ = [
    "MonteCarloRiskSimulator",
    "RiskEngine",
    "RiskSummary",
    "build_system_return_series",
    "compute_cvar",
    "compute_covar",
    "compute_delta_covar",
    "compute_drawdown",
    "compute_historical_var",
    "compute_max_drawdown_series",
    "compute_monte_carlo_cvar",
    "compute_monte_carlo_var",
    "compute_rolling_drawdown",
    "compute_var",
]
