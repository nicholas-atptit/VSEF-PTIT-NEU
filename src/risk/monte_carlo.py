"""Monte Carlo risk model for return and pnl distributions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.risk.base import RiskModel


def _confidence_key(confidence_level: float) -> str:
    pct = confidence_level * 100.0
    return str(int(round(pct))) if abs(pct - round(pct)) < 1e-9 else str(pct).replace(".", "_")


def simulate_return_paths(
    returns: pd.Series,
    *,
    horizon: int,
    simulations: int,
    random_seed: int,
    method: str = "bootstrap",
) -> np.ndarray:
    """Generate deterministic return paths from the fitted return distribution."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if simulations <= 0:
        raise ValueError("simulations must be positive")

    clean = pd.to_numeric(pd.Series(returns), errors="coerce").dropna().astype(float)
    if clean.empty:
        raise ValueError("simulate_return_paths requires non-empty returns")

    rng = np.random.default_rng(random_seed)
    if method == "bootstrap":
        sample = rng.choice(clean.to_numpy(dtype=float), size=(simulations, horizon), replace=True)
    elif method == "normal":
        mean = float(clean.mean())
        std = float(clean.std(ddof=0))
        if std <= 0:
            sample = np.full((simulations, horizon), mean, dtype=float)
        else:
            sample = rng.normal(loc=mean, scale=std, size=(simulations, horizon))
    else:
        raise ValueError(f"Unsupported Monte Carlo method '{method}'")
    return np.asarray(sample, dtype=float)


class MonteCarloRiskModel(RiskModel):
    """Monte Carlo simulation from return or residual distributions."""

    model_name = "monte_carlo"

    def __init__(
        self,
        *,
        simulations: int = 5000,
        random_seed: int = 42,
        method: str = "bootstrap",
        confidence_levels: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.simulations = int(simulations)
        self.random_seed = int(random_seed)
        self.method = str(method)
        self.confidence_levels = list(confidence_levels or [0.95, 0.99])

    def _fit_model(self, returns_series: pd.Series) -> None:
        self.returns_mean_ = float(returns_series.mean())
        self.returns_std_ = float(returns_series.std(ddof=0))

    def forecast_risk(self, horizon: int = 1) -> dict[str, Any]:
        if self.returns_ is None:
            raise RuntimeError(f"{self.model_name} is not fitted")

        paths = simulate_return_paths(
            self.returns_,
            horizon=int(horizon),
            simulations=self.simulations,
            random_seed=self.random_seed,
            method=self.method,
        )
        compounded = np.prod(1.0 + paths, axis=1) - 1.0
        equity_paths = np.cumprod(1.0 + paths, axis=1)
        running_peaks = np.maximum.accumulate(equity_paths, axis=1)
        path_drawdowns = np.min((equity_paths / running_peaks) - 1.0, axis=1)

        result: dict[str, Any] = {
            "risk_model": self.model_name,
            "horizon": int(horizon),
            "simulation_method": self.method,
            "simulations": int(self.simulations),
            "expected_return": float(np.mean(compounded)),
            "volatility": float(np.std(compounded, ddof=0)),
            "median_return": float(np.median(compounded)),
            "worst_return": float(np.min(compounded)),
            "best_return": float(np.max(compounded)),
            "mean_path_drawdown": float(np.mean(path_drawdowns)),
            "worst_path_drawdown": float(np.min(path_drawdowns)),
        }

        for conf in self.confidence_levels:
            alpha = 1.0 - float(conf)
            var_return = float(np.quantile(compounded, alpha))
            tail = compounded[compounded <= var_return]
            cvar_return = float(np.mean(tail)) if tail.size else var_return
            key = _confidence_key(float(conf))
            result[f"var_return_{key}"] = var_return
            result[f"cvar_return_{key}"] = cvar_return
            result[f"var_loss_{key}"] = float(max(0.0, -var_return))
            result[f"cvar_loss_{key}"] = float(max(0.0, -cvar_return))

        return result
