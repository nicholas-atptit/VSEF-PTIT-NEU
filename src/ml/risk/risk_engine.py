"""Aggregate rolling risk engine for per-asset and system-level metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

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
from .covar import compute_covar, compute_delta_covar


@dataclass(frozen=True)
class RiskSummary:
    per_asset: dict[str, dict[str, float | None]]
    system: dict[str, float | None]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_asset": self.per_asset,
            "system": self.system,
            "config": self.config,
        }


class RiskEngine:
    """Deterministic rolling risk engine using historical and optional Monte Carlo tails."""

    def __init__(
        self,
        *,
        window: int = 60,
        quantile: float = 0.05,
        simulations: int = 2000,
        random_seed: int = 42,
        include_monte_carlo: bool = False,
    ) -> None:
        self.window = window
        self.quantile = quantile
        self.simulations = simulations
        self.random_seed = random_seed
        self.include_monte_carlo = include_monte_carlo

    def _risk_frame(self, asset_returns: pd.Series, system_returns: pd.Series) -> pd.DataFrame:
        frame = pd.DataFrame(index=asset_returns.index)
        frame["var_q"] = compute_historical_var(asset_returns, window=self.window, quantile=self.quantile)
        frame["cvar_q"] = compute_cvar(asset_returns, window=self.window, quantile=self.quantile)
        frame["covar_q"] = compute_covar(asset_returns, system_returns, window=self.window, quantile=self.quantile)
        frame["delta_covar"] = compute_delta_covar(
            asset_returns,
            system_returns,
            window=self.window,
            quantile=self.quantile,
        )
        frame["drawdown"] = compute_drawdown(asset_returns)
        frame["max_drawdown"] = compute_max_drawdown_series(asset_returns)
        frame["rolling_drawdown"] = compute_rolling_drawdown(asset_returns, window=self.window)
        if self.include_monte_carlo:
            frame["mc_var_q"] = compute_monte_carlo_var(
                asset_returns,
                window=self.window,
                quantile=self.quantile,
                simulations=self.simulations,
                random_seed=self.random_seed,
            )
            frame["mc_cvar_q"] = compute_monte_carlo_cvar(
                asset_returns,
                window=self.window,
                quantile=self.quantile,
                simulations=self.simulations,
                random_seed=self.random_seed,
            )
        return frame

    @staticmethod
    def _latest_values(frame: pd.DataFrame) -> dict[str, float | None]:
        latest: dict[str, float | None] = {}
        for column in frame.columns:
            non_na = frame[column].dropna()
            latest[column] = None if non_na.empty else float(non_na.iloc[-1])
        return latest

    def evaluate(
        self,
        returns: pd.Series | pd.DataFrame,
        *,
        market_returns: pd.Series | None = None,
    ) -> dict[str, Any]:
        if isinstance(returns, pd.Series):
            asset_frame = returns.to_frame(name=returns.name or "asset")
        else:
            asset_frame = returns.copy()
        asset_frame = asset_frame.apply(pd.to_numeric, errors="coerce")
        system_returns = build_system_return_series(asset_frame, market_returns=market_returns)
        system_returns = system_returns.reindex(asset_frame.index)

        per_asset_frames: dict[str, pd.DataFrame] = {}
        per_asset_summary: dict[str, dict[str, float | None]] = {}
        for column in asset_frame.columns:
            frame = self._risk_frame(asset_frame[column], system_returns)
            per_asset_frames[column] = frame
            per_asset_summary[column] = self._latest_values(frame)

        system_frame = pd.DataFrame(index=asset_frame.index)
        system_frame["var_q"] = compute_historical_var(system_returns, window=self.window, quantile=self.quantile)
        system_frame["cvar_q"] = compute_cvar(system_returns, window=self.window, quantile=self.quantile)
        system_frame["drawdown"] = compute_drawdown(system_returns)
        system_frame["max_drawdown"] = compute_max_drawdown_series(system_returns)
        system_frame["rolling_drawdown"] = compute_rolling_drawdown(system_returns, window=self.window)
        if self.include_monte_carlo:
            system_frame["mc_var_q"] = compute_monte_carlo_var(
                system_returns,
                window=self.window,
                quantile=self.quantile,
                simulations=self.simulations,
                random_seed=self.random_seed,
            )
            system_frame["mc_cvar_q"] = compute_monte_carlo_cvar(
                system_returns,
                window=self.window,
                quantile=self.quantile,
                simulations=self.simulations,
                random_seed=self.random_seed,
            )

        summary = RiskSummary(
            per_asset=per_asset_summary,
            system=self._latest_values(system_frame),
            config={
                "window": self.window,
                "quantile": self.quantile,
                "simulations": self.simulations,
                "random_seed": self.random_seed,
                "include_monte_carlo": self.include_monte_carlo,
            },
        )
        return {
            "per_asset_frames": per_asset_frames,
            "system_frame": system_frame,
            "risk_summary": summary.to_dict(),
        }
