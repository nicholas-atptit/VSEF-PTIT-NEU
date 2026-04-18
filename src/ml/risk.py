"""Heuristic residual-scenario tail summaries for forecast interpretation.

This module deliberately avoids the ML model registry to cleanly separate
predictive models from post-prediction portfolio risk assessment.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


class MonteCarloRiskSimulator:
    """Residual-based Monte Carlo simulator for heuristic forecast-risk scenarios."""

    def __init__(self, simulations: int = 10000, random_seed: int = 42) -> None:
        """
        Initialize the simulator.
        
        Args:
            simulations: Number of random paths/scenarios to generate
            random_seed: Random seed guaranteeing deterministic risk metrics
        """
        self.simulations = simulations
        self.random_seed = random_seed

    def simulate_risk(
        self,
        forecast_mean: float,
        volatility_proxy: float,
        horizon: str | int = 1,
        confidence_levels: List[float] | None = None,
    ) -> Dict[str, Any]:
        """
        Simulate residual-based scenarios to calculate approximate scenario VaR and CVaR.

        Args:
            forecast_mean: Expected return over the horizon.
            volatility_proxy: Standard deviation of residuals or market vol proxy over the horizon.
            horizon: Forward timeframe identifier (e.g. "short", "mid", "long") or integer days (for metadata).
            confidence_levels: Scenario quantile levels (e.g. 0.95, 0.99). Defaults to [0.95, 0.99].

        Returns:
            Dictionary containing scenario VaR, scenario CVaR, stats, and methodology metadata.
        """
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]
            
        if volatility_proxy <= 0:
            raise ValueError("Volatility proxy must be strictly positive.")

        # Guaranteed deterministic generation per seed
        rng = np.random.default_rng(self.random_seed)

        # This is an approximate residual-driven scenario layer, not a calibrated
        # forecast-confidence model.
        scenarios = rng.normal(loc=forecast_mean, scale=volatility_proxy, size=self.simulations)
        
        # Sort for rapid quantile evaluation
        sorted_scenarios = np.sort(scenarios)

        var_metrics = {}
        cvar_metrics = {}

        for conf in confidence_levels:
            if not 0.0 < conf < 1.0:
                raise ValueError("Confidence levels must be in range (0, 1).")
            
            # For a 95% confidence VaR, we look at the lowest 5% quantile.
            alpha = 1.0 - conf
            # Compute index safely
            idx = int(np.floor(alpha * self.simulations))
            
            var_val = float(sorted_scenarios[idx])
            
            # CVaR is the expectation of returns below VaR
            # Taking the mean of all points up to `idx`
            if idx > 0:
                cvar_val = float(np.mean(sorted_scenarios[:idx]))
            else:
                cvar_val = var_val

            # Format keys cleanly, e.g., '95.0'
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
                "assumptions": (
                    "Residual-based normal scenarios around the point forecast; "
                    "not calibrated forecast confidence."
                ),
                "risk_model_type": "residual_normal_scenario_simulation",
                "uncertainty_methodology": (
                    "residual_based_normal_scenario_simulation_using_point_forecast_and_volatility_proxy"
                ),
                "calibration_status": "heuristic_not_calibrated",
                "interpretation_warning": (
                    "Scenario VaR/CVaR values are heuristic tail summaries from residual-based simulated returns. "
                    "They are not calibrated confidence intervals or guaranteed loss bounds."
                ),
                "tail_metric_context": "scenario_quantiles_of_simulated_return_distribution",
                "simulations": self.simulations,
                "horizon": horizon,
                "random_seed": self.random_seed,
            }
        }
