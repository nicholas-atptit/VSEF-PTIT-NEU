from __future__ import annotations

import pandas as pd
import pytest

from src.reporting.hardening import (
    build_cost_sensitivity_summary,
    build_phase25_stability_summary,
    build_phase3_readiness_assessment,
)


def _sample_aggregate_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": "r1",
                "core_run_id": "small_banks_h05",
                "group_name": "small_banks",
                "horizon": 5,
                "threshold": 0.005,
                "cost_mode": "baseline",
                "sizing_mode": "adaptive",
                "strategy_variant": "forecast_only",
                "cagr": 0.12,
                "sharpe": 1.00,
                "max_drawdown": -0.10,
                "turnover": 1.00,
                "total_return": 0.08,
                "delta_cagr_vs_forecast_only": 0.0,
                "delta_sharpe_vs_forecast_only": 0.0,
                "delta_max_drawdown_vs_forecast_only": 0.0,
                "delta_turnover_vs_forecast_only": 0.0,
            },
            {
                "run_id": "r1",
                "core_run_id": "small_banks_h05",
                "group_name": "small_banks",
                "horizon": 5,
                "threshold": 0.005,
                "cost_mode": "baseline",
                "sizing_mode": "adaptive",
                "strategy_variant": "forecast_plus_risk",
                "cagr": 0.10,
                "sharpe": 1.10,
                "max_drawdown": -0.06,
                "turnover": 0.70,
                "total_return": 0.07,
                "delta_cagr_vs_forecast_only": -0.02,
                "delta_sharpe_vs_forecast_only": 0.10,
                "delta_max_drawdown_vs_forecast_only": 0.04,
                "delta_turnover_vs_forecast_only": -0.30,
            },
            {
                "run_id": "r1",
                "core_run_id": "small_banks_h05",
                "group_name": "small_banks",
                "horizon": 5,
                "threshold": 0.005,
                "cost_mode": "baseline",
                "sizing_mode": "adaptive",
                "strategy_variant": "forecast_plus_risk_and_regime",
                "cagr": 0.09,
                "sharpe": 1.20,
                "max_drawdown": -0.05,
                "turnover": 0.50,
                "total_return": 0.06,
                "delta_cagr_vs_forecast_only": -0.03,
                "delta_sharpe_vs_forecast_only": 0.20,
                "delta_max_drawdown_vs_forecast_only": 0.05,
                "delta_turnover_vs_forecast_only": -0.50,
            },
            {
                "run_id": "r2",
                "core_run_id": "small_banks_h05",
                "group_name": "small_banks",
                "horizon": 5,
                "threshold": 0.005,
                "cost_mode": "elevated",
                "sizing_mode": "adaptive",
                "strategy_variant": "forecast_only",
                "cagr": 0.10,
                "sharpe": 0.90,
                "max_drawdown": -0.11,
                "turnover": 1.00,
                "total_return": 0.07,
                "delta_cagr_vs_forecast_only": 0.0,
                "delta_sharpe_vs_forecast_only": 0.0,
                "delta_max_drawdown_vs_forecast_only": 0.0,
                "delta_turnover_vs_forecast_only": 0.0,
            },
            {
                "run_id": "r2",
                "core_run_id": "small_banks_h05",
                "group_name": "small_banks",
                "horizon": 5,
                "threshold": 0.005,
                "cost_mode": "elevated",
                "sizing_mode": "adaptive",
                "strategy_variant": "forecast_plus_risk_and_regime",
                "cagr": 0.08,
                "sharpe": 1.05,
                "max_drawdown": -0.06,
                "turnover": 0.45,
                "total_return": 0.05,
                "delta_cagr_vs_forecast_only": -0.02,
                "delta_sharpe_vs_forecast_only": 0.15,
                "delta_max_drawdown_vs_forecast_only": 0.05,
                "delta_turnover_vs_forecast_only": -0.55,
            },
        ]
    )


def test_phase25_stability_summary_tracks_improvement_shares() -> None:
    summary = build_phase25_stability_summary(_sample_aggregate_results())

    regime_row = summary[summary["strategy_variant"] == "forecast_plus_risk_and_regime"].iloc[0]
    assert regime_row["share_sharpe_improved_vs_forecast_only"] == pytest.approx(1.0)
    assert regime_row["share_drawdown_improved_vs_forecast_only"] == pytest.approx(1.0)
    assert regime_row["share_turnover_reduced_vs_forecast_only"] == pytest.approx(1.0)


def test_cost_sensitivity_summary_compares_to_baseline_cost() -> None:
    summary = build_cost_sensitivity_summary(_sample_aggregate_results())

    elevated_row = summary[
        (summary["cost_mode"] == "elevated")
        & (summary["strategy_variant"] == "forecast_plus_risk_and_regime")
    ].iloc[0]
    assert elevated_row["mean_delta_sharpe_vs_baseline_cost"] == pytest.approx(-0.15)


def test_phase3_readiness_assessment_uses_conditioning_and_cost_signals() -> None:
    aggregate_results = _sample_aggregate_results()
    regime_stability_summary = pd.DataFrame(
        [
            {
                "fallback_observation_share": 0.10,
                "switch_rate": 0.20,
                "average_regime_duration": 4.0,
                "mean_max_probability": 0.75,
            }
        ]
    )
    cost_sensitivity_summary = build_cost_sensitivity_summary(aggregate_results)

    assessment = build_phase3_readiness_assessment(
        aggregate_results,
        regime_stability_summary,
        cost_sensitivity_summary,
    )

    assert assessment["recommendation"] in {"GO Phase 3", "GO Phase 3 with caveats"}
