from __future__ import annotations

import pandas as pd
import pytest

from src.reporting.summary import build_conditioning_mode_summary, build_phase2_conditioning_summary


def test_conditioning_mode_summary_computes_deltas_vs_baseline() -> None:
    strategy_metrics = pd.DataFrame(
        [
            {"strategy_variant": "forecast_only", "model_name": "linear", "cagr": 0.10, "sharpe": 1.0, "max_drawdown": -0.10, "turnover": 1.0, "win_rate": 0.50, "total_return": 0.05},
            {"strategy_variant": "forecast_plus_risk", "model_name": "linear", "cagr": 0.08, "sharpe": 1.2, "max_drawdown": -0.07, "turnover": 0.8, "win_rate": 0.55, "total_return": 0.04},
        ]
    )

    summary = build_conditioning_mode_summary(strategy_metrics)

    risk_row = summary[summary["strategy_variant"] == "forecast_plus_risk"].iloc[0]
    assert risk_row["delta_sharpe_vs_forecast_only"] == pytest.approx(0.2)
    assert risk_row["delta_max_drawdown_vs_forecast_only"] == pytest.approx(0.03)


def test_phase2_conditioning_summary_preserves_variant_rows() -> None:
    forecast_summary = pd.DataFrame(
        [
            {"model_name": "linear", "ticker": "AAA", "horizon": 5, "observations": 10, "mae": 0.01, "rmse": 0.02, "mape": 10.0, "smape": 12.0, "directional_accuracy": 0.6, "hit_rate": 0.6},
        ]
    )
    strategy_metrics = pd.DataFrame(
        [
            {"strategy_variant": "forecast_only", "model_name": "linear", "cagr": 0.10, "sharpe": 1.0, "sortino": 1.2, "max_drawdown": -0.10, "turnover": 1.0, "win_rate": 0.50, "total_return": 0.05},
            {"strategy_variant": "forecast_plus_risk_and_regime", "model_name": "linear", "cagr": 0.09, "sharpe": 1.3, "sortino": 1.5, "max_drawdown": -0.06, "turnover": 0.7, "win_rate": 0.55, "total_return": 0.045},
        ]
    )

    comparison = build_phase2_conditioning_summary(forecast_summary, strategy_metrics)

    assert len(comparison) == 2
    assert "delta_sharpe_vs_forecast_only" in comparison.columns
    assert set(comparison["strategy_variant"]) == {"forecast_only", "forecast_plus_risk_and_regime"}
