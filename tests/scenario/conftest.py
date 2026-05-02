from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.consensus import build_model_consensus_summary
from src.reporting.analysis_packets import build_analysis_packets


def sample_forecasts() -> pd.DataFrame:
    common = {
        "timestamp": pd.Timestamp("2024-01-02"),
        "ticker": "AAA",
        "y_true": 0.04,
        "target_type": "forward_return",
        "horizon": 5,
        "window_id": "window_001",
        "target_timestamp": pd.Timestamp("2024-01-09"),
        "core_run_id": "g1_h05_forward_return",
        "preset": "smoke",
        "group_name": "g1",
        "target_name": "forward_return",
        "target_column": "target_forward_return",
        "target_family": "return_regression",
        "target_tradable": True,
        "ticker_count": 1,
        "ticker_group_members": "AAA",
        "run_mode": "research_core",
    }
    return pd.DataFrame(
        [
            {
                **common,
                "model_name": "lightgbm",
                "y_pred": 0.050,
                "model_family": "boosting",
                "model_role": "primary_research",
                "model_status": "active",
                "research_priority": 10,
            },
            {
                **common,
                "model_name": "xgboost",
                "y_pred": 0.040,
                "model_family": "boosting",
                "model_role": "primary_research",
                "model_status": "active",
                "research_priority": 20,
            },
            {
                **common,
                "model_name": "weighted_ensemble",
                "y_pred": 0.045,
                "model_family": "ensemble",
                "model_role": "ensemble",
                "model_status": "derived",
                "research_priority": 0,
                "component_models": "lightgbm,xgboost",
                "component_count": 2,
            },
        ]
    )


def sample_signals() -> pd.DataFrame:
    common = {
        "timestamp": pd.Timestamp("2024-01-02"),
        "ticker": "AAA",
        "target_type": "forward_return",
        "horizon": 5,
        "window_id": "window_001",
        "core_run_id": "g1_h05_forward_return",
        "run_mode": "research_core",
    }
    return pd.DataFrame(
        [
            {**common, "model_name": "lightgbm", "signal": 1.0, "position_size": 0.4},
            {**common, "model_name": "xgboost", "signal": 1.0, "position_size": 0.4},
            {**common, "model_name": "weighted_ensemble", "signal": 1.0, "position_size": 0.4},
        ]
    )


def sample_risk() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-01-02"),
                "ticker": "AAA",
                "horizon": 5,
                "target_type": "forward_return",
                "window_id": "window_001",
                "core_run_id": "g1_h05_forward_return",
                "run_mode": "research_core",
                "risk_model": "var_cvar_drawdown_fallback",
                "vol_forecast": 0.025,
                "var_loss_95": 0.015,
                "cvar_loss_95": 0.025,
                "drawdown_state": "normal",
                "current_drawdown": -0.01,
                "max_drawdown": -0.04,
            }
        ]
    )


def sample_regime() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-01-02"),
                "ticker": "AAA",
                "horizon": 5,
                "target_type": "forward_return",
                "window_id": "window_001",
                "core_run_id": "g1_h05_forward_return",
                "run_mode": "research_core",
                "regime_label": "bull",
                "regime_prob_bull": 0.85,
                "regime_prob_bear": 0.05,
                "regime_prob_sideway": 0.10,
                "source_model": "markov_switching_threshold_fallback",
            }
        ]
    )


def sample_strategy() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "core_run_id": "g1_h05_forward_return",
                "preset": "smoke",
                "group_name": "g1",
                "horizon": 5,
                "target_name": "forward_return",
                "target_type": "forward_return",
                "target_column": "target_forward_return",
                "target_family": "return_regression",
                "target_tradable": True,
                "ticker_count": 1,
                "ticker_group_members": "AAA",
                "run_mode": "research_core",
                "model_name": "lightgbm",
                "sharpe": 1.1,
                "cagr": 0.12,
                "max_drawdown": -0.06,
                "trade_count": 4,
            }
        ]
    )


def sample_health() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model_name": "lightgbm", "health_status": "healthy", "run_success_rate": 1.0},
            {"model_name": "xgboost", "health_status": "healthy", "run_success_rate": 1.0},
            {"model_name": "weighted_ensemble", "health_status": "healthy", "run_success_rate": 1.0},
        ]
    )


@pytest.fixture
def scenario_inputs() -> dict[str, pd.DataFrame]:
    forecasts = sample_forecasts()
    signals = sample_signals()
    consensus = build_model_consensus_summary(forecasts, signals_df=signals)
    packets = build_analysis_packets(
        forecasts,
        consensus,
        risk_df=sample_risk(),
        regime_df=sample_regime(),
        signals_df=signals,
        positions_df=signals,
        strategy_metrics_df=sample_strategy(),
    )
    return {
        "forecasts": forecasts,
        "signals": signals,
        "consensus": consensus,
        "risk": sample_risk(),
        "regime": sample_regime(),
        "strategy": sample_strategy(),
        "health": sample_health(),
        "packets": packets,
    }
