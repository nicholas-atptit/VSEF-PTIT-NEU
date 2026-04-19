from __future__ import annotations

import json

import pandas as pd

from src.evaluation.consensus import build_model_consensus_summary
from src.reporting.analysis_packets import build_analysis_packets, build_decision_lane_candidates


def _forecast_rows() -> pd.DataFrame:
    common = {
        "timestamp": pd.Timestamp("2024-01-02"),
        "ticker": "AAA",
        "y_true": 0.03,
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
                "y_pred": 0.04,
                "model_family": "boosting",
                "model_role": "primary_research",
                "model_status": "active",
                "research_priority": 10,
            },
            {
                **common,
                "model_name": "xgboost",
                "y_pred": 0.03,
                "model_family": "boosting",
                "model_role": "primary_research",
                "model_status": "active",
                "research_priority": 20,
            },
            {
                **common,
                "model_name": "weighted_ensemble",
                "y_pred": 0.035,
                "model_family": "ensemble",
                "model_role": "ensemble",
                "model_status": "derived",
                "research_priority": 0,
                "component_models": "lightgbm,xgboost",
                "component_count": 2,
            },
        ]
    )


def _signal_rows() -> pd.DataFrame:
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
            {**common, "model_name": "lightgbm", "signal": 1.0, "position_size": 0.5},
            {**common, "model_name": "xgboost", "signal": 1.0, "position_size": 0.5},
            {**common, "model_name": "weighted_ensemble", "signal": 1.0, "position_size": 0.5},
        ]
    )


def _risk_rows() -> pd.DataFrame:
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
                "vol_forecast": 0.03,
                "var_loss_95": 0.02,
                "cvar_loss_95": 0.03,
                "drawdown_state": "normal",
                "current_drawdown": -0.01,
                "max_drawdown": -0.05,
            }
        ]
    )


def _regime_rows() -> pd.DataFrame:
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
                "regime_prob_bull": 1.0,
                "regime_prob_bear": 0.0,
                "regime_prob_sideway": 0.0,
                "source_model": "markov_switching_threshold_fallback",
            }
        ]
    )


def _strategy_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "core_run_id": "g1_h05_forward_return",
                "group_name": "g1",
                "horizon": 5,
                "target_name": "forward_return",
                "target_type": "forward_return",
                "run_mode": "research_core",
                "model_name": "lightgbm",
                "sharpe": 1.2,
                "cagr": 0.15,
                "max_drawdown": -0.08,
                "trade_count": 4,
            },
            {
                "core_run_id": "g1_h05_forward_return",
                "group_name": "g1",
                "horizon": 5,
                "target_name": "forward_return",
                "target_type": "forward_return",
                "run_mode": "research_core",
                "model_name": "xgboost",
                "sharpe": 0.9,
                "cagr": 0.10,
                "max_drawdown": -0.10,
                "trade_count": 4,
            },
        ]
    )


def test_model_consensus_summary_quantifies_agreement() -> None:
    consensus = build_model_consensus_summary(_forecast_rows(), signals_df=_signal_rows())

    assert len(consensus) == 1
    row = consensus.iloc[0]
    assert row["model_count"] == 3
    assert row["agreement_score"] == 1.0
    assert row["agreement_bucket"] == "high"
    assert not row["sign_conflict"]
    assert row["policy_gate_disagreement_share"] == 0.0


def test_analysis_packets_include_expected_fields_and_candidates() -> None:
    consensus = build_model_consensus_summary(_forecast_rows(), signals_df=_signal_rows())
    packets = build_analysis_packets(
        _forecast_rows(),
        consensus,
        risk_df=_risk_rows(),
        regime_df=_regime_rows(),
        signals_df=_signal_rows(),
        positions_df=_signal_rows(),
        strategy_metrics_df=_strategy_rows(),
    )

    assert len(packets) == 1
    packet = packets.iloc[0]
    assert packet["packet_id"] == "AAA|2024-01-02|h05|forward_return|research_core|g1_h05_forward_return"
    assert packet["primary_model_name"] == "lightgbm"
    assert packet["agreement_bucket"] == "high"
    assert packet["volatility_bucket"] == "medium"

    model_records = json.loads(packet["model_by_model_predictions"])
    assert {item["model_name"] for item in model_records} == {"lightgbm", "xgboost", "weighted_ensemble"}
    assert json.loads(packet["regime_summary"])["regime_label"] == "bull"
    assert json.loads(packet["risk_summary"])["risk_model"] == "var_cvar_drawdown_fallback"

    candidates = build_decision_lane_candidates(packets)
    assert len(candidates) == 1
    assert candidates.iloc[0]["ticker"] == "AAA"
    assert candidates.iloc[0]["top_policy_model"] == "lightgbm"
