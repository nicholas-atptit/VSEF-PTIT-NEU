from __future__ import annotations

import pandas as pd

from src.reporting.model_health import build_model_health_summary


def test_model_health_summary_aggregates_failures_and_weak_slices() -> None:
    execution = pd.DataFrame(
        [
            {
                "model_name": "lightgbm",
                "model_family": "boosting",
                "model_role": "primary_research",
                "model_status": "active",
                "core_run_id": "g1_h05_forward_return",
                "run_success": True,
                "warning_count": 0,
                "missing_output_count": 0,
                "failure_reason": "",
            },
            {
                "model_name": "lightgbm",
                "model_family": "boosting",
                "model_role": "primary_research",
                "model_status": "active",
                "core_run_id": "g2_h05_forward_return",
                "run_success": False,
                "warning_count": 1,
                "missing_output_count": 1,
                "failure_reason": "statsmodels_missing",
            },
        ]
    )
    forecasts = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-01-01"),
                "ticker": "AAA",
                "y_true": 0.02,
                "y_pred": -0.01,
                "model_name": "lightgbm",
                "target_type": "forward_return",
                "horizon": 5,
                "window_id": "window_001",
                "core_run_id": "g1_h05_forward_return",
                "group_name": "g1",
                "target_name": "forward_return",
                "target_column": "target_forward_return",
                "target_family": "return_regression",
                "target_tradable": True,
                "ticker_count": 1,
                "ticker_group_members": "AAA",
                "run_mode": "research_core",
                "model_family": "boosting",
                "model_role": "primary_research",
                "model_status": "active",
            },
            {
                "timestamp": pd.Timestamp("2024-01-02"),
                "ticker": "AAA",
                "y_true": 0.01,
                "y_pred": -0.02,
                "model_name": "lightgbm",
                "target_type": "forward_return",
                "horizon": 5,
                "window_id": "window_002",
                "core_run_id": "g1_h05_forward_return",
                "group_name": "g1",
                "target_name": "forward_return",
                "target_column": "target_forward_return",
                "target_family": "return_regression",
                "target_tradable": True,
                "ticker_count": 1,
                "ticker_group_members": "AAA",
                "run_mode": "research_core",
                "model_family": "boosting",
                "model_role": "primary_research",
                "model_status": "active",
            },
        ]
    )
    strategy = pd.DataFrame(
        [
            {"model_name": "lightgbm", "sharpe": -0.2, "cagr": -0.05, "max_drawdown": -0.20},
        ]
    )

    summary = build_model_health_summary(execution, forecasts, strategy)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["run_success_count"] == 1
    assert row["run_failure_count"] == 1
    assert row["warning_count_total"] == 1
    assert row["missing_output_count_total"] == 1
    assert row["failure_reasons"] == "statsmodels_missing"
    assert row["positive_slice_frequency"] == 0.0
    assert row["health_status"] == "weak"
