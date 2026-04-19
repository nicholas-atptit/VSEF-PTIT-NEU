from __future__ import annotations

import pandas as pd

from src.reporting.forecast_rehab import (
    build_feature_ablation_summary,
    build_feature_inventory_summary,
    build_forecast_quality_summary,
    build_forecast_rehab_assessment,
    build_forecast_vs_policy_summary,
    build_model_stability_summary,
)


def test_forecast_quality_and_stability_summaries_produce_expected_columns() -> None:
    slice_summary = pd.DataFrame(
        [
            {
                "group_name": "small_banks",
                "horizon": 5,
                "target_name": "forward_return",
                "target_family": "return_regression",
                "feature_family": "current_full",
                "model_name": "naive",
                "ticker": "AAA",
                "window_id": "window_001",
                "mae": 0.020,
                "rmse": 0.030,
                "directional_accuracy": 0.51,
            },
            {
                "group_name": "small_banks",
                "horizon": 5,
                "target_name": "forward_return",
                "target_family": "return_regression",
                "feature_family": "current_full",
                "model_name": "ridge",
                "ticker": "AAA",
                "window_id": "window_001",
                "mae": 0.015,
                "rmse": 0.020,
                "directional_accuracy": 0.58,
            },
            {
                "group_name": "small_banks",
                "horizon": 5,
                "target_name": "forward_return",
                "target_family": "return_regression",
                "feature_family": "current_full",
                "model_name": "ridge",
                "ticker": "BBB",
                "window_id": "window_002",
                "mae": 0.018,
                "rmse": 0.025,
                "directional_accuracy": 0.56,
            },
        ]
    )

    forecast_quality = build_forecast_quality_summary(slice_summary)
    stability = build_model_stability_summary(slice_summary)
    ablation = build_feature_ablation_summary(forecast_quality)

    assert {"tradable_slice_share", "beats_naive_rmse_share", "strong_directional_accuracy_share"} <= set(forecast_quality.columns)
    assert {"rmse_dispersion", "directional_accuracy_dispersion", "strong_slice_share"} <= set(stability.columns)
    assert {"best_rmse_model", "best_directional_model"} <= set(ablation.columns)


def test_forecast_vs_policy_summary_and_assessment_join_consistently() -> None:
    forecast_quality = pd.DataFrame(
        [
            {
                "group_name": "small_banks",
                "horizon": 5,
                "target_name": "forward_return",
                "target_family": "return_regression",
                "feature_family": "current_full",
                "model_name": "naive",
                "mean_rmse": 0.030,
                "mean_mae": 0.020,
                "mean_directional_accuracy": 0.51,
                "tradable_slice_share": 0.00,
                "strong_directional_accuracy_share": 0.00,
            },
            {
                "group_name": "small_banks",
                "horizon": 5,
                "target_name": "forward_return",
                "target_family": "return_regression",
                "feature_family": "current_full",
                "model_name": "ridge",
                "mean_rmse": 0.020,
                "mean_mae": 0.015,
                "mean_directional_accuracy": 0.58,
                "tradable_slice_share": 0.50,
                "strong_directional_accuracy_share": 0.50,
            },
        ]
    )
    strategy_metrics = pd.DataFrame(
        [
            {
                "group_name": "small_banks",
                "horizon": 5,
                "target_name": "forward_return",
                "feature_family": "current_full",
                "model_name": "naive",
                "cagr": -0.10,
                "sharpe": -1.0,
                "max_drawdown": -0.20,
                "turnover": 4.0,
                "trade_count": 10,
                "total_return": -0.05,
            },
            {
                "group_name": "small_banks",
                "horizon": 5,
                "target_name": "forward_return",
                "feature_family": "current_full",
                "model_name": "ridge",
                "cagr": 0.02,
                "sharpe": 0.3,
                "max_drawdown": -0.08,
                "turnover": 2.0,
                "trade_count": 8,
                "total_return": 0.01,
            },
        ]
    )
    feature_ablation = pd.DataFrame(
        [
            {
                "group_name": "small_banks",
                "horizon": 5,
                "target_name": "forward_return",
                "feature_family": "current_full",
                "mean_rmse": 0.025,
                "mean_directional_accuracy": 0.545,
                "strong_slice_share": 0.25,
            }
        ]
    )
    stability = pd.DataFrame(
        [
            {
                "group_name": "small_banks",
                "horizon": 5,
                "target_name": "forward_return",
                "target_family": "return_regression",
                "feature_family": "current_full",
                "model_name": "ridge",
                "directional_accuracy_dispersion": 0.02,
                "strong_slice_share": 0.50,
            }
        ]
    )

    merged = build_forecast_vs_policy_summary(forecast_quality, strategy_metrics)
    assessment = build_forecast_rehab_assessment(feature_ablation, forecast_quality, stability, merged)

    assert {"forecast_rank", "strategy_rank", "edge_but_not_monetized"} <= set(merged.columns)
    assert assessment["recommendation"] in {
        "continue forecast rehab with narrowed feature/model scope",
        "freeze weak model families and focus on best few",
        "shift from regression emphasis to direction emphasis",
        "reduce ticker universe to more favorable groups",
        "stop expansion and reconsider whether this repo has enough edge at daily frequency",
    }


def test_forecast_rehab_assessment_uses_median_group_columns_when_present() -> None:
    forecast_quality = pd.DataFrame(
        [
            {
                "group_name": "small_banks",
                "horizon": 10,
                "target_name": "direction_binary",
                "target_family": "direction_classification",
                "feature_family": "long_lag",
                "model_name": "lightgbm",
                "mean_rmse": 0.030,
                "median_rmse": 0.029,
                "mean_mae": 0.020,
                "mean_directional_accuracy": 0.54,
                "median_directional_accuracy": 0.55,
                "tradable_slice_share": 0.30,
                "strong_directional_accuracy_share": 0.40,
            },
            {
                "group_name": "mixed_large_cap",
                "horizon": 10,
                "target_name": "direction_binary",
                "target_family": "direction_classification",
                "feature_family": "long_lag",
                "model_name": "lightgbm",
                "mean_rmse": 0.031,
                "median_rmse": 0.030,
                "mean_mae": 0.021,
                "mean_directional_accuracy": 0.50,
                "median_directional_accuracy": 0.50,
                "tradable_slice_share": 0.20,
                "strong_directional_accuracy_share": 0.25,
            },
        ]
    )
    feature_ablation = pd.DataFrame(
        [
            {
                "group_name": "small_banks",
                "horizon": 10,
                "target_name": "direction_binary",
                "feature_family": "long_lag",
                "mean_rmse": 0.031,
                "median_rmse": 0.030,
                "mean_directional_accuracy": 0.53,
                "median_directional_accuracy": 0.54,
                "strong_slice_share": 0.35,
            }
        ]
    )
    stability = pd.DataFrame(
        [
            {
                "group_name": "small_banks",
                "horizon": 10,
                "target_name": "direction_binary",
                "target_family": "direction_classification",
                "feature_family": "long_lag",
                "model_name": "lightgbm",
                "directional_accuracy_dispersion": 0.03,
                "strong_slice_share": 0.40,
            }
        ]
    )
    strategy_metrics = pd.DataFrame(
        [
            {
                "group_name": "small_banks",
                "horizon": 10,
                "target_name": "direction_binary",
                "feature_family": "long_lag",
                "model_name": "lightgbm",
                "cagr": 0.01,
                "sharpe": 0.2,
                "max_drawdown": -0.06,
                "turnover": 1.5,
                "trade_count": 4,
                "total_return": 0.01,
            }
        ]
    )

    merged = build_forecast_vs_policy_summary(forecast_quality, strategy_metrics)
    assessment = build_forecast_rehab_assessment(feature_ablation, forecast_quality, stability, merged)

    assert assessment["best_group_name"] == "small_banks"
    assert assessment["best_target_name"] == "direction_binary"


def test_feature_inventory_summary_groups_registry_flags() -> None:
    inventory = pd.DataFrame(
        [
            {
                "feature_name": "bb_width",
                "category": "technical_indicator",
                "input_source": "ohlcv",
                "status": "active",
                "is_current_regression_baseline": True,
                "is_current_direction_baseline": False,
                "is_reduced_compact": True,
                "leakage_risk_note": "Computed from current and past rows only.",
            },
            {
                "feature_name": "market_return_60d",
                "category": "market_context",
                "input_source": "market/sector/breadth context",
                "status": "active",
                "is_current_regression_baseline": True,
                "is_current_direction_baseline": False,
                "is_reduced_compact": False,
                "leakage_risk_note": "Safe only if context inputs are aligned on date and never forward-filled from the future.",
            },
        ]
    )

    summary = build_feature_inventory_summary(inventory)
    assert {"feature_group", "suspected_usefulness", "suspected_risk"} <= set(summary.columns)
    assert set(summary["feature_group"]) == {"technical_indicator", "market_context"}
