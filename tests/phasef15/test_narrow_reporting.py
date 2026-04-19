from __future__ import annotations

import pandas as pd

from src.reporting.forecast_rehab_narrow import (
    build_cost_sensitivity_summary,
    build_narrow_assessment,
    build_narrow_feature_performance_summary,
    build_narrow_forecast_vs_policy_summary,
    build_narrow_model_stability_summary,
    build_narrow_target_comparison_summary,
)


def _feature_definition_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_family": "compact_v1",
                "source_family": "reduced_compact",
                "feature_count": 20,
                "requested_feature_count": 20,
                "resolved_feature_count": 20,
                "missing_feature_count": 0,
                "feature_categories": "market_context,technical_indicator",
                "rationale": "compact baseline",
                "features": "a,b",
                "missing_features": "",
            },
            {
                "feature_family": "compact_plus_longlag_v1",
                "source_family": "reduced_compact+long_lag",
                "feature_count": 28,
                "requested_feature_count": 28,
                "resolved_feature_count": 28,
                "missing_feature_count": 0,
                "feature_categories": "market_context,technical_indicator",
                "rationale": "long-lag trend overlay",
                "features": "c,d",
                "missing_features": "",
            },
        ]
    )


def _forecast_quality_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "group_name": "small_banks",
                "horizon": 5,
                "target_name": "forward_return",
                "target_family": "return_regression",
                "feature_family": "compact_v1",
                "model_name": "xgboost",
                "median_rmse": 0.03,
                "median_mae": 0.02,
                "median_directional_accuracy": 0.58,
                "strong_directional_accuracy_share": 0.50,
                "tradable_slice_share": 0.40,
            },
            {
                "group_name": "small_banks",
                "horizon": 10,
                "target_name": "forward_return",
                "target_family": "return_regression",
                "feature_family": "compact_plus_longlag_v1",
                "model_name": "lightgbm",
                "median_rmse": 0.02,
                "median_mae": 0.015,
                "median_directional_accuracy": 0.61,
                "strong_directional_accuracy_share": 0.60,
                "tradable_slice_share": 0.50,
            },
            {
                "group_name": "small_banks",
                "horizon": 10,
                "target_name": "direction_binary",
                "target_family": "direction_classification",
                "feature_family": "compact_plus_longlag_v1",
                "model_name": "lightgbm",
                "median_rmse": 0.95,
                "median_mae": 0.91,
                "median_directional_accuracy": 0.64,
                "strong_directional_accuracy_share": 0.65,
                "tradable_slice_share": 0.55,
            },
        ]
    )


def _model_stability_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "group_name": "small_banks",
                "horizon": 5,
                "target_name": "forward_return",
                "target_family": "return_regression",
                "feature_family": "compact_v1",
                "model_name": "xgboost",
                "directional_accuracy_dispersion": 0.04,
                "rmse_dispersion": 0.01,
                "positive_slice_share": 0.75,
            },
            {
                "group_name": "small_banks",
                "horizon": 10,
                "target_name": "forward_return",
                "target_family": "return_regression",
                "feature_family": "compact_plus_longlag_v1",
                "model_name": "lightgbm",
                "directional_accuracy_dispersion": 0.03,
                "rmse_dispersion": 0.01,
                "positive_slice_share": 0.80,
            },
            {
                "group_name": "small_banks",
                "horizon": 10,
                "target_name": "direction_binary",
                "target_family": "direction_classification",
                "feature_family": "compact_plus_longlag_v1",
                "model_name": "lightgbm",
                "directional_accuracy_dispersion": 0.02,
                "rmse_dispersion": 0.02,
                "positive_slice_share": 0.85,
            },
        ]
    )


def _strategy_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "group_name": "small_banks",
                "horizon": 5,
                "target_name": "forward_return",
                "target_family": "return_regression",
                "feature_family": "compact_v1",
                "model_name": "xgboost",
                "cost_mode": "baseline",
                "cost_label": "baseline",
                "sharpe": 0.4,
                "cagr": 0.05,
                "max_drawdown": -0.05,
                "turnover": 1.0,
                "trade_count": 12,
                "total_return": 0.02,
            },
            {
                "group_name": "small_banks",
                "horizon": 5,
                "target_name": "forward_return",
                "target_family": "return_regression",
                "feature_family": "compact_v1",
                "model_name": "xgboost",
                "cost_mode": "elevated",
                "cost_label": "elevated",
                "sharpe": 0.2,
                "cagr": 0.03,
                "max_drawdown": -0.06,
                "turnover": 1.0,
                "trade_count": 12,
                "total_return": 0.015,
            },
            {
                "group_name": "small_banks",
                "horizon": 10,
                "target_name": "forward_return",
                "target_family": "return_regression",
                "feature_family": "compact_plus_longlag_v1",
                "model_name": "lightgbm",
                "cost_mode": "baseline",
                "cost_label": "baseline",
                "sharpe": 0.8,
                "cagr": 0.10,
                "max_drawdown": -0.04,
                "turnover": 0.8,
                "trade_count": 8,
                "total_return": 0.04,
            },
            {
                "group_name": "small_banks",
                "horizon": 10,
                "target_name": "forward_return",
                "target_family": "return_regression",
                "feature_family": "compact_plus_longlag_v1",
                "model_name": "lightgbm",
                "cost_mode": "elevated",
                "cost_label": "elevated",
                "sharpe": 0.5,
                "cagr": 0.07,
                "max_drawdown": -0.05,
                "turnover": 0.8,
                "trade_count": 8,
                "total_return": 0.03,
            },
        ]
    )


def test_narrow_reporting_summaries_produce_expected_columns() -> None:
    feature_summary = build_narrow_feature_performance_summary(
        _feature_definition_summary(),
        _forecast_quality_summary(),
        _strategy_metrics(),
    )
    model_summary = build_narrow_model_stability_summary(
        _forecast_quality_summary(),
        _model_stability_summary(),
        _strategy_metrics(),
    )
    target_summary = build_narrow_target_comparison_summary(
        _forecast_quality_summary(),
        _strategy_metrics(),
    )
    merged = build_narrow_forecast_vs_policy_summary(
        _forecast_quality_summary(),
        _strategy_metrics(),
    )
    cost_summary = build_cost_sensitivity_summary(_strategy_metrics())

    assert {"median_sharpe_baseline", "positive_sharpe_share_elevated"} <= set(feature_summary.columns)
    assert {"median_policy_sharpe_baseline", "median_directional_accuracy_dispersion"} <= set(model_summary.columns)
    assert {"supports_policy_evaluation", "median_policy_sharpe_baseline"} <= set(target_summary.columns)
    assert {"forecast_rank", "strategy_rank", "edge_but_not_monetized"} <= set(merged.columns)
    assert {"positive_sharpe_share", "median_sharpe"} <= set(cost_summary.columns)


def test_narrow_assessment_uses_cost_and_f1_reference_context() -> None:
    feature_summary = build_narrow_feature_performance_summary(
        _feature_definition_summary(),
        _forecast_quality_summary(),
        _strategy_metrics(),
    )
    model_summary = build_narrow_model_stability_summary(
        _forecast_quality_summary(),
        _model_stability_summary(),
        _strategy_metrics(),
    )
    target_summary = build_narrow_target_comparison_summary(
        _forecast_quality_summary(),
        _strategy_metrics(),
    )
    merged = build_narrow_forecast_vs_policy_summary(
        _forecast_quality_summary(),
        _strategy_metrics(),
    )
    cost_summary = build_cost_sensitivity_summary(_strategy_metrics())
    assessment = build_narrow_assessment(
        feature_summary,
        model_summary,
        target_summary,
        merged,
        cost_summary,
        f1_reference={
            "policy_median_sharpe": -0.5,
            "policy_positive_sharpe_share": 0.25,
        },
    )

    assert assessment["best_feature_family"] == "compact_plus_longlag_v1"
    assert assessment["best_model_family"] == "lightgbm"
    assert assessment["best_target_name"] == "forward_return"
    assert assessment["phase3_blocked"] is True
