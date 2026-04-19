from __future__ import annotations

import pandas as pd
import pytest

from src.reporting.calibration import (
    build_forecast_vs_policy_summary,
    build_phase26_assessment,
    build_policy_ablation_summary,
    build_policy_cost_sensitivity_summary,
    build_policy_run_summary,
    build_regime_value_summary,
    build_sizing_calibration_summary,
    build_threshold_calibration_summary,
)


def _aggregate_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": "r1",
                "core_run_id": "banks_h05",
                "group_name": "small_banks",
                "horizon": 5,
                "threshold": 0.005,
                "cost_mode": "baseline",
                "policy_variant": "fixed_threshold_fixed_fraction",
                "policy_label": "Fixed threshold + fixed fraction",
                "policy_family": "simple_baseline",
                "strategy_variant": "forecast_only",
                "sizing_profile": "fixed_fraction_full",
                "sizing_label": "Fixed fraction 1.0",
                "cagr": 0.10,
                "sharpe": 1.00,
                "sortino": 1.20,
                "max_drawdown": -0.10,
                "calmar": 1.00,
                "turnover": 1.20,
                "win_rate": 0.55,
                "total_return": 0.08,
                "trade_count": 12,
            },
            {
                "run_id": "r1",
                "core_run_id": "banks_h05",
                "group_name": "small_banks",
                "horizon": 5,
                "threshold": 0.005,
                "cost_mode": "baseline",
                "policy_variant": "risk_only_no_regime",
                "policy_label": "Risk-only adaptive stack",
                "policy_family": "risk_stack",
                "strategy_variant": "forecast_plus_risk",
                "sizing_profile": "adaptive_current",
                "sizing_label": "Adaptive current",
                "cagr": 0.08,
                "sharpe": 1.10,
                "sortino": 1.30,
                "max_drawdown": -0.06,
                "calmar": 1.33,
                "turnover": 0.70,
                "win_rate": 0.58,
                "total_return": 0.07,
                "trade_count": 8,
            },
            {
                "run_id": "r1",
                "core_run_id": "banks_h05",
                "group_name": "small_banks",
                "horizon": 5,
                "threshold": 0.005,
                "cost_mode": "baseline",
                "policy_variant": "regime_threshold_adaptive_drawdown",
                "policy_label": "Regime threshold + risk stack",
                "policy_family": "regime_risk_stack",
                "strategy_variant": "forecast_plus_risk_and_regime",
                "sizing_profile": "adaptive_current",
                "sizing_label": "Adaptive current",
                "cagr": 0.07,
                "sharpe": 1.20,
                "sortino": 1.45,
                "max_drawdown": -0.05,
                "calmar": 1.40,
                "turnover": 0.55,
                "win_rate": 0.59,
                "total_return": 0.06,
                "trade_count": 7,
            },
            {
                "run_id": "r2",
                "core_run_id": "banks_h05",
                "group_name": "small_banks",
                "horizon": 5,
                "threshold": 0.007,
                "cost_mode": "elevated",
                "policy_variant": "fixed_threshold_fixed_fraction",
                "policy_label": "Fixed threshold + fixed fraction",
                "policy_family": "simple_baseline",
                "strategy_variant": "forecast_only",
                "sizing_profile": "fixed_fraction_full",
                "sizing_label": "Fixed fraction 1.0",
                "cagr": 0.06,
                "sharpe": 0.75,
                "sortino": 0.90,
                "max_drawdown": -0.12,
                "calmar": 0.50,
                "turnover": 1.10,
                "win_rate": 0.52,
                "total_return": 0.04,
                "trade_count": 10,
            },
            {
                "run_id": "r2",
                "core_run_id": "banks_h05",
                "group_name": "small_banks",
                "horizon": 5,
                "threshold": 0.007,
                "cost_mode": "elevated",
                "policy_variant": "risk_only_no_regime",
                "policy_label": "Risk-only adaptive stack",
                "policy_family": "risk_stack",
                "strategy_variant": "forecast_plus_risk",
                "sizing_profile": "adaptive_current",
                "sizing_label": "Adaptive current",
                "cagr": 0.05,
                "sharpe": 0.82,
                "sortino": 1.00,
                "max_drawdown": -0.08,
                "calmar": 0.63,
                "turnover": 0.65,
                "win_rate": 0.54,
                "total_return": 0.03,
                "trade_count": 7,
            },
            {
                "run_id": "r2",
                "core_run_id": "banks_h05",
                "group_name": "small_banks",
                "horizon": 5,
                "threshold": 0.007,
                "cost_mode": "elevated",
                "policy_variant": "regime_threshold_adaptive_drawdown",
                "policy_label": "Regime threshold + risk stack",
                "policy_family": "regime_risk_stack",
                "strategy_variant": "forecast_plus_risk_and_regime",
                "sizing_profile": "adaptive_current",
                "sizing_label": "Adaptive current",
                "cagr": 0.04,
                "sharpe": 0.88,
                "sortino": 1.05,
                "max_drawdown": -0.07,
                "calmar": 0.57,
                "turnover": 0.50,
                "win_rate": 0.55,
                "total_return": 0.03,
                "trade_count": 6,
            },
        ]
    )


def _positions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policy_variant": "risk_only_no_regime",
                "sizing_profile": "adaptive_current",
                "signal": 1.0,
                "position_size": 0.30,
                "size_multiplier": 0.30,
                "vol_forecast": 0.08,
                "drawdown_haircut_strength": 1.0,
                "volatility_target_scale": 1.0,
            },
            {
                "policy_variant": "risk_only_no_regime",
                "sizing_profile": "adaptive_current",
                "signal": 1.0,
                "position_size": 0.25,
                "size_multiplier": 0.25,
                "vol_forecast": 0.09,
                "drawdown_haircut_strength": 1.0,
                "volatility_target_scale": 1.0,
            },
            {
                "policy_variant": "risk_only_no_regime",
                "sizing_profile": "adaptive_capped_floor",
                "signal": 1.0,
                "position_size": 0.40,
                "size_multiplier": 0.40,
                "vol_forecast": 0.08,
                "drawdown_haircut_strength": 1.0,
                "volatility_target_scale": 1.0,
            },
            {
                "policy_variant": "regime_threshold_adaptive_drawdown",
                "sizing_profile": "adaptive_current",
                "signal": 1.0,
                "position_size": 0.20,
                "size_multiplier": 0.20,
                "vol_forecast": 0.08,
                "drawdown_haircut_strength": 1.0,
                "volatility_target_scale": 1.0,
            },
        ]
    )


def _forecast_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "group_name": "small_banks",
                "horizon": 5,
                "model_name": "ridge",
                "rmse": 0.020,
                "directional_accuracy": 0.58,
                "hit_rate": 0.58,
            },
            {
                "group_name": "small_banks",
                "horizon": 5,
                "model_name": "random_forest",
                "rmse": 0.030,
                "directional_accuracy": 0.52,
                "hit_rate": 0.52,
            },
        ]
    )


def _model_strategy_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "group_name": "small_banks",
                "horizon": 5,
                "policy_variant": "fixed_threshold_fixed_fraction",
                "sizing_profile": "fixed_fraction_full",
                "strategy_variant": "forecast_only",
                "model_name": "ridge",
                "sharpe": 0.40,
                "cagr": 0.05,
                "max_drawdown": -0.10,
                "turnover": 1.2,
                "trade_count": 10,
                "total_return": 0.03,
            },
            {
                "group_name": "small_banks",
                "horizon": 5,
                "policy_variant": "fixed_threshold_fixed_fraction",
                "sizing_profile": "fixed_fraction_full",
                "strategy_variant": "forecast_only",
                "model_name": "random_forest",
                "sharpe": 0.90,
                "cagr": 0.10,
                "max_drawdown": -0.08,
                "turnover": 1.1,
                "trade_count": 9,
                "total_return": 0.05,
            },
        ]
    )


def test_policy_run_summary_and_ablation_summary_compute_baseline_comparisons() -> None:
    strategy_metrics = _model_strategy_metrics().assign(
        policy_label="Fixed threshold + fixed fraction",
        policy_family="simple_baseline",
        sizing_label="Fixed fraction 1.0",
    )
    run_summary = build_policy_run_summary(strategy_metrics)
    assert run_summary.loc[0, "model_count"] == 2

    ablation_summary = build_policy_ablation_summary(_aggregate_results())
    risk_row = ablation_summary[
        (ablation_summary["policy_variant"] == "risk_only_no_regime")
        & (ablation_summary["sizing_profile"] == "adaptive_current")
    ].iloc[0]
    assert risk_row["share_sharpe_improved_vs_simple_baseline"] == pytest.approx(1.0)
    assert risk_row["share_drawdown_improved_vs_simple_baseline"] == pytest.approx(1.0)


def test_threshold_sizing_regime_and_forecast_policy_summaries_are_populated() -> None:
    aggregate_results = _aggregate_results()

    threshold_summary = build_threshold_calibration_summary(aggregate_results)
    sizing_summary = build_sizing_calibration_summary(aggregate_results, _positions())
    regime_summary = build_regime_value_summary(aggregate_results)
    forecast_vs_policy = build_forecast_vs_policy_summary(_forecast_summary(), _model_strategy_metrics())
    cost_summary = build_policy_cost_sensitivity_summary(aggregate_results)

    assert set(threshold_summary["threshold"]) == {0.005, 0.007}
    assert "avg_size_multiplier" in sizing_summary.columns
    assert not regime_summary.empty
    overall_regime = regime_summary[
        (regime_summary["comparison_name"] == "regime_plus_risk_vs_risk_only")
        & (regime_summary["segment_type"] == "overall")
    ].iloc[0]
    assert overall_regime["share_sharpe_improved"] == pytest.approx(1.0)
    assert "edge_but_not_monetized" in forecast_vs_policy.columns
    assert "median_delta_sharpe_vs_baseline_cost" in cost_summary.columns

    assessment = build_phase26_assessment(
        build_policy_ablation_summary(aggregate_results),
        regime_summary,
        forecast_vs_policy,
        cost_summary,
    )
    assert assessment["default_policy_candidate"] is not None
    assert assessment["phase3_blocked"] is True
