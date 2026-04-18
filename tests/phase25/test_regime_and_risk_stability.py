from __future__ import annotations

import pandas as pd
import pytest

from src.reporting.hardening import build_regime_stability_summary, build_risk_stability_summary
from src.strategy.sizing import size_positions


def test_regime_stability_summary_tracks_switches_and_fallback_share() -> None:
    regime = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02",
                "ticker": "AAA",
                "window_id": "w1",
                "regime_label": "bull",
                "regime_prob_bull": 0.8,
                "regime_prob_bear": 0.1,
                "regime_prob_sideway": 0.1,
                "source_model": "markov_switching",
            },
            {
                "timestamp": "2024-01-03",
                "ticker": "AAA",
                "window_id": "w1",
                "regime_label": "bear",
                "regime_prob_bull": 0.1,
                "regime_prob_bear": 0.8,
                "regime_prob_sideway": 0.1,
                "source_model": "markov_switching_threshold_fallback",
            },
            {
                "timestamp": "2024-01-04",
                "ticker": "AAA",
                "window_id": "w1",
                "regime_label": "bear",
                "regime_prob_bull": 0.1,
                "regime_prob_bear": 0.7,
                "regime_prob_sideway": 0.2,
                "source_model": "markov_switching_threshold_fallback",
            },
        ]
    )

    summary = build_regime_stability_summary(regime)

    row = summary.iloc[0]
    assert row["switch_count"] == 1
    assert row["fallback_observation_share"] == pytest.approx(2.0 / 3.0)
    assert row["average_regime_duration"] == pytest.approx(1.5)


def test_fixed_fraction_sizing_and_risk_summary_capture_exposure_reduction() -> None:
    signals = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02",
                "ticker": "AAA",
                "model_name": "ridge",
                "signal": 1.0,
                "threshold": 0.01,
                "y_pred": 0.02,
                "target_type": "forward_return",
                "horizon": 5,
                "window_id": "w1",
                "regime_label": "bull",
                "vol_forecast": 0.01,
                "drawdown_state": "normal",
            },
            {
                "timestamp": "2024-01-03",
                "ticker": "AAA",
                "model_name": "ridge",
                "signal": 1.0,
                "threshold": 0.01,
                "y_pred": 0.02,
                "target_type": "forward_return",
                "horizon": 5,
                "window_id": "w1",
                "regime_label": "bear",
                "vol_forecast": 0.05,
                "drawdown_state": "severe",
            },
        ]
    )

    adaptive_positions = size_positions(signals, capital_config={"max_position_size": 1.0})
    adaptive_positions["run_id"] = "adaptive_run"
    adaptive_positions["core_run_id"] = "core_1"
    adaptive_positions["group_name"] = "small_banks"
    adaptive_positions["threshold"] = 0.01
    adaptive_positions["cost_mode"] = "baseline"
    adaptive_positions["strategy_variant"] = "forecast_plus_risk"
    adaptive_positions["configured_max_position_size"] = 1.0

    fixed_positions = size_positions(
        signals,
        capital_config={"max_position_size": 1.0, "sizing_mode": "fixed_fraction", "fixed_position_size": 1.0},
    )
    assert fixed_positions["position_size"].eq(1.0).all()

    summary = build_risk_stability_summary(adaptive_positions)

    row = summary.iloc[0]
    assert adaptive_positions.loc[0, "position_size"] > adaptive_positions.loc[1, "position_size"]
    assert row["exposure_reduction_share"] == pytest.approx(0.5)
