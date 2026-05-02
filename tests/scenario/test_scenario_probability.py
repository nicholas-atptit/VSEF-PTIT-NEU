from __future__ import annotations

import pandas as pd
import pytest

from src.scenario.probability import build_scenario_probability_frame
from src.scenario.schema import SCENARIO_LABELS


def test_scenario_probabilities_sum_to_one(scenario_inputs: dict[str, pd.DataFrame]) -> None:
    probabilities = build_scenario_probability_frame(
        scenario_inputs["forecasts"],
        consensus_df=scenario_inputs["consensus"],
        risk_df=scenario_inputs["risk"],
        regime_df=scenario_inputs["regime"],
        strategy_metrics_df=scenario_inputs["strategy"],
        analysis_packets_df=scenario_inputs["packets"],
        model_health_df=scenario_inputs["health"],
    )

    assert set(probabilities["scenario_label"]) == set(SCENARIO_LABELS)
    grouped = probabilities.groupby(["ticker", "timestamp", "horizon", "target_type", "run_mode", "core_run_id"])[
        "scenario_probability"
    ].sum()
    assert grouped.tolist() == pytest.approx([1.0])


def test_missing_risk_and_regime_data_does_not_crash(scenario_inputs: dict[str, pd.DataFrame]) -> None:
    probabilities = build_scenario_probability_frame(
        scenario_inputs["forecasts"],
        consensus_df=scenario_inputs["consensus"],
        risk_df=pd.DataFrame(),
        regime_df=pd.DataFrame(),
        strategy_metrics_df=pd.DataFrame(),
        analysis_packets_df=scenario_inputs["packets"],
        model_health_df=pd.DataFrame(),
    )

    assert len(probabilities) == len(SCENARIO_LABELS)
    assert probabilities["scenario_probability"].sum() == pytest.approx(1.0)
    assert probabilities["missing_context_share"].max() > 0.0
