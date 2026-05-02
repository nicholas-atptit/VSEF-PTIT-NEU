from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.scenario import ScenarioEngineConfig, run_scenario_evaluation, write_scenario_outputs
from src.scenario.schema import SCENARIO_ARTIFACT_FILENAMES, SCENARIO_REQUIRED_FIELDS


def _run_engine(scenario_inputs: dict[str, pd.DataFrame]):
    return run_scenario_evaluation(
        forecasts_df=scenario_inputs["forecasts"],
        consensus_df=scenario_inputs["consensus"],
        risk_df=scenario_inputs["risk"],
        regime_df=scenario_inputs["regime"],
        strategy_metrics_df=scenario_inputs["strategy"],
        analysis_packets_df=scenario_inputs["packets"],
        model_health_df=scenario_inputs["health"],
        config=ScenarioEngineConfig(calibration_lookback=None),
    )


def test_all_scenario_artifacts_are_written(tmp_path: Path, scenario_inputs: dict[str, pd.DataFrame]) -> None:
    result = _run_engine(scenario_inputs)
    paths = write_scenario_outputs(tmp_path, result)

    for artifact_name, filename in SCENARIO_ARTIFACT_FILENAMES.items():
        assert artifact_name in paths
        assert (tmp_path / filename).exists()

    probability = pd.read_csv(tmp_path / "scenario_probability.csv")
    assert set(SCENARIO_REQUIRED_FIELDS).issubset(probability.columns)
    manifest = json.loads((tmp_path / "scenario_manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_type"] == "scenario_evaluation_engine_v1_manifest"
    assert manifest["dominance_authority"] == "diagnostic_only_no_buy_sell_recommendation"


def test_analysis_packets_are_enriched_with_scenario_fields(scenario_inputs: dict[str, pd.DataFrame]) -> None:
    result = _run_engine(scenario_inputs)
    packet = result.analysis_packets.iloc[0]

    assert packet["dominant_scenario"] in {"bull", "bear", "sideway", "high_volatility", "drawdown", "recovery", "uncertain"}
    assert packet["dominant_scenario_probability"] >= 0.0
    assert packet["scenario_uncertainty_score"] >= 0.0
    assert packet["scenario_dominance_score"] >= 0.0
    assert packet["scenario_confidence_bucket"] in {"high", "medium", "low", "uncalibrated", "risk_overridden"}
    assert json.loads(packet["scenario_summary"])
    assert isinstance(json.loads(packet["alternative_scenarios"]), list)


def test_scenario_engine_is_deterministic_for_identical_inputs(scenario_inputs: dict[str, pd.DataFrame]) -> None:
    first = _run_engine(scenario_inputs)
    second = _run_engine(scenario_inputs)

    pd.testing.assert_frame_equal(first.scenario_probability, second.scenario_probability)
    pd.testing.assert_frame_equal(first.scenario_rankings, second.scenario_rankings)
    pd.testing.assert_frame_equal(first.scenario_dominance_summary, second.scenario_dominance_summary)
    pd.testing.assert_frame_equal(first.scenario_uncertainty_summary, second.scenario_uncertainty_summary)
    pd.testing.assert_frame_equal(first.scenario_calibration_summary, second.scenario_calibration_summary)
    pd.testing.assert_frame_equal(first.analysis_packets, second.analysis_packets)
