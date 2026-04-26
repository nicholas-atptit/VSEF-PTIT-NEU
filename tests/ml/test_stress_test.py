from __future__ import annotations

import json

import pytest
import pandas as pd

from src.ml.benchmark.stress_test import StressTestRunner
from src.ml.data_loader import generate_mock_data
from src.ml.trainer import DualModelTrainer, HORIZON_DAYS


def _write_csv(tmp_path, ticker: str = "STRESS", num_days: int = 900) -> list:
    df = generate_mock_data(ticker=ticker, num_days=num_days)
    csv_path = tmp_path / f"{ticker}.csv"
    df.to_csv(csv_path, index=False)
    return [csv_path]


def test_stress_test_runner_generates_all_scenarios_and_outputs(tmp_path) -> None:
    files = _write_csv(tmp_path)
    runner = StressTestRunner(model_root=tmp_path / "models")
    result = runner.run(
        files=files,
        algorithms=["cart"],
        output_root=tmp_path / "models",
        report_path=tmp_path / "reports" / "stress_test.csv",
        primary_algorithm="cart",
        horizons=["short"],
        max_depth=3,
    )

    detail = result["detail"]
    summary = result["summary"]
    expected_scenarios = {
        "volatility_shock",
        "drawdown_shock",
        "liquidity_cost_shock",
        "regime_persistence_shock",
    }
    assert set(detail["stress_scenario"]) == expected_scenarios
    assert set(summary["stress_scenario"]) == expected_scenarios
    assert result["detail_path"].exists()
    assert result["summary_path"].exists()
    assert result["json_path"].exists()
    assert result["markdown_path"].exists()
    payload = json.loads(result["json_path"].read_text(encoding="utf-8"))
    assert len(payload["detail_rows"]) == len(detail)
    markdown = result["markdown_path"].read_text(encoding="utf-8")
    assert "## Scenario Assumptions" in markdown
    assert "Amplify realized volatility and risk metrics after the midpoint." in markdown

    liquidity_rows = detail[detail["stress_scenario"] == "liquidity_cost_shock"]
    assert (liquidity_rows["stress_fee"] > 0.0015).all()
    assert (liquidity_rows["stress_slippage"] > 0.002).all()


def test_stress_test_runner_is_deterministic(tmp_path) -> None:
    files = _write_csv(tmp_path, ticker="STABLE")
    runner_a = StressTestRunner(model_root=tmp_path / "models_a")
    runner_b = StressTestRunner(model_root=tmp_path / "models_b")

    result_a = runner_a.run(
        files=files,
        algorithms=["cart"],
        output_root=tmp_path / "models_a",
        report_path=tmp_path / "reports" / "stress_a.csv",
        primary_algorithm="cart",
        horizons=["short"],
        max_depth=3,
    )
    result_b = runner_b.run(
        files=files,
        algorithms=["cart"],
        output_root=tmp_path / "models_b",
        report_path=tmp_path / "reports" / "stress_b.csv",
        primary_algorithm="cart",
        horizons=["short"],
        max_depth=3,
    )

    detail_a = result_a["detail"].sort_values(["stress_scenario", "benchmark_mode"]).reset_index(drop=True)
    detail_b = result_b["detail"].sort_values(["stress_scenario", "benchmark_mode"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(detail_a, detail_b)


def test_stress_test_baseline_matches_corrected_strategy_scaling(tmp_path) -> None:
    files = _write_csv(tmp_path, ticker="STRESSALIGN")
    runner = StressTestRunner(model_root=tmp_path / "models")
    result = runner.run(
        files=files,
        algorithms=["cart"],
        output_root=tmp_path / "models",
        report_path=tmp_path / "reports" / "stress_alignment.csv",
        primary_algorithm="cart",
        horizons=["short"],
        max_depth=3,
    )

    row = result["detail"][
        (result["detail"]["benchmark_mode"] == "legacy_forecast_only")
        & (result["detail"]["stress_scenario"] == "volatility_shock")
        & (result["detail"]["ticker"] == "STRESSALIGN")
        & (result["detail"]["horizon"] == "short")
        & (result["detail"]["algorithm"] == "cart")
    ].iloc[0]

    trainer = DualModelTrainer(model_dir=tmp_path / "models" / "legacy_forecast_only")
    df = pd.read_csv(files[0])
    prepared = trainer.prepare_ticker_data(ticker="STRESSALIGN", df=df, max_sequence_length=1)
    labeled = trainer._add_targets(prepared.feature_frame)
    trainer._ensure_models_loaded("STRESSALIGN")
    manifest = trainer._manifests["STRESSALIGN"]
    algorithm_info = manifest["horizons"]["short"]["algorithms"]["cart"]
    task_columns = algorithm_info["feature_columns_by_task"]
    trend_problem = trainer._build_horizon_problem(labeled, task_columns["trend"], "short", 1)
    return_problem = trainer._build_horizon_problem(labeled, task_columns["return"], "short", 1)
    assert trend_problem is not None
    assert return_problem is not None
    trend_inputs = trend_problem["tabular"]
    return_inputs = return_problem["tabular"]
    trend_model = trainer._get_loaded_model("STRESSALIGN", "cart", "short", "trend")
    signal = trend_model.predict(trend_inputs["X_test"])
    evaluation = DualModelTrainer.evaluate_strategy_for_horizon(
        signal,
        return_inputs["y_test_return"],
        HORIZON_DAYS["short"],
        evaluator=runner.evaluator,
        config=runner.base_eval_config,
    )

    assert row["baseline_sharpe"] == pytest.approx(evaluation["metrics"]["sharpe"])
    assert row["baseline_max_drawdown"] == pytest.approx(evaluation["metrics"]["max_drawdown"])
    assert row["baseline_tail_loss"] == pytest.approx(evaluation["metrics"]["tail_loss"])
