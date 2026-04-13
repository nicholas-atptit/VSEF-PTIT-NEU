from __future__ import annotations

import json

import pytest
import pandas as pd

from src.ml.benchmark.system_benchmark import SystemBenchmarkRunner, default_benchmark_modes
from src.ml.data_loader import generate_mock_data
from src.ml.trainer import DualModelTrainer, HORIZON_DAYS


def _write_csv(tmp_path, ticker: str = "BENCH", num_days: int = 900) -> list:
    df = generate_mock_data(ticker=ticker, num_days=num_days)
    csv_path = tmp_path / f"{ticker}.csv"
    df.to_csv(csv_path, index=False)
    return [csv_path]


def test_system_benchmark_generates_all_modes_and_outputs(tmp_path) -> None:
    files = _write_csv(tmp_path)
    runner = SystemBenchmarkRunner(model_root=tmp_path / "models")
    result = runner.run(
        files=files,
        algorithms=["cart"],
        output_root=tmp_path / "models",
        report_path=tmp_path / "reports" / "system_benchmark.csv",
        primary_algorithm="cart",
        horizons=["short"],
        max_depth=3,
    )

    detail = result["detail"]
    summary = result["summary"]
    expected_modes = {
        "legacy_forecast_only",
        "forecast_plus_risk_features",
        "forecast_plus_risk_and_regime",
        "full_system",
    }
    assert set(detail["benchmark_mode"]) == expected_modes
    assert set(summary["benchmark_mode"]) == expected_modes
    assert result["detail_path"].exists()
    assert result["summary_path"].exists()
    assert result["json_path"].exists()
    assert result["markdown_path"].exists()
    markdown = result["markdown_path"].read_text(encoding="utf-8")
    assert "# System Benchmark" in markdown

    payload = json.loads(result["json_path"].read_text(encoding="utf-8"))
    assert len(payload["detail_rows"]) == len(detail)
    assert len(payload["summary_rows"]) == len(summary)
    assert detail["cumulative_return"].abs().max() < 1_000
    assert detail["cagr"].abs().max() < 1_000


def test_system_benchmark_reproducible_and_identical_splits(tmp_path) -> None:
    files = _write_csv(tmp_path, ticker="CONSISTENT")
    runner_a = SystemBenchmarkRunner(model_root=tmp_path / "models_a")
    runner_b = SystemBenchmarkRunner(model_root=tmp_path / "models_b")

    result_a = runner_a.run(
        files=files,
        algorithms=["cart"],
        output_root=tmp_path / "models_a",
        report_path=tmp_path / "reports" / "benchmark_a.csv",
        primary_algorithm="cart",
        horizons=["short"],
        max_depth=3,
    )
    result_b = runner_b.run(
        files=files,
        algorithms=["cart"],
        output_root=tmp_path / "models_b",
        report_path=tmp_path / "reports" / "benchmark_b.csv",
        primary_algorithm="cart",
        horizons=["short"],
        max_depth=3,
    )

    detail_a = result_a["detail"].sort_values(["benchmark_mode", "ticker", "horizon", "algorithm"]).reset_index(drop=True)
    detail_b = result_b["detail"].sort_values(["benchmark_mode", "ticker", "horizon", "algorithm"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(detail_a, detail_b)

    split_counts = detail_a[["benchmark_mode", "train_rows", "val_rows", "test_rows"]]
    assert split_counts["train_rows"].nunique() == 1
    assert split_counts["val_rows"].nunique() == 1
    assert split_counts["test_rows"].nunique() == 1


def test_system_benchmark_matches_trainer_strategy_scaling(tmp_path) -> None:
    files = _write_csv(tmp_path, ticker="ALIGN")
    runner = SystemBenchmarkRunner(model_root=tmp_path / "models")
    result = runner.run(
        files=files,
        algorithms=["cart"],
        output_root=tmp_path / "models",
        report_path=tmp_path / "reports" / "alignment.csv",
        primary_algorithm="cart",
        horizons=["short"],
        max_depth=3,
    )

    row = result["detail"][
        (result["detail"]["benchmark_mode"] == "legacy_forecast_only")
        & (result["detail"]["ticker"] == "ALIGN")
        & (result["detail"]["horizon"] == "short")
        & (result["detail"]["algorithm"] == "cart")
    ].iloc[0]

    trainer = DualModelTrainer(model_dir=tmp_path / "models" / "legacy_forecast_only")
    df = pd.read_csv(files[0])
    prepared = trainer.prepare_ticker_data(
        ticker="ALIGN",
        df=df,
        max_sequence_length=1,
        risk_config=default_benchmark_modes()[0].risk_config,
    )
    labeled = trainer._add_targets(prepared.feature_frame)
    problem = trainer._build_horizon_problem(labeled, prepared.base_feature_columns, "short", 1)
    assert problem is not None
    inputs = problem["tabular"]
    trend_model = trainer._get_loaded_model("ALIGN", "cart", "short", "trend")
    signal = trend_model.predict(inputs["X_test"])
    evaluation = DualModelTrainer.evaluate_strategy_for_horizon(
        signal,
        inputs["y_test_return"],
        HORIZON_DAYS["short"],
        evaluator=trainer._metrics_evaluator,
    )

    assert row["cumulative_return"] == pytest.approx(evaluation["metrics"]["cumulative_return"])
    assert row["cagr"] == pytest.approx(evaluation["metrics"]["cagr"])
    assert row["sharpe"] == pytest.approx(evaluation["metrics"]["sharpe"])
    assert row["sortino"] == pytest.approx(evaluation["metrics"]["sortino"])
    assert row["max_drawdown"] == pytest.approx(evaluation["metrics"]["max_drawdown"])
