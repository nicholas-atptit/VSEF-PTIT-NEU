from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.artifacts import (
    ARTIFACT_CREATED_BY,
    ARTIFACT_SCHEMA_VERSION,
    MANIFEST_COMPATIBILITY_VERSION,
    load_manifest,
)
from src.ml.backtest.forward_return import ForwardReturnBacktestConfig, ForwardReturnBacktestRunner
from src.ml.backtest.model_comparison import BacktestModelComparisonRunner, ModelComparisonConfig
from src.ml.inference.engine import InferenceEngine
from src.ml.metrics import (
    compute_brier_score,
    summarize_binary_probability_calibration,
    summarize_regression_residual_diagnostics,
)
from src.ml.trainer import DualModelTrainer


def _make_synthetic_ohlcv(n_rows: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(29)
    dates = pd.bdate_range(end="2026-03-20", periods=n_rows)
    close = 100.0 + np.cumsum(rng.normal(0, 0.45, n_rows))
    close = np.maximum(close, 10.0)
    return pd.DataFrame(
        {
            "time": dates,
            "open": close * (1 + rng.normal(0, 0.002, n_rows)),
            "high": close * (1 + np.abs(rng.normal(0, 0.008, n_rows))),
            "low": close * (1 - np.abs(rng.normal(0, 0.008, n_rows))),
            "close": close,
            "volume": rng.integers(100_000, 5_000_000, n_rows),
        }
    )


def test_calibration_helpers_expose_probability_and_residual_diagnostics() -> None:
    y_true = np.array([0, 1, 1, 0, 1, 0], dtype=int)
    y_prob = np.array([0.10, 0.80, 0.65, 0.40, 0.90, 0.20], dtype=float)

    brier = compute_brier_score(y_true, y_prob)
    calibration = summarize_binary_probability_calibration(y_true, y_prob, num_bins=3)
    residuals = summarize_regression_residual_diagnostics(
        actual=np.array([0.05, -0.02, 0.01, 0.03]),
        predicted=np.array([0.04, -0.01, -0.01, 0.01]),
    )

    assert brier == pytest.approx(np.mean((y_prob - y_true) ** 2))
    assert calibration["available"] is True
    assert calibration["brier_score"] == pytest.approx(brier)
    assert calibration["bins"]
    assert calibration["interpretation"].startswith("Binary probability calibration summary")
    assert residuals["available"] is True
    assert residuals["observations"] == 4
    assert "residual_q10" in residuals
    assert "interpretation" in residuals


def test_manifest_loader_normalizes_recent_legacy_manifest(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    ticker_dir = model_root / "LEG"
    ticker_dir.mkdir(parents=True, exist_ok=True)
    legacy_manifest = {
        "schema_version": 2,
        "ticker": "LEG",
        "split_config": {
            "evaluation_split_name": "test",
            "metric_source": "held_out_test",
            "validation_method": "single_validation_plus_held_out_test",
        },
        "horizons": {
            "short": {
                "algorithms": {
                    "cart": {
                        "risk_config": {
                            "risk_enabled": True,
                            "risk_confidence_levels": [0.95],
                        },
                        "calibration": {"q10": -0.02, "q50": 0.0, "q90": 0.02},
                    }
                }
            }
        },
    }
    (ticker_dir / "manifest.json").write_text(json.dumps(legacy_manifest), encoding="utf-8")

    loaded = load_manifest(model_root, "LEG")
    algo_manifest = loaded["horizons"]["short"]["algorithms"]["cart"]

    assert loaded["manifest_schema_version"] == 2
    assert loaded["compatibility_version"] == MANIFEST_COMPATIBILITY_VERSION
    assert loaded["artifact_created_by"] == ARTIFACT_CREATED_BY
    assert loaded["prediction_output_semantics"]["risk_output_field"] == "heuristic_scenario_risk"
    assert algo_manifest["risk_config"]["scenario_confidence_levels"] == [0.95]
    assert algo_manifest["calibration"]["calibration_status"] == "not_probability_calibration"
    assert "interpretation_warning" in algo_manifest["calibration"]


def test_phase3_training_manifest_adds_versioning_and_uncertainty_metadata(tmp_path: Path) -> None:
    trainer = DualModelTrainer(model_dir=tmp_path / "models")
    trainer.train(
        ticker="TEST",
        df=_make_synthetic_ohlcv(),
        algorithms=["cart"],
        risk_config={"simulations": 300, "confidence_levels": [0.95], "random_seed": 13},
    )

    manifest = trainer._manifests["TEST"]
    algo_manifest = manifest["horizons"]["short"]["algorithms"]["cart"]
    risk_config = algo_manifest["risk_config"]

    assert manifest["manifest_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert manifest["compatibility_version"] == MANIFEST_COMPATIBILITY_VERSION
    assert manifest["artifact_created_by"] == ARTIFACT_CREATED_BY
    assert manifest["prediction_output_semantics"]["uncertainty_methodology"].startswith("residual_based")
    assert risk_config["uncertainty_methodology"].startswith("residual_based")
    assert "interpretation_warning" in risk_config
    assert algo_manifest["calibration"]["calibration_status"] == "not_probability_calibration"
    assert algo_manifest["calibration_diagnostics"]["regression_residuals"]["available"] is True
    assert "scenario_risk" in algo_manifest["calibration_diagnostics"]
    assert "feature_columns_by_task" in algo_manifest
    assert set(algo_manifest["feature_columns_by_task"]) == {"trend", "profit", "return"}
    assert algo_manifest["feature_columns_by_task"]["trend"]
    assert algo_manifest["feature_columns_by_task"]["return"]
    assert "final_task_feature_sets" in manifest["feature_governance"]
    assert "feature_selection_evidence" in manifest["feature_governance"]
    assert manifest["feature_governance"]["sentiment_policy"]["enabled_by_default"] is False
    assert manifest["feature_governance"]["sentiment_policy"]["approved_for_main_pipeline"] is False


def test_inference_engine_surfaces_phase3_uncertainty_boundaries(tmp_path: Path) -> None:
    trainer = DualModelTrainer(model_dir=tmp_path / "models")
    df = _make_synthetic_ohlcv()
    trainer.train(
        ticker="TEST",
        df=df,
        algorithms=["cart"],
        risk_config={"simulations": 300, "confidence_levels": [0.95], "random_seed": 9},
    )

    engine = InferenceEngine(model_root=tmp_path / "models")
    history = df.copy()
    history["ticker"] = "TEST"
    prediction = engine.predict_ticker("TEST", history, horizon="short")

    assert prediction["manifest_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert prediction["compatibility_version"] == MANIFEST_COMPATIBILITY_VERSION
    assert prediction["artifact_created_by"] == ARTIFACT_CREATED_BY
    assert prediction["uncertainty_methodology"].startswith("residual_based")
    assert "not calibrated" in prediction["risk_interpretation_warning"]
    assert prediction["risk_assessment"] is prediction["heuristic_scenario_risk"]
    assert prediction["heuristic_scenario_risk"]["metadata"]["uncertainty_methodology"].startswith("residual_based")


def test_model_comparison_run_config_marks_comparability_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ModelComparisonConfig(
        tickers=["AAA"],
        train_start="2025-01-01",
        train_end="2025-06-30",
        eval_start="2025-07-01",
        eval_end="2025-07-31",
        output_dir=str(tmp_path / "comparison"),
        algorithms=["cart"],
    )
    runner = BacktestModelComparisonRunner(config)
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-07-01", "2025-07-02"]),
            "ticker": ["AAA", "AAA"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.5],
            "volume": [1000.0, 1200.0],
        }
    )
    comparison_frame = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "model_name": ["cart", "cart"],
            "date": pd.to_datetime(["2025-07-01", "2025-07-02"]),
            "actual_close": [100.0, 101.5],
            "predicted_close": [100.4, 101.2],
            "predicted_close_baseline": [100.1, 100.7],
            "actual_direction": [1, 1],
            "predicted_direction": [1, 1],
            "predicted_direction_baseline": [0, 0],
        }
    )

    monkeypatch.setattr(runner, "_fetch_start", lambda: pd.Timestamp("2024-12-01"))
    monkeypatch.setattr(runner, "_fetch_histories", lambda start, end: ({"AAA": history.copy()}, pd.DataFrame([{"ticker": "AAA", "rows": 2}])))
    monkeypatch.setattr(runner, "_build_context_sources", lambda start, end: {})
    monkeypatch.setattr(runner, "_resolve_available_algorithms", lambda: (["cart"], []))
    monkeypatch.setattr(
        runner,
        "_train_algorithm_ticker",
        lambda trainer, algorithm, ticker, ticker_history, context: {"report_rows": [{"ticker": ticker, "algorithm": algorithm}]},
    )
    monkeypatch.setattr(
        runner,
        "_evaluate_algorithm_ticker",
        lambda trainer, algorithm, ticker, ticker_history, context: comparison_frame.copy(),
    )

    result = runner.run()
    run_config = result["run_config"]

    assert run_config["benchmark_basis"] == "shared_close_level_regression_task_with_directional_sign_check"
    assert run_config["comparable_tasks_only"] is True
    assert run_config["evaluation_context"] == "fixed_window_holdout_target_date_model_family_comparison"
    assert run_config["metric_semantics"]["financial_performance_metrics_included"] is False
    assert run_config["metric_semantics"]["heuristic_scenario_risk_included"] is False


def test_forward_return_run_config_marks_comparability_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ForwardReturnBacktestConfig(
        tickers=["AAA"],
        train_start="2025-01-01",
        train_end="2025-06-30",
        eval_start="2025-07-01",
        eval_end="2025-07-31",
        output_dir=str(tmp_path / "forward"),
        algorithms=["cart"],
        horizons=["3d"],
    )
    runner = ForwardReturnBacktestRunner(config)
    monkeypatch.setattr(runner, "_render_horizon_charts", lambda comparison_df, metrics_df, charts_dir: {})

    comparison_df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "model_name": ["cart", "cart"],
            "target_date": pd.to_datetime(["2025-07-10", "2025-07-11"]),
            "prediction_date": pd.to_datetime(["2025-07-07", "2025-07-08"]),
            "actual_return": [0.02, -0.01],
            "predicted_return": [0.01, -0.02],
            "naive_predicted_return": [0.0, 0.0],
            "absolute_error": [0.01, 0.01],
            "absolute_error_naive": [0.02, 0.01],
        }
    )
    metrics_df = pd.DataFrame(
        {
            "ticker": ["OVERALL"],
            "model_name": ["cart"],
            "rmse": [0.01],
            "mape": [10.0],
            "directional_accuracy": [0.5],
        }
    )
    ranking_df = pd.DataFrame(
        {
            "model_name": ["cart"],
            "rmse": [0.01],
            "mape": [10.0],
            "directional_accuracy": [0.5],
            "average_rank": [1.0],
        }
    )
    fetch_summary = pd.DataFrame([{"ticker": "AAA", "rows": 2}])
    training_df = pd.DataFrame([{"ticker": "AAA", "algorithm": "cart"}])

    result = runner._write_horizon_artifacts(
        horizon_name="3d",
        horizon_days=3,
        comparison_df=comparison_df,
        metrics_df=metrics_df,
        ranking_df=ranking_df,
        fetch_summary=fetch_summary,
        training_df=training_df,
        available_algorithms=["cart"],
        skipped_algorithms=[],
    )

    run_config = result["run_config"]
    assert run_config["benchmark_basis"].startswith("forward_return_regression")
    assert run_config["comparable_tasks_only"] is True
    assert run_config["evaluation_context"] == "fixed_window_target_date_forward_return_backtest"
    assert run_config["metric_semantics"]["financial_performance_metrics_included"] is False
    assert run_config["metric_semantics"]["heuristic_scenario_risk_included"] is False
