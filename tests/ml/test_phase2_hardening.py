from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.datasets.loader import DatasetLoader
from src.ml.artifacts import load_manifest
from src.ml.backtest.forward_return import _compute_error_metrics
from src.ml.benchmark.evaluator import MetricsEvaluator
from src.ml.inference.engine import InferenceEngine
from src.ml.metrics import compute_prediction_error_metrics
from src.ml.trainer import DualModelTrainer
from src.ml.training.baseline_model import BaselineModel
from src.reporting.reports.daily_report import DailyReportGenerator


EXPECTED_FEATURE_DEPENDENCY_BEHAVIOR = "local_deterministic_numpy_pandas_computation"


def _make_synthetic_ohlcv(n_rows: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    dates = pd.bdate_range(end="2026-03-20", periods=n_rows)
    trend = np.linspace(0.0, 8.0, n_rows)
    cyclical = 4.0 * np.sin(np.linspace(0.0, 18.0 * np.pi, n_rows))
    close = 100.0 + trend + cyclical + np.cumsum(rng.normal(0, 0.25, n_rows))
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


def test_canonical_prediction_metrics_match_wrappers() -> None:
    actual = pd.Series([0.05, -0.02, 0.01, 0.03, -0.01])
    predicted = pd.Series([0.04, -0.01, -0.01, 0.02, 0.00])

    canonical = compute_prediction_error_metrics(actual, predicted, include_residual_std=True)
    forward = _compute_error_metrics(actual, predicted)
    benchmark = MetricsEvaluator().evaluate_prediction_quality(
        predicted_returns=predicted.to_numpy(),
        realized_returns=actual.to_numpy(),
        predicted_direction=np.sign(predicted.to_numpy()),
        realized_direction=np.sign(actual.to_numpy()),
    )
    regression = DualModelTrainer._regression_metrics(actual.to_numpy(), predicted.to_numpy())

    assert forward["mae"] == pytest.approx(canonical["mae"])
    assert forward["rmse"] == pytest.approx(canonical["rmse"])
    assert forward["mape"] == pytest.approx(canonical["mape"])
    assert forward["directional_accuracy"] == pytest.approx(canonical["directional_accuracy"])
    assert benchmark["mae"] == pytest.approx(round(canonical["mae"], 6))
    assert benchmark["rmse"] == pytest.approx(round(canonical["rmse"], 6))
    assert benchmark["directional_accuracy"] == pytest.approx(round(canonical["directional_accuracy"], 6))
    assert regression["mae"] == pytest.approx(canonical["mae"])
    assert regression["rmse"] == pytest.approx(canonical["rmse"])
    assert regression["residual_std"] == pytest.approx(canonical["residual_std"])


def test_legacy_scaffolds_require_explicit_opt_in() -> None:
    with pytest.raises(RuntimeError, match="legacy scaffold"):
        DatasetLoader(["AAA"])
    with pytest.raises(RuntimeError, match="legacy scaffold"):
        BaselineModel()

    loader = DatasetLoader(["AAA"], allow_legacy=True)
    baseline = BaselineModel(allow_legacy=True)

    assert loader.symbols == ["AAA"]
    assert baseline.model is None


def test_manifest_round_trip_includes_phase2_reproducibility_metadata(tmp_path: Path) -> None:
    trainer = DualModelTrainer(model_dir=tmp_path / "models")
    df = _make_synthetic_ohlcv()
    trainer.train(
        ticker="TEST",
        df=df,
        algorithms=["cart"],
        risk_config={"simulations": 200, "confidence_levels": [0.95], "random_seed": 5},
    )

    manifest = trainer._manifests["TEST"]
    loaded = load_manifest(tmp_path / "models", "TEST")
    algo_manifest = loaded["horizons"]["short"]["algorithms"]["cart"]

    assert loaded["feature_generation"]["technical_indicator_dependency_behavior"] == EXPECTED_FEATURE_DEPENDENCY_BEHAVIOR
    assert loaded["training_backend"]["authoritative_dependency_manifest"] == "pyproject.toml"
    assert loaded["prediction_output_semantics"]["risk_output_field"] == "heuristic_scenario_risk"
    assert loaded["target_definition"]["task_bundle"] == "trend_classification_profit_classification_forward_return_regression"
    assert loaded["evaluation_semantics"]["metric_source"] == manifest["evaluation_semantics"]["metric_source"]
    assert algo_manifest["model_type"] == "cart"
    assert algo_manifest["prediction_output_semantics"]["risk_output_aliases"] == ["risk_assessment"]
    assert algo_manifest["training_backend"]["tuning_backend"] == "model_factory_defaults_only"
    assert algo_manifest["evaluation_metadata"]["evaluation_split_name"] == "test"


def test_inference_engine_and_daily_report_surface_semantic_metadata(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    trainer = DualModelTrainer(model_dir=model_root)
    df = _make_synthetic_ohlcv()
    trainer.train(
        ticker="TEST",
        df=df,
        algorithms=["cart"],
        risk_config={"simulations": 200, "confidence_levels": [0.95], "random_seed": 7},
    )

    engine = InferenceEngine(model_root=model_root)
    history = df.copy()
    history["ticker"] = "TEST"
    prediction = engine.predict_ticker("TEST", history, horizon="short")

    assert prediction["evaluation_split_name"] == "test"
    assert prediction["metric_source"] == "held_out_test"
    assert prediction["validation_method"] == "single_validation_plus_held_out_test"
    assert prediction["risk_semantics"] == "heuristic_scenario_risk_not_calibrated_confidence"
    assert prediction["risk_calibration_status"] == "heuristic_not_calibrated"
    assert prediction["feature_dependency_behavior"] == EXPECTED_FEATURE_DEPENDENCY_BEHAVIOR

    report = DailyReportGenerator().generate(pd.DataFrame([prediction]))
    assert "Heuristic Scenario Risk" in report
    assert "held_out_test / test" in report
    assert "not calibrated forecast confidence" in report
    assert "per-name inference summary" in report
