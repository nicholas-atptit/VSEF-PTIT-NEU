from __future__ import annotations

import builtins
import tomllib
from pathlib import Path

import pandas as pd
import pytest
import numpy as np

from src.ml.backtest.strategy_backtest import (
    PORTFOLIO_CAPITAL_MODEL,
    aggregate_active_position_portfolio,
)
from src.ml.features.technical import TechnicalFeatures, TechnicalIndicatorDependencyError
from src.ml.pipelines.training_pipeline import TrainingPipeline
from src.ml.trainer import DualModelTrainer


def _make_synthetic_ohlcv(n_rows: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range(end="2026-03-20", periods=n_rows)
    close = 100.0 + np.cumsum(rng.normal(0, 0.4, n_rows))
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


def _train_cart_fixture(tmp_path, *, risk_config: dict | None = None, ticker: str = "TEST"):
    trainer = DualModelTrainer(model_dir=tmp_path / f"{ticker.lower()}_models")
    df = _make_synthetic_ohlcv()
    trainer.train(
        ticker=ticker,
        df=df,
        algorithms=["cart"],
        risk_config=risk_config,
    )
    features = trainer.compute_features_for_ticker(ticker, df)
    return trainer, df, features


def test_legacy_training_pipeline_is_explicitly_blocked() -> None:
    with pytest.raises(RuntimeError, match="deprecated legacy path"):
        TrainingPipeline(["AAA"])


def test_technical_indicator_generation_fails_loudly_without_pandas_ta(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({"close": [10.0, 10.5, 10.2, 10.8]})
    original_import = builtins.__import__

    def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pandas_ta":
            raise ImportError("simulated missing pandas_ta")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _patched_import)

    with pytest.raises(TechnicalIndicatorDependencyError, match="pandas-ta"):
        TechnicalFeatures.add_all_indicators(frame)


def test_portfolio_aggregation_uses_active_positions_only_and_cash_when_flat() -> None:
    index = pd.to_datetime(["2026-01-02", "2026-01-05"])
    pivot_return = pd.DataFrame(
        {"AAA": [0.10, 0.01], "BBB": [0.00, 0.00]},
        index=index,
    )
    pivot_position = pd.DataFrame(
        {"AAA": [1.0, 0.0], "BBB": [0.0, 0.0]},
        index=index,
    )

    portfolio_return, portfolio_position, active_count, cash_weight = aggregate_active_position_portfolio(
        pivot_return,
        pivot_position,
    )

    assert portfolio_return.iloc[0] == pytest.approx(0.10)
    assert portfolio_return.iloc[1] == pytest.approx(0.0)
    assert portfolio_position.tolist() == [1.0, 0.0]
    assert active_count.tolist() == [1, 0]
    assert cash_weight.tolist() == [0.0, 1.0]
    assert PORTFOLIO_CAPITAL_MODEL == "equal_weight_active_positions_with_cash_when_flat"


def test_predict_labels_risk_as_heuristic_scenario_output(tmp_path) -> None:
    trainer, _, features = _train_cart_fixture(
        tmp_path,
        risk_config={
            "simulations": 500,
            "confidence_levels": [0.95],
            "random_seed": 11,
        },
    )
    prediction = trainer.predict("TEST", features, horizon="short")

    assert "heuristic_scenario_risk" in prediction
    assert "risk_assessment" in prediction
    assert prediction["risk_assessment"] is prediction["heuristic_scenario_risk"]
    metadata = prediction["heuristic_scenario_risk"]["metadata"]
    assert metadata["risk_model_type"] == "residual_normal_scenario_simulation"
    assert metadata["calibration_status"] == "heuristic_not_calibrated"
    assert metadata["output_field"] == "heuristic_scenario_risk"
    assert metadata["deprecated_aliases"] == ["risk_assessment"]


def test_risk_override_false_disables_manifest_enabled_risk(tmp_path) -> None:
    trainer, _, features = _train_cart_fixture(
        tmp_path,
        risk_config={"simulations": 500, "confidence_levels": [0.95], "random_seed": 11},
    )

    prediction = trainer.predict("TEST", features, horizon="short", risk_config={"risk_enabled": False})

    assert "heuristic_scenario_risk" not in prediction
    assert "risk_assessment" not in prediction


def test_risk_override_true_enables_manifest_disabled_risk(tmp_path) -> None:
    trainer, _, features = _train_cart_fixture(tmp_path, risk_config=None)

    prediction = trainer.predict(
        "TEST",
        features,
        horizon="short",
        risk_config={"risk_enabled": True, "confidence_levels": [0.95], "simulations": 250},
    )

    assert "heuristic_scenario_risk" in prediction
    assert prediction["heuristic_scenario_risk"]["metadata"]["simulations"] == 250


def test_non_enablement_risk_override_does_not_change_default_behavior(tmp_path) -> None:
    trainer, _, features = _train_cart_fixture(tmp_path, risk_config=None)

    prediction = trainer.predict(
        "TEST",
        features,
        horizon="short",
        risk_config={"confidence_levels": [0.95], "simulations": 250},
    )

    assert "heuristic_scenario_risk" not in prediction
    assert "risk_assessment" not in prediction


def test_old_manifest_risk_confidence_levels_still_work(tmp_path) -> None:
    trainer, _, features = _train_cart_fixture(
        tmp_path,
        risk_config={"simulations": 500, "confidence_levels": [0.95], "random_seed": 11},
    )
    algo_manifest = trainer._manifests["TEST"]["horizons"]["short"]["algorithms"]["cart"]["risk_config"]
    algo_manifest["risk_confidence_levels"] = algo_manifest.pop("scenario_confidence_levels")

    prediction = trainer.predict("TEST", features, horizon="short")

    assert "heuristic_scenario_risk" in prediction
    assert "95.0" in prediction["heuristic_scenario_risk"]["var"]


def test_training_manifests_distinguish_holdout_and_validation_split_semantics(tmp_path) -> None:
    # Explicit-split training uses the full governed feature path, including long warmup windows.
    # Use a longer synthetic history so the test exercises manifest semantics rather than row-count rejection.
    df = _make_synthetic_ohlcv(n_rows=520)

    holdout_trainer = DualModelTrainer(model_dir=tmp_path / "holdout_models")
    holdout_trainer.train(ticker="HOLD", df=df, algorithms=["cart"])
    holdout_manifest = holdout_trainer._manifests["HOLD"]
    holdout_algo = holdout_manifest["horizons"]["short"]["algorithms"]["cart"]

    assert holdout_manifest["split_config"]["evaluation_split_name"] == "test"
    assert holdout_manifest["split_config"]["metric_source"] == "held_out_test"
    assert holdout_algo["evaluation_metadata"]["evaluation_split_name"] == "test"
    assert holdout_algo["evaluation_metadata"]["metric_source"] == "held_out_test"
    assert holdout_algo["evaluation_metadata"]["validation_method"] == "single_validation_plus_held_out_test"
    assert "test_window" in holdout_algo["evaluation_metadata"]

    explicit_trainer = DualModelTrainer(model_dir=tmp_path / "explicit_models")
    explicit_trainer.train_explicit_split(
        ticker="EXPL",
        df=df,
        algorithms=["cart"],
        train_start=df["time"].iloc[20],
        train_end=df["time"].iloc[-30],
        horizon_name="short",
        horizon_days=5,
    )
    explicit_manifest = explicit_trainer._manifests["EXPL"]
    explicit_algo = explicit_manifest["horizons"]["short"]["algorithms"]["cart"]

    assert explicit_manifest["split_config"]["evaluation_split_name"] == "validation"
    assert explicit_manifest["split_config"]["metric_source"] == "validation_window"
    assert explicit_algo["evaluation_metadata"]["evaluation_split_name"] == "validation"
    assert explicit_algo["evaluation_metadata"]["metric_source"] == "validation_window"
    assert explicit_algo["evaluation_metadata"]["validation_method"] == "explicit_date_window_with_validation_tail"
    assert "validation_window" in explicit_algo["evaluation_metadata"]


def test_ml_dependency_manifests_keep_core_and_optional_rl_split() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    requirements_lines = [
        line.strip()
        for line in (repo_root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    core_dependencies = set(pyproject["project"]["dependencies"])
    optional_rl = set(pyproject["project"]["optional-dependencies"]["ml-rl"])

    assert "pandas-ta>=0.3.14b0" in core_dependencies
    assert "pandas-ta>=0.3.14b0" in requirements_lines
    assert "vnstock_data" in core_dependencies
    assert any(line.startswith("vnstock_data") for line in requirements_lines)
    assert "vnstock>=3.0" not in core_dependencies
    assert all(not line.startswith("vnstock>=3.0") for line in requirements_lines)
    assert "gymnasium>=0.29" in optional_rl
    assert "stable-baselines3>=2.3" in optional_rl
    assert "gymnasium>=0.29" not in requirements_lines
    assert "stable-baselines3>=2.3" not in requirements_lines
