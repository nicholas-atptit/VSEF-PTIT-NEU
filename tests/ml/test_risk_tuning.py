from __future__ import annotations

import json

import pytest
import pandas as pd

from src.ml.benchmark.risk_tuning import RiskTuningRunner
from src.ml.benchmark.system_benchmark import BenchmarkModeSpec
from src.ml.data_loader import generate_mock_data
from src.ml.trainer import DualModelTrainer, HORIZON_DAYS


def _write_csv(tmp_path, ticker: str = "TUNE", num_days: int = 900) -> list:
    df = generate_mock_data(ticker=ticker, num_days=num_days)
    csv_path = tmp_path / f"{ticker}.csv"
    df.to_csv(csv_path, index=False)
    return [csv_path]


def test_risk_tuning_runner_outputs_best_params_and_manifests(tmp_path) -> None:
    files = _write_csv(tmp_path)
    runner = RiskTuningRunner(model_root=tmp_path / "models")
    result = runner.run(
        files=files,
        algorithms=["cart"],
        output_root=tmp_path / "models",
        report_path=tmp_path / "reports" / "risk_tuning.csv",
        primary_algorithm="cart",
        horizons=["short"],
        max_trials=1,
        max_depth=3,
    )

    best = result["best_params"]
    assert 0.01 <= best["covar_quantile"] <= 0.10
    assert 20 <= best["covar_window"] <= 120
    assert 0.1 <= best["risk_penalty_strength"] <= 3.0
    assert result["csv_path"].exists()
    assert result["json_path"].exists()
    assert result["markdown_path"].exists()
    assert result["markdown_path"] == tmp_path / "reports" / "risk_tuning_report.md"
    assert not result["trials"].empty

    manifest_path = result["best_model_root"] / "TUNE" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    advanced = manifest["advanced_risk"]
    assert advanced["enable_risk_allocation"] is True
    assert advanced["enable_regime_switching"] is True
    assert advanced["covar_quantile"] == best["covar_quantile"]
    assert advanced["covar_window"] == best["covar_window"]
    assert advanced["risk_penalty_strength"] == best["risk_penalty_strength"]


def test_risk_tuning_uses_corrected_strategy_scaling(tmp_path) -> None:
    files = _write_csv(tmp_path, ticker="VALALIGN")
    runner = RiskTuningRunner(model_root=tmp_path / "models")
    candidate = {
        "risk_enabled": True,
        "enable_covar": True,
        "enable_risk_engine": True,
        "enable_regime_detection": True,
        "enable_regime_switching": True,
        "enable_risk_allocation": True,
        "covar_quantile": 0.05,
        "covar_window": 60,
        "risk_penalty_strength": 1.0,
        "high_vol_threshold": 0.03,
        "crisis_drawdown_threshold": -0.12,
        "crisis_delta_covar_threshold": 0.015,
        "high_vol_exposure_cut": 0.6,
        "crisis_exposure_cut": 0.25,
        "regime_method": "threshold",
        "random_seed": 42,
        "simulations": 10000,
        "confidence_levels": [0.95, 0.99],
    }

    score, rows = runner._evaluate_candidate(
        files=files,
        algorithms=["cart"],
        output_root=tmp_path / "trial_eval",
        risk_config=candidate,
        primary_algorithm="cart",
        horizons=["short"],
        sequence_length=20,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=1e-3,
        batch_size=32,
        epochs=30,
        patience=5,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion=None,
    )

    row = next(item for item in rows if item["ticker"] == "VALALIGN" and item["horizon"] == "short" and item["algorithm"] == "cart")
    trainer = DualModelTrainer(model_dir=tmp_path / "trial_eval" / "VALALIGN")
    df = pd.read_csv(files[0])
    prepared = trainer.prepare_ticker_data(
        ticker="VALALIGN",
        df=df,
        max_sequence_length=20,
        risk_config=candidate,
    )
    labeled = trainer._add_targets(prepared.feature_frame)
    trainer._ensure_models_loaded("VALALIGN")
    manifest = trainer._manifests["VALALIGN"]
    algorithm_info = manifest["horizons"]["short"]["algorithms"]["cart"]
    task_columns = algorithm_info["feature_columns_by_task"]
    trend_problem = trainer._build_horizon_problem(labeled, task_columns["trend"], "short", 20)
    return_problem = trainer._build_horizon_problem(labeled, task_columns["return"], "short", 20)
    assert trend_problem is not None
    assert return_problem is not None
    trend_inputs = trend_problem["tabular"]
    return_inputs = return_problem["tabular"]
    trend_model = trainer._get_loaded_model("VALALIGN", "cart", "short", "trend")
    predicted_direction = trend_model.predict(trend_inputs["X_val"])
    signal = runner.signal_builder._build_signal(
        BenchmarkModeSpec(
            name="full_system",
            description="Risk/regime/allocation system under tuning.",
            risk_config=candidate,
        ),
        "VALALIGN",
        predicted_direction,
        trend_inputs["val_feature_frame"].reset_index(drop=True),
    )
    evaluation = DualModelTrainer.evaluate_strategy_for_horizon(
        signal,
        return_inputs["y_val_return"],
        HORIZON_DAYS["short"],
        evaluator=runner.evaluator,
        config=runner.eval_config,
    )
    expected_score = runner._score(evaluation["metrics"], evaluation["trade_stats"])

    assert row["score"] == pytest.approx(expected_score)
    assert score == pytest.approx(expected_score)
