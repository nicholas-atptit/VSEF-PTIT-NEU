from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.engine.experiment_orchestrator import ExperimentOrchestrator, PREDICTION_COLUMNS


def _write_config(path: Path, output_root: Path) -> None:
    payload = {
        "experiment": {
            "id": "EXP-UNIT-001",
            "name": "Unit experiment",
            "description": "Unit test",
            "phase": 1,
            "owner": "tests",
            "created_at": "2026-05-09",
            "seed": 42,
        },
        "data": {
            "provider": "vnstock_data",
            "frequency": "daily",
            "universe": ["FPT"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-12",
            "schema": ["date", "ticker", "open", "high", "low", "close", "volume"],
        },
        "target": {"column": "close", "task_type": "regression", "horizons": [1]},
        "features": {"enabled": True, "feature_sets": ["ohlcv_basic"]},
        "models": {"enabled": True, "include": ["ets"]},
        "baselines": {"enabled": True, "include": ["persistence"]},
        "evaluation": {
            "method": "fixed_chronological_split",
            "train_start": "2024-01-01",
            "train_end": "2024-01-06",
            "test_start": "2024-01-07",
            "test_end": "2024-01-10",
            "metrics": ["mae", "rmse", "directional_accuracy"],
        },
        "risk": {"enabled": False, "methods": []},
        "outputs": {
            "root_dir": str(output_root),
            "save_predictions": True,
            "save_metrics": True,
            "save_manifest": True,
            "save_charts": False,
            "save_summary": True,
        },
        "runtime": {
            "fail_fast": False,
            "max_workers": 1,
            "log_level": "INFO",
            "allow_test_output_override": True,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _ohlcv() -> pd.DataFrame:
    close = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0]
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(close), freq="D"),
            "ticker": ["FPT"] * len(close),
            "open": close,
            "high": [value + 1.0 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
            "volume": [1000] * len(close),
        }
    )


def test_orchestrator_writes_standard_artifacts(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "experiment.yaml"
    output_root = tmp_path / "outputs" / "experiments"
    _write_config(config_path, output_root)

    orchestrator = ExperimentOrchestrator(str(config_path))
    monkeypatch.setattr(orchestrator, "_fetch_ohlcv", lambda ticker: _ohlcv())

    def fake_run_models(ticker: str, horizon: int, data: pd.DataFrame | None = None) -> list[pd.DataFrame]:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-07", periods=4, freq="D"),
                "ticker": [ticker] * 4,
                "horizon": [horizon] * 4,
                "model_name": ["ets"] * 4,
                "model_type": ["model"] * 4,
                "y_true": [17.0, 18.0, 19.0, 20.0],
                "y_pred": [16.5, 17.5, 19.5, 20.5],
                "predicted_direction": [1, 1, 1, 1],
                "actual_direction": [1, 1, 1, 1],
                "notes": ["unit_model"] * 4,
            }
        )
        return [frame[PREDICTION_COLUMNS]]

    monkeypatch.setattr(orchestrator, "run_models", fake_run_models)

    result = orchestrator.run()

    output_dir = output_root / "EXP-UNIT-001"
    assert result["status"] == "completed"
    assert (output_dir / "config" / "resolved_config.yaml").exists()
    assert (output_dir / "manifests" / "run_manifest.json").exists()
    assert (output_dir / "metrics" / "metrics.csv").exists()
    assert (output_dir / "reports" / "summary.md").exists()

    metrics = pd.read_csv(output_dir / "metrics" / "metrics.csv")
    assert {"model", "baseline"} <= set(metrics["model_type"])


def test_orchestrator_rejects_unsupported_model(tmp_path) -> None:
    config_path = tmp_path / "experiment.yaml"
    output_root = tmp_path / "outputs" / "experiments"
    _write_config(config_path, output_root)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["models"]["include"] = ["random_forest"]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    orchestrator = ExperimentOrchestrator(str(config_path))
    orchestrator.load_config()

    try:
        orchestrator.validate_config()
    except ValueError as exc:
        assert "outside the Phase 0 frozen registry" in str(exc)
    else:
        raise AssertionError("validate_config should reject unsupported models")


def test_orchestrator_expands_phase2_feature_aliases(tmp_path) -> None:
    orchestrator = ExperimentOrchestrator(str(tmp_path / "experiment.yaml"))
    orchestrator.config = {
        "target": {"column": "close", "task_type": "price_forecast"},
        "features": {"enabled": True, "feature_sets": ["default_ohlcv", "technical_basic"]},
    }

    frame = orchestrator._build_supervised_frame(_ohlcv(), horizon=1)

    assert frame.attrs["feature_columns"] == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return_1",
        "high_low_range",
        "close_open_return",
        "ma_3",
        "ma_5",
    ]
    assert orchestrator.warnings == []
