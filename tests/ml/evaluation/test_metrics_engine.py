from __future__ import annotations

import pandas as pd

from src.ml.evaluation import MetricsEngine


def test_metrics_engine_outputs_model_and_baseline_rows() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D").tolist() * 2,
            "ticker": ["FPT"] * 8,
            "horizon": [1] * 8,
            "model_name": ["ets"] * 4 + ["persistence"] * 4,
            "model_type": ["model"] * 4 + ["baseline"] * 4,
            "y_true": [10.0, 12.0, 14.0, 16.0] * 2,
            "y_pred": [11.0, 12.0, 13.0, 15.0, 10.0, 11.0, 13.0, 15.0],
            "actual_direction": [1, 1, 1, 1] * 2,
            "predicted_direction": [1, 1, -1, 1, 1, 1, 1, 1],
        }
    )

    metrics = MetricsEngine().compute(predictions, experiment_id="EXP-TEST", run_id="RUN-1")

    assert {"model", "baseline"} <= set(metrics["model_type"])
    assert {"mae", "rmse", "mape", "directional_accuracy"} <= set(metrics["metric_name"])
    mae = metrics[
        (metrics["model_name"] == "ets")
        & (metrics["metric_group"] == "forecast")
        & (metrics["metric_name"] == "mae")
    ].iloc[0]
    assert mae["metric_value"] == 0.75
    assert mae["sample_size"] == 4


def test_metrics_engine_returns_null_with_reason_when_metric_unavailable() -> None:
    predictions = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "ticker": ["FPT"],
            "horizon": [1],
            "model_name": ["ets"],
            "model_type": ["model"],
            "y_true": [None],
            "y_pred": [None],
        }
    )

    metrics = MetricsEngine().compute(predictions, experiment_id="EXP-TEST", run_id="RUN-1")

    mae = metrics[(metrics["metric_name"] == "mae")].iloc[0]
    assert pd.isna(mae["metric_value"])
    assert mae["notes"] == "no_valid_y_true_y_pred_pairs"
