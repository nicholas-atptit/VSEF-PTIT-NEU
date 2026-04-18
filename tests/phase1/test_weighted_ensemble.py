from __future__ import annotations

import pandas as pd
import pytest

from src.ensemble.weighted import WeightedEnsembleModel


def test_weighted_ensemble_combines_predictions_by_explicit_weights() -> None:
    linear = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02"]),
            "ticker": ["AAA"],
            "y_true": [0.01],
            "y_pred": [0.02],
            "model_name": ["linear"],
            "target_type": ["forward_return"],
            "horizon": [1],
            "window_id": ["w1"],
        }
    )
    ridge = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02"]),
            "ticker": ["AAA"],
            "y_true": [0.01],
            "y_pred": [0.00],
            "model_name": ["ridge"],
            "target_type": ["forward_return"],
            "horizon": [1],
            "window_id": ["w1"],
        }
    )

    combined = WeightedEnsembleModel().combine(
        [linear, ridge],
        context={"model_weights": {"linear": 0.75, "ridge": 0.25}},
    )

    assert combined.loc[0, "model_name"] == "weighted_ensemble"
    assert combined.loc[0, "y_pred"] == pytest.approx(0.015)
