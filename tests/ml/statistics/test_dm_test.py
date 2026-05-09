from __future__ import annotations

import pandas as pd

from src.ml.statistics import diebold_mariano_test


def test_dm_test_detects_lower_loss_with_real_p_value() -> None:
    model_errors = [0.10, -0.20, 0.15, -0.10, 0.20, -0.15, 0.10, -0.20, 0.15, -0.10, 0.20, -0.15]
    baseline_errors = [0.80, -0.70, 0.90, -0.85, 0.75, -0.95, 0.80, -0.70, 0.90, -0.85, 0.75, -0.95]

    result = diebold_mariano_test(model_errors, baseline_errors, loss="squared", horizon=1)

    assert result["sample_size"] == 12
    assert result["mean_loss_model"] < result["mean_loss_baseline"]
    assert result["mean_loss_diff"] < 0
    assert result["p_value"] is not None


def test_dm_test_returns_warning_for_small_sample() -> None:
    result = diebold_mariano_test([1.0, 2.0], [1.1, 1.8], loss="absolute", horizon=1)

    assert result["p_value"] is None
    assert result["warning"].startswith("insufficient_sample_size")


def test_dm_test_drops_missing_pairs() -> None:
    result = diebold_mariano_test(
        pd.Series([1.0, None, 1.2, 1.0, 0.8, 1.1, 1.3, 0.9, 1.0, 1.2, 0.7]),
        pd.Series([1.5, 1.4, 1.6, 1.4, 1.7, 1.5, 1.8, 1.4, 1.6, 1.5, 1.7]),
        loss="absolute",
        horizon=1,
    )

    assert result["sample_size"] == 10
    assert result["dropped_count"] == 1
    assert "dropped_missing_pairs" in result["warning"]
