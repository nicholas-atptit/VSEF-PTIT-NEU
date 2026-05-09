from __future__ import annotations

import pandas as pd

from src.ml.baselines import BaselineRegistry


def _ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="D"),
            "ticker": ["FPT"] * 8,
            "open": [10, 11, 12, 13, 14, 15, 16, 17],
            "high": [11, 12, 13, 14, 15, 16, 17, 18],
            "low": [9, 10, 11, 12, 13, 14, 15, 16],
            "close": [10, 11, 12, 13, 14, 15, 16, 17],
            "volume": [1000] * 8,
        }
    )


def test_baseline_registry_lists_required_baselines() -> None:
    registry = BaselineRegistry()
    assert set(registry.list_baselines()) == {
        "moving_average_rule",
        "persistence",
        "random_direction",
        "zero_return",
    }


def test_random_direction_is_seed_controlled() -> None:
    registry = BaselineRegistry()
    config = {"target": {"column": "close", "task_type": "regression"}, "seed": 123}

    first = registry.run_baseline("random_direction", _ohlcv(), 1, config)
    second = registry.run_baseline("random_direction", _ohlcv(), 1, config)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["model_type"]) == {"baseline"}
    assert {"date", "ticker", "horizon", "model_name", "y_true", "y_pred"} <= set(first.columns)


def test_persistence_uses_current_close_for_price_target() -> None:
    registry = BaselineRegistry()
    result = registry.run_baseline(
        "persistence",
        _ohlcv(),
        1,
        {"target": {"column": "close", "task_type": "regression"}},
    )

    assert result.iloc[0]["y_pred"] == 10
    assert result.iloc[0]["y_true"] == 11
    assert set(result["model_name"]) == {"persistence"}
