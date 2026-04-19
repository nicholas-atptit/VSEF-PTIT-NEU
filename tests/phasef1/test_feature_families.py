from __future__ import annotations

import pandas as pd

from src.evaluation.forecast_rehab import (
    CURRENT_DIRECTION_FEATURES,
    CURRENT_REGRESSION_FEATURES,
    REDUCED_COMPACT_FEATURES,
    build_feature_family_columns,
    build_feature_inventory_table,
)


def _frame(columns: list[str]) -> pd.DataFrame:
    payload = {column: [1.0, 2.0, 3.0] for column in columns}
    payload["ticker"] = ["AAA", "AAA", "AAA"]
    payload["timestamp"] = pd.date_range("2024-01-01", periods=3, freq="B")
    return pd.DataFrame(payload)


def test_current_full_is_task_specific() -> None:
    frame = _frame(
        sorted(
            set(CURRENT_REGRESSION_FEATURES)
            | set(CURRENT_DIRECTION_FEATURES)
        )
    )

    regression_columns = build_feature_family_columns(frame, family_name="current_full", target_name="forward_return")
    direction_columns = build_feature_family_columns(frame, family_name="current_full", target_name="direction_binary")

    assert set(regression_columns) == set(CURRENT_REGRESSION_FEATURES)
    assert set(direction_columns) == set(CURRENT_DIRECTION_FEATURES)


def test_reduced_compact_uses_union_of_current_baselines() -> None:
    frame = _frame(sorted(set(REDUCED_COMPACT_FEATURES)))
    resolved = build_feature_family_columns(frame, family_name="reduced_compact", target_name="forward_return")
    assert set(resolved) == set(REDUCED_COMPACT_FEATURES)


def test_short_lag_excludes_long_memory_but_keeps_short_lags() -> None:
    frame = _frame(
        [
            "close_lag_1",
            "close_lag_3",
            "close_lag_5",
            "rolling_volatility_20",
            "rolling_volatility_60",
            "sma_200",
            "bb_width",
            "market_return_60d",
            "sector_return_20d",
            "turnover_ratio_20",
        ]
    )

    resolved = build_feature_family_columns(frame, family_name="short_lag", target_name="forward_return")

    assert "close_lag_1" in resolved
    assert "close_lag_3" in resolved
    assert "close_lag_5" not in resolved
    assert "rolling_volatility_60" not in resolved
    assert "sma_200" not in resolved


def test_long_lag_keeps_long_memory_features() -> None:
    frame = _frame(
        [
            "close_lag_1",
            "close_lag_5",
            "rolling_volatility_20",
            "rolling_volatility_60",
            "sma_200",
            "bb_width",
            "market_return_60d",
            "sector_return_20d",
            "turnover_ratio_20",
        ]
    )

    resolved = build_feature_family_columns(frame, family_name="long_lag", target_name="forward_return")

    assert "close_lag_5" in resolved
    assert "rolling_volatility_60" in resolved
    assert "sma_200" in resolved


def test_feature_inventory_table_excludes_target_support_columns() -> None:
    frame = _frame(["bb_width", "daily_return", "current_log_return_1d", "current_direction_1d"])
    inventory = build_feature_inventory_table(frame)

    assert "bb_width" in set(inventory["feature_name"])
    assert "daily_return" not in set(inventory["feature_name"])
    assert "current_log_return_1d" not in set(inventory["feature_name"])
    assert "current_direction_1d" not in set(inventory["feature_name"])
