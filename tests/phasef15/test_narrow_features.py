from __future__ import annotations

import pandas as pd

from src.evaluation.forecast_rehab_narrow import (
    build_narrow_feature_summary,
    resolve_narrow_feature_family_columns,
)


def _feature_frame(columns: list[str]) -> pd.DataFrame:
    values = {column: [0.0, 1.0] for column in columns}
    values["timestamp"] = pd.to_datetime(["2024-01-01", "2024-01-02"])
    values["close"] = [10.0, 10.5]
    return pd.DataFrame(values)


def test_narrow_feature_resolution_returns_explicit_subset() -> None:
    frame = _feature_frame(
        [
            "log_return",
            "momentum_5",
            "dist_ma_20",
            "rsi_14",
            "macd_signal",
            "close_return_10d",
            "rolling_volatility_20",
            "foreign_net_value_ratio",
            "foreign_participation_20",
            "foreign_flow_intensity_zscore_20",
        ]
    )
    resolved = resolve_narrow_feature_family_columns(frame, family_name="tech_core_v1")
    assert resolved == [
        "log_return",
        "momentum_5",
        "dist_ma_20",
        "rsi_14",
        "macd_signal",
        "close_return_10d",
        "rolling_volatility_20",
        "foreign_net_value_ratio",
        "foreign_participation_20",
        "foreign_flow_intensity_zscore_20",
    ]


def test_narrow_feature_summary_tracks_missing_and_resolved_columns() -> None:
    frame = _feature_frame(
        [
            "m_ret_5d",
            "m_ret_20d",
            "declining_share",
            "pct_above_ma20",
            "pct_above_ma50",
            "range_20",
            "rolling_min_5",
            "dist_ma_20",
            "dist_ma_60",
            "ema_50",
            "close_to_sma_200",
            "turnover_ma_60",
            "macd_signal",
            "bb_width",
            "close_return_10d",
            "rolling_volatility_60",
            "market_return_60d",
            "breadth_thrust_10",
            "new_high_low_spread_5",
            "up_down_volume_ratio_5",
        ]
    )
    summary = build_narrow_feature_summary(frame)
    compact_row = summary.loc[summary["feature_family"] == "compact_v1"].iloc[0]
    assert int(compact_row["resolved_feature_count"]) == 20
    assert int(compact_row["missing_feature_count"]) == 0
    assert "compact baseline" in str(compact_row["rationale"]).lower()
