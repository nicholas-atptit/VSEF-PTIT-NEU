from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.backtest.feature_governance_review import classify_feature_governance
from src.ml.data_loader import apply_context_features
from src.ml.feature_engineering import FeatureEngineer


def _base_ohlcv(dates: pd.DatetimeIndex) -> pd.DataFrame:
    close = pd.Series(np.linspace(100.0, 110.0, len(dates)))
    return pd.DataFrame(
        {
            "ticker": "AAA",
            "date": dates,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1_000.0, 2_000.0, len(dates)),
        }
    )


def test_breadth_context_join_does_not_pull_future_rows_backward() -> None:
    df = _base_ohlcv(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    breadth = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03"]),
            "market_breadth": [0.75],
            "breadth_member_count": [40],
            "pct_above_ma20": [0.60],
        }
    )

    result = apply_context_features(df, "AAA", breadth_df=breadth)
    by_date = result.set_index("date")

    assert by_date.loc[pd.Timestamp("2024-01-02"), "market_breadth"] == 0.0
    assert by_date.loc[pd.Timestamp("2024-01-03"), "market_breadth"] == pytest.approx(0.75)
    assert by_date.loc[pd.Timestamp("2024-01-04"), "market_breadth"] == 0.0


def test_macro_context_uses_backward_asof_alignment_only() -> None:
    df = _base_ohlcv(pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-06"]))
    macro = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-05"]),
            "fx_usdvnd": [24_100.0, 24_200.0],
        }
    )

    result = apply_context_features(df, "AAA", macro_df=macro)
    by_date = result.set_index("date")

    assert pd.isna(by_date.loc[pd.Timestamp("2024-01-02"), "fx_usdvnd"])
    assert by_date.loc[pd.Timestamp("2024-01-04"), "fx_usdvnd"] == pytest.approx(24_100.0)
    assert by_date.loc[pd.Timestamp("2024-01-06"), "fx_usdvnd"] == pytest.approx(24_200.0)


def test_foreign_flow_join_is_exact_date_and_ticker_scoped() -> None:
    df = _base_ohlcv(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    foreign_flow = pd.DataFrame(
        {
            "ticker": ["BBB", "AAA"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "foreign_net_value": [99_000.0, 12_500.0],
        }
    )

    result = apply_context_features(df, "AAA", foreign_flow_df=foreign_flow)
    by_date = result.set_index("date")

    assert by_date.loc[pd.Timestamp("2024-01-02"), "foreign_net_value"] == 0.0
    assert bool(by_date.loc[pd.Timestamp("2024-01-02"), "foreign_flow_context_missing"]) is True
    assert by_date.loc[pd.Timestamp("2024-01-03"), "foreign_net_value"] == pytest.approx(12_500.0)
    assert bool(by_date.loc[pd.Timestamp("2024-01-03"), "foreign_flow_context_available"]) is True
    assert by_date.loc[pd.Timestamp("2024-01-04"), "foreign_net_value"] == 0.0
    assert bool(by_date.loc[pd.Timestamp("2024-01-04"), "foreign_flow_context_missing"]) is True


def test_feature_engineer_forward_fill_does_not_backfill_context_from_future() -> None:
    df = _base_ohlcv(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    df["market_breadth"] = [np.nan, 0.5, 0.6]

    features = FeatureEngineer().build_feature_frame(df, build_mode="fast_core_mode")

    assert pd.isna(features.loc[0, "market_breadth"])
    assert features.loc[1, "market_breadth"] == pytest.approx(0.5)


def test_turnover_ma_60_uses_current_and_past_rows_only() -> None:
    dates = pd.bdate_range("2024-01-02", periods=70)
    df = _base_ohlcv(dates)
    changed_future = df.copy()
    changed_future.loc[60:, "volume"] = 1_000_000.0

    baseline = FeatureEngineer().build_feature_frame(df, build_mode="fast_core_mode")
    modified = FeatureEngineer().build_feature_frame(changed_future, build_mode="fast_core_mode")

    expected_turnover = (df.loc[:59, "close"] * df.loc[:59, "volume"]).mean()
    assert baseline.loc[59, "turnover_ma_60"] == pytest.approx(expected_turnover)
    assert modified.loc[59, "turnover_ma_60"] == pytest.approx(baseline.loc[59, "turnover_ma_60"])


def test_governance_distinguishes_local_flow_from_joined_context() -> None:
    turnover_review = classify_feature_governance("turnover_ma_60")
    foreign_review = classify_feature_governance("foreign_net_value_ratio")
    breadth_review = classify_feature_governance("breadth_thrust_10")

    assert turnover_review["governance_category"] == "safe_trailing"
    assert turnover_review["risk_level"] == "low"
    assert turnover_review["recommended_action"] == "keep"
    assert turnover_review["is_context_feature"] is False

    assert foreign_review["governance_category"] == "requires_review"
    assert foreign_review["recommended_action"] == "review_timing"
    assert foreign_review["is_context_feature"] is True

    assert breadth_review["governance_category"] == "requires_review"
    assert breadth_review["recommended_action"] == "review_timing"
