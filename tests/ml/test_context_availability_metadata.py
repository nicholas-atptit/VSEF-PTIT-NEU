from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.backtest.feature_governance_review import classify_feature_governance
from src.ml.data_loader import apply_context_features
from src.ml.feature_engineering import FeatureEngineer


CONTEXT_METADATA_COLUMNS = {
    "breadth_context_available",
    "breadth_context_source_date",
    "breadth_context_missing",
    "foreign_flow_context_available",
    "foreign_flow_context_source_date",
    "foreign_flow_context_missing",
}


def _base_ohlcv(dates: pd.DatetimeIndex) -> pd.DataFrame:
    close = pd.Series(np.linspace(100.0, 105.0, len(dates)))
    return pd.DataFrame(
        {
            "ticker": "AAA",
            "date": dates,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1_000.0, 1_500.0, len(dates)),
        }
    )


def test_breadth_exact_date_join_sets_availability_metadata() -> None:
    df = _base_ohlcv(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    breadth = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            "market_breadth": [0.0, 0.4],
            "breadth_member_count": [30, 32],
        }
    )

    result = apply_context_features(df, "AAA", breadth_df=breadth)
    by_date = result.set_index("date")

    assert bool(by_date.loc[pd.Timestamp("2024-01-02"), "breadth_context_available"]) is True
    assert bool(by_date.loc[pd.Timestamp("2024-01-02"), "breadth_context_missing"]) is False
    assert by_date.loc[pd.Timestamp("2024-01-02"), "breadth_context_source_date"] == pd.Timestamp("2024-01-02")
    assert by_date.loc[pd.Timestamp("2024-01-04"), "breadth_context_source_date"] == pd.Timestamp("2024-01-04")


def test_breadth_missing_zero_is_distinguishable_from_measured_zero() -> None:
    df = _base_ohlcv(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    breadth = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "market_breadth": [0.0],
            "up_volume": [0.0],
            "down_volume": [0.0],
        }
    )

    result = apply_context_features(df, "AAA", breadth_df=breadth)
    by_date = result.set_index("date")

    assert by_date.loc[pd.Timestamp("2024-01-02"), "market_breadth"] == 0.0
    assert bool(by_date.loc[pd.Timestamp("2024-01-02"), "breadth_context_available"]) is True
    assert bool(by_date.loc[pd.Timestamp("2024-01-02"), "breadth_context_missing"]) is False

    assert by_date.loc[pd.Timestamp("2024-01-03"), "market_breadth"] == 0.0
    assert bool(by_date.loc[pd.Timestamp("2024-01-03"), "breadth_context_available"]) is False
    assert bool(by_date.loc[pd.Timestamp("2024-01-03"), "breadth_context_missing"]) is True
    assert pd.isna(by_date.loc[pd.Timestamp("2024-01-03"), "breadth_context_source_date"])


def test_foreign_flow_exact_ticker_date_join_sets_availability_metadata() -> None:
    df = _base_ohlcv(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    foreign_flow = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "AAA"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "foreign_net_value": [0.0, 99_000.0, 10_000.0],
        }
    )

    result = apply_context_features(df, "AAA", foreign_flow_df=foreign_flow)
    by_date = result.set_index("date")

    assert by_date.loc[pd.Timestamp("2024-01-02"), "foreign_net_value"] == 0.0
    assert bool(by_date.loc[pd.Timestamp("2024-01-02"), "foreign_flow_context_available"]) is True
    assert bool(by_date.loc[pd.Timestamp("2024-01-03"), "foreign_flow_context_available"]) is False
    assert bool(by_date.loc[pd.Timestamp("2024-01-03"), "foreign_flow_context_missing"]) is True
    assert by_date.loc[pd.Timestamp("2024-01-03"), "foreign_net_value"] == 0.0
    assert by_date.loc[pd.Timestamp("2024-01-04"), "foreign_flow_context_source_date"] == pd.Timestamp("2024-01-04")


def test_foreign_flow_provenance_columns_are_not_joined_as_features() -> None:
    df = _base_ohlcv(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    foreign_flow = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "date": pd.to_datetime(["2024-01-02"]),
            "foreign_net_value": [12_000.0],
            "source": ["vnstock_data.Trading.foreign_trade"],
            "source_date": pd.to_datetime(["2024-01-02"]),
            "retrieved_at": ["2026-04-26T16:03:56Z"],
            "provider": ["vnstock_data"],
            "coverage_note": ["provider-backed test row"],
        }
    )

    result = apply_context_features(df, "AAA", foreign_flow_df=foreign_flow)

    assert {"source", "source_date", "retrieved_at", "provider", "coverage_note"}.isdisjoint(result.columns)
    assert result.loc[0, "foreign_net_value"] == 12_000.0
    assert result.loc[1, "foreign_net_value"] == 0.0
    assert bool(result.loc[1, "foreign_flow_context_missing"]) is True


def test_future_dated_context_rows_are_not_pulled_backward() -> None:
    df = _base_ohlcv(pd.to_datetime(["2024-01-02"]))
    future_date = pd.Timestamp("2024-01-03")
    breadth = pd.DataFrame({"date": [future_date], "market_breadth": [0.9]})
    foreign_flow = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "date": [future_date],
            "foreign_net_value": [100_000.0],
        }
    )

    result = apply_context_features(df, "AAA", breadth_df=breadth, foreign_flow_df=foreign_flow)
    row = result.iloc[0]

    assert row["market_breadth"] == 0.0
    assert bool(row["breadth_context_available"]) is False
    assert bool(row["foreign_flow_context_available"]) is False
    assert row["foreign_net_value"] == 0.0


def test_context_metadata_columns_are_excluded_from_model_features() -> None:
    df = _base_ohlcv(pd.bdate_range("2024-01-02", periods=70))
    breadth = pd.DataFrame(
        {
            "date": df["date"],
            "market_breadth": np.linspace(-0.2, 0.2, len(df)),
            "breadth_member_count": 50,
        }
    )
    foreign_flow = pd.DataFrame(
        {
            "ticker": "AAA",
            "date": df["date"],
            "foreign_net_value": np.linspace(-1_000.0, 1_000.0, len(df)),
        }
    )
    contextual = apply_context_features(df, "AAA", breadth_df=breadth, foreign_flow_df=foreign_flow)
    feature_frame = FeatureEngineer().build_feature_frame(contextual, build_mode="fast_core_mode")

    feature_columns = FeatureEngineer().get_feature_columns(feature_frame)

    assert CONTEXT_METADATA_COLUMNS <= set(feature_frame.columns)
    assert CONTEXT_METADATA_COLUMNS.isdisjoint(feature_columns)
    assert bool(feature_frame.loc[0, "breadth_context_available"]) is True
    assert feature_frame.loc[0, "breadth_context_source_date"] == df.loc[0, "date"]


def test_feature_governance_references_context_availability_metadata() -> None:
    breadth_review = classify_feature_governance("breadth_thrust_10")
    foreign_review = classify_feature_governance("foreign_net_value_ratio")

    assert "breadth_context_available" in breadth_review["source_hint"]
    assert "missing-context fallback" in breadth_review["reason"]
    assert "foreign_flow_context_available" in foreign_review["source_hint"]
    assert "missing-context fallback" in foreign_review["reason"]
