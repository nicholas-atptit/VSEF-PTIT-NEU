"""Trailing and lagged feature builders that do not read future rows."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _ordered(frame: pd.DataFrame, asset_column: str, timestamp_column: str) -> pd.DataFrame:
    return frame.sort_values([asset_column, timestamp_column]).copy()


def build_momentum_features(frame: pd.DataFrame, *, windows: Iterable[int] = (3, 5, 10, 20), asset_column: str = "asset_code", timestamp_column: str = "asof_timestamp", close_column: str = "close") -> pd.DataFrame:
    out = _ordered(frame, asset_column, timestamp_column)
    grouped = out.groupby(asset_column, group_keys=False)[close_column]
    for window in windows:
        out[f"momentum_{window}"] = grouped.pct_change(window, fill_method=None).shift(1)
    return out


def build_relative_strength_features(frame: pd.DataFrame, *, benchmark_return_columns: Iterable[str], return_column: str = "return_1_lag_1") -> pd.DataFrame:
    out = frame.copy()
    for column in benchmark_return_columns:
        out[f"relative_strength_vs_{column}"] = out[return_column] - out[column]
    return out


def build_market_context_features(frame: pd.DataFrame, *, market_code: str = "VNINDEX", asset_column: str = "asset_code", timestamp_column: str = "asof_timestamp", close_column: str = "close") -> pd.DataFrame:
    out = _ordered(frame, asset_column, timestamp_column)
    market = out[out[asset_column].eq(market_code)][[timestamp_column, close_column]].copy()
    market[f"{market_code}_lag_return_1"] = market[close_column].pct_change(fill_method=None).shift(1)
    return out.merge(market[[timestamp_column, f"{market_code}_lag_return_1"]], on=timestamp_column, how="left")


def build_volume_volatility_features(frame: pd.DataFrame, *, windows: Iterable[int] = (5, 10, 20), asset_column: str = "asset_code", timestamp_column: str = "asof_timestamp", close_column: str = "close", volume_column: str = "volume") -> pd.DataFrame:
    out = _ordered(frame, asset_column, timestamp_column)
    grouped = out.groupby(asset_column, group_keys=False)
    returns = grouped[close_column].pct_change(fill_method=None)
    for window in windows:
        out[f"rolling_volatility_{window}"] = returns.groupby(out[asset_column]).rolling(window, min_periods=2).std().reset_index(level=0, drop=True).shift(1)
        trailing_volume = grouped[volume_column].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
        out[f"volume_ratio_{window}"] = out[volume_column] / trailing_volume.replace(0.0, np.nan)
    return out


def build_range_features(frame: pd.DataFrame, *, windows: Iterable[int] = (5, 10, 20), asset_column: str = "asset_code", timestamp_column: str = "asof_timestamp") -> pd.DataFrame:
    out = _ordered(frame, asset_column, timestamp_column)
    out["range_pct"] = (out["high"] - out["low"]) / out["close"].replace(0.0, np.nan)
    grouped_range = out.groupby(asset_column)["range_pct"]
    for window in windows:
        out[f"range_mean_{window}"] = grouped_range.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
    return out


def build_combined_features(frame: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
    out = build_momentum_features(frame, **kwargs)
    out = build_volume_volatility_features(out, **kwargs)
    return build_range_features(out, **kwargs)
