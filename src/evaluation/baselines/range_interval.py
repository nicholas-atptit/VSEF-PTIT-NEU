from __future__ import annotations

import pandas as pd


def atr_range_band(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14, multiplier: float = 1.0) -> pd.DataFrame:
    previous_close = pd.to_numeric(close, errors="coerce").shift(1)
    true_range = pd.concat([(high - low).abs(), (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
    width = true_range.rolling(window, min_periods=1).mean().shift(1) * multiplier
    return pd.DataFrame({"low": previous_close - width, "high": previous_close + width})


def rolling_volatility_range_band(close: pd.Series, window: int = 20, multiplier: float = 1.96) -> pd.DataFrame:
    numeric = pd.to_numeric(close, errors="coerce")
    center = numeric.shift(1)
    width = numeric.pct_change(fill_method=None).rolling(window, min_periods=2).std().shift(1) * center * multiplier
    return pd.DataFrame({"low": center - width, "high": center + width})


def historical_quantile_range_band(returns: pd.Series, close: pd.Series, window: int = 60, lower: float = 0.05, upper: float = 0.95) -> pd.DataFrame:
    numeric_returns = pd.to_numeric(returns, errors="coerce")
    center = pd.to_numeric(close, errors="coerce").shift(1)
    low_return = numeric_returns.rolling(window, min_periods=5).quantile(lower).shift(1)
    high_return = numeric_returns.rolling(window, min_periods=5).quantile(upper).shift(1)
    return pd.DataFrame({"low": center * (1.0 + low_return), "high": center * (1.0 + high_return)})
