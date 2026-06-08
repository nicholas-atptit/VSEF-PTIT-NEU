from __future__ import annotations

import pandas as pd


def random_walk(close: pd.Series) -> pd.Series:
    return pd.to_numeric(close, errors="coerce").shift(1)


def last_close(close: pd.Series) -> pd.Series:
    return random_walk(close)


def historical_mean_return(returns: pd.Series) -> pd.Series:
    return pd.to_numeric(returns, errors="coerce").expanding(min_periods=1).mean().shift(1)


def rolling_mean_return(returns: pd.Series, window: int = 20) -> pd.Series:
    return pd.to_numeric(returns, errors="coerce").rolling(window, min_periods=1).mean().shift(1)


def rolling_median_return(returns: pd.Series, window: int = 20) -> pd.Series:
    return pd.to_numeric(returns, errors="coerce").rolling(window, min_periods=1).median().shift(1)
