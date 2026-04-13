"""Core rolling risk analytics with time-series-safe alignment."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _as_series(values: pd.Series | Iterable[float], name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(list(values))
    series.name = name
    return pd.to_numeric(series, errors="coerce")


def build_system_return_series(
    asset_returns: pd.Series | pd.DataFrame,
    market_returns: pd.Series | None = None,
) -> pd.Series:
    """Use market returns when provided, otherwise fall back to equal-weight asset returns."""

    if market_returns is not None and not market_returns.empty:
        system = pd.to_numeric(market_returns.copy(), errors="coerce")
        system.name = "system_return"
        return system

    if isinstance(asset_returns, pd.DataFrame):
        numeric = asset_returns.apply(pd.to_numeric, errors="coerce")
        system = numeric.mean(axis=1)
    else:
        system = _as_series(asset_returns, name="system_return")
    system.name = "system_return"
    return system


def compute_historical_var(
    returns: pd.Series | Iterable[float],
    *,
    window: int = 60,
    quantile: float = 0.05,
    min_periods: int | None = None,
) -> pd.Series:
    series = _as_series(returns, name="var_q")
    min_periods = min_periods or window
    return series.rolling(window=window, min_periods=min_periods).quantile(quantile).rename("var_q")


def compute_cvar(
    returns: pd.Series | Iterable[float],
    *,
    window: int = 60,
    quantile: float = 0.05,
    min_periods: int | None = None,
) -> pd.Series:
    series = _as_series(returns, name="cvar_q")
    min_periods = min_periods or window

    def _tail_mean(values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return np.nan
        cutoff = np.nanquantile(arr, quantile)
        tail = arr[arr <= cutoff]
        if len(tail) == 0:
            return float(cutoff)
        return float(np.nanmean(tail))

    return series.rolling(window=window, min_periods=min_periods).apply(_tail_mean, raw=True).rename("cvar_q")


def _simulate_paths(
    series: pd.Series,
    *,
    window: int,
    simulations: int,
    random_seed: int,
    min_periods: int,
) -> tuple[np.ndarray, np.ndarray]:
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean().to_numpy(dtype=float)
    rolling_std = series.rolling(window=window, min_periods=min_periods).std(ddof=0).to_numpy(dtype=float)
    valid = np.isfinite(rolling_mean) & np.isfinite(rolling_std) & (rolling_std > 0)
    rng = np.random.default_rng(random_seed)
    z = rng.standard_normal((len(series), simulations))
    scenarios = np.full((len(series), simulations), np.nan, dtype=float)
    scenarios[valid] = rolling_mean[valid, None] + rolling_std[valid, None] * z[valid]
    return scenarios, valid


def compute_monte_carlo_var(
    returns: pd.Series | Iterable[float],
    *,
    window: int = 60,
    quantile: float = 0.05,
    simulations: int = 2000,
    random_seed: int = 42,
    min_periods: int | None = None,
) -> pd.Series:
    series = _as_series(returns, name="mc_var_q")
    min_periods = min_periods or window
    scenarios, valid = _simulate_paths(
        series,
        window=window,
        simulations=simulations,
        random_seed=random_seed,
        min_periods=min_periods,
    )
    values = np.full(len(series), np.nan, dtype=float)
    if np.any(valid):
        values[valid] = np.nanquantile(scenarios[valid], quantile, axis=1)
    return pd.Series(values, index=series.index, name="mc_var_q")


def compute_monte_carlo_cvar(
    returns: pd.Series | Iterable[float],
    *,
    window: int = 60,
    quantile: float = 0.05,
    simulations: int = 2000,
    random_seed: int = 42,
    min_periods: int | None = None,
) -> pd.Series:
    series = _as_series(returns, name="mc_cvar_q")
    min_periods = min_periods or window
    scenarios, valid = _simulate_paths(
        series,
        window=window,
        simulations=simulations,
        random_seed=random_seed,
        min_periods=min_periods,
    )
    values = np.full(len(series), np.nan, dtype=float)
    if np.any(valid):
        var_vals = np.nanquantile(scenarios[valid], quantile, axis=1)
        tail_mask = scenarios[valid] <= var_vals[:, None]
        tail_sum = np.where(tail_mask, scenarios[valid], 0.0).sum(axis=1)
        tail_count = tail_mask.sum(axis=1).clip(min=1)
        values[valid] = tail_sum / tail_count
    return pd.Series(values, index=series.index, name="mc_cvar_q")


def compute_drawdown(returns: pd.Series | Iterable[float]) -> pd.Series:
    series = _as_series(returns, name="drawdown").fillna(0.0)
    equity_curve = (1.0 + series).cumprod()
    running_peak = equity_curve.cummax()
    return (equity_curve / running_peak - 1.0).rename("drawdown")


def compute_max_drawdown_series(returns: pd.Series | Iterable[float]) -> pd.Series:
    return compute_drawdown(returns).cummin().rename("max_drawdown")


def compute_rolling_drawdown(
    returns: pd.Series | Iterable[float],
    *,
    window: int = 60,
    min_periods: int | None = None,
) -> pd.Series:
    series = _as_series(returns, name="rolling_drawdown").fillna(0.0)
    min_periods = min_periods or window
    equity_curve = (1.0 + series).cumprod()
    rolling_peak = equity_curve.rolling(window=window, min_periods=min_periods).max()
    return (equity_curve / rolling_peak - 1.0).rename("rolling_drawdown")
