"""Custom Performance Metrics for Backtesting.

Calculates:
- Standby / Null Rate
- LLM Veto Rate
- Risk-Adjusted Returns (Sharpe, Sortino)
"""

from __future__ import annotations

import numpy as np


def calculate_veto_standby_rates(
    total_signals: int,
    standby_count: int,
    veto_count: int,
) -> dict[str, float | str]:
    """Calculate rates and issue warnings if thresholds are breached."""
    if total_signals == 0:
        return {"error": "No signals to measure"}

    standby_rate = standby_count / total_signals
    veto_rate = veto_count / total_signals

    warning = ""
    if standby_rate > 0.95:
        warning = "CRITICAL WARNING: Standby Rate > 95%. System is too cowardly. Sub-optimal parameterization."

    return {
        "total_signals_evaluated": total_signals,
        "standby_rate": round(standby_rate, 4),
        "veto_rate": round(veto_rate, 4),
        "warning": warning,
    }


def calculate_risk_adjusted_returns(
    daily_returns: list[float],
    risk_free_rate: float = 0.04,  # e.g., 4% bond yield
) -> dict[str, float]:
    """Calculate Sharpe, Sortino, and Max Drawdown.

    Requires an array of daily percentage returns (e.g. 0.015 for +1.5%).
    """
    if not daily_returns or len(daily_returns) < 2:
        return {"error": 1.0}

    returns_array = np.array(daily_returns)
    avg_return = np.mean(returns_array)
    std_dev = np.std(returns_array)

    # Annualization factor (approx 252 trading days)
    annualized_return = avg_return * 252
    annualized_vol = std_dev * np.sqrt(252)

    # Sharpe Ratio
    sharpe_ratio = 0.0
    if annualized_vol > 0:
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol

    # Sortino Ratio (only downside volatility)
    downside_returns = returns_array[returns_array < 0]
    sortino_ratio = 0.0
    if len(downside_returns) > 0:
        downside_std = np.std(downside_returns)
        annualized_downside_vol = downside_std * np.sqrt(252)
        if annualized_downside_vol > 0:
            sortino_ratio = (annualized_return - risk_free_rate) / annualized_downside_vol

    # Max Drawdown
    cumulative_returns = np.cumprod(1 + returns_array)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown)

    return {
        "annualized_return": round(annualized_return, 4),
        "annualized_volatility": round(annualized_vol, 4),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "sortino_ratio": round(sortino_ratio, 4),
        "max_drawdown": round(max_drawdown, 4),
    }
