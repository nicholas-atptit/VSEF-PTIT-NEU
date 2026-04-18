"""Custom Performance Metrics for Backtesting.

Calculates:
- Standby / Null Rate
- LLM Veto Rate
- Risk-Adjusted Returns (Sharpe, Sortino)
"""

from __future__ import annotations

import numpy as np
from src.ml.metrics import compute_max_drawdown, compute_sharpe_ratio, compute_sortino_ratio


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
    """Compatibility wrapper over the canonical ML metric helpers."""
    if not daily_returns or len(daily_returns) < 2:
        return {"error": 1.0}

    returns_array = np.array(daily_returns)
    avg_return = np.mean(returns_array)
    std_dev = np.std(returns_array)

    # Annualization factor (approx 252 trading days)
    annualized_return = avg_return * 252
    annualized_vol = std_dev * np.sqrt(252)

    # Sharpe Ratio
    sharpe_ratio = compute_sharpe_ratio(returns_array - (risk_free_rate / 252.0))

    # Sortino Ratio (only downside volatility)
    sortino_ratio = compute_sortino_ratio(returns_array - (risk_free_rate / 252.0))

    # Max Drawdown
    cumulative_returns = np.cumprod(1 + returns_array)
    max_drawdown = compute_max_drawdown(cumulative_returns)

    return {
        "annualized_return": round(annualized_return, 4),
        "annualized_volatility": round(annualized_vol, 4),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "sortino_ratio": round(sortino_ratio, 4),
        "max_drawdown": round(max_drawdown, 4),
    }
