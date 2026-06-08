"""Reusable offline forecast baselines."""

from .direction import always_down, always_up, lag1_direction, majority_class
from .range_interval import atr_range_band, historical_quantile_range_band, rolling_volatility_range_band
from .return_price import historical_mean_return, last_close, random_walk, rolling_mean_return, rolling_median_return

__all__ = [
    "always_down",
    "always_up",
    "atr_range_band",
    "historical_mean_return",
    "historical_quantile_range_band",
    "lag1_direction",
    "last_close",
    "majority_class",
    "random_walk",
    "rolling_mean_return",
    "rolling_median_return",
    "rolling_volatility_range_band",
]
