"""Price-range and prediction-interval metrics."""

from __future__ import annotations

import math

import numpy as np

from ._common import paired_numeric


def pinball_loss(actual: object, predicted_quantile: object, quantile: float) -> float:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    arrays = paired_numeric(actual, predicted_quantile)
    if not arrays or len(arrays[0]) == 0:
        return math.nan
    error = arrays[0] - arrays[1]
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def interval_metrics(actual: object, low: object, high: object, *, alpha: float = 0.1) -> dict[str, float | int]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    arrays = paired_numeric(actual, low, high)
    if not arrays or len(arrays[0]) == 0:
        return {"rows": 0, "interval_coverage": math.nan, "average_interval_width": math.nan, "low_breach_rate": math.nan, "high_breach_rate": math.nan, "winkler_score": math.nan, "lower_pinball_loss": math.nan, "upper_pinball_loss": math.nan}
    true, lower, upper = arrays
    if np.any(lower > upper):
        raise ValueError("low interval bound exceeds high interval bound")
    width = upper - lower
    low_breach = true < lower
    high_breach = true > upper
    winkler = width + (2.0 / alpha) * (lower - true) * low_breach + (2.0 / alpha) * (true - upper) * high_breach
    return {
        "rows": int(len(true)),
        "interval_coverage": float(np.mean((true >= lower) & (true <= upper))),
        "average_interval_width": float(np.mean(width)),
        "low_breach_rate": float(np.mean(low_breach)),
        "high_breach_rate": float(np.mean(high_breach)),
        "winkler_score": float(np.mean(winkler)),
        "lower_pinball_loss": pinball_loss(true, lower, alpha / 2.0),
        "upper_pinball_loss": pinball_loss(true, upper, 1.0 - alpha / 2.0),
    }
