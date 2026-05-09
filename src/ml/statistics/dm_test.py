"""Diebold-Mariano forecast comparison tests.

The helpers in this module are intentionally conservative. They drop missing
error pairs, report the dropped count, and return warnings instead of p-values
when the available sample or loss-differential variance is not adequate.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

try:  # pragma: no cover - covered by runtime environment rather than unit tests
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None


MIN_SAMPLE_SIZE = 10


def squared_error(errors: Iterable[float]) -> np.ndarray:
    """Return squared-error losses for forecast errors."""
    values = _as_float_array(errors)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.square(values)


def absolute_error(errors: Iterable[float]) -> np.ndarray:
    """Return absolute-error losses for forecast errors."""
    values = _as_float_array(errors)
    return np.abs(values)


def dm_test_from_errors(
    errors_model: Iterable[float],
    errors_baseline: Iterable[float],
    loss: str = "squared",
    horizon: int = 1,
    alternative: str = "two_sided",
) -> dict[str, Any]:
    """Alias for :func:`diebold_mariano_test` for call-site readability."""
    return diebold_mariano_test(
        errors_model,
        errors_baseline,
        loss=loss,
        horizon=horizon,
        alternative=alternative,
    )


def diebold_mariano_test(
    errors_model: Iterable[float],
    errors_baseline: Iterable[float],
    loss: str = "squared",
    horizon: int = 1,
    alternative: str = "two_sided",
) -> dict[str, Any]:
    """Compare model and baseline forecast losses with a DM statistic.

    Negative ``mean_loss_diff`` means the model has lower average loss than the
    baseline. ``alternative="less"`` tests that model loss is lower than
    baseline loss. The default two-sided test is used by Phase 6 artifacts.
    """
    loss_name = _normalize_loss(loss)
    alternative_name = _normalize_alternative(alternative)
    horizon_value = max(int(horizon or 1), 1)

    model = _as_float_array(errors_model)
    baseline = _as_float_array(errors_baseline)
    pair_count = int(min(len(model), len(baseline)))
    model = model[:pair_count]
    baseline = baseline[:pair_count]

    valid_mask = np.isfinite(model) & np.isfinite(baseline)
    dropped_count = int(pair_count - valid_mask.sum())
    model = model[valid_mask]
    baseline = baseline[valid_mask]
    sample_size = int(len(model))

    base_result: dict[str, Any] = {
        "dm_statistic": None,
        "p_value": None,
        "mean_loss_model": None,
        "mean_loss_baseline": None,
        "mean_loss_diff": None,
        "effect_size": None,
        "alternative": alternative_name,
        "loss": loss_name,
        "horizon": horizon_value,
        "sample_size": sample_size,
        "dropped_count": dropped_count,
        "significant_05": False,
        "significant_10": False,
        "warning": "",
    }

    if sample_size < MIN_SAMPLE_SIZE:
        base_result["warning"] = f"insufficient_sample_size:{sample_size}<" f"{MIN_SAMPLE_SIZE}"
        return base_result

    model_loss = _loss_values(model, loss_name)
    baseline_loss = _loss_values(baseline, loss_name)
    diff = model_loss - baseline_loss
    diff = diff[np.isfinite(diff)]
    sample_size = int(len(diff))
    if sample_size < MIN_SAMPLE_SIZE:
        base_result["sample_size"] = sample_size
        base_result["warning"] = f"insufficient_finite_loss_diff:{sample_size}<" f"{MIN_SAMPLE_SIZE}"
        return base_result

    mean_model = float(np.mean(model_loss))
    mean_baseline = float(np.mean(baseline_loss))
    mean_diff = float(np.mean(diff))
    with np.errstate(over="ignore", invalid="ignore"):
        diff_std = float(np.std(diff, ddof=1))
    base_result.update(
        {
            "mean_loss_model": mean_model,
            "mean_loss_baseline": mean_baseline,
            "mean_loss_diff": mean_diff,
            "effect_size": _safe_divide(mean_diff, diff_std),
            "sample_size": sample_size,
        }
    )

    variance = _newey_west_long_run_variance(diff, max_lag=min(horizon_value - 1, sample_size - 1))
    if not math.isfinite(variance) or variance <= 0.0:
        base_result["warning"] = "invalid_or_zero_loss_differential_variance"
        return base_result

    dm_statistic = mean_diff / math.sqrt(variance / sample_size)
    p_value = _p_value(dm_statistic, sample_size, alternative_name)
    base_result.update(
        {
            "dm_statistic": float(dm_statistic),
            "p_value": None if p_value is None else float(p_value),
            "significant_05": bool(p_value is not None and p_value < 0.05),
            "significant_10": bool(p_value is not None and p_value < 0.10),
        }
    )
    if dropped_count:
        base_result["warning"] = f"dropped_missing_pairs:{dropped_count}"
    return base_result


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def _normalize_loss(loss: str) -> str:
    normalized = str(loss or "squared").strip().lower()
    aliases = {"se": "squared", "squared_error": "squared", "ae": "absolute", "absolute_error": "absolute"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"squared", "absolute"}:
        raise ValueError("loss must be one of: squared, absolute")
    return normalized


def _normalize_alternative(alternative: str) -> str:
    normalized = str(alternative or "two_sided").strip().lower()
    aliases = {"two-sided": "two_sided", "less_than": "less", "greater_than": "greater"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"two_sided", "less", "greater"}:
        raise ValueError("alternative must be one of: two_sided, less, greater")
    return normalized


def _loss_values(errors: np.ndarray, loss: str) -> np.ndarray:
    if loss == "squared":
        return squared_error(errors)
    if loss == "absolute":
        return absolute_error(errors)
    raise ValueError("unsupported loss")


def _newey_west_long_run_variance(diff: np.ndarray, max_lag: int) -> float:
    centered = diff - np.mean(diff)
    n = len(centered)
    if n <= 1:
        return float("nan")
    with np.errstate(over="ignore", invalid="ignore"):
        gamma0 = float(np.dot(centered, centered) / n)
    variance = gamma0
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        with np.errstate(over="ignore", invalid="ignore"):
            autocov = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - (lag / (max_lag + 1.0))
        variance += 2.0 * weight * autocov
    return variance


def _p_value(statistic: float, sample_size: int, alternative: str) -> float | None:
    if not math.isfinite(statistic):
        return None
    if scipy_stats is not None:
        dist = scipy_stats.t(df=max(sample_size - 1, 1))
        if alternative == "two_sided":
            return float(2.0 * dist.sf(abs(statistic)))
        if alternative == "less":
            return float(dist.cdf(statistic))
        return float(dist.sf(statistic))

    cdf = _normal_cdf(statistic)
    if alternative == "two_sided":
        return float(2.0 * min(cdf, 1.0 - cdf))
    if alternative == "less":
        return float(cdf)
    return float(1.0 - cdf)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
        return None
    return float(numerator / denominator)
