"""Bootstrap confidence intervals for diagnostic evaluation metrics."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd


MIN_BOOTSTRAP_SAMPLE_SIZE = 2
SMALL_SAMPLE_WARNING_SIZE = 10


def bootstrap_mean_ci(
    values: Iterable[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate a percentile bootstrap CI for the arithmetic mean."""
    return bootstrap_metric_ci(
        values,
        metric_fn=lambda sample: float(np.mean(sample)),
        metric_name="mean",
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )


def bootstrap_hit_ratio_ci(
    outcomes: Iterable[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate a percentile bootstrap CI for positive-outcome hit ratio."""
    clean, dropped_count = _clean_values(outcomes)
    binary = (clean > 0).astype(float)
    result = bootstrap_metric_ci(
        binary,
        metric_fn=lambda sample: float(np.mean(sample)),
        metric_name="hit_ratio",
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    result["dropped_count"] = dropped_count
    return result


def bootstrap_metric_ci(
    values: Iterable[float],
    metric_fn: Callable[[np.ndarray], float],
    metric_name: str = "metric",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate a percentile bootstrap CI for an arbitrary numeric metric."""
    clean, dropped_count = _clean_values(values)
    sample_size = int(len(clean))
    result: dict[str, Any] = {
        "metric_name": metric_name,
        "estimate": None,
        "ci_lower": None,
        "ci_upper": None,
        "confidence": float(confidence),
        "n_bootstrap": int(n_bootstrap),
        "sample_size": sample_size,
        "seed": int(seed),
        "dropped_count": dropped_count,
        "warning": "",
    }
    if sample_size < MIN_BOOTSTRAP_SAMPLE_SIZE:
        result["warning"] = f"insufficient_sample_size:{sample_size}<" f"{MIN_BOOTSTRAP_SAMPLE_SIZE}"
        return result

    estimate = _safe_metric(metric_fn, clean)
    result["estimate"] = estimate
    if estimate is None:
        result["warning"] = "metric_estimate_not_finite"
        return result

    n_bootstrap = max(int(n_bootstrap), 1)
    rng = np.random.default_rng(int(seed))
    boot_values: list[float] = []
    for _ in range(n_bootstrap):
        sample = rng.choice(clean, size=sample_size, replace=True)
        value = _safe_metric(metric_fn, sample)
        if value is not None:
            boot_values.append(value)

    if not boot_values:
        result["warning"] = "no_finite_bootstrap_replicates"
        return result

    alpha = (1.0 - float(confidence)) / 2.0
    lower, upper = np.quantile(np.asarray(boot_values, dtype=float), [alpha, 1.0 - alpha])
    result["ci_lower"] = float(lower)
    result["ci_upper"] = float(upper)
    warnings: list[str] = []
    if sample_size < SMALL_SAMPLE_WARNING_SIZE:
        warnings.append(f"small_sample_ci_unstable:{sample_size}<" f"{SMALL_SAMPLE_WARNING_SIZE}")
    if len(boot_values) < n_bootstrap:
        warnings.append(f"dropped_nonfinite_bootstrap_replicates:{n_bootstrap - len(boot_values)}")
    if dropped_count:
        warnings.append(f"dropped_nan_values:{dropped_count}")
    result["warning"] = ";".join(warnings)
    return result


def bootstrap_from_dataframe(
    frame: pd.DataFrame,
    value_column: str,
    group_by: list[str] | tuple[str, ...] | None = None,
    metric_name: str = "mean",
    metric_fn: Callable[[np.ndarray], float] | None = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute bootstrap CIs for one value column, optionally by groups."""
    columns = [
        "group_key",
        "metric_name",
        "estimate",
        "ci_lower",
        "ci_upper",
        "confidence",
        "n_bootstrap",
        "sample_size",
        "seed",
        "warning",
    ]
    if frame is None or frame.empty or value_column not in frame.columns:
        return pd.DataFrame(columns=columns)

    fn = metric_fn or (lambda sample: float(np.mean(sample)))
    groups = list(group_by or [])
    rows: list[dict[str, Any]] = []
    if groups:
        iterator = frame.groupby(groups, dropna=False, sort=True)
    else:
        iterator = [("ALL", frame)]

    for keys, group in iterator:
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_parts = [f"{column}={value}" for column, value in zip(groups or ["ALL"], keys, strict=False)]
        result = bootstrap_metric_ci(
            group[value_column],
            metric_fn=fn,
            metric_name=metric_name,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed,
        )
        result["group_key"] = "|".join(key_parts) if key_parts else "ALL"
        rows.append(result)
    return pd.DataFrame(rows, columns=columns)


def _clean_values(values: Iterable[float]) -> tuple[np.ndarray, int]:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    clean = series.dropna().astype(float).to_numpy()
    return clean, int(series.isna().sum())


def _safe_metric(metric_fn: Callable[[np.ndarray], float], sample: np.ndarray) -> float | None:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(metric_fn(sample))
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value
