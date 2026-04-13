"""Rolling CoVaR and Delta-CoVaR estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core_risk import compute_historical_var

try:
    import statsmodels.api as sm

    HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    HAS_STATSMODELS = False


def compute_var(
    returns: pd.Series,
    *,
    window: int = 60,
    quantile: float = 0.05,
    min_periods: int | None = None,
) -> pd.Series:
    return compute_historical_var(returns, window=window, quantile=quantile, min_periods=min_periods).rename("var_q")


def _rolling_quantile_regression_components(
    asset_returns: pd.Series,
    system_returns: pd.Series,
    *,
    window: int,
    quantile: float,
) -> tuple[pd.Series, pd.Series]:
    asset, system = asset_returns.align(system_returns, join="inner")
    distress = pd.Series(np.nan, index=asset.index, name="covar_q")
    baseline = pd.Series(np.nan, index=asset.index, name="covar_median")
    asset_var = compute_var(asset, window=window, quantile=quantile)

    for end in range(window - 1, len(asset)):
        start = end - window + 1
        x_window = asset.iloc[start : end + 1]
        y_window = system.iloc[start : end + 1]
        mask = x_window.notna() & y_window.notna()
        if int(mask.sum()) < max(20, window // 2):
            continue
        x_values = x_window.loc[mask].to_numpy(dtype=float)
        y_values = y_window.loc[mask].to_numpy(dtype=float)
        design = sm.add_constant(x_values, has_constant="add")
        try:
            fitted = sm.QuantReg(y_values, design).fit(q=quantile, max_iter=1000, p_tol=1e-6)
        except Exception:
            continue

        current_var = asset_var.iloc[end]
        current_median = float(np.nanmedian(x_values))
        distress.iloc[end] = float(fitted.predict([1.0, current_var])[0])
        baseline.iloc[end] = float(fitted.predict([1.0, current_median])[0])

    return distress, baseline


def _rolling_quantile_approximation_components(
    asset_returns: pd.Series,
    system_returns: pd.Series,
    *,
    window: int,
    quantile: float,
) -> tuple[pd.Series, pd.Series]:
    asset, system = asset_returns.align(system_returns, join="inner")
    distress = pd.Series(np.nan, index=asset.index, name="covar_q")
    baseline = pd.Series(np.nan, index=asset.index, name="covar_median")

    for end in range(window - 1, len(asset)):
        start = end - window + 1
        a_window = asset.iloc[start : end + 1]
        s_window = system.iloc[start : end + 1]
        mask = a_window.notna() & s_window.notna()
        if int(mask.sum()) < max(20, window // 2):
            continue

        a_values = a_window.loc[mask].to_numpy(dtype=float)
        s_values = s_window.loc[mask].to_numpy(dtype=float)

        asset_var = float(np.nanquantile(a_values, quantile))
        asset_median = float(np.nanmedian(a_values))
        median_band = max(float(np.nanstd(a_values)) * 0.25, 1e-9)

        distress_mask = a_values <= asset_var
        baseline_mask = np.abs(a_values - asset_median) <= median_band
        if not np.any(distress_mask):
            distress_mask = a_values <= np.nanquantile(a_values, min(quantile * 1.5, 0.2))
        if not np.any(baseline_mask):
            baseline_mask = np.abs(a_values - asset_median) == np.abs(a_values - asset_median).min()

        distress.iloc[end] = float(np.nanquantile(s_values[distress_mask], quantile))
        baseline.iloc[end] = float(np.nanquantile(s_values[baseline_mask], quantile))

    return distress, baseline


def _compute_covar_components(
    asset_returns: pd.Series,
    system_returns: pd.Series,
    *,
    window: int = 60,
    quantile: float = 0.05,
) -> tuple[pd.Series, pd.Series]:
    if HAS_STATSMODELS:
        distress, baseline = _rolling_quantile_regression_components(
            asset_returns,
            system_returns,
            window=window,
            quantile=quantile,
        )
        if distress.notna().sum() > 0:
            return distress.rename("covar_q"), baseline.rename("covar_median")
    distress, baseline = _rolling_quantile_approximation_components(
        asset_returns,
        system_returns,
        window=window,
        quantile=quantile,
    )
    return distress.rename("covar_q"), baseline.rename("covar_median")


def compute_covar(
    asset_returns: pd.Series,
    system_returns: pd.Series,
    *,
    window: int = 60,
    quantile: float = 0.05,
) -> pd.Series:
    distress, _ = _compute_covar_components(
        asset_returns,
        system_returns,
        window=window,
        quantile=quantile,
    )
    return distress.rename("covar_q")


def compute_delta_covar(
    asset_returns: pd.Series,
    system_returns: pd.Series,
    *,
    window: int = 60,
    quantile: float = 0.05,
) -> pd.Series:
    distress, baseline = _compute_covar_components(
        asset_returns,
        system_returns,
        window=window,
        quantile=quantile,
    )
    return (distress - baseline).rename("delta_covar")
