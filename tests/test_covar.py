from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.risk.covar import compute_covar, compute_delta_covar, compute_var


def _make_returns(n_rows: int = 160) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2025-01-01", periods=n_rows)
    system = pd.Series(rng.normal(0.0005, 0.012, n_rows), index=index, name="system")
    asset = pd.Series(0.8 * system + rng.normal(0.0, 0.01, n_rows), index=index, name="asset")
    shock_start = min(70, max(n_rows - 5, 0))
    shock_end = min(shock_start + 5, n_rows)
    if shock_end > shock_start:
        asset.iloc[shock_start:shock_end] -= 0.03
        system.iloc[shock_start:shock_end] -= 0.015
    return asset, system


def test_covar_alignment_and_window_behavior() -> None:
    asset, system = _make_returns()

    var_q = compute_var(asset, window=40, quantile=0.05)
    covar_q = compute_covar(asset, system, window=40, quantile=0.05)
    delta = compute_delta_covar(asset, system, window=40, quantile=0.05)

    assert list(var_q.index) == list(asset.index)
    assert list(covar_q.index) == list(asset.index)
    assert list(delta.index) == list(asset.index)
    assert var_q.iloc[:39].isna().all()
    assert covar_q.iloc[:39].isna().all()
    assert delta.iloc[:39].isna().all()
    assert covar_q.iloc[80:].notna().any()
    assert delta.iloc[80:].notna().any()


def test_covar_no_future_leakage_prefix_invariance() -> None:
    asset, system = _make_returns()
    full = compute_delta_covar(asset, system, window=35, quantile=0.05)
    prefix = compute_delta_covar(asset.iloc[:120], system.iloc[:120], window=35, quantile=0.05)

    np.testing.assert_allclose(
        full.iloc[:120].to_numpy(dtype=float),
        prefix.to_numpy(dtype=float),
        equal_nan=True,
    )


def test_covar_small_dataset_is_stable() -> None:
    asset, system = _make_returns(n_rows=12)
    covar_q = compute_covar(asset, system, window=30, quantile=0.05)
    delta = compute_delta_covar(asset, system, window=30, quantile=0.05)

    assert len(covar_q) == 12
    assert len(delta) == 12
    assert covar_q.isna().all()
    assert delta.isna().all()
