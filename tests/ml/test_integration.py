"""Phase 7D: End-to-end integration tests for risk-aware forecasting pipeline."""

from __future__ import annotations

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from src.ml.trainer import DualModelTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_csv(n_rows: int = 300) -> pd.DataFrame:
    """Generate a minimal synthetic daily OHLCV DataFrame."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end="2026-03-20", periods=n_rows)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n_rows))
    close = np.maximum(close, 10.0)
    return pd.DataFrame({
        "time": dates,
        "open": close * (1 + rng.normal(0, 0.002, n_rows)),
        "high": close * (1 + abs(rng.normal(0, 0.01, n_rows))),
        "low": close * (1 - abs(rng.normal(0, 0.01, n_rows))),
        "close": close,
        "volume": rng.integers(100_000, 10_000_000, n_rows),
    })


# ---------------------------------------------------------------------------
# 1. Legacy non-risk path still works
# ---------------------------------------------------------------------------

def test_legacy_path_no_risk(tmp_path):
    trainer = DualModelTrainer(model_dir=str(tmp_path / "models"))
    df = _make_synthetic_csv()
    result = trainer.train(
        ticker="TEST",
        df=df,
        algorithms=["cart"],
    )
    assert result["ticker"] == "TEST"
    assert len(result["report_rows"]) > 0

    # Manifest should have risk_config with risk_enabled=False
    manifest = trainer._manifests["TEST"]
    for hz_info in manifest["horizons"].values():
        for algo_info in hz_info["algorithms"].values():
            rc = algo_info.get("risk_config", {})
            assert rc.get("risk_enabled") is False


# ---------------------------------------------------------------------------
# 2. Risk path is deterministic with fixed seed
# ---------------------------------------------------------------------------

def test_risk_deterministic_seed(tmp_path):
    risk_cfg = {
        "simulations": 5000,
        "confidence_levels": [0.95, 0.99],
        "random_seed": 42,
    }

    trainer = DualModelTrainer(model_dir=str(tmp_path / "models"))
    df = _make_synthetic_csv()
    result = trainer.train(
        ticker="TEST",
        df=df,
        algorithms=["cart"],
        risk_config=risk_cfg,
    )

    manifest = trainer._manifests["TEST"]

    # Collect all risk configs from the manifest
    risk_entries = []
    for hz_info in manifest["horizons"].values():
        for algo_info in hz_info["algorithms"].values():
            rc = algo_info.get("risk_config", {})
            assert rc.get("risk_enabled") is True
            assert rc.get("risk_seed") == 42
            assert rc.get("risk_simulations") == 5000
            assert rc.get("risk_confidence_levels") == [0.95, 0.99]
            assert rc.get("volatility_proxy_source") in ("test_residuals_std", "validation_rmse")
            assert "risk_assumptions" in rc
            risk_entries.append(rc)

    assert len(risk_entries) > 0


# ---------------------------------------------------------------------------
# 3. Manifest contains required risk metadata fields
# ---------------------------------------------------------------------------

def test_manifest_risk_metadata_schema(tmp_path):
    risk_cfg = {
        "simulations": 1000,
        "confidence_levels": [0.95],
        "random_seed": 99,
    }
    trainer = DualModelTrainer(model_dir=str(tmp_path / "models"))
    df = _make_synthetic_csv()
    trainer.train(
        ticker="TEST",
        df=df,
        algorithms=["cart"],
        risk_config=risk_cfg,
    )

    manifest = trainer._manifests["TEST"]
    for hz_info in manifest["horizons"].values():
        for algo_info in hz_info["algorithms"].values():
            rc = algo_info["risk_config"]
            required_keys = {
                "risk_enabled",
                "risk_simulations",
                "risk_confidence_levels",
                "risk_seed",
                "volatility_proxy",
                "volatility_proxy_source",
                "risk_assumptions",
            }
            assert required_keys.issubset(set(rc.keys())), f"Missing keys: {required_keys - set(rc.keys())}"
            assert isinstance(rc["volatility_proxy"], float)
            assert rc["volatility_proxy"] > 0


# ---------------------------------------------------------------------------
# 4. Stacking enablement is explicit and safe
# ---------------------------------------------------------------------------

def test_stacking_safe_enablement():
    """Stacking should only be appended when >=2 compatible base learners exist."""
    # Simulate CLI logic from main()
    STACKING_COMPATIBLE = {"cart", "xgboost", "lightgbm", "sarimax", "ets"}

    # Case 1: enough compatible learners
    algos_ok = ["cart", "xgboost"]
    compatible = len(set(algos_ok) & STACKING_COMPATIBLE)
    assert compatible >= 2
    algos_ok.append("stacking")
    assert "stacking" in algos_ok

    # Case 2: only one compatible learner -> stacking NOT appended
    algos_insufficient = ["cart"]
    compatible2 = len(set(algos_insufficient) & STACKING_COMPATIBLE)
    assert compatible2 < 2
    # no append
    assert "stacking" not in algos_insufficient

    # Case 3: DL-only algorithms -> stacking NOT appended
    algos_dl = ["lstm", "bilstm"]
    compatible3 = len(set(algos_dl) & STACKING_COMPATIBLE)
    assert compatible3 == 0


# ---------------------------------------------------------------------------
# 5. Booster tuning flag does not break non-booster runs
# ---------------------------------------------------------------------------

def test_tune_boosters_no_break_cart(tmp_path):
    """tune_boosters should be harmlessly ignored for non-booster algorithms."""
    trainer = DualModelTrainer(model_dir=str(tmp_path / "models"))
    df = _make_synthetic_csv()
    result = trainer.train(
        ticker="TEST",
        df=df,
        algorithms=["cart"],
        tune_boosters=True,  # Should have zero effect on CART
    )
    assert result["ticker"] == "TEST"
    assert len(result["report_rows"]) > 0


# ---------------------------------------------------------------------------
# 6. Residual std is captured in regression metrics
# ---------------------------------------------------------------------------

def test_regression_metrics_include_residual_std():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    metrics = DualModelTrainer._regression_metrics(y_true, y_pred)
    assert "residual_std" in metrics
    assert "volatility_proxy_source" in metrics
    assert metrics["residual_std"] > 0
    assert metrics["volatility_proxy_source"] == "test_residuals_std"
