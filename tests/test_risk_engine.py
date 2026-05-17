from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.data_loader import generate_mock_data
from src.ml.risk.risk_engine import RiskEngine
from src.ml.trainer import DualModelTrainer


def _asset_matrix(n_rows: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(21)
    index = pd.bdate_range("2025-01-01", periods=n_rows)
    base = rng.normal(0.0004, 0.01, n_rows)
    return pd.DataFrame(
        {
            "AAA": base + rng.normal(0.0, 0.006, n_rows),
            "BBB": 0.6 * base + rng.normal(0.0, 0.008, n_rows),
        },
        index=index,
    )


def test_risk_engine_no_leakage_and_alignment() -> None:
    returns = _asset_matrix()
    market = returns.mean(axis=1).rename("market")
    engine = RiskEngine(window=40, quantile=0.05)

    full = engine.evaluate(returns["AAA"], market_returns=market)["per_asset_frames"]["AAA"]
    prefix = engine.evaluate(returns["AAA"].iloc[:120], market_returns=market.iloc[:120])["per_asset_frames"]["AAA"]

    assert list(full.index) == list(returns.index)
    np.testing.assert_allclose(
        full.iloc[:120][["var_q", "cvar_q", "covar_q", "delta_covar", "rolling_drawdown"]].to_numpy(dtype=float),
        prefix[["var_q", "cvar_q", "covar_q", "delta_covar", "rolling_drawdown"]].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_risk_engine_equal_weight_fallback_and_small_samples() -> None:
    returns = _asset_matrix(n_rows=20)
    engine = RiskEngine(window=30, quantile=0.05)
    result = engine.evaluate(returns)

    assert set(result["per_asset_frames"]) == {"AAA", "BBB"}
    assert "system_frame" in result
    assert len(result["system_frame"]) == 20
    assert result["system_frame"]["var_q"].isna().all()
    assert result["risk_summary"]["system"]["var_q"] is None


def test_trainer_boosters_receive_optional_risk_and_regime_features(tmp_path) -> None:
    pytest.importorskip("xgboost")

    trainer = DualModelTrainer(model_dir=tmp_path / "models")
    df = generate_mock_data(ticker="BOOST", num_days=900)
    trainer.train(
        ticker="BOOST",
        df=df,
        algorithms=["cart", "xgboost"],
        horizons=["short"],
        risk_config={
            "risk_enabled": False,
            "enable_covar": True,
            "enable_risk_engine": True,
            "enable_regime_detection": True,
            "enable_regime_switching": True,
            "enable_risk_allocation": True,
            "covar_quantile": 0.05,
            "covar_window": 45,
            "regime_method": "threshold",
            "risk_penalty_strength": 1.2,
        },
    )

    manifest = trainer._manifests["BOOST"]
    short_algos = manifest["horizons"]["short"]["algorithms"]
    cart_cols = short_algos["cart"]["feature_columns"]
    xgb_cols = short_algos["xgboost"]["feature_columns"]
    cart_task_cols = short_algos["cart"]["feature_columns_by_task"]
    xgb_task_cols = short_algos["xgboost"]["feature_columns_by_task"]

    for column in ("var_q", "cvar_q", "covar_q", "delta_covar", "rolling_drawdown", "regime_label", "regime_probability"):
        assert column not in cart_cols
        assert column in xgb_cols
        assert column not in cart_task_cols["trend"]
        assert column not in cart_task_cols["return"]
        assert column in xgb_task_cols["trend"] or column in xgb_task_cols["return"]

    assert manifest["risk_summary"]
    assert manifest["regime_distribution"]
