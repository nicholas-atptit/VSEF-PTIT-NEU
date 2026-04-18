from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk.garch import GARCHRiskModel


def test_garch_forecast_interface_returns_vol_and_tail_metrics() -> None:
    returns = pd.Series(np.random.default_rng(42).normal(0.001, 0.02, 260))
    risk = GARCHRiskModel(distribution="normal").fit(returns).forecast_risk(horizon=5)

    assert risk["risk_model"] == "garch"
    assert risk["vol_forecast"] > 0.0
    assert risk["cvar_loss_95"] >= risk["var_loss_95"] >= 0.0
    assert risk["cvar_loss_99"] >= risk["var_loss_99"] >= 0.0
    assert risk["drawdown_state"] in {"normal", "elevated", "severe"}

