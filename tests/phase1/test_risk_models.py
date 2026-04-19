from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.var_cvar import VaRCVaRRiskModel, compute_var_cvar


def test_compute_var_cvar_matches_empirical_tail_definition() -> None:
    distribution = pd.Series([-0.05, -0.02, 0.0, 0.01, 0.03])
    metrics = compute_var_cvar(distribution, 0.80)

    expected_var = float(np.quantile(distribution, 0.20))
    expected_tail = float(distribution[distribution <= expected_var].mean())

    assert metrics["var_return"] == pytest.approx(expected_var)
    assert metrics["cvar_return"] == pytest.approx(expected_tail)
    assert metrics["cvar_loss"] >= metrics["var_loss"] >= 0.0


def test_var_cvar_risk_model_outputs_positive_loss_metrics() -> None:
    returns = pd.Series(np.linspace(-0.04, 0.03, 120))
    model = VaRCVaRRiskModel().fit(returns)
    risk = model.forecast_risk(horizon=5)

    assert risk["var_loss_95"] >= 0.0
    assert risk["cvar_loss_95"] >= risk["var_loss_95"]
