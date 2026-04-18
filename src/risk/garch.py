"""GARCH-based volatility forecasting for Phase 2 risk conditioning."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.risk.base import RiskModel
from src.risk.drawdown import DrawdownRiskModel


def _confidence_key(confidence_level: float) -> str:
    pct = confidence_level * 100.0
    return str(int(round(pct))) if abs(pct - round(pct)) < 1e-9 else str(pct).replace(".", "_")


def _drawdown_state(current_drawdown: float, *, elevated: float = -0.05, severe: float = -0.10) -> str:
    if float(current_drawdown) <= float(severe):
        return "severe"
    if float(current_drawdown) <= float(elevated):
        return "elevated"
    return "normal"


class GARCHRiskModel(RiskModel):
    """Forecast conditional volatility and tail losses from realized returns."""

    model_name = "garch"

    def __init__(
        self,
        *,
        p: int = 1,
        q: int = 1,
        mean: str = "Constant",
        distribution: str = "t",
        confidence_levels: list[float] | None = None,
        drawdown_elevated_threshold: float = -0.05,
        drawdown_severe_threshold: float = -0.10,
    ) -> None:
        super().__init__()
        self.p = int(p)
        self.q = int(q)
        self.mean = mean
        self.distribution = distribution
        self.confidence_levels = list(confidence_levels or [0.95, 0.99])
        self.drawdown_elevated_threshold = float(drawdown_elevated_threshold)
        self.drawdown_severe_threshold = float(drawdown_severe_threshold)
        self._result: Any = None
        self._std_resid = pd.Series(dtype=float)
        self._mean_return = 0.0

    @staticmethod
    def _arch_model() -> Any:
        try:
            from arch import arch_model
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "GARCHRiskModel requires arch. Install arch in the active interpreter."
            ) from exc
        return arch_model

    def _fit_model(self, returns_series: pd.Series) -> None:
        scaled_returns = returns_series.astype(float) * 100.0
        if len(scaled_returns) < max(50, self.p + self.q + 10):
            raise ValueError(f"{self.model_name} requires at least 50 usable return observations")

        arch_model = self._arch_model()
        model = arch_model(
            scaled_returns,
            mean=self.mean,
            vol="GARCH",
            p=self.p,
            q=self.q,
            dist=self.distribution,
            rescale=False,
        )
        self._result = model.fit(disp="off")
        self._std_resid = pd.Series(self._result.std_resid, dtype=float).dropna().reset_index(drop=True)
        if self._std_resid.empty:
            raise ValueError(f"{self.model_name} produced no usable standardized residuals")
        self._mean_return = float(self._result.params.get("mu", 0.0)) / 100.0

    def forecast_risk(self, horizon: int = 1) -> dict[str, Any]:
        if self.returns_ is None or self._result is None:
            raise RuntimeError(f"{self.model_name} is not fitted")
        if int(horizon) <= 0:
            raise ValueError("horizon must be positive")

        forecast = self._result.forecast(horizon=int(horizon), reindex=False)
        variance_path = pd.Series(forecast.variance.iloc[-1], dtype=float).iloc[: int(horizon)]
        cumulative_variance = float(variance_path.sum()) / (100.0 ** 2)
        vol_forecast = float(np.sqrt(max(cumulative_variance, 0.0)))
        expected_return = float(self._mean_return * int(horizon))

        drawdown_metrics = DrawdownRiskModel().fit(self.returns_).forecast_risk(horizon=horizon)
        result: dict[str, Any] = {
            "risk_model": self.model_name,
            "horizon": int(horizon),
            "distribution": self.distribution,
            "expected_return": expected_return,
            "vol_forecast": vol_forecast,
            "volatility": vol_forecast,
            "current_drawdown": float(drawdown_metrics.get("current_drawdown", 0.0)),
            "max_drawdown": float(drawdown_metrics.get("max_drawdown", 0.0)),
            "drawdown_state": _drawdown_state(
                float(drawdown_metrics.get("current_drawdown", 0.0)),
                elevated=self.drawdown_elevated_threshold,
                severe=self.drawdown_severe_threshold,
            ),
        }

        for conf in self.confidence_levels:
            alpha = 1.0 - float(conf)
            z_var = float(np.quantile(self._std_resid, alpha))
            tail = self._std_resid[self._std_resid <= z_var]
            z_cvar = float(tail.mean()) if not tail.empty else z_var
            var_return = float(expected_return + (vol_forecast * z_var))
            cvar_return = float(expected_return + (vol_forecast * z_cvar))
            key = _confidence_key(float(conf))
            result[f"var_return_{key}"] = var_return
            result[f"cvar_return_{key}"] = cvar_return
            result[f"var_loss_{key}"] = float(max(0.0, -var_return))
            result[f"cvar_loss_{key}"] = float(max(0.0, -cvar_return))

        return result

    def get_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "p": self.p,
            "q": self.q,
            "mean": self.mean,
            "distribution": self.distribution,
            "confidence_levels": list(self.confidence_levels),
            "drawdown_elevated_threshold": self.drawdown_elevated_threshold,
            "drawdown_severe_threshold": self.drawdown_severe_threshold,
        }
