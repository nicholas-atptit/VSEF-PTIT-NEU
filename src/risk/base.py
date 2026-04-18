"""Base contracts for Phase 1 risk models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class RiskModel(ABC):
    """Risk model contract operating on returns or pnl distributions."""

    model_name = "base_risk"

    def __init__(self, *, model_name: str | None = None) -> None:
        self.model_name = model_name or self.model_name
        self.config: dict[str, Any] = {}
        self.returns_: pd.Series | None = None

    def fit(self, returns_series: pd.Series, config: dict[str, Any] | None = None) -> "RiskModel":
        numeric = pd.to_numeric(pd.Series(returns_series), errors="coerce").dropna().astype(float)
        if numeric.empty:
            raise ValueError(f"{self.model_name} received no usable returns")
        self.config = dict(config or {})
        self.returns_ = numeric
        self._fit_model(numeric)
        return self

    @abstractmethod
    def _fit_model(self, returns_series: pd.Series) -> None:
        """Optional fit hook for stateful risk models."""

    @abstractmethod
    def forecast_risk(self, horizon: int = 1) -> dict[str, Any]:
        """Return risk estimates for the fitted return distribution."""

