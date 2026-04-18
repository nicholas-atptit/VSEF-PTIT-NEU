"""ETS forecast model under the shared Phase 1 contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.forecast.base import ForecastModel


class ETSForecastModel(ForecastModel):
    """Explicit ETS wrapper for walk-forward evaluation."""

    model_name = "ets"
    requires_features = False

    def __init__(
        self,
        *,
        trend: str | None = "add",
        damped_trend: bool = False,
        seasonal: str | None = None,
        seasonal_periods: int | None = None,
        target_type: str = "forward_return",
    ) -> None:
        super().__init__(target_type=target_type)
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self._results: Any = None

    @staticmethod
    def _ets_class() -> Any:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "ETSForecastModel requires statsmodels. Install statsmodels to use this baseline."
            ) from exc
        return ExponentialSmoothing

    def _fit_model(self, train_frame: pd.DataFrame) -> None:
        target = self._coerce_numeric_series(train_frame[self.target_column]).dropna()
        if target.empty:
            raise ValueError(f"{self.model_name} requires non-null training targets")
        trend = self.trend
        if trend == "mul" and (target <= 0).any():
            trend = "add"
        ets_cls = self._ets_class()
        model = ets_cls(
            endog=target.to_numpy(dtype=float),
            trend=trend,
            damped_trend=self.damped_trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        )
        self._results = model.fit(optimized=True)

    def _predict_values(self, frame: pd.DataFrame) -> np.ndarray:
        if self._results is None:
            raise RuntimeError(f"{self.model_name} is not fitted")
        forecast = self._results.forecast(steps=len(frame))
        return np.asarray(forecast, dtype=float)

    def predict_in_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        self._ensure_fitted()
        frame = self._prepare_frame(df, require_target=False)
        fitted = np.asarray(self._results.fittedvalues, dtype=float)
        if len(fitted) >= len(frame):
            fitted = fitted[-len(frame) :]
        else:
            padding = np.full(len(frame) - len(fitted), np.nan, dtype=float)
            fitted = np.concatenate([padding, fitted])
        return self._build_forecast_frame(frame, fitted)

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update(
            {
                "trend": self.trend,
                "damped_trend": bool(self.damped_trend),
                "seasonal": self.seasonal,
                "seasonal_periods": self.seasonal_periods,
            }
        )
        return metadata
