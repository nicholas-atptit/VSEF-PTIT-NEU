"""SARIMAX forecast model under the shared Phase 1 contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.forecast.base import ForecastModel


class SARIMAXForecastModel(ForecastModel):
    """Explicit SARIMAX forecast wrapper for walk-forward evaluation."""

    model_name = "sarimax"

    def __init__(
        self,
        *,
        order: tuple[int, int, int] = (1, 0, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: str | None = "c",
        target_type: str = "forward_return",
    ) -> None:
        super().__init__(target_type=target_type)
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend
        self._feature_fill_values = pd.Series(dtype=float)
        self._results: Any = None

    @staticmethod
    def _statsmodels_sarimax() -> Any:
        try:
            import statsmodels.api as sm
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "SARIMAXForecastModel requires statsmodels. Install statsmodels to use this baseline."
            ) from exc
        return sm.tsa.SARIMAX

    def _prepare_exog(self, frame: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        features = frame[self.feature_columns].apply(self._coerce_numeric_series)
        if fit:
            self._feature_fill_values = features.median(axis=0, skipna=True).fillna(0.0).astype(float)
        elif self._feature_fill_values.empty:
            raise RuntimeError(f"{self.model_name} does not have fitted exogenous fill values")
        return features.fillna(self._feature_fill_values)

    def _fit_model(self, train_frame: pd.DataFrame) -> None:
        target = self._coerce_numeric_series(train_frame[self.target_column])
        exog = self._prepare_exog(train_frame, fit=True)
        mask = target.notna() & exog.notna().all(axis=1)
        if int(mask.sum()) <= max(self.order[0] + self.order[1] + self.order[2], 3):
            raise ValueError(f"{self.model_name} does not have enough usable rows for order={self.order}")

        sarimax_cls = self._statsmodels_sarimax()
        model = sarimax_cls(
            endog=target.loc[mask].to_numpy(dtype=float),
            exog=exog.loc[mask].to_numpy(dtype=float),
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._results = model.fit(disp=False)

    def _predict_values(self, frame: pd.DataFrame) -> np.ndarray:
        if self._results is None:
            raise RuntimeError(f"{self.model_name} is not fitted")
        exog = self._prepare_exog(frame, fit=False)
        forecast = self._results.forecast(steps=len(frame), exog=exog.to_numpy(dtype=float))
        return np.asarray(forecast, dtype=float)

    def predict_in_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        self._ensure_fitted()
        frame = self._prepare_frame(df, require_target=False)
        self._validate_columns(frame, require_target=False)
        exog = self._prepare_exog(frame, fit=False)
        fitted = self._results.predict(start=0, end=len(frame) - 1, exog=exog.to_numpy(dtype=float))
        return self._build_forecast_frame(frame, np.asarray(fitted, dtype=float))

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update(
            {
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "trend": self.trend,
                "feature_fill_values": self._feature_fill_values.to_dict(),
            }
        )
        return metadata
