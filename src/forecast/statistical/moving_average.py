"""Moving-average statistical baseline under the shared forecast contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.forecast.base import ForecastModel


class MovingAverageForecastModel(ForecastModel):
    """Predict the next horizon from a trailing average of observable returns."""

    model_name = "moving_average"
    requires_features = False

    def __init__(
        self,
        *,
        lookback: int = 5,
        min_periods: int = 1,
        value_column: str = "daily_return",
        fallback_column: str = "close_return_1d",
        target_type: str = "forward_return",
    ) -> None:
        super().__init__(target_type=target_type)
        self.lookback = int(lookback)
        self.min_periods = int(min_periods)
        self.value_column = value_column
        self.fallback_column = fallback_column
        self._history_source = pd.Series(dtype=float)
        self._fallback_prediction = 0.0

    def _extract_source(self, frame: pd.DataFrame) -> pd.Series:
        for column in (self.value_column, self.fallback_column, self.target_column):
            if column and column in frame.columns:
                return self._coerce_numeric_series(frame[column])
        return pd.Series(np.nan, index=frame.index, dtype=float)

    def _fit_model(self, train_frame: pd.DataFrame) -> None:
        if self.lookback <= 0:
            raise ValueError("lookback must be positive")
        source = self._extract_source(train_frame).dropna()
        target = self._coerce_numeric_series(train_frame[self.target_column]).dropna()
        if target.empty:
            raise ValueError(f"{self.model_name} requires at least one non-null target value")
        self._history_source = source.astype(float).reset_index(drop=True)
        self._fallback_prediction = float(target.tail(max(1, self.lookback)).mean())

    def _predict_values(self, frame: pd.DataFrame) -> np.ndarray:
        source = self._extract_source(frame).astype(float).reset_index(drop=True)
        combined = pd.concat([self._history_source, source], ignore_index=True)
        rolling = combined.rolling(window=self.lookback, min_periods=self.min_periods).mean()
        prediction = rolling.iloc[-len(frame) :].reset_index(drop=True).fillna(self._fallback_prediction)
        return prediction.to_numpy(dtype=float)

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update(
            {
                "lookback": int(self.lookback),
                "min_periods": int(self.min_periods),
                "value_column": self.value_column,
                "fallback_column": self.fallback_column,
                "fallback_prediction": float(self._fallback_prediction),
            }
        )
        return metadata
