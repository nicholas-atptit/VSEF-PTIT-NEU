"""Naive statistical baseline under the shared forecast contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.forecast.base import ForecastModel


class NaiveForecastModel(ForecastModel):
    """Predict the next horizon from the most recent observable return."""

    model_name = "naive"
    requires_features = False

    def __init__(
        self,
        *,
        value_column: str = "daily_return",
        fallback_column: str = "close_return_1d",
        target_type: str = "forward_return",
    ) -> None:
        super().__init__(target_type=target_type)
        self.value_column = value_column
        self.fallback_column = fallback_column
        self.last_observed_target = 0.0

    def _fit_model(self, train_frame: pd.DataFrame) -> None:
        target = self._coerce_numeric_series(train_frame[self.target_column]).dropna()
        if target.empty:
            raise ValueError(f"{self.model_name} requires at least one non-null target value")
        self.last_observed_target = float(target.iloc[-1])

    def _predict_values(self, frame: pd.DataFrame) -> np.ndarray:
        source = None
        for column in (self.value_column, self.fallback_column):
            if column in frame.columns:
                source = self._coerce_numeric_series(frame[column])
                break
        if source is None:
            source = pd.Series(self.last_observed_target, index=frame.index, dtype=float)
        prediction = source.fillna(self.last_observed_target).astype(float)
        return prediction.to_numpy(dtype=float)

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update(
            {
                "value_column": self.value_column,
                "fallback_column": self.fallback_column,
                "last_observed_target": float(self.last_observed_target),
            }
        )
        return metadata
