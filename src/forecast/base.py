"""Shared forecast-model contracts for the Phase 1 architecture."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from src.core.contracts import validate_forecast_frame


class ForecastModel(ABC):
    """Explicit forecast model contract based on train/test DataFrames."""

    model_name = "base_forecast"
    requires_features = True

    def __init__(
        self,
        *,
        model_name: str | None = None,
        target_type: str = "forward_return",
    ) -> None:
        self.model_name = model_name or self.model_name
        self.target_type = target_type
        self.feature_columns: list[str] = []
        self.target_column: str | None = None
        self.horizon: int = 1
        self.config: dict[str, Any] = {}
        self._is_fitted = False

    def fit(
        self,
        train_df: pd.DataFrame,
        features: list[str],
        target: str,
        horizon: int,
        config: dict[str, Any] | None = None,
    ) -> "ForecastModel":
        train_frame = self._prepare_frame(train_df, require_target=True)
        self.feature_columns = list(features or [])
        self.target_column = str(target)
        self.horizon = int(horizon)
        self.config = dict(config or {})
        if self.requires_features and not self.feature_columns:
            raise ValueError(f"{self.model_name} requires at least one feature column")
        self._validate_columns(train_frame)
        self._fit_model(train_frame)
        self._is_fitted = True
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        self._ensure_fitted()
        test_frame = self._prepare_frame(test_df, require_target=False)
        self._validate_columns(test_frame, require_target=False)
        return self._build_forecast_frame(test_frame, self._predict_values(test_frame))

    def predict_in_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        self._ensure_fitted()
        in_sample = self._prepare_frame(df, require_target=False)
        self._validate_columns(in_sample, require_target=False)
        return self._build_forecast_frame(in_sample, self._predict_values(in_sample))

    def get_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "target_type": self.target_type,
            "target_column": self.target_column,
            "feature_columns": list(self.feature_columns),
            "horizon": int(self.horizon),
            "config": dict(self.config),
        }

    @abstractmethod
    def _fit_model(self, train_frame: pd.DataFrame) -> None:
        """Fit the model using the prepared training frame."""

    @abstractmethod
    def _predict_values(self, frame: pd.DataFrame) -> np.ndarray:
        """Return a prediction array aligned to the input frame rows."""

    def _ensure_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.model_name} is not fitted")

    def _prepare_frame(self, df: pd.DataFrame, *, require_target: bool) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Forecast model received an empty frame")
        prepared = df.copy()
        if "timestamp" not in prepared.columns:
            if "date" in prepared.columns:
                prepared = prepared.rename(columns={"date": "timestamp"})
            else:
                raise ValueError("Forecast model requires a timestamp/date column")
        if "ticker" not in prepared.columns:
            raise ValueError("Forecast model requires a ticker column")
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
        if prepared["timestamp"].isna().any():
            raise ValueError("Forecast model received invalid timestamps")
        prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
        if prepared["ticker"].nunique() != 1:
            raise ValueError(
                f"{self.model_name} expects a single ticker per fit/predict call; "
                f"got {prepared['ticker'].nunique()}"
            )
        if require_target and self.target_column and self.target_column not in prepared.columns:
            raise ValueError(f"Missing target column '{self.target_column}'")
        return prepared.sort_values("timestamp").reset_index(drop=True)

    def _validate_columns(self, frame: pd.DataFrame, *, require_target: bool = True) -> None:
        missing_features = [column for column in self.feature_columns if column not in frame.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns for {self.model_name}: {missing_features}")
        if require_target and self.target_column and self.target_column not in frame.columns:
            raise ValueError(f"Missing target column '{self.target_column}'")

    def _build_forecast_frame(self, frame: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
        y_true = (
            pd.to_numeric(frame[self.target_column], errors="coerce").astype(float)
            if self.target_column and self.target_column in frame.columns
            else pd.Series(np.nan, index=frame.index, dtype=float)
        )
        result = pd.DataFrame(
            {
                "timestamp": frame["timestamp"].to_numpy(),
                "ticker": frame["ticker"].to_numpy(),
                "y_true": y_true.to_numpy(),
                "y_pred": np.asarray(predictions, dtype=float),
                "model_name": self.model_name,
                "target_type": self.target_type,
                "horizon": int(self.horizon),
                "window_id": frame.get("window_id", pd.Series("unassigned", index=frame.index)).astype(str).to_numpy(),
            }
        )
        if "target_timestamp" in frame.columns:
            result["target_timestamp"] = pd.to_datetime(frame["target_timestamp"], errors="coerce").to_numpy()
        return validate_forecast_frame(result)


class SklearnForecastModel(ForecastModel):
    """Convenience base for sklearn-compatible regression estimators."""

    estimator_cls: type[Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model_name=kwargs.pop("model_name", None), target_type=kwargs.pop("target_type", "forward_return"))
        self.estimator_params = dict(kwargs)
        self.estimator = self._build_estimator()

    def _build_estimator(self) -> Any:
        if self.estimator_cls is None:
            raise NotImplementedError("estimator_cls must be defined")
        return self.estimator_cls(**self.estimator_params)

    def _fit_model(self, train_frame: pd.DataFrame) -> None:
        target = pd.to_numeric(train_frame[self.target_column], errors="coerce")
        features = train_frame[self.feature_columns].apply(pd.to_numeric, errors="coerce")
        mask = target.notna() & features.notna().all(axis=1)
        if not mask.any():
            raise ValueError(f"{self.model_name} could not find usable rows after numeric coercion")
        self.estimator.fit(features.loc[mask], target.loc[mask])

    def _predict_values(self, frame: pd.DataFrame) -> np.ndarray:
        features = frame[self.feature_columns].apply(pd.to_numeric, errors="coerce")
        if features.isna().any().any():
            raise ValueError(f"{self.model_name} received NaN features during prediction")
        return np.asarray(self.estimator.predict(features), dtype=float)

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata["estimator_params"] = dict(self.estimator_params)
        metadata["estimator_class"] = self.estimator.__class__.__name__
        return metadata

