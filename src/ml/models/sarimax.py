"""SARIMAX wrapper for structural time series forecasting.

**Statistical Model Contract for SARIMAX:**
- **target input shape (y)**: 1D array of shape `(n_samples,)`
- **exogenous input shape (X)**: 2D array of shape `(n_samples, n_features)`
- **forecast behavior**: 
    - `predict(X)` executes a dynamic multi-step out-of-sample forecast for `steps=len(X)`, using `X` as `exog`.
- **task adaptation**:
    - `regression`: Directly forecasts future returns.
    - `classification`: Forecasts binary labels treating them continuously, then applies clipping/sigmoid mapping to satisfy `predict_proba` requirements.
- **artifact save/load**: Uses `statsmodels` native `save`/`load` via pickle to preserve full state, with metadata adjacent.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from .base import BaseModel


class SarimaxModel(BaseModel):
    """SARIMAX model wrapper matching the unified ML inference facade."""

    algorithm_name = "sarimax"

    def __init__(
        self,
        *,
        task: str = "regression",
        order: tuple[int, int, int] = (1, 0, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: str | None = "c",
        **_: Any,
    ) -> None:
        self.task = task
        self.order = order
        self.seasonal_order = seasonal_order
        self.convergence_warnings: list[str] = []
        self.trend = trend
        self.model_results: Any = None
        self.params = {
            "task": task,
            "order": order,
            "seasonal_order": seasonal_order,
            "trend": trend,
        }

    @classmethod
    def get_model_capabilities(cls) -> Dict[str, Any]:
        return {
            "algorithm": cls.algorithm_name,
            "model_family": "statistical",
            "requires_sequence_data": False,
            "supports_exogenous_features": True,
            "artifact_type": "joblib",
        }

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any = None,
        y_val: Any = None,
    ) -> None:
        y = np.asarray(y_train, dtype=float)
        exog = np.asarray(X_train, dtype=float) if X_train is not None and len(X_train) > 0 else None

        model = sm.tsa.SARIMAX(
            endog=y,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            self.model_results = model.fit(disp=False)
            
            for w in captured_warnings:
                if issubclass(w.category, ConvergenceWarning):
                    self.convergence_warnings.append(str(w.message))

    def _get_forecast(self, X: Any) -> np.ndarray:
        if self.model_results is None:
            raise RuntimeError("Model is not fitted")
        
        exog = np.asarray(X, dtype=float) if X is not None and len(X) > 0 else None
        steps = len(X) if X is not None else 1
        
        forecast = self.model_results.forecast(steps=steps, exog=exog)
        return np.asarray(forecast, dtype=float)

    def predict(self, X: Any) -> np.ndarray:
        forecast = self._get_forecast(X)
        if self.task == "classification":
            return (forecast > 0.5).astype(int)
        return forecast

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.task != "classification":
            raise NotImplementedError("SARIMAX regression models do not expose predict_proba")
        
        forecast = self._get_forecast(X)
        # Heuristic mapping: Map continuous regression labels on [0,1] to probabilities via clipping
        p_up = np.clip(forecast, 0.001, 0.999)
        p_down = 1.0 - p_up
        
        return np.vstack([p_down, p_up]).T

    def save(self, artifact_path: Path) -> None:
        if self.model_results is None:
            raise RuntimeError("Cannot save an unfitted SARIMAX model")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        # We save the results wrapped in joblib
        joblib.dump(self.model_results, artifact_path)
        joblib.dump(self.get_artifact_metadata(), artifact_path.with_suffix(".meta.joblib"))

    @classmethod
    def load(cls, artifact_path: Path) -> "SarimaxModel":
        metadata: Dict[str, Any] = joblib.load(artifact_path.with_suffix(".meta.joblib"))
        instance = cls(**metadata["params"])
        instance.model_results = joblib.load(artifact_path)
        return instance

    def get_artifact_metadata(self) -> Dict[str, Any]:
        metadata = {
            "algorithm": self.algorithm_name,
            "task": self.task,
            "params": self.params,
            "convergence_warnings": self.convergence_warnings,
        }
        if self.task == "classification":
            metadata["heuristic_probabilities"] = True
            
        return metadata
