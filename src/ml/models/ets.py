"""ETS wrapper for structural time series forecasting.

**Statistical Model Contract for ETS:**
- **target input shape (y)**: 1D array of shape `(n_samples,)`
- **exogenous input shape (X)**: COMPLETELY IGNORED (ETS does not natively support exog).
- **forecast behavior**: 
    - `predict(X)` executes a dynamic multi-step out-of-sample forecast for `steps=len(X)`. The input `X` is ONLY used to infer this horizon length under the shared trainer contract.
- **task adaptation**:
    - `regression`: Directly forecasts future returns.
    - `classification`: Forecasts binary labels treating them continuously, then applies clipping/sigmoid mapping.
- **artifact save/load**: Uses `joblib` on the statsmodels results wrapper.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from .base import BaseModel


class EtsModel(BaseModel):
    """ETS model wrapper matching the unified ML inference facade."""

    algorithm_name = "ets"

    def __init__(
        self,
        *,
        task: str = "regression",
        trend: str | None = "add",
        damped_trend: bool = False,
        seasonal: str | None = None,
        seasonal_periods: int | None = None,
        **_: Any,
    ) -> None:
        self.task = task
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.convergence_warnings: list[str] = []
        self.model_results: Any = None
        self.params = {
            "task": task,
            "trend": trend,
            "damped_trend": damped_trend,
            "seasonal": seasonal,
            "seasonal_periods": seasonal_periods,
        }

    @classmethod
    def get_model_capabilities(cls) -> Dict[str, Any]:
        return {
            "algorithm": cls.algorithm_name,
            "model_family": "statistical",
            "requires_sequence_data": False,
            "supports_exogenous_features": False,
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
        # For ETS, statsmodels requires strictly positive values if multiplicative trend is used
        if self.trend == 'mul' and np.any(y <= 0):
            # Fallback to additive if incompatible
            self.trend = 'add'
            self.params['trend'] = 'add'

        model = ExponentialSmoothing(
            endog=y,
            trend=self.trend,
            damped_trend=self.damped_trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        )

        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            self.model_results = model.fit()

            for w in captured_warnings:
                if issubclass(w.category, ConvergenceWarning):
                    self.convergence_warnings.append(str(w.message))

    def _get_forecast(self, X: Any) -> np.ndarray:
        if self.model_results is None:
            raise RuntimeError("Model is not fitted")
        
        # ETS doesn't use X, but the length determines the number of steps to forecast
        steps = len(X) if X is not None else 1
        forecast = self.model_results.forecast(steps=steps)
        return np.asarray(forecast, dtype=float)

    def predict(self, X: Any) -> np.ndarray:
        forecast = self._get_forecast(X)
        if self.task == "classification":
            return (forecast > 0.5).astype(int)
        return forecast

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.task != "classification":
            raise NotImplementedError("ETS regression models do not expose predict_proba")
        
        forecast = self._get_forecast(X)
        # Heuristic mapping: Map continuous regression labels on [0,1] to probabilities via clipping
        p_up = np.clip(forecast, 0.001, 0.999)
        p_down = 1.0 - p_up
        
        return np.vstack([p_down, p_up]).T

    def save(self, artifact_path: Path) -> None:
        if self.model_results is None:
            raise RuntimeError("Cannot save an unfitted ETS model")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model_results, artifact_path)
        joblib.dump(self.get_artifact_metadata(), artifact_path.with_suffix(".meta.joblib"))

    @classmethod
    def load(cls, artifact_path: Path) -> "EtsModel":
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
