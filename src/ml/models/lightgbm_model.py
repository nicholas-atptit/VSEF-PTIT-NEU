"""LightGBM wrapper for classification and regression."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from .base import BaseModel


class LightgbmModel(BaseModel):
    """LightGBM wrapper matching the unified ML inference facade."""

    algorithm_name = "lightgbm"

    def __init__(
        self,
        *,
        task: str = "classification",
        n_estimators: int = 100,
        max_depth: int | None = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        num_leaves: int = 31,
        random_state: int = 42,
        tuned: bool = False,
        tuning_backend: str = "none",
        validation_method: str = "time_series_split",
        **_: Any,
    ) -> None:
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is not installed. Run process with optional dependencies installed.")
            
        self.task = task
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth if max_depth is not None else -1,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "num_leaves": num_leaves,
            "random_state": random_state,
        }
        self.tuned = tuned
        self.tuning_backend = tuning_backend
        self.validation_method = validation_method
        self.model = self._build_estimator()

    @classmethod
    def get_model_capabilities(cls) -> Dict[str, Any]:
        return {
            "algorithm": cls.algorithm_name,
            "model_family": "boosting",
            "requires_sequence_data": False,
            "supports_exogenous_features": True,
            "artifact_type": "joblib",
        }

    def _build_estimator(self) -> Any:
        valid_params = {key: value for key, value in self.params.items() if value is not None}
        if self.task == "classification":
            return lgb.LGBMClassifier(**valid_params)
        if self.task == "regression":
            return lgb.LGBMRegressor(**valid_params)
        raise ValueError("task must be 'classification' or 'regression'")

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any = None,
        y_val: Any = None,
    ) -> None:
        # LightGBM handles callbacks for early stopping if needed in the future
        if X_val is not None and y_val is not None and len(X_val) > 0:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(X_train, y_train, eval_set=eval_set)
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model.predict(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.task != "classification":
            raise NotImplementedError("LightGBM regression models do not expose predict_proba")
        return np.asarray(self.model.predict_proba(X))

    def save(self, artifact_path: Path) -> None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, artifact_path)
        joblib.dump(self.get_artifact_metadata(), artifact_path.with_suffix(".meta.joblib"))

    @classmethod
    def load(cls, artifact_path: Path) -> "LightgbmModel":
        metadata: Dict[str, Any] = joblib.load(artifact_path.with_suffix(".meta.joblib"))
        instance = cls(**metadata["params"])
        instance.task = metadata["task"]
        instance.model = joblib.load(artifact_path)
        return instance

    def get_artifact_metadata(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm_name,
            "model_family": "boosting",
            "task": self.task,
            "tuning_enabled": self.tuned,
            "tuning_backend": self.tuning_backend,
            "validation_method": self.validation_method,
            "params": {"task": self.task, "tuned": self.tuned, "tuning_backend": self.tuning_backend, "validation_method": self.validation_method, **self.params},
        }
