"""CART wrappers for classification and regression."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .base import BaseModel


class CartModel(BaseModel):
    """Unified CART wrapper used by the training and inference pipeline."""

    def __init__(
        self,
        *,
        task: str = "classification",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str | None = None,
        random_state: int = 42,
        **_: Any,
    ) -> None:
        self.task = task
        self.params = {
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "criterion": criterion,
            "random_state": random_state,
        }
        self.model = self._build_estimator()

    def _build_estimator(self) -> DecisionTreeClassifier | DecisionTreeRegressor:
        valid_params = {key: value for key, value in self.params.items() if value is not None}
        if self.task == "classification":
            return DecisionTreeClassifier(**valid_params)
        if self.task == "regression":
            return DecisionTreeRegressor(**valid_params)
        raise ValueError("task must be 'classification' or 'regression'")

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any = None,
        y_val: Any = None,
    ) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model.predict(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.task != "classification":
            raise NotImplementedError("CART regression models do not expose predict_proba")
        return np.asarray(self.model.predict_proba(X))

    def save(self, artifact_path: Path) -> None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, artifact_path)
        joblib.dump(self.get_artifact_metadata(), artifact_path.with_suffix(".meta.joblib"))

    @classmethod
    def load(cls, artifact_path: Path) -> "CartModel":
        metadata: Dict[str, Any] = joblib.load(artifact_path.with_suffix(".meta.joblib"))
        instance = cls(**metadata["params"])
        instance.task = metadata["task"]
        instance.model = joblib.load(artifact_path)
        return instance

    def get_artifact_metadata(self) -> Dict[str, Any]:
        return {
            "algorithm": "cart",
            "task": self.task,
            "params": {"task": self.task, **self.params},
        }
