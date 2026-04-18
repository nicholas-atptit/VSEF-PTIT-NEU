"""Stacking Ensemble wrapper for structural ML forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import copy

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit

from .base import BaseModel
from .factory import create_model


class StackingModel(BaseModel):
    """Time-series safe stacking ensemble.
    
    If optional models like xgboost or lightgbm are requested but unavailable,
    the model gracefully degrades by skipping them and fitting only the supported
    models, ensuring predictability and robustness in different environments.
    """

    algorithm_name = "stacking"

    def __init__(
        self,
        *,
        task: str = "classification",
        base_learners: List[str] | None = None,
        n_splits: int = 5,
        **kwargs: Any,
    ) -> None:
        self.task = task
        # Limit to Stack v1 learners
        self.base_learners_names = base_learners or ["cart", "xgboost", "lightgbm", "sarimax", "ets"]
        self.n_splits = n_splits
        self.base_kwargs = kwargs

        self.meta_model: Any = None
        self.fitted_base_models: List[BaseModel] = []
        self.params = {
            "task": task,
            "base_learners": self.base_learners_names,
            "n_splits": n_splits,
            **self.base_kwargs
        }

    @classmethod
    def get_model_capabilities(cls) -> Dict[str, Any]:
        return {
            "algorithm": cls.algorithm_name,
            "model_family": "ensemble",
            "requires_sequence_data": False,
            "supports_exogenous_features": True,
            "artifact_type": "joblib",
        }

    def _create_meta_model(self) -> Any:
        if self.task == "classification":
            return LogisticRegression(class_weight="balanced", random_state=42)
        if self.task == "regression":
            return Ridge(random_state=42)
        raise ValueError("task must be 'classification' or 'regression'")

    def _predict_base(self, model: BaseModel, X: Any) -> np.ndarray:
        if self.task == "classification":
            # For classification, use probability of positive class as the meta-feature
            probs = model.predict_proba(X)
            if probs.shape[1] == 1:
                # If a CV fold had only one class (usually 0), the internal model might only return 1 column
                return np.zeros(len(X))
            return probs[:, 1]
        return model.predict(X)

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any = None,
        y_val: Any = None,
    ) -> None:
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float)
        n_samples = len(X)

        if n_samples < self.n_splits * 2:
            self.n_splits = max(2, n_samples // 10)
            
        # Filter supported learners upfront
        supported_learners = []
        for algo in self.base_learners_names:
            try:
                # Cheap test to check dependencies by instantiating without training
                create_model(algo, task=self.task, **self.base_kwargs)
                supported_learners.append(algo)
            except ImportError:
                continue
                
        self.base_learners_names = supported_learners
        
        if not self.base_learners_names:
            raise RuntimeError("No base learners are available for Stacking.")

        ts_cv = TimeSeriesSplit(n_splits=self.n_splits)
        
        # Meta-features matrix for the validation folds
        meta_features = np.zeros((n_samples, len(self.base_learners_names)))
        valid_indices = []

        # 1. Out-of-fold meta-feature generation using strictly forward-looking split
        for train_idx, val_idx in ts_cv.split(X):
            X_cv_train, X_cv_val = X[train_idx], X[val_idx]
            y_cv_train, _ = y[train_idx], y[val_idx]
            
            valid_indices.extend(val_idx)

            for j, algo in enumerate(self.base_learners_names):
                base_model = create_model(algo, task=self.task, **self.base_kwargs)
                base_model.fit(X_cv_train, y_cv_train)
                meta_features[val_idx, j] = self._predict_base(base_model, X_cv_val)

        valid_indices = np.array(valid_indices)
        X_meta = meta_features[valid_indices]
        y_meta = y[valid_indices]

        # 2. Train meta-model on OOF predictions
        self.meta_model = self._create_meta_model()
        self.meta_model.fit(X_meta, y_meta)

        # 3. Train all base learners on the FULL training set for future out-of-sample predictions
        self.fitted_base_models = []
        for algo in self.base_learners_names:
            base_model = create_model(algo, task=self.task, **self.base_kwargs)
            base_model.fit(X, y)
            self.fitted_base_models.append(base_model)

    def _execute_with_fallback(self, X: Any, is_proba: bool = False) -> np.ndarray:
        num_models = len(self.fitted_base_models)
        meta_features = np.zeros((len(X), num_models))
        
        self.last_healthy_count = 0
        self.last_failed_count = 0
        self.last_fallback_policy = "none"
        
        healthy_predictions = []
        for j, model in enumerate(self.fitted_base_models):
            try:
                if is_proba:
                    val = model.predict_proba(X)
                else:
                    val = self._predict_base(model, X)
                
                if np.isnan(val).any() or np.isinf(val).any():
                    raise ValueError("NaN/Inf in prediction")
                    
                if not is_proba:
                    meta_features[:, j] = val
                else:
                    # For meta features, we extract the positive probability if classification
                    if val.ndim == 2 and val.shape[1] > 1:
                        meta_features[:, j] = val[:, 1]
                    else:
                        meta_features[:, j] = val.ravel()
                        
                healthy_predictions.append(val)
                self.last_healthy_count += 1
            except Exception:
                meta_features[:, j] = np.nan
                self.last_failed_count += 1
                
        meta_success = False
        meta_pred = None
        if self.last_failed_count == 0:
            try:
                if is_proba:
                    meta_pred = self.meta_model.predict_proba(meta_features)
                else:
                    meta_pred = self.meta_model.predict(meta_features)
                    
                if not (np.isnan(meta_pred).any() or np.isinf(meta_pred).any()):
                    meta_success = True
            except Exception:
                pass
                
        if meta_success:
            self.last_fallback_policy = "none"
            return meta_pred
            
        if self.last_healthy_count == 0:
            self.last_fallback_policy = "failed"
            raise RuntimeError("[stacking_base_model_failed] 0 healthy base models available")
            
        if self.last_healthy_count == 1:
            self.last_fallback_policy = "single_model"
            return healthy_predictions[0]
            
        self.last_fallback_policy = "average"
        stack = np.stack(healthy_predictions, axis=0)
        return np.mean(stack, axis=0)

    def predict(self, X: Any) -> np.ndarray:
        return self._execute_with_fallback(X, is_proba=False)

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.task != "classification":
            raise NotImplementedError("Stacking regression models do not expose predict_proba")
        return self._execute_with_fallback(X, is_proba=True)

    def save(self, artifact_path: Path) -> None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        # We need to save the meta-model and all fitted base models
        payload = {
            "meta_model": self.meta_model,
        }
        
        # Save base models recursively into temporary paths or just let joblib pickle them if supported.
        # But wait, SarimaxModel and EtsModel use joblib internally and save the statsmodels wrapper cleanly
        # when pickled by Python, but sometimes statsmodels is finicky. It's safe to pickle them in one payload
        # because statsmodels results objects support native pickling.
        payload["fitted_base_models"] = self.fitted_base_models
        
        joblib.dump(payload, artifact_path)
        joblib.dump(self.get_artifact_metadata(), artifact_path.with_suffix(".meta.joblib"))

    @classmethod
    def load(cls, artifact_path: Path) -> "StackingModel":
        metadata: Dict[str, Any] = joblib.load(artifact_path.with_suffix(".meta.joblib"))
        instance = cls(**metadata["params"])
        payload = joblib.load(artifact_path)
        
        instance.meta_model = payload["meta_model"]
        instance.fitted_base_models = payload["fitted_base_models"]
        return instance

    def get_artifact_metadata(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm_name,
            "model_family": "ensemble",
            "task": self.task,
            "params": self.params,
            "meta_model_type": type(self.meta_model).__name__ if self.meta_model else "None",
            "base_learners": self.base_learners_names,
            "n_splits": self.n_splits,
            "validation_split": "TimeSeriesSplit",
        }
