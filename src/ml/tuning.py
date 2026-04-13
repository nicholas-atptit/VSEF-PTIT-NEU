"""Optuna-based hyperparameter tuning with time-series validation."""

from __future__ import annotations

import logging
from typing import Any, Dict, Type

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, mean_squared_error

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from .models.base import BaseModel

logger = logging.getLogger(__name__)


def optimize_hyperparameters(
    model_cls: Type[BaseModel],
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    n_trials: int = 20,
    n_splits: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run Optuna optimization using time-series split cross-validation.
    
    Args:
        model_cls: The specific BaseModel class to tune
        X: The full feature array (ordered temporally)
        y: The target array (ordered temporally)
        task: "classification" or "regression"
        n_trials: Number of Optuna trials
        n_splits: Number of TimeSeriesSplits
        random_state: Random seed for reproducible search
        
    Returns:
        Best hyperparameters dictionary.
    """
    if not HAS_OPTUNA:
        logger.warning("Optuna is not installed; skipping hyperparameter tuning.")
        return {}

    algo_name = getattr(model_cls, "algorithm_name", "").lower()
    if algo_name not in {"xgboost", "lightgbm"}:
        logger.warning(f"Tuning not implemented for {algo_name}, returning empty params.")
        return {}

    ts_cv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
        if algo_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": random_state,
                "task": task,
            }
        else: # lightgbm
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": random_state,
                "task": task,
            }

        scores = []
        for train_idx, val_idx in ts_cv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = model_cls(**params)
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

            if task == "classification":
                # Ensure we handle classes cleanly. Use predict_proba for log_loss
                preds = model.predict_proba(X_val)
                # If only one class in y_val, log_loss might complain, handle carefully
                try:
                    score = log_loss(y_val, preds, labels=[0, 1])
                except ValueError:
                    # Fallback if log_loss fails due to y_val homogeneity
                    hard_preds = model.predict(X_val)
                    score = float(np.mean(hard_preds != y_val))
            else:
                preds = model.predict(X_val)
                score = mean_squared_error(y_val, preds)
                
            scores.append(score)

        return float(np.mean(scores))

    # Optuna runs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    return study.best_params
