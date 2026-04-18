"""Deprecated baseline scaffold retained only for historical reference."""

from __future__ import annotations

import os
import pandas as pd
from typing import Dict, Any, Optional
from src.utils.logging import get_logger
from config.settings import get_settings

# from catboost import CatBoostRegressor, Pool
# from lightgbm import LGBMRegressor

logger = get_logger(__name__)

class BaselineModel:
    """Legacy baseline helper that is no longer part of the supported ML path."""

    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        *,
        allow_legacy: bool = False,
    ) -> None:
        if not allow_legacy:
            raise RuntimeError(
                "BaselineModel is legacy scaffold and is intentionally blocked by default. "
                "Use DualModelTrainer for supported training workflows. "
                "Pass allow_legacy=True only for historical debugging."
            )
        self.params = model_params or {}
        self.model = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train a basic RandomForestRegressor as baseline."""
        from sklearn.ensemble import RandomForestRegressor
        
        # Filter numeric features only
        X = X_train.select_dtypes(include=['number'])
        logger.info("training_baseline_forest", rows=len(X), features=X.columns.tolist())
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, **self.params)
        self.model.fit(X, y_train)
        logger.info("model_training_complete")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model on Mean Absolute Error."""
        from sklearn.metrics import mean_absolute_error
        
        if self.model is None:
            return {"error": 0.0}
            
        X = X_test.select_dtypes(include=['number'])
        preds = self.model.predict(X)
        mae = mean_absolute_error(y_test, preds)
        
        logger.info("model_evaluation_complete", mae=float(mae))
        return {"mae": float(mae)}

    def save(self, path: str) -> None:
        """Serialize model to disk.
        
        TODO: Use os.path logic to ensure directory exists.
        """
        logger.info("saving_model", path=path)
        pass
