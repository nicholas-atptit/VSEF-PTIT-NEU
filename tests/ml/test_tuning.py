import pytest
import numpy as np
from unittest.mock import patch, MagicMock

import src.ml.tuning as tuning
from src.ml.tuning import optimize_hyperparameters
from src.ml.models.factory import create_model

# Dummy test classes matching BaseModel footprint for isolated tuning tests
class DummyTunedModel:
    algorithm_name = "xgboost"
    
    def __init__(self, **kwargs):
        self.params = kwargs
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        pass
        
    def predict(self, X):
        return np.zeros(len(X))
        
    def predict_proba(self, X):
        return np.vstack([np.ones(len(X)), np.zeros(len(X))]).T


def test_tuning_graceful_skip():
    # Force HAS_OPTUNA to False
    original_has_optuna = tuning.HAS_OPTUNA
    tuning.HAS_OPTUNA = False
    
    try:
        X = np.random.rand(50, 3)
        y = np.random.rand(50)
        best_params = optimize_hyperparameters(DummyTunedModel, X, y, task="regression")
        assert best_params == {}
    finally:
        tuning.HAS_OPTUNA = original_has_optuna


@pytest.mark.skipif(not tuning.HAS_OPTUNA, reason="optuna not installed")
def test_tuning_uses_time_series_split():
    # We will mock TimeSeriesSplit to intercept its call
    X = np.random.rand(50, 3)
    y = np.random.rand(50)
    
    with patch("src.ml.tuning.TimeSeriesSplit") as mock_tscv:
        mock_instance = MagicMock()
        mock_instance.split.return_value = [
            (np.array([0, 1, 2]), np.array([3, 4])),
        ]
        mock_tscv.return_value = mock_instance
        
        # We also mock optuna so it runs extremely fast (1 trial)
        best_params = optimize_hyperparameters(
            DummyTunedModel, 
            X, y, 
            task="regression", 
            n_trials=1, 
            n_splits=3
        )
        
        mock_tscv.assert_called_once_with(n_splits=3)
        mock_instance.split.assert_called_once()
        assert isinstance(best_params, dict)

def test_booster_models_metadata_captures_tuning():
    try:
        import xgboost
    except ImportError:
        pytest.skip("xgboost not installed")
        
    # Verify the tracking of tuning params in class
    model = create_model("xgboost", task="classification", tuned=True, validation_method="time_series_split_optuna")
    meta = model.get_artifact_metadata()
    
    assert meta["tuning_enabled"] is True
    assert meta["validation_method"] == "time_series_split_optuna"
    assert meta["params"]["tuned"] is True
