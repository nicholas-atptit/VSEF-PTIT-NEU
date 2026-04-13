import pytest
import numpy as np

pytest.importorskip("statsmodels.api")

from src.ml.models.sarimax import SarimaxModel
from src.ml.models.ets import EtsModel

def test_sarimax_model(tmp_path):
    X_train = np.random.rand(100, 3)
    y_train = np.random.rand(100)
    
    model = SarimaxModel(task="regression")
    model.fit(X_train, y_train)
    
    # Predict 5 steps based on test features length 5
    X_test = np.random.rand(5, 3)
    preds = model.predict(X_test)
    assert preds.shape == (5,)
    
    # Test class mapping
    class_model = SarimaxModel(task="classification")
    y_class = (y_train > 0.5).astype(int)
    class_model.fit(X_train, y_class)
    class_preds = class_model.predict(X_test)
    assert class_preds.shape == (5,)
    assert set(np.unique(class_preds)).issubset({0, 1})
    
    probs = class_model.predict_proba(X_test)
    assert probs.shape == (5, 2)
    assert np.all((probs >= 0) & (probs <= 1))
    
    # Save/load logic
    artifact_path = tmp_path / "sarimax_class.joblib"
    class_model.save(artifact_path)
    loaded_model = SarimaxModel.load(artifact_path)
    loaded_probs = loaded_model.predict_proba(X_test)
    np.testing.assert_array_almost_equal(probs, loaded_probs)
    
    # Check heuristic metadata flag
    meta = loaded_model.get_artifact_metadata()
    assert meta["heuristic_probabilities"] is True
    assert "convergence_warnings" in meta

def test_ets_model(tmp_path):
    # ETS shouldn't be affected by X shape but we pass it anyway.
    X_train = np.random.rand(100, 3)
    # Ensure positive targets since multiplicative trend demands strict positives if it were used
    y_train = np.random.rand(100) + 1.0 
    
    model = EtsModel(task="regression", trend="add")
    model.fit(X_train, y_train)
    
    X_test = np.random.rand(3, 3)
    preds = model.predict(X_test)
    assert preds.shape == (3,)
    
    # Classification tests
    class_model = EtsModel(task="classification", trend="add")
    # For classification label [0, 1] ETS can handle them if additive
    y_class = (np.random.rand(100) > 0.5).astype(int)
    class_model.fit(X_train, y_class)
    probs = class_model.predict_proba(X_test)
    assert probs.shape == (3, 2)
    assert np.all((probs >= 0) & (probs <= 1))
    
    # Save/load logic
    artifact_path = tmp_path / "ets_class.joblib"
    class_model.save(artifact_path)
    loaded_model = EtsModel.load(artifact_path)
    loaded_probs = loaded_model.predict_proba(X_test)
    np.testing.assert_array_almost_equal(probs, loaded_probs)
    
    # Check heuristic metadata flag
    meta = loaded_model.get_artifact_metadata()
    assert meta["heuristic_probabilities"] is True
    assert "convergence_warnings" in meta
