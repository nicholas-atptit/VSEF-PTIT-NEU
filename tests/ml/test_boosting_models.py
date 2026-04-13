import pytest
import numpy as np

from src.ml.models.factory import create_model

try:
    import xgboost
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


@pytest.mark.skipif(not HAS_XGB, reason="xgboost not installed")
def test_xgboost_model(tmp_path):
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100)
    
    # Regression
    model = create_model("xgboost", task="regression", n_estimators=10)
    assert model.get_artifact_metadata()["algorithm"] == "xgboost"
    model.fit(X_train, y_train)
    preds = model.predict(X_train[:5])
    assert preds.shape == (5,)
    
    # Classification
    y_class = (y_train > 0.5).astype(int)
    class_model = create_model("xgboost", task="classification", n_estimators=10)
    class_model.fit(X_train, y_class)
    
    probs = class_model.predict_proba(X_train[:5])
    assert probs.shape == (5, 2)
    
    # Save Load
    artifact_path = tmp_path / "xgb.joblib"
    class_model.save(artifact_path)
    loaded = class_model.load(artifact_path)
    np.testing.assert_array_almost_equal(probs, loaded.predict_proba(X_train[:5]))


@pytest.mark.skipif(not HAS_LGB, reason="lightgbm not installed")
def test_lightgbm_model(tmp_path):
    # Lightgbm is extremely sensitive to dataset bounds, provide more rows
    X_train = np.random.rand(200, 5)
    y_train = np.random.rand(200)
    
    # Regression
    model = create_model("lightgbm", task="regression", n_estimators=10)
    assert model.get_artifact_metadata()["algorithm"] == "lightgbm"
    model.fit(X_train, y_train)
    preds = model.predict(X_train[:5])
    assert preds.shape == (5,)
    
    # Classification
    y_class = (y_train > 0.5).astype(int)
    class_model = create_model("lightgbm", task="classification", n_estimators=10)
    class_model.fit(X_train, y_class)
    
    probs = class_model.predict_proba(X_train[:5])
    assert probs.shape == (5, 2)
    
    # Save Load
    artifact_path = tmp_path / "lgb.joblib"
    class_model.save(artifact_path)
    loaded = class_model.load(artifact_path)
    np.testing.assert_array_almost_equal(probs, loaded.predict_proba(X_train[:5]))
