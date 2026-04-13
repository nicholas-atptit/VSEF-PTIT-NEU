import pytest
import numpy as np

from src.ml.models.factory import create_model

def test_stacking_model_time_series_safe(tmp_path):
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100)
    
    # Use regression for simplicity
    # Constraint: v1 must limit base learners.
    base_learners = ["cart", "sarimax", "ets"]
    
    try:
        import xgboost
        base_learners.append("xgboost")
    except ImportError:
        pass
        
    try:
        import lightgbm
        base_learners.append("lightgbm")
    except ImportError:
        pass

    model = create_model("stacking", task="regression", base_learners=base_learners, max_depth=3, n_estimators=10)
    
    assert model.get_artifact_metadata()["algorithm"] == "stacking"
    
    model.fit(X_train, y_train)
    
    assert len(model.fitted_base_models) == len(model.base_learners_names)
    assert set(model.base_learners_names).issubset(set(base_learners))
    
    # Predict
    preds = model.predict(X_train[:5])
    assert preds.shape == (5,)
    
    # Artifact save/load
    artifact_path = tmp_path / "stacking.joblib"
    model.save(artifact_path)
    loaded = model.load(artifact_path)
    
    np.testing.assert_array_almost_equal(preds, loaded.predict(X_train[:5]))
    
    # Check meta tracking
    meta = model.get_artifact_metadata()
    assert meta["meta_model_type"] == "Ridge"
    assert meta["n_splits"] == 5
    assert "TimeSeriesSplit" in meta["validation_split"]
    assert set(meta["base_learners"]) == set(model.base_learners_names)


def test_stacking_classification(tmp_path):
    X_train = np.random.rand(100, 3)
    y_train = (np.random.rand(100) > 0.5).astype(int)
    
    model = create_model("stacking", task="classification", base_learners=["cart"])
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_train[:5])
    assert probs.shape == (5, 2)
    assert np.all((probs >= 0) & (probs <= 1))
    
    meta = model.get_artifact_metadata()
    assert meta["meta_model_type"] == "LogisticRegression"

def test_stacking_edge_cases():
    # 1. Very short sample => graceful n_splits downgrade
    X_short = np.random.rand(8, 2)
    y_short = np.random.rand(8)
    
    model1 = create_model("stacking", task="regression", base_learners=["cart"], n_splits=5)
    model1.fit(X_short, y_short)
    meta1 = model1.get_artifact_metadata()
    # 8 samples cannot support 5 splits cleanly, should downgrade (max(2, 8 // 10) -> 2)
    assert meta1["n_splits"] < 5
    
    # 2. Classification fold stability with weak distribution
    X_class = np.random.rand(100, 2)
    # create highly imbalanced target where mostly 0
    y_class = np.zeros(100)
    y_class[90:] = 1  # only 10 positive cases at the very end
    
    model2 = create_model("stacking", task="classification", base_learners=["cart"], n_splits=3)
    # Shouldn't crash even if fold validation misses a class entirely
    # The LogRegression uses class_weight='balanced'
    model2.fit(X_class, y_class)
    assert model2.predict_proba(X_class[:2]).shape == (2, 2)
