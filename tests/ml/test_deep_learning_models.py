import pytest
import numpy as np

from src.ml.models.factory import create_model

pytest.importorskip("torch")

def test_lstm_sequence_compatibility(tmp_path):
    # Deep learning sequence models expect 3D data: (batch, seq_len, features)
    X_train = np.random.rand(100, 10, 5) # 100 samples, 10 sequence length, 5 features
    y_train = np.random.rand(100)
    
    # Regression
    model = create_model("lstm", task="regression", hidden_size=8, num_layers=1, epochs=2)
    assert model.get_artifact_metadata()["algorithm"] == "lstm"
    assert model.get_model_capabilities()["requires_sequence_data"] is True
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    preds = model.predict(X_train[:5])
    assert preds.shape == (5,)
    
    # Check save/load artifact compatibility using .pt torch save format
    artifact_path = tmp_path / "lstm.pt"
    model.save(artifact_path)
    
    loaded = model.load(artifact_path)
    loaded_preds = loaded.predict(X_train[:5])
    
    np.testing.assert_array_almost_equal(preds, loaded_preds, decimal=4)

def test_bilstm_sequence_compatibility(tmp_path):
    X_train = np.random.rand(50, 10, 5)
    y_train = (np.random.rand(50) > 0.5).astype(int)
    
    # Classification
    model = create_model("bilstm", task="classification", hidden_size=8, num_layers=1, epochs=2)
    assert model.get_artifact_metadata()["algorithm"] == "bilstm"
    assert model.get_model_capabilities()["requires_sequence_data"] is True
    
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_train[:5])
    assert probs.shape == (5, 2)
    assert np.all((probs >= 0) & (probs <= 1))
    
    artifact_path = tmp_path / "bilstm.pt"
    model.save(artifact_path)
    loaded = model.load(artifact_path)
    
    np.testing.assert_array_almost_equal(probs, loaded.predict_proba(X_train[:5]), decimal=4)
