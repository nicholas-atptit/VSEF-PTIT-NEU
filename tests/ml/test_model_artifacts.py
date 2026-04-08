from __future__ import annotations

import numpy as np
import pytest

from src.ml.models.cart import CartModel


def test_cart_save_and_load_round_trip(tmp_path) -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    model = CartModel(task="classification", max_depth=2)
    model.fit(X, y)
    artifact_path = tmp_path / "cart.joblib"
    model.save(artifact_path)

    loaded = CartModel.load(artifact_path)
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))


def test_lstm_save_and_load_round_trip(tmp_path) -> None:
    pytest.importorskip("torch")
    from src.ml.models.lstm import LstmModel

    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(12, 4, 3))
    y_train = np.array([0, 1] * 6)
    X_val = rng.normal(size=(4, 4, 3))
    y_val = np.array([0, 1, 0, 1])

    model = LstmModel(
        task="classification",
        sequence_length=4,
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
        batch_size=4,
        epochs=2,
        patience=1,
    )
    model.fit(X_train, y_train, X_val, y_val)
    artifact_path = tmp_path / "lstm.pt"
    model.save(artifact_path)

    loaded = LstmModel.load(artifact_path)
    np.testing.assert_array_equal(model.predict(X_val), loaded.predict(X_val))


def test_bilstm_save_and_load_round_trip(tmp_path) -> None:
    pytest.importorskip("torch")
    from src.ml.models.bilstm import BiLstmModel

    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(12, 4, 3))
    y_train = np.array([0, 1] * 6)
    X_val = rng.normal(size=(4, 4, 3))
    y_val = np.array([1, 0, 1, 0])

    model = BiLstmModel(
        task="classification",
        sequence_length=4,
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
        batch_size=4,
        epochs=2,
        patience=1,
    )
    model.fit(X_train, y_train, X_val, y_val)
    artifact_path = tmp_path / "bilstm.pt"
    model.save(artifact_path)

    loaded = BiLstmModel.load(artifact_path)
    np.testing.assert_array_equal(model.predict(X_val), loaded.predict(X_val))
