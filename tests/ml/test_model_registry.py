import pytest

from src.ml.models.base import BaseModel
from src.ml.models.cart import CartModel
from src.ml.models.factory import _resolve_model, supported_algorithms

try:
    from src.ml.models.lstm import LstmModel
    from src.ml.models.bilstm import BiLstmModel
    HAS_TORCH = True
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    LstmModel = None
    BiLstmModel = None
    HAS_TORCH = False

def test_base_model_capabilities():
    caps = BaseModel.get_model_capabilities()
    assert caps["algorithm"] == "unknown"
    assert caps["model_family"] == "unknown"

def test_cart_capabilities():
    caps = CartModel.get_model_capabilities()
    assert caps["algorithm"] == "cart"
    assert caps["model_family"] == "tree"
    assert not caps["requires_sequence_data"]
    assert caps["supports_exogenous_features"]
    assert caps["artifact_type"] == "joblib"

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_lstm_capabilities():
    caps = LstmModel.get_model_capabilities()
    assert caps["algorithm"] == "lstm"
    assert caps["model_family"] == "deep_learning"
    assert caps["requires_sequence_data"]
    assert caps["supports_exogenous_features"]
    assert caps["artifact_type"] == "torch"

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_bilstm_capabilities():
    caps = BiLstmModel.get_model_capabilities()
    assert caps["algorithm"] == "bilstm"
    assert caps["model_family"] == "deep_learning"
    assert caps["requires_sequence_data"]
    assert caps["supports_exogenous_features"]
    assert caps["artifact_type"] == "torch"

def test_factory_supported_algorithms():
    algos = supported_algorithms()
    assert "cart" in algos
    assert "lstm" in algos
    assert "bilstm" in algos

def test_factory_resolve_capabilities():
    for algo in supported_algorithms():
        try:
            model_cls = _resolve_model(algo)
        except ModuleNotFoundError as exc:
            if exc.name in {"torch", "statsmodels"}:
                continue
            raise
        caps = model_cls.get_model_capabilities()
        assert caps["algorithm"] == algo
        assert caps["model_family"] in ["tree", "deep_learning", "statistical", "boosting", "ensemble"]
