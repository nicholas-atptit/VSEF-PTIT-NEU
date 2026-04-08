"""Central registry for supported ML algorithms."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .base import BaseModel


def _cart_cls():
    from .cart import CartModel

    return CartModel


def _lstm_cls():
    from .lstm import LstmModel

    return LstmModel


def _bilstm_cls():
    from .bilstm import BiLstmModel

    return BiLstmModel


MODEL_REGISTRY: dict[str, Callable[[], type[BaseModel]]] = {
    "cart": _cart_cls,
    "lstm": _lstm_cls,
    "bilstm": _bilstm_cls,
}


def _resolve_model(name: str) -> type[BaseModel]:
    algorithm = name.lower()
    if algorithm not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported algorithm '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[algorithm]()


def create_model(name: str, **kwargs) -> BaseModel:
    return _resolve_model(name)(**kwargs)


def load_model(name: str, artifact_path: str | Path) -> BaseModel:
    return _resolve_model(name).load(Path(artifact_path))


def supported_algorithms() -> list[str]:
    return sorted(MODEL_REGISTRY)
