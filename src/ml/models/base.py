"""Shared SQLAlchemy base objects and ML model interface contracts."""

from __future__ import annotations

import abc
import datetime as dt
from pathlib import Path
from typing import Any, Dict

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for SQLAlchemy ORM models."""


class TimestampMixin:
    """Mixin that adds created_at and updated_at columns."""

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class BaseModel(abc.ABC):
    """Abstract contract for trainable ML models."""

    @abc.abstractmethod
    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any = None,
        y_val: Any = None,
    ) -> None:
        """Train the model."""

    @abc.abstractmethod
    def predict(self, X: Any) -> Any:
        """Return model predictions."""

    def predict_proba(self, X: Any) -> Any:  # pragma: no cover
        """Return class probabilities when supported."""
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, artifact_path: Path) -> None:
        """Persist model weights and companion metadata."""

    @classmethod
    @abc.abstractmethod
    def load(cls, artifact_path: Path) -> "BaseModel":
        """Load a persisted model bundle."""

    @classmethod
    def get_model_capabilities(cls) -> Dict[str, Any]:
        """Return declared capabilities of this model class."""
        return {
            "algorithm": "unknown",
            "model_family": "unknown",
            "requires_sequence_data": False,
            "supports_exogenous_features": False,
            "artifact_type": "unknown",
        }

    @abc.abstractmethod
    def get_artifact_metadata(self) -> Dict[str, Any]:
        """Return serializable metadata required for reloading and auditing."""

    def metadata(self) -> Dict[str, Any]:
        """Compatibility alias for older callers."""
        return self.get_artifact_metadata()
