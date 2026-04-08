"""Model package exports.

Keep this module intentionally lightweight because importing any submodule under
``src.ml.models`` executes this package initializer first.
"""

from src.ml.models.base import Base, BaseModel, TimestampMixin

__all__ = ["Base", "BaseModel", "TimestampMixin"]
