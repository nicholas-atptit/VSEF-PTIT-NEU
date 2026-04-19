"""Base contracts for Phase 1 forecast ensembles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from src.core.contracts import validate_forecast_frame


class EnsembleModel(ABC):
    """Ensemble contract combining compatible forecast frames."""

    model_name = "base_ensemble"

    def __init__(self, *, model_name: str | None = None) -> None:
        self.model_name = model_name or self.model_name

    @abstractmethod
    def combine(
        self,
        prediction_frames: list[pd.DataFrame],
        context: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Combine forecast frames into a single forecast frame."""

    @staticmethod
    def _validated_frames(prediction_frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
        if not prediction_frames:
            raise ValueError("At least one prediction frame is required for ensemble combination")
        return [validate_forecast_frame(frame) for frame in prediction_frames]

