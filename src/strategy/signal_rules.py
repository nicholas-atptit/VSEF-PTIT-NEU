"""Base strategy contract plus shared signal/position helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from src.core.contracts import validate_position_frame, validate_signal_frame


class StrategyModel(ABC):
    """Strategy contract built on forecast and risk outputs."""

    model_name = "base_strategy"

    def __init__(self, *, model_name: str | None = None) -> None:
        self.model_name = model_name or self.model_name

    @abstractmethod
    def generate_signal(
        self,
        forecast_df: pd.DataFrame,
        risk_df: pd.DataFrame | None = None,
        regime_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate signals from forecast outputs."""

    @abstractmethod
    def size_positions(
        self,
        signal_df: pd.DataFrame,
        risk_df: pd.DataFrame | None = None,
        capital_config: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Convert signals into sized positions."""

    @staticmethod
    def validate_signal_output(frame: pd.DataFrame) -> pd.DataFrame:
        return validate_signal_frame(frame)

    @staticmethod
    def validate_position_output(frame: pd.DataFrame) -> pd.DataFrame:
        return validate_position_frame(frame)
