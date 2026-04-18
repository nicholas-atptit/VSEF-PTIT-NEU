"""Execution policy that turns forecasts into signals and sized positions."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.strategy.signal_rules import StrategyModel
from src.strategy.sizing import size_positions
from src.strategy.thresholding import generate_threshold_signals


class BasicExecutionPolicy(StrategyModel):
    """Conservative threshold-and-size execution policy for Phase 1."""

    model_name = "basic_execution_policy"

    def __init__(
        self,
        *,
        threshold: float = 0.0,
        allow_short: bool = False,
        capital_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.threshold = max(float(threshold), 0.0)
        self.allow_short = bool(allow_short)
        self.capital_config = dict(capital_config or {})

    def generate_signal(
        self,
        forecast_df: pd.DataFrame,
        risk_df: pd.DataFrame | None = None,
        regime_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        signal_df = generate_threshold_signals(
            forecast_df,
            threshold=self.threshold,
            allow_short=self.allow_short,
        )
        if risk_df is not None and not risk_df.empty:
            passthrough = risk_df.copy()
            join_keys = [column for column in ("timestamp", "ticker", "model_name", "window_id") if column in signal_df.columns and column in passthrough.columns]
            if join_keys:
                signal_df = signal_df.merge(passthrough, on=join_keys, how="left", suffixes=("", "_risk"))
        if regime_df is not None and not regime_df.empty:
            passthrough = regime_df.copy()
            join_keys = [column for column in ("timestamp", "ticker") if column in signal_df.columns and column in passthrough.columns]
            if join_keys:
                signal_df = signal_df.merge(passthrough, on=join_keys, how="left", suffixes=("", "_regime"))
        return self.validate_signal_output(signal_df)

    def size_positions(
        self,
        signal_df: pd.DataFrame,
        risk_df: pd.DataFrame | None = None,
        capital_config: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        merged_capital_config = dict(self.capital_config)
        merged_capital_config.update(dict(capital_config or {}))
        positions = size_positions(
            signal_df,
            risk_df=risk_df,
            capital_config=merged_capital_config,
        )
        return self.validate_position_output(positions)
