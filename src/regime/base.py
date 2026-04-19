"""Shared regime-model contracts for the Phase 2 architecture."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from src.core.contracts import validate_regime_frame


class RegimeModel(ABC):
    """Explicit regime contract built on time-ordered single-ticker history."""

    model_name = "base_regime"

    def __init__(
        self,
        *,
        model_name: str | None = None,
        return_column: str = "daily_return",
        price_column: str = "close",
    ) -> None:
        self.model_name = model_name or self.model_name
        self.return_column = return_column
        self.price_column = price_column
        self.config: dict[str, Any] = {}
        self.ticker_: str | None = None
        self.window_id_: str = "unassigned"
        self.source_model_: str = self.model_name
        self._is_fitted = False

    def fit(
        self,
        history_df: pd.DataFrame,
        config: dict[str, Any] | None = None,
    ) -> "RegimeModel":
        prepared = self._prepare_frame(history_df)
        self.config = dict(config or {})
        self.ticker_ = str(prepared["ticker"].iloc[0])
        self.window_id_ = str(self.config.get("window_id", "unassigned"))
        self.source_model_ = self.model_name
        self._fit_model(prepared)
        self._is_fitted = True
        return self

    def predict(self, history_df: pd.DataFrame) -> pd.DataFrame:
        self._ensure_fitted()
        prepared = self._prepare_frame(history_df)
        probabilities = self._predict_probabilities(prepared)
        return self._build_regime_frame(prepared, probabilities)

    def predict_in_sample(self, history_df: pd.DataFrame) -> pd.DataFrame:
        return self.predict(history_df)

    def get_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "return_column": self.return_column,
            "price_column": self.price_column,
            "ticker": self.ticker_,
            "window_id": self.window_id_,
            "source_model": self.source_model_,
            "config": dict(self.config),
        }

    @staticmethod
    def _coerce_numeric_series(values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return numeric.astype(float)

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Regime model received an empty frame")
        prepared = df.copy()
        if "timestamp" not in prepared.columns:
            if "date" in prepared.columns:
                prepared = prepared.rename(columns={"date": "timestamp"})
            else:
                raise ValueError("Regime model requires a timestamp/date column")
        if "ticker" not in prepared.columns:
            raise ValueError("Regime model requires a ticker column")
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
        if prepared["timestamp"].isna().any():
            raise ValueError("Regime model received invalid timestamps")
        prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
        if prepared["ticker"].nunique() != 1:
            raise ValueError(
                f"{self.model_name} expects a single ticker per fit/predict call; "
                f"got {prepared['ticker'].nunique()}"
            )
        if self.return_column not in prepared.columns:
            if self.price_column not in prepared.columns:
                raise ValueError(
                    f"Regime model requires either '{self.return_column}' or '{self.price_column}'"
                )
            prepared[self.return_column] = self._coerce_numeric_series(prepared[self.price_column]).pct_change()
        else:
            prepared[self.return_column] = self._coerce_numeric_series(prepared[self.return_column])
        return prepared.sort_values("timestamp").reset_index(drop=True)

    def _ensure_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.model_name} is not fitted")

    def _build_regime_frame(
        self,
        frame: pd.DataFrame,
        probabilities: pd.DataFrame,
    ) -> pd.DataFrame:
        if len(probabilities) != len(frame):
            raise ValueError("Regime probabilities must align to the input frame length")
        result = pd.DataFrame(
            {
                "timestamp": frame["timestamp"].to_numpy(),
                "ticker": frame["ticker"].to_numpy(),
                "regime_label": probabilities["regime_label"].astype(str).to_numpy(),
                "regime_prob_bull": pd.to_numeric(probabilities["regime_prob_bull"], errors="coerce").astype(float).to_numpy(),
                "regime_prob_bear": pd.to_numeric(probabilities["regime_prob_bear"], errors="coerce").astype(float).to_numpy(),
                "regime_prob_sideway": pd.to_numeric(probabilities["regime_prob_sideway"], errors="coerce").astype(float).to_numpy(),
                "source_model": self.source_model_,
                "window_id": frame.get("window_id", pd.Series(self.window_id_, index=frame.index)).astype(str).to_numpy(),
            }
        )
        return validate_regime_frame(result)

    @abstractmethod
    def _fit_model(self, history_frame: pd.DataFrame) -> None:
        """Fit the regime model on the provided in-sample history."""

    @abstractmethod
    def _predict_probabilities(self, history_frame: pd.DataFrame) -> pd.DataFrame:
        """Return bull/bear/sideway probabilities for each row in the history frame."""
