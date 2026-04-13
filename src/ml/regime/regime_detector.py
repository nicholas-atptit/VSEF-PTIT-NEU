"""Rule-based market regime detection using rolling risk metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REGIME_TO_CODE = {"NORMAL": 0, "HIGH_VOL": 1, "CRISIS": 2}


@dataclass(frozen=True)
class RegimeDetectionResult:
    labels: pd.Series
    encoded_labels: pd.Series
    probabilities: pd.DataFrame


class RegimeDetector:
    """Threshold-based regime detector with deterministic probability scores."""

    def __init__(
        self,
        *,
        method: str = "threshold",
        high_vol_threshold: float = 0.03,
        crisis_drawdown_threshold: float = -0.12,
        crisis_delta_covar_threshold: float = 0.015,
    ) -> None:
        self.method = method
        self.high_vol_threshold = high_vol_threshold
        self.crisis_drawdown_threshold = crisis_drawdown_threshold
        self.crisis_delta_covar_threshold = crisis_delta_covar_threshold

    @staticmethod
    def _safe_series(values: pd.Series | None, index: pd.Index, fill: float = 0.0) -> pd.Series:
        if values is None:
            return pd.Series(fill, index=index)
        return pd.to_numeric(values.reindex(index), errors="coerce").fillna(fill)

    def detect(
        self,
        *,
        volatility: pd.Series,
        drawdown: pd.Series,
        delta_covar: pd.Series | None = None,
    ) -> RegimeDetectionResult:
        index = volatility.index
        vol = self._safe_series(volatility, index=index)
        dd = self._safe_series(drawdown, index=index)
        dc = self._safe_series(delta_covar, index=index)

        crisis_mask = (dd <= self.crisis_drawdown_threshold) | (dc >= self.crisis_delta_covar_threshold)
        high_vol_mask = (~crisis_mask) & (vol >= self.high_vol_threshold)

        labels = pd.Series("NORMAL", index=index, dtype="object")
        labels.loc[high_vol_mask] = "HIGH_VOL"
        labels.loc[crisis_mask] = "CRISIS"

        vol_score = (vol / max(self.high_vol_threshold, 1e-9)).clip(lower=0.0, upper=3.0)
        dd_score = (-dd / max(abs(self.crisis_drawdown_threshold), 1e-9)).clip(lower=0.0, upper=3.0)
        dc_score = (dc / max(self.crisis_delta_covar_threshold, 1e-9)).clip(lower=0.0, upper=3.0)

        crisis_score = ((dd_score + dc_score) / 2.0).clip(0.0, 1.0)
        high_vol_score = (vol_score - crisis_score * 0.5).clip(0.0, 1.0)
        normal_score = (1.0 - np.maximum(crisis_score, high_vol_score)).clip(0.0, 1.0)

        probs = pd.DataFrame(
            {
                "NORMAL": normal_score,
                "HIGH_VOL": high_vol_score,
                "CRISIS": crisis_score,
            },
            index=index,
        )
        probs = probs.div(probs.sum(axis=1).replace(0.0, 1.0), axis=0)
        encoded = labels.map(REGIME_TO_CODE).astype(float).rename("regime_label")
        return RegimeDetectionResult(
            labels=labels.rename("regime_name"),
            encoded_labels=encoded,
            probabilities=probs,
        )

    def detect_from_frame(
        self,
        frame: pd.DataFrame,
        *,
        volatility_col: str = "rolling_volatility_20",
        drawdown_col: str = "rolling_drawdown",
        delta_covar_col: str = "delta_covar",
    ) -> RegimeDetectionResult:
        volatility = frame[volatility_col] if volatility_col in frame.columns else pd.Series(index=frame.index, dtype=float)
        drawdown = frame[drawdown_col] if drawdown_col in frame.columns else pd.Series(index=frame.index, dtype=float)
        delta_covar = frame[delta_covar_col] if delta_covar_col in frame.columns else pd.Series(index=frame.index, dtype=float)
        return self.detect(
            volatility=pd.to_numeric(volatility, errors="coerce"),
            drawdown=pd.to_numeric(drawdown, errors="coerce"),
            delta_covar=pd.to_numeric(delta_covar, errors="coerce"),
        )
