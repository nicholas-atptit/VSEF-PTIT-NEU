"""Weighted forecast ensemble for the Phase 1 strategy layer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.ensemble.base import EnsembleModel


class WeightedEnsembleModel(EnsembleModel):
    """Combine compatible forecast frames using explicit or metric-derived weights."""

    model_name = "weighted_ensemble"

    def __init__(
        self,
        *,
        metric_column: str = "rmse",
        inverse_metric: bool = True,
        epsilon: float = 1e-9,
    ) -> None:
        super().__init__()
        self.metric_column = metric_column
        self.inverse_metric = bool(inverse_metric)
        self.epsilon = float(epsilon)

    def _resolve_weights(
        self,
        validated_frames: list[pd.DataFrame],
        context: dict[str, Any] | None,
    ) -> dict[str, float]:
        model_names = [str(frame["model_name"].iloc[0]) for frame in validated_frames]
        if context and isinstance(context.get("model_weights"), dict):
            raw = {str(key): float(value) for key, value in context["model_weights"].items()}
        elif context and isinstance(context.get("forecast_summary"), pd.DataFrame):
            summary = context["forecast_summary"].copy()
            if self.metric_column not in summary.columns or "model_name" not in summary.columns:
                raise ValueError(
                    f"forecast_summary must include 'model_name' and '{self.metric_column}' for weighted ensemble"
                )
            grouped = (
                summary.groupby("model_name", sort=True)[self.metric_column]
                .mean()
                .dropna()
                .astype(float)
            )
            if grouped.empty:
                raw = {name: 1.0 for name in model_names}
            elif self.inverse_metric:
                raw = {str(name): float(1.0 / max(value, self.epsilon)) for name, value in grouped.items()}
            else:
                raw = {str(name): float(value) for name, value in grouped.items()}
        else:
            raw = {name: 1.0 for name in model_names}

        total = float(sum(max(value, 0.0) for value in raw.values()))
        if total <= 0:
            return {name: 1.0 / len(model_names) for name in model_names}
        return {name: max(float(raw.get(name, 0.0)), 0.0) / total for name in model_names}

    def combine(
        self,
        prediction_frames: list[pd.DataFrame],
        context: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        validated_frames = self._validated_frames(prediction_frames)
        weights = self._resolve_weights(validated_frames, context)

        combined = pd.concat(validated_frames, ignore_index=True)
        combined["ensemble_weight"] = combined["model_name"].map(weights).fillna(0.0).astype(float)
        group_keys = [
            "timestamp",
            "ticker",
            "target_type",
            "horizon",
            "window_id",
        ]
        if "target_timestamp" in combined.columns:
            group_keys.append("target_timestamp")

        rows: list[dict[str, Any]] = []
        for keys, group in combined.groupby(group_keys, sort=True, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_keys, keys))
            weight_sum = float(group["ensemble_weight"].sum())
            normalized_weight = (
                group["ensemble_weight"] / weight_sum
                if weight_sum > 0
                else pd.Series(1.0 / len(group), index=group.index, dtype=float)
            )
            y_true = pd.to_numeric(group["y_true"], errors="coerce").dropna()
            row.update(
                {
                    "y_true": float(y_true.iloc[0]) if not y_true.empty else np.nan,
                    "y_pred": float((pd.to_numeric(group["y_pred"], errors="coerce") * normalized_weight).sum()),
                    "model_name": self.model_name,
                    "component_models": ",".join(sorted(group["model_name"].astype(str).unique())),
                    "component_count": int(group["model_name"].nunique()),
                    "weight_sum": float(weight_sum),
                }
            )
            rows.append(row)

        if not rows:
            raise ValueError("Weighted ensemble did not produce any combined rows")
        return pd.DataFrame(rows).sort_values(group_keys).reset_index(drop=True)
