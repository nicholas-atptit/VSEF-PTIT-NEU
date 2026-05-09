"""Deterministic baseline registry for Phase 1 experiment runs.

Baselines are comparison evidence only. They are not forecasting models in the
Phase 0 frozen model registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


OHLCV_COLUMNS = {"date", "ticker", "open", "high", "low", "close", "volume"}


@dataclass(frozen=True)
class BaselineSpec:
    """Registered baseline metadata."""

    name: str
    description: str
    runner: Callable[[pd.DataFrame, int, dict[str, Any]], pd.DataFrame]


class BaselineRegistry:
    """Registry and execution facade for deterministic baseline methods."""

    def __init__(self) -> None:
        self._registry: dict[str, BaselineSpec] = {
            "persistence": BaselineSpec(
                name="persistence",
                description="Predict the future close or target value from the latest observed value.",
                runner=self._run_persistence,
            ),
            "zero_return": BaselineSpec(
                name="zero_return",
                description="Predict zero return, or current close for price targets.",
                runner=self._run_zero_return,
            ),
            "random_direction": BaselineSpec(
                name="random_direction",
                description="Seed-controlled random up/down direction baseline.",
                runner=self._run_random_direction,
            ),
            "moving_average_rule": BaselineSpec(
                name="moving_average_rule",
                description="Predict from a trailing moving average window.",
                runner=self._run_moving_average_rule,
            ),
        }

    def list_baselines(self) -> list[str]:
        """Return supported baseline names."""
        return sorted(self._registry)

    def get_baseline(self, name: str) -> BaselineSpec:
        """Return a registered baseline spec."""
        key = str(name).strip().lower()
        if key not in self._registry:
            raise ValueError(f"Unsupported baseline '{name}'. Available: {self.list_baselines()}")
        return self._registry[key]

    def run_baseline(self, name: str, data: pd.DataFrame, horizon: int, config: dict[str, Any]) -> pd.DataFrame:
        """Run a baseline and return standardized prediction rows."""
        if int(horizon) <= 0:
            raise ValueError("horizon must be a positive integer")
        self._validate_input(data)
        spec = self.get_baseline(name)
        return spec.runner(data.copy(), int(horizon), dict(config or {}))

    @staticmethod
    def _validate_input(data: pd.DataFrame) -> None:
        missing = OHLCV_COLUMNS - set(data.columns)
        if missing:
            raise ValueError(f"Baseline data is missing OHLCV columns: {sorted(missing)}")
        if data.empty:
            raise ValueError("Baseline data is empty")

    @staticmethod
    def _base_frame(data: pd.DataFrame, horizon: int, config: dict[str, Any]) -> pd.DataFrame:
        target_cfg = dict(config.get("target") or {})
        target_column = str(target_cfg.get("column", "close")).lower()
        task_type = str(target_cfg.get("task_type", "regression")).lower()

        frame = data.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame["future_close"] = frame["close"].shift(-int(horizon))
        frame["current_close"] = frame["close"]
        frame["observed_return"] = frame["close"].pct_change()
        frame["future_return"] = (frame["future_close"] / frame["current_close"]) - 1.0

        if target_column in {"forward_return", "return", "future_return"} or "return" in target_column:
            frame["y_true"] = frame["future_return"]
            frame["prediction_reference"] = 0.0
            frame["target_kind"] = "return"
        elif task_type == "classification" or "direction" in target_column:
            frame["y_true"] = np.where(frame["future_close"] > frame["current_close"], 1.0, 0.0)
            frame["prediction_reference"] = 0.5
            frame["target_kind"] = "direction"
        else:
            frame["y_true"] = frame["future_close"]
            frame["prediction_reference"] = frame["current_close"]
            frame["target_kind"] = "price"

        frame = frame.dropna(subset=["y_true", "current_close"]).reset_index(drop=True)
        return frame

    @staticmethod
    def _finalize(
        frame: pd.DataFrame,
        *,
        horizon: int,
        model_name: str,
        y_pred: pd.Series | np.ndarray,
        notes: str,
    ) -> pd.DataFrame:
        output = frame.copy()
        output["y_pred"] = pd.to_numeric(pd.Series(y_pred, index=output.index), errors="coerce")

        price_reference = pd.to_numeric(output["current_close"], errors="coerce")
        if "target_kind" in output.columns and (output["target_kind"] == "return").all():
            output["actual_direction"] = np.sign(pd.to_numeric(output["y_true"], errors="coerce"))
            output["predicted_direction"] = np.sign(output["y_pred"])
        elif "target_kind" in output.columns and (output["target_kind"] == "direction").all():
            output["actual_direction"] = np.where(output["y_true"] >= 0.5, 1, -1)
            output["predicted_direction"] = np.where(output["y_pred"] >= 0.5, 1, -1)
        else:
            output["actual_direction"] = np.sign(pd.to_numeric(output["y_true"], errors="coerce") - price_reference)
            output["predicted_direction"] = np.sign(output["y_pred"] - price_reference)

        output["horizon"] = int(horizon)
        output["model_name"] = model_name
        output["model_type"] = "baseline"
        output["notes"] = notes
        return output[
            [
                "date",
                "ticker",
                "horizon",
                "model_name",
                "model_type",
                "y_true",
                "y_pred",
                "predicted_direction",
                "actual_direction",
                "notes",
            ]
        ].reset_index(drop=True)

    def _run_persistence(self, data: pd.DataFrame, horizon: int, config: dict[str, Any]) -> pd.DataFrame:
        frame = self._base_frame(data, horizon, config)
        if frame.empty:
            return self._empty_prediction_frame()
        if (frame["target_kind"] == "return").all():
            y_pred = frame["observed_return"].fillna(0.0)
        elif (frame["target_kind"] == "direction").all():
            y_pred = np.where(frame["observed_return"].fillna(0.0) >= 0.0, 1.0, 0.0)
        else:
            y_pred = frame["current_close"]
        return self._finalize(
            frame,
            horizon=horizon,
            model_name="persistence",
            y_pred=y_pred,
            notes="latest_observed_value",
        )

    def _run_zero_return(self, data: pd.DataFrame, horizon: int, config: dict[str, Any]) -> pd.DataFrame:
        frame = self._base_frame(data, horizon, config)
        if frame.empty:
            return self._empty_prediction_frame()
        if (frame["target_kind"] == "return").all():
            y_pred = pd.Series(0.0, index=frame.index)
        elif (frame["target_kind"] == "direction").all():
            y_pred = pd.Series(0.5, index=frame.index)
        else:
            y_pred = frame["current_close"]
        return self._finalize(
            frame,
            horizon=horizon,
            model_name="zero_return",
            y_pred=y_pred,
            notes="zero_return_or_current_close",
        )

    def _run_random_direction(self, data: pd.DataFrame, horizon: int, config: dict[str, Any]) -> pd.DataFrame:
        frame = self._base_frame(data, horizon, config)
        if frame.empty:
            return self._empty_prediction_frame()
        seed = int(config.get("seed", 42))
        rng = np.random.default_rng(seed + int(horizon))
        direction = rng.choice([-1.0, 1.0], size=len(frame))
        if (frame["target_kind"] == "return").all():
            scale = frame["observed_return"].abs().replace(0.0, np.nan).median()
            scale = 0.001 if pd.isna(scale) else float(scale)
            y_pred = direction * scale
        elif (frame["target_kind"] == "direction").all():
            y_pred = np.where(direction > 0, 1.0, 0.0)
        else:
            y_pred = frame["current_close"] * (1.0 + direction * 0.001)
        return self._finalize(
            frame,
            horizon=horizon,
            model_name="random_direction",
            y_pred=y_pred,
            notes=f"seed_controlled_random_direction_seed={seed}",
        )

    def _run_moving_average_rule(self, data: pd.DataFrame, horizon: int, config: dict[str, Any]) -> pd.DataFrame:
        frame = self._base_frame(data, horizon, config)
        if frame.empty:
            return self._empty_prediction_frame()
        baselines_cfg = dict(config.get("baselines") or {})
        params = dict(baselines_cfg.get("params") or {}).get("moving_average_rule", {})
        window = int(params.get("window", config.get("moving_average_window", 5)))
        window = max(1, window)
        if (frame["target_kind"] == "return").all():
            y_pred = frame["observed_return"].rolling(window=window, min_periods=1).mean().fillna(0.0)
        elif (frame["target_kind"] == "direction").all():
            rolling_return = frame["observed_return"].rolling(window=window, min_periods=1).mean().fillna(0.0)
            y_pred = np.where(rolling_return >= 0.0, 1.0, 0.0)
        else:
            y_pred = frame["current_close"].rolling(window=window, min_periods=1).mean()
        return self._finalize(
            frame,
            horizon=horizon,
            model_name="moving_average_rule",
            y_pred=y_pred,
            notes=f"moving_average_window={window}",
        )

    @staticmethod
    def _empty_prediction_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "horizon",
                "model_name",
                "model_type",
                "y_true",
                "y_pred",
                "predicted_direction",
                "actual_direction",
                "notes",
            ]
        )
