"""Contract helpers shared across forecast, risk, ensemble, and strategy layers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


FORECAST_REQUIRED_COLUMNS = (
    "timestamp",
    "ticker",
    "y_true",
    "y_pred",
    "model_name",
    "target_type",
    "horizon",
    "window_id",
)

SIGNAL_REQUIRED_COLUMNS = (
    "timestamp",
    "ticker",
    "model_name",
    "signal",
    "threshold",
)

POSITION_REQUIRED_COLUMNS = (
    "timestamp",
    "ticker",
    "model_name",
    "signal",
    "position_size",
)


@dataclass(frozen=True)
class WalkForwardWindow:
    """Explicit train/test window definition used by the walk-forward evaluator."""

    window_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    gap_size: int = 0
    train_rows: int = 0
    test_rows: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "window_id": self.window_id,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "gap_size": int(self.gap_size),
            "train_rows": int(self.train_rows),
            "test_rows": int(self.test_rows),
        }


def _ensure_datetime(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    prepared = frame.copy()
    prepared[column] = pd.to_datetime(prepared[column], errors="coerce")
    return prepared


def validate_forecast_frame(
    frame: pd.DataFrame,
    *,
    require_y_true: bool = False,
) -> pd.DataFrame:
    """Validate and normalize the shared forecast output schema."""

    missing = [column for column in FORECAST_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Forecast frame missing required columns: {missing}")

    prepared = _ensure_datetime(frame, "timestamp")
    if prepared["timestamp"].isna().any():
        raise ValueError("Forecast frame contains invalid timestamps")

    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    prepared["model_name"] = prepared["model_name"].astype(str)
    prepared["target_type"] = prepared["target_type"].astype(str)
    prepared["window_id"] = prepared["window_id"].astype(str)
    prepared["horizon"] = pd.to_numeric(prepared["horizon"], errors="raise").astype(int)
    prepared["y_pred"] = pd.to_numeric(prepared["y_pred"], errors="coerce").astype(float)
    prepared["y_true"] = pd.to_numeric(prepared["y_true"], errors="coerce").astype(float)
    if prepared["y_pred"].isna().any():
        raise ValueError("Forecast frame contains invalid predictions")
    if require_y_true and prepared["y_true"].isna().any():
        raise ValueError("Forecast frame requires non-null y_true values")

    if "target_timestamp" in prepared.columns:
        prepared = _ensure_datetime(prepared, "target_timestamp")
    return prepared.sort_values(["timestamp", "ticker", "model_name"]).reset_index(drop=True)


def validate_signal_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in SIGNAL_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Signal frame missing required columns: {missing}")
    prepared = _ensure_datetime(frame, "timestamp")
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    prepared["model_name"] = prepared["model_name"].astype(str)
    prepared["signal"] = pd.to_numeric(prepared["signal"], errors="coerce").fillna(0.0).astype(float)
    prepared["threshold"] = pd.to_numeric(prepared["threshold"], errors="coerce").fillna(0.0).astype(float)
    return prepared.sort_values(["timestamp", "ticker", "model_name"]).reset_index(drop=True)


def validate_position_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in POSITION_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Position frame missing required columns: {missing}")
    prepared = _ensure_datetime(frame, "timestamp")
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    prepared["model_name"] = prepared["model_name"].astype(str)
    prepared["signal"] = pd.to_numeric(prepared["signal"], errors="coerce").fillna(0.0).astype(float)
    prepared["position_size"] = pd.to_numeric(prepared["position_size"], errors="coerce").fillna(0.0).astype(float)
    return prepared.sort_values(["timestamp", "ticker", "model_name"]).reset_index(drop=True)

