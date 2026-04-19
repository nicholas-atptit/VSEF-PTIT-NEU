"""Unified walk-forward evaluation backbone for Phase 1 forecasts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.core.contracts import WalkForwardWindow, validate_forecast_frame
from src.evaluation.targets import apply_target_spec, build_target_spec
from src.forecast.base import ForecastModel
from src.ml.features.registry import resolve_feature_set, resolve_task_feature_set
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PreparedEvaluationData:
    """Prepared per-ticker dataset used by the walk-forward evaluator."""

    ticker: str
    frame: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    source: str


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for leakage-safe expanding or rolling walk-forward evaluation."""

    tickers: list[str]
    horizon: int
    train_size: int = 252
    test_size: int = 21
    step_size: int = 21
    gap_size: int = 0
    expanding_window: bool = True
    prepared_dir: str = "data/processed/ml_5y"
    raw_dir: str = "data/daily_market_split_data"
    feature_columns: list[str] | None = None
    target_column: str = "target_forward_return"
    target_type: str = "forward_return"
    target_params: dict[str, Any] | None = None
    start_date: str | None = None
    end_date: str | None = None
    seed: int = 42
    max_windows: int | None = None


def add_forward_return_target(
    frame: pd.DataFrame,
    *,
    horizon: int,
    target_column: str = "target_forward_return",
) -> pd.DataFrame:
    """Add a forward-return target and aligned target timestamp."""

    prepared = frame.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
    prepared = prepared.sort_values("timestamp").reset_index(drop=True)
    prepared[target_column] = (
        pd.to_numeric(prepared["close"], errors="coerce").shift(-int(horizon))
        / pd.to_numeric(prepared["close"], errors="coerce")
        - 1.0
    )
    prepared["target_timestamp"] = prepared["timestamp"].shift(-int(horizon))
    prepared["daily_return"] = pd.to_numeric(prepared["close"], errors="coerce").pct_change()
    return prepared


def compute_smape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    actual_series = pd.to_numeric(pd.Series(actual), errors="coerce")
    predicted_series = pd.to_numeric(pd.Series(predicted), errors="coerce")
    mask = actual_series.notna() & predicted_series.notna()
    if not mask.any():
        return float("nan")
    denom = actual_series.loc[mask].abs() + predicted_series.loc[mask].abs()
    valid = denom > 0
    if not valid.any():
        return float("nan")
    return float((200.0 * (actual_series.loc[mask].loc[valid] - predicted_series.loc[mask].loc[valid]).abs() / denom.loc[valid]).mean())


def compute_mape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    actual_series = pd.to_numeric(pd.Series(actual), errors="coerce")
    predicted_series = pd.to_numeric(pd.Series(predicted), errors="coerce")
    mask = actual_series.notna() & predicted_series.notna() & (actual_series != 0)
    if not mask.any():
        return float("nan")
    return float((((actual_series.loc[mask] - predicted_series.loc[mask]).abs()) / actual_series.loc[mask].abs()).mean() * 100.0)


def compute_forecast_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Compute forecast metrics on a standardized forecast frame."""

    validated = validate_forecast_frame(frame, require_y_true=True)
    errors = validated["y_pred"] - validated["y_true"]
    directional = np.sign(validated["y_pred"]) == np.sign(validated["y_true"])
    return {
        "observations": int(len(validated)),
        "mae": float(errors.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mape": compute_mape(validated["y_true"], validated["y_pred"]),
        "smape": compute_smape(validated["y_true"], validated["y_pred"]),
        "directional_accuracy": float(directional.mean()),
        "hit_rate": float(directional.mean()),
    }


def summarize_forecasts(
    forecast_df: pd.DataFrame,
    *,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize forecast metrics by model/ticker/horizon or another grouping."""

    validated = validate_forecast_frame(forecast_df, require_y_true=True)
    group_columns = group_columns or ["model_name", "ticker", "horizon"]
    rows: list[dict[str, Any]] = []
    for keys, group in validated.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row.update(compute_forecast_metrics(group))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=[*(group_columns or []), "observations", "mae", "rmse", "mape", "smape", "directional_accuracy", "hit_rate"])
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


class WalkForwardSplitter:
    """Generate explicit, non-overlapping train/test windows on a time axis."""

    def __init__(self, config: WalkForwardConfig) -> None:
        self.config = config

    def split(self, frame: pd.DataFrame) -> list[WalkForwardWindow]:
        if frame.empty:
            raise ValueError("Cannot build walk-forward windows from an empty frame")

        target_column = self.config.target_column
        eligible = frame.loc[pd.to_numeric(frame[target_column], errors="coerce").notna(), "timestamp"]
        unique_dates = pd.Index(pd.to_datetime(eligible, errors="coerce").dropna().unique()).sort_values()
        if len(unique_dates) < (self.config.train_size + self.config.gap_size + self.config.test_size):
            raise ValueError(
                "Insufficient rows for walk-forward evaluation: "
                f"dates={len(unique_dates)} train={self.config.train_size} gap={self.config.gap_size} test={self.config.test_size}"
            )

        windows: list[WalkForwardWindow] = []
        cursor = int(self.config.train_size)
        while cursor + self.config.gap_size + self.config.test_size <= len(unique_dates):
            train_start_idx = 0 if self.config.expanding_window else max(0, cursor - int(self.config.train_size))
            train_dates = unique_dates[train_start_idx:cursor]
            test_start_idx = cursor + int(self.config.gap_size)
            test_dates = unique_dates[test_start_idx : test_start_idx + int(self.config.test_size)]
            train_mask = frame["timestamp"].isin(train_dates)
            test_mask = frame["timestamp"].isin(test_dates)
            window = WalkForwardWindow(
                window_id=f"window_{len(windows) + 1:03d}",
                train_start=pd.Timestamp(train_dates.min()),
                train_end=pd.Timestamp(train_dates.max()),
                test_start=pd.Timestamp(test_dates.min()),
                test_end=pd.Timestamp(test_dates.max()),
                gap_size=int(self.config.gap_size),
                train_rows=int(train_mask.sum()),
                test_rows=int(test_mask.sum()),
            )
            windows.append(window)
            if self.config.max_windows and len(windows) >= int(self.config.max_windows):
                break
            cursor += int(self.config.step_size)

        if not windows:
            raise ValueError("No walk-forward windows were generated from the requested configuration")
        return windows


class WalkForwardEvaluator:
    """Evaluate forecast models side-by-side on a shared leakage-safe backbone."""

    def __init__(self, config: WalkForwardConfig) -> None:
        self.config = config
        self.splitter = WalkForwardSplitter(config)
        self.prepared_dir = Path(config.prepared_dir)
        self.raw_dir = Path(config.raw_dir)
        self._dataset_cache: dict[str, PreparedEvaluationData] = {}

    def _resolve_feature_columns(self, frame: pd.DataFrame) -> list[str]:
        if self.config.feature_columns is not None:
            selected = [column for column in self.config.feature_columns if column in frame.columns]
        else:
            selected = resolve_task_feature_set(
                "regression_forecasting",
                available_columns=frame.columns,
            )
            if not selected:
                selected = resolve_feature_set(
                    "forecast_core_features",
                    available_columns=frame.columns,
                )
        if self.config.feature_columns is None and not selected:
            raise ValueError("No approved forecast feature columns were available in the prepared dataset")
        return selected

    def _load_prepared_frame(self, ticker: str) -> tuple[pd.DataFrame, str]:
        prepared_path = self.prepared_dir / f"{ticker.upper()}.csv"
        if prepared_path.exists():
            frame = pd.read_csv(prepared_path, low_memory=False)
            logger.info("phase1_dataset_loaded", ticker=ticker, source="prepared_csv", path=str(prepared_path))
            return frame, "prepared_csv"

        raw_path = self.raw_dir / f"{ticker.upper()}.csv"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"No prepared or raw dataset found for {ticker}. Checked {prepared_path} and {raw_path}."
            )

        logger.info("phase1_dataset_fallback", ticker=ticker, source="raw_csv_feature_build", path=str(raw_path))
        raw_df = pd.read_csv(raw_path)
        from src.ml.trainer import DualModelTrainer

        trainer = DualModelTrainer()
        prepared = trainer.prepare_ticker_data(
            ticker=ticker,
            df=raw_df,
            max_sequence_length=1,
            window_start=pd.to_datetime(raw_df["date"], errors="coerce").min(),
            window_end=pd.to_datetime(raw_df["date"], errors="coerce").max(),
        )
        return prepared.feature_frame, "raw_csv_feature_build"

    def load_ticker_data(self, ticker: str) -> PreparedEvaluationData:
        normalized_ticker = str(ticker).upper().strip()
        cached = self._dataset_cache.get(normalized_ticker)
        if cached is not None:
            return cached
        frame, source = self._load_prepared_frame(ticker)
        prepared = frame.copy()
        if "timestamp" not in prepared.columns:
            if "date" not in prepared.columns:
                raise ValueError(f"{ticker} dataset is missing both timestamp and date columns")
            prepared = prepared.rename(columns={"date": "timestamp"})
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
        if prepared["timestamp"].isna().all():
            raise ValueError(f"{ticker} dataset does not contain usable timestamps")
        if "ticker" not in prepared.columns:
            prepared["ticker"] = ticker.upper()
        prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
        prepared = prepared[prepared["ticker"] == ticker.upper()].copy()
        if self.config.start_date:
            prepared = prepared[prepared["timestamp"] >= pd.Timestamp(self.config.start_date)]
        if self.config.end_date:
            prepared = prepared[prepared["timestamp"] <= pd.Timestamp(self.config.end_date)]
        prepared = prepared.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
        if prepared.empty:
            raise ValueError(f"{ticker} dataset is empty after date filtering")

        if not {"open", "high", "low", "close", "volume"} <= set(prepared.columns):
            raise ValueError(f"{ticker} dataset is missing required OHLCV columns")

        target_spec = build_target_spec(
            self.config.target_type,
            target_column=self.config.target_column,
            **dict(self.config.target_params or {}),
        )
        prepared = apply_target_spec(
            prepared,
            horizon=self.config.horizon,
            target_spec=target_spec,
        )
        feature_columns = self._resolve_feature_columns(prepared)
        dataset = PreparedEvaluationData(
            ticker=ticker.upper(),
            frame=prepared.reset_index(drop=True),
            feature_columns=feature_columns,
            target_column=target_spec.target_column,
            source=source,
        )
        self._dataset_cache[normalized_ticker] = dataset
        return dataset

    def build_windows(self, dataset: PreparedEvaluationData) -> list[WalkForwardWindow]:
        return self.splitter.split(dataset.frame)

    def evaluate(
        self,
        models: list[ForecastModel],
    ) -> dict[str, Any]:
        if not models:
            raise ValueError("At least one forecast model is required")

        all_predictions: list[pd.DataFrame] = []
        window_rows: list[dict[str, Any]] = []
        datasets: dict[str, PreparedEvaluationData] = {}

        for ticker in [value.upper().strip() for value in self.config.tickers]:
            dataset = self.load_ticker_data(ticker)
            datasets[ticker] = dataset
            windows = self.build_windows(dataset)
            frame = dataset.frame.copy()
            for window in windows:
                train_mask = (frame["timestamp"] >= window.train_start) & (frame["timestamp"] <= window.train_end)
                test_mask = (frame["timestamp"] >= window.test_start) & (frame["timestamp"] <= window.test_end)
                train_df = frame.loc[train_mask].copy()
                test_df = frame.loc[test_mask].copy()
                if train_df.empty or test_df.empty:
                    continue
                train_df["window_id"] = window.window_id
                test_df["window_id"] = window.window_id

                for base_model in models:
                    model = deepcopy(base_model)
                    model.fit(
                        train_df=train_df,
                        features=dataset.feature_columns,
                        target=dataset.target_column,
                        horizon=self.config.horizon,
                        config={"seed": self.config.seed, "window_id": window.window_id},
                    )
                    predictions = model.predict(test_df)
                    predictions["source"] = dataset.source
                    all_predictions.append(validate_forecast_frame(predictions))
                    window_rows.append(
                        {
                            "ticker": ticker,
                            "window_id": window.window_id,
                            "model_name": model.model_name,
                            "train_start": str(window.train_start.date()),
                            "train_end": str(window.train_end.date()),
                            "test_start": str(window.test_start.date()),
                            "test_end": str(window.test_end.date()),
                            "train_rows": int(len(train_df)),
                            "test_rows": int(len(test_df)),
                            "feature_count": int(len(dataset.feature_columns)),
                            "source": dataset.source,
                        }
                    )

        if not all_predictions:
            raise ValueError("The walk-forward evaluator did not generate any predictions")

        forecast_df = pd.concat(all_predictions, ignore_index=True)
        forecast_df = validate_forecast_frame(forecast_df, require_y_true=True)
        summary_df = summarize_forecasts(forecast_df)
        window_df = pd.DataFrame(window_rows).sort_values(["ticker", "window_id", "model_name"]).reset_index(drop=True)
        horizon_summary_df = summarize_forecasts(
            forecast_df,
            group_columns=["model_name", "horizon"],
        )
        return {
            "forecasts": forecast_df,
            "forecast_summary": summary_df,
            "forecast_summary_by_horizon": horizon_summary_df,
            "window_summary": window_df,
            "datasets": datasets,
        }
