from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.walkforward import (
    WalkForwardConfig,
    WalkForwardEvaluator,
    WalkForwardSplitter,
    _raw_ohlcv_date_bounds,
)
from src.forecast.base import ForecastModel


class BoundaryCheckModel(ForecastModel):
    model_name = "boundary_check"
    requires_features = False

    def _fit_model(self, train_frame: pd.DataFrame) -> None:
        self.train_end = pd.Timestamp(train_frame["timestamp"].max())

    def _predict_values(self, frame: pd.DataFrame) -> np.ndarray:
        test_start = pd.Timestamp(frame["timestamp"].min())
        if not self.train_end < test_start:
            raise AssertionError("train/test leakage detected")
        return np.zeros(len(frame), dtype=float)


def test_walkforward_splitter_produces_disjoint_train_and_test_windows() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=40, freq="B"),
            "target_forward_return": [0.01] * 40,
        }
    )
    config = WalkForwardConfig(
        tickers=["AAA"],
        horizon=1,
        train_size=20,
        test_size=5,
        step_size=5,
        gap_size=1,
    )
    windows = WalkForwardSplitter(config).split(frame)

    for window in windows:
        train_mask = (frame["timestamp"] >= window.train_start) & (frame["timestamp"] <= window.train_end)
        test_mask = (frame["timestamp"] >= window.test_start) & (frame["timestamp"] <= window.test_end)
        assert not set(frame.loc[train_mask, "timestamp"]).intersection(set(frame.loc[test_mask, "timestamp"]))
        assert window.train_end < window.test_start


def test_raw_ohlcv_date_bounds_accepts_time_column() -> None:
    frame = pd.DataFrame(
        {
            "time": ["2024-01-03", "2024-01-01", "bad-date"],
            "close": [10.0, 10.5, 11.0],
        }
    )

    window_start, window_end = _raw_ohlcv_date_bounds(frame)

    assert window_start == pd.Timestamp("2024-01-01")
    assert window_end == pd.Timestamp("2024-01-03")


def test_walkforward_evaluator_keeps_train_and_test_boundaries_separate(tmp_path) -> None:
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=80, freq="B"),
            "ticker": ["AAA"] * 80,
            "open": np.linspace(100, 120, 80),
            "high": np.linspace(101, 121, 80),
            "low": np.linspace(99, 119, 80),
            "close": np.linspace(100, 120, 80),
            "volume": np.linspace(1000, 2000, 80),
            "daily_return": np.linspace(-0.01, 0.02, 80),
            "close_return_1d": np.linspace(-0.01, 0.02, 80),
        }
    )
    frame.to_csv(prepared_dir / "AAA.csv", index=False)

    evaluator = WalkForwardEvaluator(
        WalkForwardConfig(
            tickers=["AAA"],
            horizon=3,
            train_size=30,
            test_size=10,
            step_size=10,
            max_windows=2,
            prepared_dir=str(prepared_dir),
            feature_columns=["daily_return"],
        )
    )
    result = evaluator.evaluate([BoundaryCheckModel()])

    assert not result["forecasts"].empty
    assert result["forecasts"]["model_name"].eq("boundary_check").all()
