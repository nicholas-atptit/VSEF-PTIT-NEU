from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.targets import apply_target_spec, build_target_spec
from src.evaluation.walkforward import WalkForwardConfig, WalkForwardEvaluator
from src.forecast.base import ForecastModel


class BoundaryCheckModel(ForecastModel):
    model_name = "boundary_check"
    requires_features = False

    def _fit_model(self, train_frame: pd.DataFrame) -> None:
        self.train_end = pd.Timestamp(train_frame["timestamp"].max())

    def _predict_values(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.train_end < pd.Timestamp(frame["timestamp"].min()):
            raise AssertionError("train/test leakage detected")
        return np.zeros(len(frame), dtype=float)


def test_apply_target_spec_forward_return_matches_formula() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="B"),
            "close": [100.0, 102.0, 101.0, 104.0, 106.0, 108.0],
        }
    )
    spec = build_target_spec("forward_return")
    result = apply_target_spec(frame, horizon=2, target_spec=spec)

    expected = frame["close"].shift(-2) / frame["close"] - 1.0
    pd.testing.assert_series_equal(
        pd.to_numeric(result["target_forward_return"], errors="coerce"),
        pd.to_numeric(expected, errors="coerce"),
        check_names=False,
    )
    assert result["target_timestamp"].iloc[0] == pd.Timestamp("2024-01-03")
    assert result["target_forward_return"].iloc[-2:].isna().all()


def test_apply_target_spec_direction_binary_is_signed_and_leak_safe() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="B"),
            "close": [100.0, 101.0, 99.0, 100.0, 103.0],
        }
    )
    spec = build_target_spec("direction_binary")
    result = apply_target_spec(frame, horizon=1, target_spec=spec)

    assert set(result["target_direction_binary"].dropna().unique()).issubset({-1.0, 0.0, 1.0})
    assert result["target_direction_binary"].iloc[-1] != result["target_direction_binary"].iloc[-1]
    assert result["target_direction_binary"].iloc[0] == 1.0
    assert result["target_direction_binary"].iloc[1] == -1.0


def test_walkforward_evaluator_supports_direction_target_without_leakage(tmp_path) -> None:
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
            feature_columns=[],
            target_column="target_direction_binary",
            target_type="direction_binary",
        )
    )
    result = evaluator.evaluate([BoundaryCheckModel(target_type="direction_binary")])

    assert not result["forecasts"].empty
    assert result["forecasts"]["target_type"].eq("direction_binary").all()
