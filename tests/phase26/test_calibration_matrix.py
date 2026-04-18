from __future__ import annotations

import pandas as pd

from src.evaluation.calibration import (
    build_phase26_core_frame,
    build_phase26_matrix_config,
    build_phase26_sweep_frame,
)
from src.evaluation.walkforward import WalkForwardConfig, WalkForwardSplitter


def test_medium_calibration_matrix_covers_requested_thresholds_and_ablation_labels() -> None:
    matrix = build_phase26_matrix_config("medium")
    core_frame = build_phase26_core_frame(matrix)
    sweep_frame = build_phase26_sweep_frame(matrix)

    assert len(core_frame) == 9
    assert len(sweep_frame) == 1170
    assert set(matrix["ablation_labels"]) == {"A", "B", "C", "D", "E", "F", "G", "H"}
    assert set(matrix["thresholds"]) == {0.001, 0.003, 0.005, 0.007, 0.01}
    assert set(sweep_frame["sizing_profile"]) >= {
        "fixed_fraction_full",
        "adaptive_current",
        "adaptive_capped_floor",
        "adaptive_lighter_vol",
        "adaptive_lighter_drawdown",
    }


def test_phase26_matrix_preserves_non_overlapping_walkforward_configuration() -> None:
    matrix = build_phase26_matrix_config("medium")
    evaluation = matrix["evaluation_config"]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=400, freq="B"),
            "target_forward_return": [0.01] * 400,
        }
    )
    splitter = WalkForwardSplitter(
        WalkForwardConfig(
            tickers=["AAA"],
            horizon=5,
            train_size=int(evaluation["train_size"]),
            test_size=int(evaluation["test_size"]),
            step_size=int(evaluation["step_size"]),
            gap_size=int(evaluation["gap_size"]),
            max_windows=2,
        )
    )
    windows = splitter.split(frame)

    assert len(windows) >= 2
    for window in windows:
        assert window.train_end < window.test_start
