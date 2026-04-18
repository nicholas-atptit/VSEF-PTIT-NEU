from __future__ import annotations

import pandas as pd

from src.evaluation.walkforward import WalkForwardConfig, WalkForwardSplitter


def test_walkforward_splitter_still_prevents_train_test_overlap() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=60, freq="B"),
            "target_forward_return": [0.01] * 60,
        }
    )
    config = WalkForwardConfig(
        tickers=["AAA"],
        horizon=1,
        train_size=20,
        test_size=10,
        step_size=10,
        gap_size=2,
        expanding_window=False,
    )
    windows = WalkForwardSplitter(config).split(frame)

    for window in windows:
        train_mask = (frame["timestamp"] >= window.train_start) & (frame["timestamp"] <= window.train_end)
        test_mask = (frame["timestamp"] >= window.test_start) & (frame["timestamp"] <= window.test_end)
        assert window.train_end < window.test_start
        assert not set(frame.loc[train_mask, "timestamp"]).intersection(set(frame.loc[test_mask, "timestamp"]))

