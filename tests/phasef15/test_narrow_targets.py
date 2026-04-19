from __future__ import annotations

import pandas as pd

from src.evaluation.targets import build_target_spec, apply_target_spec


def test_narrow_scope_target_generation_remains_leakage_safe_for_horizon_ten() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=15, freq="D"),
            "close": [100.0 + float(index) for index in range(15)],
        }
    )
    target_spec = build_target_spec("forward_return")
    prepared = apply_target_spec(frame, horizon=10, target_spec=target_spec)

    expected = frame.loc[10, "close"] / frame.loc[0, "close"] - 1.0
    assert abs(float(prepared.loc[0, target_spec.target_column]) - float(expected)) < 1e-12
    assert prepared[target_spec.target_column].tail(10).isna().all()
    assert prepared["target_timestamp"].iloc[0] == frame["timestamp"].iloc[10]
