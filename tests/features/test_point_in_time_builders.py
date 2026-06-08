import pandas as pd

from src.features.builders import build_momentum_features, build_range_features


def test_feature_builders_only_use_trailing_values():
    frame = pd.DataFrame(
        {
            "asset_code": ["AAA"] * 5,
            "asof_timestamp": pd.date_range("2026-01-01", periods=5),
            "close": [10.0, 11.0, 12.0, 13.0, 1000.0],
            "high": [11.0, 12.0, 13.0, 14.0, 1001.0],
            "low": [9.0, 10.0, 11.0, 12.0, 999.0],
        }
    )
    momentum = build_momentum_features(frame, windows=(2,))
    ranges = build_range_features(frame, windows=(2,))
    assert pd.isna(momentum.loc[2, "momentum_2"])
    assert momentum.loc[3, "momentum_2"] == (12.0 / 10.0) - 1.0
    assert ranges.loc[3, "range_mean_2"] < 1.0
