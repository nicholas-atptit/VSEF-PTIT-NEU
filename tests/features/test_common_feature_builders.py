import pandas as pd

from src.features.builders import build_momentum_features


def test_momentum_is_lagged_and_asset_local():
    frame = pd.DataFrame(
        {
            "asset_code": ["A", "A", "A", "B", "B", "B"],
            "asof_timestamp": list(pd.date_range("2025-01-01", periods=3)) * 2,
            "close": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )
    built = build_momentum_features(frame, windows=(1,))
    assert pd.isna(built[built["asset_code"].eq("B")].iloc[0]["momentum_1"])
