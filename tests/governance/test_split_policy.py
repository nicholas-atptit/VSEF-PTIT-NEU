import pandas as pd

from src.governance.split_policy import assign_split


def test_feature_and_target_must_share_strict_period():
    result = assign_split(
        pd.DataFrame(
            {
                "feature_timestamp": ["2023-12-31", "2024-06-01", "2024-12-31", "2025-01-01"],
                "target_timestamp": ["2024-01-01", "2024-06-02", "2025-01-01", "2025-01-02"],
            }
        )
    )
    assert result["split"].tolist() == [pd.NA, "validation", pd.NA, "final"]
