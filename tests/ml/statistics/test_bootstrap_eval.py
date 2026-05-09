from __future__ import annotations

import pandas as pd

from src.ml.statistics import bootstrap_from_dataframe, bootstrap_hit_ratio_ci, bootstrap_mean_ci


def test_bootstrap_mean_ci_is_reproducible() -> None:
    values = [0.01, -0.02, 0.03, 0.04, -0.01, 0.02, 0.01, -0.03, 0.05, 0.02]

    first = bootstrap_mean_ci(values, n_bootstrap=200, seed=7)
    second = bootstrap_mean_ci(values, n_bootstrap=200, seed=7)

    assert first["estimate"] == second["estimate"]
    assert first["ci_lower"] == second["ci_lower"]
    assert first["ci_upper"] == second["ci_upper"]
    assert first["sample_size"] == 10


def test_bootstrap_hit_ratio_converts_positive_outcomes() -> None:
    result = bootstrap_hit_ratio_ci([0.1, -0.2, 0.0, 0.3, 0.4], n_bootstrap=100, seed=42)

    assert result["metric_name"] == "hit_ratio"
    assert result["estimate"] == 0.6
    assert result["sample_size"] == 5
    assert "small_sample_ci_unstable" in result["warning"]


def test_bootstrap_from_dataframe_groups_rows() -> None:
    frame = pd.DataFrame(
        {
            "policy_id": ["a", "a", "b", "b"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = bootstrap_from_dataframe(frame, "value", group_by=["policy_id"], n_bootstrap=50)

    assert set(result["group_key"]) == {"policy_id=a", "policy_id=b"}
    assert set(result["sample_size"]) == {2}
