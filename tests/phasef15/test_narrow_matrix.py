from __future__ import annotations

from src.evaluation.forecast_rehab_narrow import (
    BASELINE_ONLY_MODELS,
    COMPARATOR_MODELS,
    PRIMARY_MODELS,
    build_narrow_core_frame,
    build_narrow_matrix_config,
    build_narrow_scope_table,
)


def test_narrow_scope_table_locks_expected_dimensions() -> None:
    scope = build_narrow_scope_table()
    assert set(scope["dimension"]) == {
        "ticker_group",
        "horizons",
        "feature_families",
        "model_families",
        "target_framing",
    }


def test_narrow_matrix_config_medium_is_bounded_to_small_banks() -> None:
    config = build_narrow_matrix_config("medium")
    assert [group["group_name"] for group in config["ticker_groups"]] == ["small_banks"]
    assert config["horizons"] == [5, 10]
    assert config["target_names"] == ["forward_return", "direction_binary"]
    assert config["primary_models"] == PRIMARY_MODELS
    assert config["comparator_models"] == COMPARATOR_MODELS
    assert config["baseline_only_models"] == BASELINE_ONLY_MODELS
    assert set(config["cost_modes"]) == {"baseline", "elevated"}


def test_narrow_core_frame_has_expected_medium_shape() -> None:
    config = build_narrow_matrix_config("medium")
    core = build_narrow_core_frame(config)
    assert len(core) == 16
    assert core["group_name"].nunique() == 1
    assert core["horizon"].tolist().count(5) == 8
    assert core["horizon"].tolist().count(10) == 8
